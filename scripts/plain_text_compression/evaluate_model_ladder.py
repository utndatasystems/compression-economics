#!/usr/bin/env python3
"""Benchmark pinned pretrained causal models using canonical result schema v1."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from pyroaring import BitMap
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.encoding import LLMCompressor, LLMDecompressor
from src.model_ladder import DEFAULT_CATALOG, ModelLadderEntry, load_model_ladder
from src.result_schema import (
    SCHEMA_NAME, SCHEMA_VERSION, CoderSpec, DatasetSpec,
    PlainTextCompressionResult, PredictorSpec, SizeBreakdown, SymbolCounts,
    TimingBreakdown, TokenizerSpec, local_execution_spec,
)
from src.utils import save_global_mask_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--input", type=Path, default=Path("data/text8"))
    parser.add_argument(
        "--input-bytes", type=int, default=4096,
        help="Common UTF-8 source-byte prefix supplied to every tokenizer.",
    )
    parser.add_argument("--model", action="append", help="Catalog name; repeat to select rungs.")
    parser.add_argument(
        "--include-optional", action="store_true",
        help="Select every rung when --model is omitted, including multi-GiB models.",
    )
    parser.add_argument("--context-length", type=int, action="append", dest="context_lengths")
    parser.add_argument("--batch-size", type=int, action="append", dest="batch_sizes")
    parser.add_argument("--warmups", type=int, default=0)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("artifacts/papers/cidr-2027/model-survey/model-ladder"),
    )
    return parser.parse_args()


def common_utf8_prefix(path: Path, byte_limit: int) -> tuple[str, bytes]:
    """Return at most byte_limit bytes without replacing a partial UTF-8 codepoint."""
    if byte_limit <= 0:
        raise ValueError("input-bytes must be positive")
    raw = path.read_bytes()[:byte_limit]
    while raw:
        try:
            return raw.decode("utf-8"), raw
        except UnicodeDecodeError as error:
            if error.start == 0:
                raise ValueError("input does not begin with valid UTF-8") from error
            raw = raw[:error.start]
    raise ValueError("input-byte prefix is empty")


def split_contiguous(tokens: list[int], batch_size: int) -> list[list[int]]:
    if batch_size < 1 or batch_size > len(tokens):
        raise ValueError("batch size must be between 1 and the model's input-symbol count")
    width, extra = divmod(len(tokens), batch_size)
    batches, start = [], 0
    for index in range(batch_size):
        end = start + width + (index < extra)
        batches.append(tokens[start:end])
        start = end
    return batches


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            requested = "cuda"
        elif torch.backends.mps.is_available():
            requested = "mps"
        else:
            requested = "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is unavailable")
    return torch.device(requested)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class CausalPredictor:
    """Deterministic next-token scorer shared by encoding and decoding."""

    def __init__(
        self, entry: ModelLadderEntry, tokenizer, allowed_ids: list[int], device: torch.device,
        *, local_files_only: bool,
    ) -> None:
        common = {
            "cache_dir": ".cache", "local_files_only": local_files_only,
            "trust_remote_code": entry.trust_remote_code,
        }
        self.tokenizer = tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(
            entry.model_id, revision=entry.revision, dtype="auto", **common
        ).eval().to(device)
        self.device = device
        self.family = entry.family
        self.allowed_ids = allowed_ids
        self.index = torch.tensor(allowed_ids, dtype=torch.long, device=device)
        self.dtype = str(next(self.model.parameters()).dtype).removeprefix("torch.")
        self.total_parameters = sum(parameter.numel() for parameter in self.model.parameters())

    def probabilities(self, contexts: list[list[int]]) -> tuple[object, int]:
        max_length = max(map(len, contexts))
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else 0
        inputs = torch.full(
            (len(contexts), max_length), pad_id, dtype=torch.long, device=self.device
        )
        mask = torch.zeros_like(inputs)
        lengths = []
        for row, context in enumerate(contexts):
            length = len(context)
            lengths.append(length)
            inputs[row, :length] = torch.tensor(context, dtype=torch.long, device=self.device)
            mask[row, :length] = 1
        with torch.inference_mode():
            model_args = {"input_ids": inputs, "use_cache": False}
            if self.family != "ssm":
                model_args["attention_mask"] = mask
            logits = self.model(**model_args).logits
            rows = torch.arange(len(contexts), device=self.device)
            last_logits = logits[rows, torch.tensor(lengths, device=self.device) - 1]
            scores = last_logits.index_select(-1, self.index)
            probabilities = torch.softmax(scores.float(), dim=-1).cpu().numpy()
        # Dense padded positions are the model tokens actually submitted.
        return probabilities, len(contexts) * max_length


def run_roundtrip(
    predictor: CausalPredictor, tokens: list[int], *, batch_size: int,
    context_length: int,
) -> tuple[list[int], float, float, int]:
    batches = split_contiguous(tokens, batch_size)
    dense_ids = {token: index for index, token in enumerate(predictor.allowed_ids)}
    compressor = LLMCompressor(algorithm="AC", alphabet_size=len(dense_ids))
    encode_work = 0
    synchronize(predictor.device)
    started = time.perf_counter()
    for step in range(max(map(len, batches)) - 1):
        active = [batch for batch in batches if step + 1 < len(batch)]
        contexts = [batch[max(0, step + 1 - context_length):step + 1] for batch in active]
        probabilities, work = predictor.probabilities(contexts)
        encode_work += work
        for batch, probs in zip(active, probabilities):
            compressor.next_token(dense_ids[batch[step + 1]], probs)
    bits = compressor.compress()
    synchronize(predictor.device)
    encode_seconds = time.perf_counter() - started

    decompressor = LLMDecompressor(bits, algorithm="AC", alphabet_size=len(dense_ids))
    decoded = [[batch[0]] for batch in batches]
    decode_work = 0
    synchronize(predictor.device)
    started = time.perf_counter()
    for step in range(max(map(len, batches)) - 1):
        active = [index for index, batch in enumerate(batches) if step + 1 < len(batch)]
        contexts = [decoded[index][-context_length:] for index in active]
        probabilities, work = predictor.probabilities(contexts)
        decode_work += work
        for index, probs in zip(active, probabilities):
            decoded[index].append(predictor.allowed_ids[decompressor.decompress(probs)])
    synchronize(predictor.device)
    decode_seconds = time.perf_counter() - started
    reconstructed = [token for batch in decoded for token in batch]
    if reconstructed != tokens:
        raise AssertionError("model-ladder token round trip failed")
    if encode_work != decode_work:
        raise AssertionError("encoder and decoder submitted different model-token work")
    return bits, encode_seconds, decode_seconds, encode_work


def main() -> None:
    args = parse_args()
    catalog = load_model_ladder(args.catalog)
    selected = catalog.select(args.model, include_optional=args.include_optional)
    if not selected:
        raise ValueError("model selection is empty")
    contexts = args.context_lengths or [128]
    batch_sizes = args.batch_sizes or [32]
    if any(value <= 0 for value in contexts + batch_sizes):
        raise ValueError("context lengths and batch sizes must be positive")
    if len(set(contexts)) != len(contexts) or len(set(batch_sizes)) != len(batch_sizes):
        raise ValueError("context lengths and batch sizes must not contain duplicates")
    if args.warmups < 0 or args.repetitions < 1:
        raise ValueError("warmups must be nonnegative and repetitions positive")
    text, source_bytes = common_utf8_prefix(args.input, args.input_bytes)
    source_hash = hashlib.sha256(source_bytes).hexdigest()
    device = resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for entry in selected:
        print(f"[{entry.name}] loading {entry.model_id}@{entry.revision[:12]}", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(
            entry.tokenizer_id, revision=entry.tokenizer_revision, cache_dir=".cache",
            local_files_only=args.local_files_only, trust_remote_code=entry.trust_remote_code,
        )
        tokenization_started = time.perf_counter()
        tokens = tokenizer.encode(text, add_special_tokens=False)
        tokenization_seconds = time.perf_counter() - tokenization_started
        restored = tokenizer.decode(tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        if restored.encode("utf-8") != source_bytes:
            raise ValueError(f"{entry.name} tokenizer does not round-trip the common byte prefix")
        allowed_ids = sorted(set(tokens))
        bitmap = BitMap(allowed_ids).serialize()
        predictor = CausalPredictor(
            entry, tokenizer, allowed_ids, device, local_files_only=args.local_files_only
        )
        if predictor.total_parameters != entry.total_parameters:
            raise ValueError(
                f"{entry.name} catalog has {entry.total_parameters} parameters but "
                f"the pinned runtime model materialized {predictor.total_parameters}"
            )
        for context_length in contexts:
            for batch_size in batch_sizes:
                if batch_size > len(tokens):
                    raise ValueError(f"batch size {batch_size} exceeds {entry.name}'s {len(tokens)} tokens")
                for _ in range(args.warmups):
                    run_roundtrip(predictor, tokens, batch_size=batch_size, context_length=context_length)
                for repetition in range(args.repetitions):
                    bits, encode_seconds, decode_seconds, encode_work = run_roundtrip(
                        predictor, tokens, batch_size=batch_size, context_length=context_length
                    )
                    stem = f"{entry.name}-c{context_length}-b{batch_size}-r{repetition}"
                    stream_path = args.output_dir / f"{stem}.bin"
                    seed_tokens = [batch[0] for batch in split_contiguous(tokens, batch_size)]
                    stream_args = SimpleNamespace(
                        input_path=str(args.input), output_path=str(stream_path),
                        model_name=entry.model_id, model_revision=entry.revision,
                        tokenizer_name=entry.tokenizer_id,
                        tokenizer_revision=entry.tokenizer_revision,
                        trust_remote_code=entry.trust_remote_code,
                        context_length=context_length, first_n_tokens=len(tokens),
                        retain_tokens=context_length, use_kv_cache=False,
                        batch_size=batch_size, encoding="AC", reduce_tokens=True,
                        engine="transformer", lora_path=None, pmatic_delta=None, pmatic_r=None,
                    )
                    save_global_mask_file(stream_args, seed_tokens, bits, bitmap)
                    payload_bytes = math.ceil(len(bits) / 8)
                    seed_bytes = 4 * batch_size
                    framing_bytes = stream_path.stat().st_size - payload_bytes - seed_bytes - len(bitmap)
                    row = PlainTextCompressionResult(
                        dataset=DatasetSpec(
                            name=args.input.name, path=str(args.input), split="byte_prefix",
                            sha256=source_hash, source_bytes=len(source_bytes),
                        ),
                        tokenizer=TokenizerSpec(
                            name=entry.tokenizer_id, kind="pretrained", vocabulary_size=len(tokenizer),
                            revision=entry.tokenizer_revision, state_bytes=None,
                        ),
                        predictor=PredictorSpec(
                            name=entry.name, family=entry.family, context_length=context_length,
                            revision=entry.revision, dtype=predictor.dtype,
                            total_parameters=entry.total_parameters,
                            active_parameters=entry.active_parameters, model_state_bytes=None,
                            training_mode="pretrained",
                        ),
                        coder=CoderSpec(name="AC", probability_total=262144),
                        execution=replace(
                            local_execution_spec(batch_size=batch_size, seed=args.seed),
                            device=device.type, precision=predictor.dtype,
                        ),
                        counts=SymbolCounts(
                            input_symbols=len(tokens), encoded_symbols=len(tokens) - batch_size,
                            model_input_tokens=encode_work,
                        ),
                        sizes=SizeBreakdown(
                            payload_bits=len(bits), payload_bytes=payload_bytes,
                            framing_bytes=framing_bytes, bitmap_bytes=len(bitmap), seed_bytes=seed_bytes,
                            tokenizer_bytes=None, model_bytes=None, adapter_bytes=0,
                        ),
                        timings=TimingBreakdown(
                            tokenization_seconds=tokenization_seconds,
                            encode_seconds=encode_seconds, decode_seconds=decode_seconds,
                        ),
                        roundtrip_valid=True, repetition=repetition,
                        artifacts={"stream": str(stream_path)},
                        notes={
                            "architecture": entry.architecture,
                            "runtime_materialized_parameters": predictor.total_parameters,
                            "parameter_count_policy": catalog.parameter_count_policy,
                            "source_selection": "common UTF-8 byte prefix",
                        },
                    ).to_dict()
                    rows.append(row)
                    print(json.dumps(row, sort_keys=True), flush=True)
        del predictor
        if device.type == "cuda":
            torch.cuda.empty_cache()

    output = {
        "schema_name": SCHEMA_NAME, "schema_version": SCHEMA_VERSION,
        "catalog": str(args.catalog), "catalog_version": catalog.catalog_version,
        "source_policy": "same UTF-8 input-byte prefix for every tokenizer",
        "matrix": {
            "models": [entry.name for entry in selected], "context_lengths": contexts,
            "batch_sizes": batch_sizes, "warmups": args.warmups,
            "repetitions": args.repetitions,
        },
        "results": rows,
    }
    (args.output_dir / "results.json").write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
