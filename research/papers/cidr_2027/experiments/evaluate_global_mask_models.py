#!/usr/bin/env python3
"""Evaluate CIDR token models with main.py global-mask stream accounting."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from dataclasses import asdict
from pathlib import Path
import sys
import time
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from pyroaring import BitMap
from transformers import AutoTokenizer

from src.encoding import LLMCompressor, LLMDecompressor
from src.models import NGramPredictor
from src.predictors import (
    build_predictor, expand_context_specs, survey_model_specs,
    train_neural_predictor, train_ngram_predictor,
)
from src.result_schema import (
    SCHEMA_NAME,
    SCHEMA_VERSION,
    CoderSpec,
    DatasetSpec,
    PlainTextCompressionResult,
    PredictorSpec,
    SizeBreakdown,
    SymbolCounts,
    TimingBreakdown,
    TokenizerSpec,
    local_execution_spec,
)
from src.utils import save_global_mask_file


TOKEN_SPECS = tuple(spec for spec in survey_model_specs() if spec.symbol_kind == "token")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/text8"))
    parser.add_argument("--target-tokens", type=int, default=100_000)
    parser.add_argument("--training-tokens", type=int, default=100_000)
    parser.add_argument("--training-mode", choices=("disjoint", "same_data"), action="append")
    parser.add_argument("--model", choices=[spec.name for spec in TOKEN_SPECS], action="append")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument(
        "--batch-size", dest="batch_sizes", type=int, action="append",
        help="Inference batch size; repeat to define a sweep (default: 128).",
    )
    parser.add_argument(
        "--context-length", dest="context_lengths", type=int, action="append",
        help=(
            "Neural predictor context; repeat to define a sweep. N-gram context "
            "remains fixed by order (default: contexts in the model catalog)."
        ),
    )
    parser.add_argument("--training-batch-size", type=int, default=128)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--warmups", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/papers/cidr-2027/model-survey/global-mask-text8"))
    parser.add_argument("--model-dir", type=Path, help="Root for trained checkpoints; model-family directories are created beneath it.")
    return parser.parse_args()


def token_prefix(path: Path, count: int) -> tuple[list[int], bytes, object]:
    raw = path.read_bytes()[: count * 6]
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", cache_dir=".cache", local_files_only=True)
    tokenizer.model_max_length = 10**9
    tokens = tokenizer.encode(raw.decode("utf-8"), add_special_tokens=False)
    if len(tokens) < count:
        raise ValueError("input does not contain enough tokens")
    selected = tokens[:count]
    decoded = tokenizer.decode(selected, skip_special_tokens=False, clean_up_tokenization_spaces=False).encode("utf-8")
    return selected, decoded, tokenizer


def split_contiguous(tokens: list[int], batch_size: int) -> list[list[int]]:
    if batch_size < 1 or batch_size > len(tokens):
        raise ValueError("batch_size must be between 1 and target token count")
    base, extra = divmod(len(tokens), batch_size)
    batches, start = [], 0
    for index in range(batch_size):
        end = start + base + (index < extra)
        batches.append(tokens[start:end])
        start = end
    return batches


def probabilities(model, contexts: list[list[int]]):
    """Score one decoder step without constructing an autograd graph."""
    with torch.inference_mode():
        logits = model.logits(contexts)
        return torch.softmax(logits.float(), dim=-1).cpu().numpy()


def save_checkpoint(model, spec, path: Path, *, mode: str, token_ids: list[int], target_tokens: int) -> int:
    """Persist exactly the shared decoder state and return its byte length."""
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "format_version": 1,
        "model_spec": asdict(spec),
        "training_mode": mode,
        "target_tokens": target_tokens,
        "vocabulary_token_ids": token_ids,
    }
    if isinstance(model, NGramPredictor):
        path = path.with_suffix(".pkl")
        path.write_bytes(pickle.dumps({
            **metadata, "kind": "ngram", "counts": dict(model._counts),
        }, protocol=5))
    else:
        path = path.with_suffix(".pt")
        torch.save({**metadata, "kind": "torch", "state_dict": model.state_dict()}, path)
    return path.stat().st_size


def train_data(mode: str, target: list[int], disjoint: list[int], allowed: set[int]) -> tuple[list[int], int]:
    if mode == "same_data":
        return target, 0
    # The arithmetic alphabet is target-only, exactly as main.py's global mask.
    # Dropping outside-alphabet symbols avoids injecting an unencodable UNK symbol.
    filtered = [token for token in disjoint if token in allowed]
    if len(filtered) < 2:
        raise ValueError("disjoint training has fewer than two target-alphabet tokens")
    return filtered, len(disjoint) - len(filtered)


def _model_input_tokens(model, contexts: list[list[int]]) -> int:
    """Count symbols actually presented to one predictor invocation."""
    if isinstance(model, NGramPredictor):
        return sum(min(len(context), model.context_length) for context in contexts)
    # WindowModel materializes a dense, left-padded fixed-width tensor.
    return len(contexts) * model.context_length


def run_stream(model, batches: list[list[int]], *, label: str) -> tuple[list[int], list[list[int]], float, float, int]:
    encoder = LLMCompressor(algorithm="AC", alphabet_size=model.vocabulary_size)
    started = time.perf_counter()
    steps = max(len(batch) for batch in batches) - 1
    report_every = max(1, steps // 10)
    encode_model_input_tokens = 0
    print(f"  Encoding {steps:,} synchronized steps...", flush=True)
    for step in range(steps):
        active = [batch for batch in batches if step + 1 < len(batch)]
        contexts = [batch[max(0, step + 1 - model.context_length):step + 1] for batch in active]
        encode_model_input_tokens += _model_input_tokens(model, contexts)
        scores = probabilities(model, contexts)
        for batch, probs in zip(active, scores):
            encoder.next_token(batch[step + 1], probs)
        if (step + 1) % report_every == 0 or step + 1 == steps:
            print(f"  {label}: encoded {step + 1:,}/{steps:,} steps", flush=True)
    bits = encoder.compress()
    encode_seconds = time.perf_counter() - started

    decoder = LLMDecompressor(bits, algorithm="AC", alphabet_size=model.vocabulary_size)
    decoded = [[batch[0]] for batch in batches]
    decode_model_input_tokens = 0
    started = time.perf_counter()
    print(f"  Decoding {steps:,} synchronized steps for round-trip verification...", flush=True)
    for step in range(steps):
        active = [(index, batch) for index, batch in enumerate(batches) if step + 1 < len(batch)]
        contexts = [decoded[index][-model.context_length:] for index, _ in active]
        decode_model_input_tokens += _model_input_tokens(model, contexts)
        scores = probabilities(model, contexts)
        for (index, _), probs in zip(active, scores):
            decoded[index].append(decoder.decompress(probs))
        if (step + 1) % report_every == 0 or step + 1 == steps:
            print(f"  {label}: decoded {step + 1:,}/{steps:,} steps", flush=True)
    decode_seconds = time.perf_counter() - started
    if decoded != batches:
        raise AssertionError("decoder token round trip failed")
    if encode_model_input_tokens != decode_model_input_tokens:
        raise AssertionError("encoder and decoder performed different predictor work")
    return bits, decoded, encode_seconds, decode_seconds, encode_model_input_tokens


def main() -> None:
    args = parse_args()
    batch_sizes = args.batch_sizes or [128]
    if any(size < 1 or size > args.target_tokens for size in batch_sizes):
        raise ValueError("batch sizes must be between 1 and target token count")
    if len(set(batch_sizes)) != len(batch_sizes):
        raise ValueError("batch sizes must not contain duplicates")
    if args.training_batch_size < 1:
        raise ValueError("training batch size must be positive")
    if args.repetitions < 1 or args.warmups < 0:
        raise ValueError("repetitions must be positive and warmups nonnegative")
    modes = args.training_mode or ["disjoint", "same_data"]
    requested = [spec for spec in TOKEN_SPECS if not args.model or spec.name in args.model]
    selected = list(expand_context_specs(requested, args.context_lengths))
    total = args.target_tokens + (args.training_tokens if "disjoint" in modes else 0)
    print(f"Loading and tokenizing enough text8 for {total:,} tokens...", flush=True)
    original, _, tokenizer = token_prefix(args.input, total)
    target_original = original[:args.target_tokens]
    target_bytes = tokenizer.decode(target_original, skip_special_tokens=False, clean_up_tokenization_spaces=False).encode("utf-8")
    allowed_ids = sorted(set(target_original))
    dense = {token: index for index, token in enumerate(allowed_ids)}
    target = [dense[token] for token in target_original]
    disjoint_original = original[args.target_tokens:args.target_tokens + args.training_tokens]
    bitmap = BitMap(allowed_ids).serialize()
    conditions = len(modes) * len(selected) * len(batch_sizes) * args.repetitions
    print(
        f"Target: {len(target_original):,} tokens, {len(target_bytes):,} bytes; "
        f"bitmap: {len(bitmap):,} bytes; measured conditions: {conditions}",
        flush=True,
    )
    rows = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_root = args.model_dir or Path("artifacts/models")

    for mode in modes:
        raw_training, dropped = train_data(mode, target_original, disjoint_original, set(allowed_ids))
        training = [dense[token] for token in raw_training]
        for spec in selected:
            print(f"[{mode}] {spec.name}: constructing and training...", flush=True)
            torch.manual_seed(args.seed)
            model = build_predictor(spec, len(allowed_ids))
            started = time.perf_counter()
            if isinstance(model, NGramPredictor):
                train_ngram_predictor(model, training)
            else:
                train_neural_predictor(
                    model, training, epochs=args.epochs,
                    batch_size=args.training_batch_size,
                    learning_rate=args.learning_rate, seed=args.seed,
                )
            training_seconds = time.perf_counter() - started
            print(f"[{mode}] {spec.name}: training completed in {training_seconds:.2f}s", flush=True)
            checkpoint_dir = model_root / spec.family / f"text8-global-mask-n{args.target_tokens}"
            checkpoint_base = checkpoint_dir / f"{mode}-{spec.name}"
            model_bytes = save_checkpoint(model, spec, checkpoint_base, mode=mode, token_ids=allowed_ids, target_tokens=args.target_tokens)
            checkpoint_path = checkpoint_base.with_suffix(".pkl" if isinstance(model, NGramPredictor) else ".pt")
            print(f"[{mode}] {spec.name}: saved decoder checkpoint to {checkpoint_path}", flush=True)
            total_parameters = (
                sum(parameter.numel() for parameter in model.parameters())
                if isinstance(model, torch.nn.Module) else 0
            )
            for batch_size in batch_sizes:
                target_batches = split_contiguous(target, batch_size)
                offsets = [
                    sum(len(batch) for batch in target_batches[:index])
                    for index in range(len(target_batches))
                ]
                seed_tokens = [target_original[offset] for offset in offsets]
                for warmup in range(args.warmups):
                    run_stream(
                        model, target_batches,
                        label=f"[{mode}] {spec.name} b{batch_size} warmup{warmup}",
                    )
                for repetition in range(args.repetitions):
                    label = f"[{mode}] {spec.name} b{batch_size} r{repetition}"
                    bits, _, encode_seconds, decode_seconds, model_input_tokens = run_stream(
                        model, target_batches, label=label
                    )
                    stream_path = args.output_dir / f"{mode}-{spec.name}-b{batch_size}-r{repetition}.bin"
                    stream_args = SimpleNamespace(
                        input_path=str(args.input), output_path=str(stream_path),
                        model_name=f"survey/{spec.name}", context_length=spec.context_length,
                        first_n_tokens=args.target_tokens, retain_tokens=spec.context_length,
                        use_kv_cache=False, batch_size=batch_size, encoding="AC",
                        reduce_tokens=True, engine="survey", lora_path=None,
                        pmatic_delta=None, pmatic_r=None,
                    )
                    save_global_mask_file(stream_args, seed_tokens, bits, bitmap)
                    stored_bytes = stream_path.stat().st_size
                    payload_bytes = (len(bits) + 7) // 8
                    seed_bytes = 4 * len(seed_tokens)
                    framing_bytes = stored_bytes - len(bitmap) - payload_bytes - seed_bytes
                    if framing_bytes < 0:
                        raise AssertionError("stream components exceed stored stream size")
                    row = PlainTextCompressionResult(
                        dataset=DatasetSpec(
                            name=args.input.name,
                            path=str(args.input),
                            split="target",
                            sha256=hashlib.sha256(target_bytes).hexdigest(),
                            source_bytes=len(target_bytes),
                        ),
                        tokenizer=TokenizerSpec(
                            name="Qwen/Qwen2.5-0.5B",
                            kind="pretrained_bpe",
                            vocabulary_size=len(tokenizer),
                            state_bytes=None,
                        ),
                        predictor=PredictorSpec(
                            name=spec.name,
                            family=spec.family,
                            context_length=spec.context_length,
                            dtype=(str(next(model.parameters()).dtype) if isinstance(model, torch.nn.Module) else None),
                            total_parameters=total_parameters,
                            active_parameters=total_parameters,
                            model_state_bytes=model_bytes,
                            training_mode=mode,
                            training_batch_size=args.training_batch_size,
                            training_epochs=args.epochs,
                            learning_rate=args.learning_rate,
                        ),
                        coder=CoderSpec(name="AC", probability_total=262_144),
                        execution=local_execution_spec(batch_size=batch_size, seed=args.seed),
                        counts=SymbolCounts(
                            input_symbols=len(target_original),
                            encoded_symbols=len(target_original) - len(seed_tokens),
                            model_input_tokens=model_input_tokens,
                        ),
                        sizes=SizeBreakdown(
                            payload_bits=len(bits),
                            payload_bytes=payload_bytes,
                            framing_bytes=framing_bytes,
                            bitmap_bytes=len(bitmap),
                            seed_bytes=seed_bytes,
                            tokenizer_bytes=None,
                            model_bytes=model_bytes,
                            adapter_bytes=0,
                        ),
                        timings=TimingBreakdown(
                            training_seconds=training_seconds,
                            encode_seconds=encode_seconds,
                            decode_seconds=decode_seconds,
                        ),
                        roundtrip_valid=True,
                        repetition=repetition,
                        artifacts={
                            "stream": str(stream_path),
                            "model_checkpoint": str(checkpoint_path),
                        },
                        notes={
                            "training_symbols": len(training),
                            "disjoint_training_tokens_dropped_for_bitmap": dropped,
                            "bitmap_policy": "global bitmap of target token IDs",
                        },
                    ).to_dict()
                    rows.append(row)
                    print(json.dumps(row, sort_keys=True))
    result = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "target_policy": "first target_tokens from input",
        "bitmap_policy": "global bitmap of target token IDs, matching main.py reduce_tokens",
        "framing_policy": "src.utils.save_global_mask_file",
        "matrix": {
            "models": [spec.name for spec in selected],
            "neural_context_lengths": args.context_lengths,
            "batch_sizes": batch_sizes,
            "repetitions": args.repetitions,
            "warmups": args.warmups,
        },
        "training": {
            "batch_size": args.training_batch_size,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
        },
        "results": rows,
    }
    (args.output_dir / "results.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
