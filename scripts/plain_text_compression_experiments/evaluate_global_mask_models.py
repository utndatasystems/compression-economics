#!/usr/bin/env python3
"""Evaluate CIDR token models with main.py global-mask stream accounting."""

from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import asdict
from pathlib import Path
import time
from types import SimpleNamespace

import torch
from pyroaring import BitMap
from transformers import AutoTokenizer

from src.encoding import LLMCompressor, LLMDecompressor
from src.models import NGramPredictor
from src.predictors import (
    build_predictor, survey_model_specs, train_neural_predictor, train_ngram_predictor,
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
    parser.add_argument("--batch-size", type=int, default=128)
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


def run_stream(model, batches: list[list[int]], *, label: str) -> tuple[list[int], list[list[int]], float, float]:
    encoder = LLMCompressor(algorithm="AC", alphabet_size=model.vocabulary_size)
    started = time.perf_counter()
    steps = max(len(batch) for batch in batches) - 1
    report_every = max(1, steps // 10)
    print(f"  Encoding {steps:,} synchronized steps...", flush=True)
    for step in range(steps):
        active = [batch for batch in batches if step + 1 < len(batch)]
        contexts = [batch[max(0, step + 1 - model.context_length):step + 1] for batch in active]
        scores = probabilities(model, contexts)
        for batch, probs in zip(active, scores):
            encoder.next_token(batch[step + 1], probs)
        if (step + 1) % report_every == 0 or step + 1 == steps:
            print(f"  {label}: encoded {step + 1:,}/{steps:,} steps", flush=True)
    bits = encoder.compress()
    encode_seconds = time.perf_counter() - started

    decoder = LLMDecompressor(bits, algorithm="AC", alphabet_size=model.vocabulary_size)
    decoded = [[batch[0]] for batch in batches]
    started = time.perf_counter()
    print(f"  Decoding {steps:,} synchronized steps for round-trip verification...", flush=True)
    for step in range(steps):
        active = [(index, batch) for index, batch in enumerate(batches) if step + 1 < len(batch)]
        contexts = [decoded[index][-model.context_length:] for index, _ in active]
        scores = probabilities(model, contexts)
        for (index, _), probs in zip(active, scores):
            decoded[index].append(decoder.decompress(probs))
        if (step + 1) % report_every == 0 or step + 1 == steps:
            print(f"  {label}: decoded {step + 1:,}/{steps:,} steps", flush=True)
    decode_seconds = time.perf_counter() - started
    if decoded != batches:
        raise AssertionError("decoder token round trip failed")
    return bits, decoded, encode_seconds, decode_seconds


def main() -> None:
    args = parse_args()
    modes = args.training_mode or ["disjoint", "same_data"]
    selected = [spec for spec in TOKEN_SPECS if not args.model or spec.name in args.model]
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
    print(f"Target: {len(target_original):,} tokens, {len(target_bytes):,} bytes; bitmap: {len(bitmap):,} bytes; conditions: {len(modes) * len(selected)}", flush=True)
    target_batches = split_contiguous(target, args.batch_size)
    seed_tokens = [target_original[sum(len(batch) for batch in target_batches[:index])] for index in range(len(target_batches))]
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
                train_neural_predictor(model, training, epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate, seed=args.seed)
            training_seconds = time.perf_counter() - started
            print(f"[{mode}] {spec.name}: training completed in {training_seconds:.2f}s", flush=True)
            checkpoint_dir = model_root / spec.family / f"text8-global-mask-n{args.target_tokens}"
            checkpoint_base = checkpoint_dir / f"{mode}-{spec.name}"
            model_bytes = save_checkpoint(model, spec, checkpoint_base, mode=mode, token_ids=allowed_ids, target_tokens=args.target_tokens)
            checkpoint_path = checkpoint_base.with_suffix(".pkl" if isinstance(model, NGramPredictor) else ".pt")
            print(f"[{mode}] {spec.name}: saved decoder checkpoint to {checkpoint_path}", flush=True)
            bits, _, encode_seconds, decode_seconds = run_stream(model, target_batches, label=f"[{mode}] {spec.name}")
            stream_path = args.output_dir / f"{mode}-{spec.name}.bin"
            stream_args = SimpleNamespace(
                input_path=str(args.input), output_path=str(stream_path),
                model_name=f"cidr/{spec.name}", context_length=spec.context_length,
                first_n_tokens=args.target_tokens, retain_tokens=spec.context_length,
                use_kv_cache=False, batch_size=args.batch_size, encoding="AC",
                reduce_tokens=True, engine="survey", lora_path=None,
                pmatic_delta=None, pmatic_r=None,
            )
            save_global_mask_file(stream_args, seed_tokens, bits, bitmap)
            stored_bytes = stream_path.stat().st_size
            source_mib = len(target_bytes) / (1024 ** 2)
            row = {
                "model": spec.name, "training_mode": mode,
                "target_tokens": len(target_original), "source_bytes": len(target_bytes),
                "global_bitmap_bytes": len(bitmap), "payload_bits": len(bits),
                "payload_bytes": (len(bits) + 7) // 8, "stream_framing_bytes": stored_bytes - len(bitmap) - (len(bits) + 7) // 8,
                "stored_stream_bytes": stored_bytes, "model_state_bytes": model_bytes,
                "shared_model_compression_factor": len(target_bytes) / stored_bytes,
                "self_contained_compression_factor": len(target_bytes) / (stored_bytes + model_bytes),
                "training_seconds": training_seconds, "encode_seconds": encode_seconds,
                "decode_seconds": decode_seconds,
                "encode_mib_per_second": source_mib / encode_seconds,
                "decode_mib_per_second": source_mib / decode_seconds,
                "train_plus_encode_mib_per_second": source_mib / (training_seconds + encode_seconds),
                "disjoint_training_tokens_dropped_for_bitmap": dropped,
                "roundtrip_valid": True, "accounting_mode": "global_bitmap_framed",
                "stream": str(stream_path),
                "model_checkpoint": str(checkpoint_path),
            }
            rows.append(row)
            print(json.dumps(row, sort_keys=True))
    result = {
        "schema_version": 1,
        "target_policy": "first target_tokens from input",
        "bitmap_policy": "global bitmap of target token IDs, matching main.py reduce_tokens",
        "framing_policy": "src.utils.save_global_mask_file",
        "results": rows,
    }
    (args.output_dir / "results.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
