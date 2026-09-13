#!/usr/bin/env python3
"""Measure decoder-verified predictive compression on a text8 continuation."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from src.benchmark.predictive_codec import compress_and_verify
from src.models import NGramPredictor
from src.predictors import build_predictor, survey_model_specs, train_neural_predictor, train_ngram_predictor


def parse_args() -> argparse.Namespace:
    specs = survey_model_specs()
    parser = argparse.ArgumentParser(description="Train on one text8 prefix and code the following byte excerpt.")
    parser.add_argument("--input", type=Path, default=Path("data/text8"))
    parser.add_argument("--train-bytes", type=int, default=10_000)
    parser.add_argument("--compress-bytes", type=int, default=10_000)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model", action="append", choices=["all", *(s.name for s in specs)])
    return parser.parse_args()


def remap(values: list[int]) -> tuple[list[int], int]:
    ids = {value: index for index, value in enumerate(sorted(set(values)))}
    return [ids[value] for value in values], len(ids)


def rate(size: int, seconds: float) -> float:
    return size / (1024 ** 2) / seconds if seconds else float("inf")


def main() -> None:
    args = parse_args()
    if min(args.train_bytes, args.compress_bytes, args.epochs, args.batch_size) < 1:
        raise ValueError("byte counts, epochs, and batch size must be positive")
    torch.manual_seed(args.seed)
    raw = args.input.read_bytes()
    needed = args.train_bytes + args.compress_bytes
    if len(raw) < needed:
        raise ValueError("input is shorter than the requested split")
    train_raw, target_raw = raw[:args.train_bytes], raw[args.train_bytes:needed]
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", cache_dir=".cache", local_files_only=True)
    streams = [
        ("byte", list(train_raw), list(target_raw)),
        ("token", tokenizer.encode(train_raw.decode("utf-8"), add_special_tokens=False),
         tokenizer.encode(target_raw.decode("utf-8"), add_special_tokens=False)),
    ]
    chosen, results = set(args.model or ["all"]), []
    for kind, train_original, target_original in streams:
        if kind == "byte":
            train, target, vocabulary_size = train_original, target_original, 256
        else:
            mapped, vocabulary_size = remap(train_original + target_original)
            train, target = mapped[:len(train_original)], mapped[len(train_original):]
        for spec in survey_model_specs():
            if spec.symbol_kind != kind or ("all" not in chosen and spec.name not in chosen):
                continue
            model = build_predictor(spec, vocabulary_size)
            started = time.perf_counter()
            if isinstance(model, NGramPredictor):
                train_ngram_predictor(model, train)
            else:
                train_neural_predictor(model, train, epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate, seed=args.seed)
            training_seconds = time.perf_counter() - started
            coded = compress_and_verify(model, target)
            if not coded.roundtrip_valid:
                raise RuntimeError("arithmetic decoder failed round-trip validation")
            row = {
                "model": spec.name, "symbol_kind": kind, "source_bytes": len(target_raw),
                "source_symbols": len(target), "payload_bits": coded.payload_bits,
                "payload_bytes": coded.payload_bytes,
                "payload_compression_ratio": len(target_raw) / coded.payload_bytes,
                "encode_mib_per_second": rate(len(target_raw), coded.encode_seconds),
                "decode_mib_per_second": rate(len(target_raw), coded.decode_seconds),
                "training_seconds": training_seconds, "roundtrip_valid": True,
                "accounting_mode": "shared_model_payload_only",
            }
            results.append(row)
            print(json.dumps(row, sort_keys=True))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"results": results}, indent=2) + "\n")
    print("\nSummary (ratio > 1 compresses; shared-model payload only):")
    for row in results:
        print(f'{row["model"]:<30} ratio={row["payload_compression_ratio"]:.3f} encode={row["encode_mib_per_second"]:.3f} MiB/s decode={row["decode_mib_per_second"]:.3f} MiB/s')


if __name__ == "__main__":
    main()
