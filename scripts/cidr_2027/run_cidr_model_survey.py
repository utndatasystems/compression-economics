#!/usr/bin/env python3
"""Train and score the compact CIDR model survey on a contiguous text excerpt."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from transformers import AutoTokenizer
import torch

from src.models import NGramPredictor
from src.predictors import (
    bits_per_symbol,
    build_predictor,
    survey_model_specs,
    train_neural_predictor,
    train_ngram_predictor,
)


def arguments() -> argparse.Namespace:
    specs = survey_model_specs()
    parser = argparse.ArgumentParser(
        description="Train CIDR survey predictors and report held-out bits/symbol."
    )
    parser.add_argument("--input", type=Path, default=Path("data/text8"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--train-symbols", type=int, default=2_048)
    parser.add_argument("--eval-symbols", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument(
        "--model", action="append", choices=["all", *(spec.name for spec in specs)],
        help="Repeat to select models; defaults to all.",
    )
    return parser.parse_args()


def remap(symbols: list[int]) -> tuple[list[int], int]:
    """Create a compact fixed alphabet known to the evaluator."""
    alphabet = sorted(set(symbols))
    lookup = {symbol: index for index, symbol in enumerate(alphabet)}
    return [lookup[symbol] for symbol in symbols], len(alphabet)


def token_symbols(source: bytes, count: int) -> list[int]:
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-0.5B", cache_dir=".cache", local_files_only=True
    )
    # text8 is UTF-8/ASCII. Decoding here is only for token-model evaluation;
    # byte models consume source directly.
    return tokenizer.encode(source.decode("utf-8"), add_special_tokens=False)[:count]


def main() -> None:
    args = arguments()
    if min(args.train_symbols, args.eval_symbols, args.epochs, args.batch_size) < 1:
        raise ValueError("train/eval symbols, epochs, and batch size must be positive")
    torch.manual_seed(args.seed)
    source = args.input.read_bytes()
    requested = set(args.model or ["all"])
    specs = [
        spec for spec in survey_model_specs()
        if "all" in requested or spec.name in requested
    ]
    needed = args.train_symbols + args.eval_symbols
    byte_data = list(source[:needed])
    if len(byte_data) < needed:
        raise ValueError("input is shorter than the requested byte excerpt")
    token_data = token_symbols(source[: needed * 16], needed)
    if len(token_data) < needed:
        raise ValueError("input is shorter than the requested token excerpt")

    results = []
    for kind, data in (("byte", byte_data), ("token", token_data)):
        mapped, vocabulary_size = remap(data)
        if kind == "byte":
            # Byte models retain the full 256-symbol alphabet, rather than the
            # observed excerpt alphabet, because that is their coding alphabet.
            mapped, vocabulary_size = data, 256
        train = mapped[:args.train_symbols]
        evaluation = mapped[args.train_symbols:]
        for spec in (item for item in specs if item.symbol_kind == kind):
            started = time.perf_counter()
            model = build_predictor(spec, vocabulary_size)
            if isinstance(model, NGramPredictor):
                train_ngram_predictor(model, train)
                train_losses = None
            else:
                train_losses = train_neural_predictor(
                    model, train, epochs=args.epochs, batch_size=args.batch_size,
                    learning_rate=args.learning_rate, seed=args.seed,
                )
            result = {
                "model": spec.name,
                "symbol_kind": kind,
                "vocabulary_size": vocabulary_size,
                "train_symbols": len(train),
                "evaluation_symbols": len(evaluation),
                "train_loss_nats": train_losses[-1] if train_losses else None,
                "evaluation_bits_per_symbol": bits_per_symbol(
                    model, evaluation
                ),
                "elapsed_seconds": time.perf_counter() - started,
            }
            results.append(result)
            print(json.dumps(result, sort_keys=True))
    report = {
        "input": str(args.input),
        "excerpt_policy": "contiguous prefix; train then held-out continuation",
        "tokenizer": "Qwen/Qwen2.5-0.5B (local cache)",
        "token_vocabulary_policy": "observed train+evaluation alphabet remapped to dense IDs",
        "results": results,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print("\nSummary (lower bits/symbol is better):")
    for row in sorted(results, key=lambda row: row["evaluation_bits_per_symbol"]):
        print(f'{row["model"]:<30} {row["evaluation_bits_per_symbol"]:8.3f} b/symbol')


if __name__ == "__main__":
    main()
