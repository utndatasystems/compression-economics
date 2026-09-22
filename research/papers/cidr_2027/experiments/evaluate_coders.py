#!/usr/bin/env python3
"""Run a matched entropy-coder study over one frozen probability trace."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.coder_benchmark import (
    CODERS,
    benchmark_coder,
    load_probability_trace,
    save_probability_trace,
    synthetic_probability_trace,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--trace", type=Path, help="NPZ with probabilities and target symbols.")
    source.add_argument(
        "--synthetic-symbols", type=int,
        help="Generate a deterministic smoke trace with this many symbols.",
    )
    parser.add_argument("--alphabet-size", type=int, default=64)
    parser.add_argument("--coder", choices=CODERS, action="append")
    parser.add_argument("--frequency-total", type=int, default=262144)
    parser.add_argument("--ans-block-size", type=int, default=256)
    parser.add_argument("--ans-lanes", type=int, default=4)
    parser.add_argument("--pmatic-delta", type=float, default=1e-3)
    parser.add_argument(
        "--perturbation-scale", type=float, action="append", dest="perturbation_scales",
        help="Decoder-only Gaussian probability noise; repeat for a sweep.",
    )
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("artifacts/papers/cidr-2027/coder-comparison"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.trace:
        trace = load_probability_trace(args.trace)
        trace_source = str(args.trace)
    else:
        trace = synthetic_probability_trace(
            args.synthetic_symbols, args.alphabet_size, seed=args.seed
        )
        trace_path = args.output_dir / "synthetic-trace.npz"
        save_probability_trace(trace_path, trace)
        trace_source = str(trace_path)
    coders = args.coder or list(CODERS)
    scales = args.perturbation_scales or [1e-8, 1e-6, 1e-4]
    rows = []
    for coder in coders:
        print(f"Benchmarking {coder} on trace {trace.sha256[:12]}...", flush=True)
        row = benchmark_coder(
            trace, coder, total=args.frequency_total,
            ans_block_size=args.ans_block_size, ans_lanes=args.ans_lanes,
            pmatic_delta=args.pmatic_delta, perturbation_scales=scales,
            seed=args.seed,
        )
        rows.append(row)
        (args.output_dir / f"{coder.lower()}-result.json").write_text(
            json.dumps(row, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(row, sort_keys=True), flush=True)
    result = {
        "benchmark_name": "matched-predictive-coder-comparison",
        "benchmark_version": 1,
        "trace": trace_source,
        "trace_sha256": trace.sha256,
        "trace_shape": [trace.symbol_count, trace.alphabet_size],
        "invariants": {
            "predictor_probabilities_frozen": True,
            "target_symbols_frozen": True,
            "frequency_total_shared": args.frequency_total,
            "predictor_time_excluded": True,
            "rank_codecs_are_probability_equivalent": False,
        },
        "results": rows,
    }
    (args.output_dir / "results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
