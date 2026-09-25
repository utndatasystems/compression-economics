#!/usr/bin/env python3
"""Measure GIL-free MSAC range encoding on fixed quantized intervals."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.coding.parallel_ac import encode_intervals_packed, prepare_parallel_encoder


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--streams", type=int, default=4)
    parser.add_argument("--symbols-per-stream", type=int, default=150_000)
    parser.add_argument("--alphabet-size", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=147)
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 2, 4])
    args = parser.parse_args()
    if min(args.streams, args.symbols_per_stream, args.alphabet_size, args.repeats, *args.workers) < 1:
        parser.error("all sizes and worker counts must be positive")
    if args.alphabet_size > (1 << 18):
        parser.error("alphabet size must not exceed the frequency total")
    total = args.alphabet_size
    prepare_parallel_encoder(32, total)
    rng = np.random.default_rng(args.seed)
    lows = [rng.integers(0, total, size=args.symbols_per_stream, dtype=np.int64)
            for _ in range(args.streams)]
    intervals = [(low, low + 1) for low in lows]
    expected_digest = None
    rows = []
    for workers in args.workers:
        elapsed = []
        digest = None
        for _ in range(args.repeats):
            start = time.perf_counter()
            with ThreadPoolExecutor(max_workers=min(workers, args.streams)) as pool:
                streams = list(pool.map(
                    lambda pair: encode_intervals_packed(pair[0], pair[1], total, 32),
                    intervals,
                ))
            elapsed.append(time.perf_counter() - start)
            current_digest = hashlib.sha256(b"".join(
                data.tobytes() + int(bits).to_bytes(8, "big") for data, bits in streams
            )).hexdigest()
            if digest is not None and digest != current_digest:
                raise AssertionError("nondeterministic range encoding")
            digest = current_digest
        if expected_digest is not None and digest != expected_digest:
            raise AssertionError("worker count changed the encoded streams")
        expected_digest = digest
        median = statistics.median(elapsed)
        rows.append({
            "workers": min(workers, args.streams),
            "median_seconds": median,
            "range_symbols_per_second": args.streams * args.symbols_per_stream / median,
            "encoded_streams_sha256": digest,
        })
    print(json.dumps({
        "benchmark": "parallel_msac_range_kernel",
        "symbols": args.streams * args.symbols_per_stream,
        "streams": args.streams,
        "alphabet_size": total,
        "repeats": args.repeats,
        "seed": args.seed,
        "excludes": ["model_inference", "probability_quantization", "archive_framing", "decoder"],
        "results": rows,
    }, indent=2))


if __name__ == "__main__":
    main()
