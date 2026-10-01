"""Benchmark the byte-exact CUDA MSAC v2 interval encoder."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from src.coding.cuda_ac import encode_intervals_cuda, encode_intervals_cuda_raw
from src.coding.multistream_ac import MultistreamACEncoder
from src.coding.target_interval import target_intervals_from_probs_tensor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=512)
    parser.add_argument("--streams", type=int, default=32)
    parser.add_argument("--alphabet", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    if args.steps < 1 or args.streams < 1 or not 2 <= args.alphabet < 262144:
        parser.error("invalid trace dimensions")
    if args.repeats < 1:
        parser.error("repeats must be positive")

    rng = np.random.default_rng(2027)
    flat_rows = rng.dirichlet(
        np.ones(args.alphabet), size=args.steps * args.streams
    ).astype(np.float32)
    flat_targets = rng.integers(
        0, args.alphabet, size=args.steps * args.streams, dtype=np.int64
    )
    rows = torch.tensor(flat_rows, device="cuda")
    targets = torch.tensor(flat_targets, device="cuda")
    lows, highs, totals, _ = target_intervals_from_probs_tensor(rows, targets)
    shape = (args.steps, args.streams)
    lows = lows.view(shape).contiguous()
    highs = highs.view(shape).contiguous()
    totals = totals.view(shape).contiguous()
    counts = args.steps - (torch.arange(args.streams, device="cuda") % 4)
    counts = counts.to(torch.int64).contiguous()
    capacity = max(1, (args.steps * 33 + 8) // 8)

    # Build outside all timing regions and validate the complete archive bytes.
    archive_metrics = {}
    cuda_archive = encode_intervals_cuda(
        lows, highs, totals, counts, metrics=archive_metrics
    )
    host_lows = lows.cpu().numpy()
    host_highs = highs.cpu().numpy()
    host_totals = totals.cpu().numpy()
    host_counts = counts.cpu().tolist()
    reference = MultistreamACEncoder(args.streams, target_interval=True)
    reference_started = time.perf_counter()
    for step in range(args.steps):
        for stream, count in enumerate(host_counts):
            if step < count:
                reference.encode_interval(
                    stream,
                    int(host_lows[step, stream]),
                    int(host_highs[step, stream]),
                    int(host_totals[step, stream]),
                )
    reference_archive = reference.finish()
    reference_seconds = time.perf_counter() - reference_started
    if cuda_archive != reference_archive:
        raise AssertionError("CUDA archive differs from the Python reference")

    # Warm the allocator and operator, then time only device work with events.
    encode_intervals_cuda_raw(
        lows, highs, totals, counts, workspace_bytes=capacity
    )
    torch.cuda.synchronize()
    samples = []
    for _ in range(args.repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        _, _, byte_counts, errors = encode_intervals_cuda_raw(
            lows, highs, totals, counts, workspace_bytes=capacity
        )
        end.record()
        end.synchronize()
        if errors.count_nonzero().item():
            raise RuntimeError("CUDA encoder reported an error")
        if byte_counts.max().item() > capacity:
            raise RuntimeError("benchmark workspace was too small")
        samples.append(start.elapsed_time(end) / 1000.0)

    symbols = sum(host_counts)
    median = statistics.median(samples)
    result = {
        "seed": 2027,
        "steps": args.steps,
        "streams": args.streams,
        "alphabet": args.alphabet,
        "symbols": symbols,
        "archive_bytes": len(cuda_archive),
        "archive_sha256": hashlib.sha256(cuda_archive).hexdigest(),
        "byte_exact_python_reference": True,
        "python_reference_seconds": reference_seconds,
        "operator_seconds": samples,
        "operator_median_seconds": median,
        "operator_median_symbols_per_sec": symbols / median,
        "archive_wrapper_metrics": archive_metrics,
        "torch_version": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "timing_scope": (
            "warm device operator including output zero-fill and range kernel; "
            "excludes quantization, host transfer, CRC, framing, and JIT build"
        ),
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
