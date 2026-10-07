"""Device-only ablation of PyTorch versus fused target-interval construction."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from src.coding.cuda_ac import (
    encode_intervals_cuda,
    encode_intervals_cuda_raw,
    target_intervals_from_probs_cuda,
    target_intervals_from_probs_cuda_v3,
    target_intervals_from_probs_cuda_v4,
)
from src.coding.target_interval import target_intervals_from_probs_tensor


def _time_cuda(operation, repeats: int) -> list[float]:
    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        operation()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / 1000.0)
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=4096)
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

    generator = torch.Generator(device="cuda").manual_seed(2029)
    rows = args.steps * args.streams
    probabilities = torch.rand(
        (rows, args.alphabet),
        dtype=torch.float32,
        device="cuda",
        generator=generator,
    )
    probabilities /= probabilities.sum(dim=1, keepdim=True)
    targets = torch.randint(
        args.alphabet, (rows,), dtype=torch.int64,
        device="cuda", generator=generator,
    )
    counts = torch.full(
        (args.streams,), args.steps, dtype=torch.int64, device="cuda"
    )
    capacity = max(1, (args.steps * 33 + 8) // 8)

    def torch_intervals():
        return target_intervals_from_probs_tensor(
            probabilities, targets, validate=False
        )

    def fused_intervals():
        return target_intervals_from_probs_cuda(
            probabilities, targets, validate=False
        )

    def compact_intervals():
        return target_intervals_from_probs_cuda_v3(
            probabilities, targets, validate=False
        )

    def exact_fixed_intervals():
        return target_intervals_from_probs_cuda_v4(
            probabilities, targets, validate=False
        )

    expected = torch_intervals()
    actual = fused_intervals()
    compact = compact_intervals()
    exact_fixed = exact_fixed_intervals()
    torch.cuda.synchronize()
    if any(not torch.equal(left, right) for left, right in zip(expected, actual)):
        raise AssertionError("fused intervals differ from the PyTorch baseline")
    if any(not torch.equal(left, right) for left, right in zip(expected, compact)):
        raise AssertionError("CUDA v3 intervals differ from the PyTorch baseline")
    if not torch.all(exact_fixed[2] == 262144):
        raise AssertionError("CUDA v4 did not produce fixed-total intervals")

    def reshape(intervals):
        return tuple(
            value.view(args.steps, args.streams).contiguous()
            for value in intervals[:3]
        )

    baseline_lows, baseline_highs, baseline_totals = reshape(expected)
    fused_lows, fused_highs, fused_totals = reshape(actual)
    compact_lows, compact_highs, compact_totals = reshape(compact)
    exact_lows, exact_highs, exact_totals = reshape(exact_fixed)
    baseline_archive = encode_intervals_cuda(
        baseline_lows, baseline_highs, baseline_totals, counts
    )
    fused_archive = encode_intervals_cuda(
        fused_lows, fused_highs, fused_totals, counts
    )
    compact_archive = encode_intervals_cuda(
        compact_lows, compact_highs, compact_totals, counts
    )
    exact_archive = encode_intervals_cuda(
        exact_lows,
        exact_highs,
        exact_totals,
        counts,
        exact_fixed=True,
    )
    if baseline_archive != fused_archive:
        raise AssertionError("fused path changed the encoded archive")
    if baseline_archive != compact_archive:
        raise AssertionError("CUDA v3 changed the encoded archive")

    def combined(interval_function):
        lows, highs, totals = reshape(interval_function())
        return encode_intervals_cuda_raw(
            lows, highs, totals, counts, workspace_bytes=capacity
        )

    # Warm both allocation/operator paths before timing.
    torch_intervals()
    fused_intervals()
    compact_intervals()
    exact_fixed_intervals()
    combined(torch_intervals)
    combined(fused_intervals)
    combined(compact_intervals)
    combined(exact_fixed_intervals)
    torch.cuda.synchronize()

    interval_samples = {
        "cuda_torch": _time_cuda(torch_intervals, args.repeats),
        "cuda_fused": _time_cuda(fused_intervals, args.repeats),
        "cuda_v3": _time_cuda(compact_intervals, args.repeats),
        "cuda_v4": _time_cuda(exact_fixed_intervals, args.repeats),
    }
    combined_samples = {
        "cuda_torch": _time_cuda(
            lambda: combined(torch_intervals), args.repeats
        ),
        "cuda_fused": _time_cuda(
            lambda: combined(fused_intervals), args.repeats
        ),
        "cuda_v3": _time_cuda(
            lambda: combined(compact_intervals), args.repeats
        ),
        "cuda_v4": _time_cuda(
            lambda: combined(exact_fixed_intervals), args.repeats
        ),
    }

    def summarize(samples):
        median = statistics.median(samples)
        return {
            "samples_seconds": samples,
            "median_seconds": median,
            "target_intervals_per_second": rows / median,
            "probability_values_per_second": rows * args.alphabet / median,
        }

    interval_summaries = {
        name: summarize(samples) for name, samples in interval_samples.items()
    }
    combined_summaries = {
        name: summarize(samples) for name, samples in combined_samples.items()
    }
    interval_ratio = (
        interval_summaries["cuda_torch"]["median_seconds"]
        / interval_summaries["cuda_fused"]["median_seconds"]
    )
    combined_ratio = (
        combined_summaries["cuda_torch"]["median_seconds"]
        / combined_summaries["cuda_fused"]["median_seconds"]
    )
    v3_interval_ratio = (
        interval_summaries["cuda_fused"]["median_seconds"]
        / interval_summaries["cuda_v3"]["median_seconds"]
    )
    v3_combined_ratio = (
        combined_summaries["cuda_fused"]["median_seconds"]
        / combined_summaries["cuda_v3"]["median_seconds"]
    )
    result = {
        "seed": 2029,
        "steps": args.steps,
        "streams": args.streams,
        "alphabet": args.alphabet,
        "target_intervals": rows,
        "repeats": args.repeats,
        "experimental_factor": "target_interval_quantizer",
        "controlled_encoder": "encode_intervals_cuda_raw",
        "exact_interval_match": True,
        "exact_archive_match": True,
        "archive_match_scope": ["cuda_torch", "cuda_fused", "cuda_v3"],
        "archive_bytes": len(fused_archive),
        "archive_sha256": hashlib.sha256(fused_archive).hexdigest(),
        "cuda_v4_archive_bytes": len(exact_archive),
        "cuda_v4_archive_sha256": hashlib.sha256(exact_archive).hexdigest(),
        "interval_construction": interval_summaries,
        "interval_construction_plus_encoding": combined_summaries,
        "fused_vs_cuda_torch": {
            "interval_construction_speedup": interval_ratio,
            "interval_construction_time_reduction_fraction": 1 - 1 / interval_ratio,
            "construction_plus_encoding_speedup": combined_ratio,
            "construction_plus_encoding_time_reduction_fraction": 1 - 1 / combined_ratio,
        },
        "cuda_v3_vs_cuda_fused": {
            "interval_construction_speedup": v3_interval_ratio,
            "interval_construction_time_reduction_fraction": (
                1 - 1 / v3_interval_ratio
            ),
            "construction_plus_encoding_speedup": v3_combined_ratio,
            "construction_plus_encoding_time_reduction_fraction": (
                1 - 1 / v3_combined_ratio
            ),
        },
        "gpu": torch.cuda.get_device_name(),
        "torch_version": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "timing_scope": (
            "warm CUDA-event device time including allocations; excludes JIT, "
            "host transfer, CRC, and archive framing"
        ),
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
