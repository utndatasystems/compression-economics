"""Frozen-trace MSAC v2 host/device decode comparison (no model inference)."""

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from src.coding.device_ac import (
    CudaCDFMultistreamACDecoder,
    DeviceMultistreamACDecoder,
)
from src.coding.multistream_ac import MultistreamACDecoder, MultistreamACEncoder
from src.coding.target_interval import target_intervals_from_probs_tensor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--streams", type=int, default=4)
    parser.add_argument("--alphabet", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--backends", nargs="+",
        choices=["host", "torch_device", "cuda_cdf"],
        help="Optional subset for isolated profiling; defaults to all valid backends",
    )
    parser.add_argument("--output", type=Path, help="Optional JSON result path")
    args = parser.parse_args()
    if args.steps < 1 or args.streams < 1 or not 2 <= args.alphabet < 262144 or args.repeats < 1:
        parser.error("invalid trace dimensions or repetition count")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    device = torch.device(args.device)
    rng = np.random.default_rng(2027)
    rows = rng.dirichlet(np.ones(args.alphabet), size=(args.steps, args.streams)).astype(np.float32)
    rows_device = torch.as_tensor(rows, device=device)
    targets = np.empty((args.steps, args.streams), dtype=np.int64)
    for step in range(args.steps):
        for stream in range(args.streams):
            targets[step, stream] = rng.choice(args.alphabet, p=rows[step, stream].astype(np.float64) /
                                                   rows[step, stream].sum(dtype=np.float64))
    encoder = MultistreamACEncoder(args.streams, target_interval=True)
    for step in range(args.steps):
        probabilities = rows_device[step]
        low, high, total, _ = target_intervals_from_probs_tensor(
            probabilities, torch.as_tensor(targets[step], device=device))
        for stream in range(args.streams):
            encoder.encode_interval(stream, int(low[stream]), int(high[stream]), int(total[stream]))
    archive = encoder.finish()
    active = [True] * args.streams
    backends = args.backends or ["host", "torch_device"]
    if args.backends is None and device.type == "cuda":
        backends.append("cuda_cdf")
    if device.type != "cuda" and "cuda_cdf" in backends:
        parser.error("cuda_cdf requires --device cuda")
    samples = {backend: [] for backend in backends}
    cuda_stage_samples = []
    runs = [(backend, 0) for backend in backends]
    timed_orders = []
    for repeat in range(1, args.repeats + 1):
        offset = (repeat - 1) % len(backends)
        order = backends[offset:] + backends[:offset]
        if ((repeat - 1) // len(backends)) % 2:
            order.reverse()
        timed_orders.append(order)
        runs.extend((backend, repeat) for backend in order)
    for backend, repeat in runs:
        if backend == "host":
            decoder = MultistreamACDecoder(archive)
        elif backend == "torch_device":
            decoder = DeviceMultistreamACDecoder(archive, device)
        else:
            decoder = CudaCDFMultistreamACDecoder(archive, device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        decoded_rows = []
        for step in range(args.steps):
            if backend == "host":
                host_rows = rows_device[step].cpu().numpy() if device.type == "cuda" else rows[step]
                actual = [decoder.decode(stream, host_rows[stream])
                          for stream in range(args.streams)]
            else:
                actual = decoder.decode(rows_device[step], active).cpu().tolist()
            decoded_rows.append(actual)
        decoder.assert_complete()
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        if not np.array_equal(np.asarray(decoded_rows), targets):
            raise AssertionError("decoder disagrees with source trace")
        if repeat:
            samples[backend].append(elapsed)
            if backend == "cuda_cdf":
                cuda_stage_samples.append(decoder.metrics)
    medians = {
        backend: {
            "seconds": statistics.median(values),
            "symbols_per_second": args.steps * args.streams / statistics.median(values),
        }
        for backend, values in samples.items()
    }
    result = {
        "seed": 2027, "dtype": "float32", "torch_version": torch.__version__,
        "cpu": next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                     if line.startswith("model name")), platform.processor()),
        "timing_scope": "warm decoder; per-step probability handoff and AC decode; no model or archive I/O; exactness checked outside timed loop",
        "exact_recovery": True,
        "device": args.device, "steps": args.steps, "streams": args.streams,
        "alphabet": args.alphabet, "archive_bytes": len(archive),
        "backends": backends,
        "timed_condition_orders": timed_orders,
        "samples_seconds": samples,
        "medians": medians,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
    }
    if "cuda_cdf" in medians:
        result["cuda_cdf_stage_medians"] = {
            key: statistics.median(sample[key] for sample in cuda_stage_samples)
            for key in ("quantization_seconds", "kernel_seconds", "decode_steps")
        }
        if "torch_device" in medians:
            result["cuda_cdf_vs_torch_device"] = {
                "speedup": (
                    medians["torch_device"]["seconds"]
                    / medians["cuda_cdf"]["seconds"]
                )
            }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
