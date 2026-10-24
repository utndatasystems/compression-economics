"""Isolate CUDA model inference from arithmetic coding on one live trace.

This runner answers two deliberately narrow questions for the coding survey:

1. Does the buffered v3 arithmetic coder affect measured model inference time?
2. How long does that coder take when replayed alone on a frozen live trace?

The no-coder condition executes the same tokenization, model inference, target
selection, and v3 interval construction as the integrated condition. Its
encoder is a no-op: it does not launch the final batched coder kernel.
A separate, untimed capture pass materializes the frozen trace used by the
coder-only measurements.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time
from types import SimpleNamespace
from typing import Iterator

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

import src.global_mask_compressor as compression_pipeline
from src.coding.cuda_ac import encode_intervals_cuda


_CAPTURED_TRACE: dict[str, np.ndarray] | None = None


def _no_op_encoder(lows, highs, totals, counts, *, metrics=None, **kwargs) -> bytes:
    """Match the v3 terminal handoff without launching the CUDA coder."""
    if metrics is not None:
        metrics.update({
            "kernel_seconds": 0.0,
            "device_to_host_seconds": 0.0,
            "device_to_host_bytes": 0,
            "archive_bytes": 1,
            "total_seconds": 0.0,
            "benchmark_no_op": True,
        })
    return b"\x00"


def _capture_encoder(lows, highs, totals, counts, *, metrics=None, **kwargs) -> bytes:
    """Materialize the buffered v3 trace in an explicitly untimed pass."""
    global _CAPTURED_TRACE
    torch.cuda.synchronize(lows.device)
    _CAPTURED_TRACE = {
        "lows": lows.cpu().numpy(),
        "highs": highs.cpu().numpy(),
        "totals": totals.cpu().numpy(),
        "counts": counts.cpu().numpy(),
    }
    if metrics is not None:
        metrics.update({
            "kernel_seconds": 0.0,
            "device_to_host_seconds": 0.0,
            "device_to_host_bytes": sum(a.nbytes for a in _CAPTURED_TRACE.values()),
            "archive_bytes": 1,
            "total_seconds": 0.0,
            "benchmark_trace_capture": True,
        })
    return b"\x00"


@contextmanager
def _encoder_override(encoder) -> Iterator[None]:
    original = compression_pipeline.encode_intervals_cuda
    compression_pipeline.encode_intervals_cuda = encoder
    try:
        yield
    finally:
        compression_pipeline.encode_intervals_cuda = original


def _balanced_orders(conditions: list[str], repeats: int) -> list[list[str]]:
    orders = []
    for repeat in range(repeats):
        offset = repeat % len(conditions)
        order = conditions[offset:] + conditions[:offset]
        if (repeat // len(conditions)) % 2:
            order = list(reversed(order))
        orders.append(order)
    return orders


def _compression_args(cli) -> SimpleNamespace:
    return SimpleNamespace(
        mode="compress",
        input_path=None,
        text_input=cli.benchmark_text,
        output_path=None,
        model_name=cli.model,
        model_revision=cli.model_revision,
        tokenizer_name=None,
        tokenizer_revision=None,
        trust_remote_code=False,
        lora_path=None,
        is_mamba=False,
        is_seq2seq=False,
        engine="transformer",
        encoding="AC_TARGET_INTERVAL",
        frequency_quantizer="reference",
        ac_backend="cuda",
        target_interval_quantizer="cuda_v3",
        ac_threads=None,
        ac_layout="standard",
        reduce_tokens=True,
        first_n_tokens=cli.tokens,
        batch_size=cli.batch_size,
        context_length=cli.context_length,
        retain_tokens=cli.retain_tokens,
        use_kv_cache=True,
        spec_k=None,
    )


def _run_live(cli, condition: str) -> tuple[dict, bytes]:
    override = _no_op_encoder if condition == "inference_only" else None
    if override is None:
        result = compression_pipeline.run_global_mask_compression(
            _compression_args(cli)
        )
    else:
        with _encoder_override(override):
            result = compression_pipeline.run_global_mask_compression(
                _compression_args(cli)
            )
    seeds, archive, bitmap, stats, _ = result
    sample = {
        "inference_seconds": stats["inference_time"],
        "compression_phase_seconds": stats["compression_time"],
        "input_symbols": stats["input_symbols_count"],
        "model_input_tokens": stats["model_input_tokens_count"],
        "inference_input_symbols_per_second": (
            stats["input_symbols_count"] / stats["inference_time"]
        ),
        "interval_conversion_seconds": stats["interval_conversion_seconds"],
        "coder_kernel_seconds": stats["ac_time"],
        "seed_count": len(seeds),
        "bitmap_bytes": len(bitmap),
    }
    if condition == "integrated":
        sample.update({
            "archive_bytes": len(archive),
            "archive_sha256": hashlib.sha256(archive).hexdigest(),
        })
    gc.collect()
    torch.cuda.empty_cache()
    return sample, archive


def _capture_trace(cli) -> dict[str, np.ndarray]:
    global _CAPTURED_TRACE
    _CAPTURED_TRACE = None
    with _encoder_override(_capture_encoder):
        compression_pipeline.run_global_mask_compression(_compression_args(cli))
    if _CAPTURED_TRACE is None:
        raise RuntimeError("the capture pass did not produce a trace")
    trace = _CAPTURED_TRACE
    _validate_trace(trace)
    return trace


def _validate_trace(trace: dict[str, np.ndarray]) -> None:
    required = {"lows", "highs", "totals", "counts"}
    if set(trace) != required:
        raise ValueError(f"trace fields must be exactly {sorted(required)}")
    lows, highs, totals, counts = (trace[name] for name in (
        "lows", "highs", "totals", "counts"
    ))
    if lows.ndim != 2 or lows.shape != highs.shape or lows.shape != totals.shape:
        raise ValueError("interval arrays must have one common [steps, streams] shape")
    if counts.shape != (lows.shape[1],):
        raise ValueError("counts must contain one value per stream")
    if any(array.dtype != np.int64 for array in (lows, highs, totals, counts)):
        raise ValueError("all frozen-trace arrays must use int64")
    if np.any(counts < 0) or np.any(counts > lows.shape[0]):
        raise ValueError("stream count outside frozen trace")
    valid = np.arange(lows.shape[0])[:, None] < counts[None, :]
    if np.any(lows[valid] < 0) or np.any(highs[valid] <= lows[valid]):
        raise ValueError("invalid target interval")
    if np.any(highs[valid] > totals[valid]):
        raise ValueError("target interval exceeds its total")


def _trace_sha256(trace: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in ("lows", "highs", "totals", "counts"):
        array = np.ascontiguousarray(trace[name].astype("<i8", copy=False))
        digest.update(name.encode("ascii") + b"\x00")
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _save_trace(path: Path, trace: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **trace)


def _load_trace(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        trace = {name: archive[name] for name in archive.files}
    _validate_trace(trace)
    return trace


def _device_trace(trace: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
    return {
        name: torch.as_tensor(array, device="cuda").contiguous()
        for name, array in trace.items()
    }


def _run_frozen_coder(trace: dict[str, torch.Tensor]) -> tuple[dict, bytes]:
    torch.cuda.synchronize()
    started = time.perf_counter()
    metrics: dict = {}
    archive = encode_intervals_cuda(
        trace["lows"], trace["highs"], trace["totals"], trace["counts"],
        metrics=metrics, exact_fixed=False,
    )
    torch.cuda.synchronize()
    wall_seconds = time.perf_counter() - started
    symbols = int(trace["counts"].sum().item())
    return ({
        "wall_seconds": wall_seconds,
        "kernel_seconds": metrics["kernel_seconds"],
        "wall_symbols_per_second": symbols / wall_seconds,
        "kernel_symbols_per_second": symbols / metrics["kernel_seconds"],
        "symbols": symbols,
        "archive_bytes": len(archive),
        "archive_sha256": hashlib.sha256(archive).hexdigest(),
        "metrics": metrics,
    }, archive)


def _medians(samples: dict[str, list[dict]], keys: tuple[str, ...]) -> dict:
    return {
        condition: {
            key: statistics.median(sample[key] for sample in values)
            for key in keys
        }
        for condition, values in samples.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=REPO_ROOT / "data/text8")
    parser.add_argument("--input-characters", type=int, default=1048576)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--model-revision")
    parser.add_argument("--tokens", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--retain-tokens", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--trace", type=Path, help="reuse an existing frozen NPZ trace")
    parser.add_argument("--save-trace", type=Path)
    parser.add_argument("--coder-only", action="store_true", help="skip live model runs")
    parser.add_argument("--output", type=Path)
    cli = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    if cli.repeats < 1 or cli.tokens <= cli.batch_size:
        parser.error("repeats must be positive and tokens must exceed batch size")
    if cli.trace is not None:
        trace = _load_trace(cli.trace)
        trace_source = str(cli.trace)
        cli.benchmark_text = None
    else:
        if cli.coder_only:
            parser.error("--coder-only requires --trace")
        if not cli.input.is_file():
            parser.error(f"input does not exist: {cli.input}")
        with cli.input.open("r", encoding="utf-8", newline="") as input_file:
            cli.benchmark_text = input_file.read(cli.input_characters)
        if not cli.benchmark_text:
            parser.error("input prefix is empty")
        trace_source = "untimed live capture pass"

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    torch.manual_seed(2027)

    live_samples = None
    live_medians = None
    integrated_archive: bytes | None = None
    live_orders = None
    if not cli.coder_only:
        conditions = ["integrated", "inference_only"]
        # One warm-up per condition; these are deliberately omitted from output
        # samples but establish steady-state kernels and allocators.
        for condition in conditions:
            _run_live(cli, condition)
        live_samples = {condition: [] for condition in conditions}
        live_orders = _balanced_orders(conditions, cli.repeats)
        integrated_hash = None
        for order in live_orders:
            for condition in order:
                sample, archive = _run_live(cli, condition)
                live_samples[condition].append(sample)
                if condition == "integrated":
                    archive_hash = hashlib.sha256(archive).hexdigest()
                    if integrated_hash is not None and archive_hash != integrated_hash:
                        raise AssertionError("integrated runs produced different archives")
                    integrated_hash = archive_hash
                    integrated_archive = archive
        live_medians = _medians(live_samples, (
            "inference_seconds", "compression_phase_seconds",
            "inference_input_symbols_per_second", "interval_conversion_seconds",
            "coder_kernel_seconds",
        ))
        if cli.trace is None:
            trace = _capture_trace(cli)
            if cli.save_trace is not None:
                _save_trace(cli.save_trace, trace)

    device_trace = _device_trace(trace)
    # Warm the exact v3 coder path before collecting frozen-trace samples.
    _, warm_archive = _run_frozen_coder(device_trace)
    if integrated_archive is not None and integrated_archive != warm_archive:
        raise AssertionError("frozen trace archive differs from integrated archive")

    coder_samples = []
    for _ in range(cli.repeats):
        sample, archive = _run_frozen_coder(device_trace)
        if archive != warm_archive:
            raise AssertionError("timed frozen coder archive differs")
        coder_samples.append(sample)
    coder_medians = {
        key: statistics.median(sample[key] for sample in coder_samples)
        for key in (
            "wall_seconds", "kernel_seconds", "wall_symbols_per_second",
            "kernel_symbols_per_second",
        )
    }

    result = {
        "experiment": "inference_coder_isolation_v1",
        "question": {
            "live": "Does the buffered v3 arithmetic coder affect inference time?",
            "frozen": "How long does arithmetic coding alone take on the same trace?",
        },
        "model": None if cli.coder_only else cli.model,
        "model_revision": cli.model_revision,
        "input": None if cli.coder_only else str(cli.input),
        "input_characters": None if cli.coder_only else len(cli.benchmark_text),
        "input_sha256": None if cli.coder_only else hashlib.sha256(
            cli.benchmark_text.encode("utf-8")
        ).hexdigest(),
        "tokens": cli.tokens,
        "batch_size": cli.batch_size,
        "context_length": cli.context_length,
        "retain_tokens": cli.retain_tokens,
        "repeats": cli.repeats,
        "trace": {
            "source": trace_source,
            "sha256": _trace_sha256(trace),
            "steps": int(trace["lows"].shape[0]),
            "streams": int(trace["lows"].shape[1]),
            "symbols": int(trace["counts"].sum()),
            "saved_path": str(cli.save_trace) if cli.save_trace else None,
        },
        "live_condition_definitions": {
            "integrated": "v3 interval construction followed by the buffered CUDA coder",
            "inference_only": "identical model and v3 interval construction with the final coder disabled",
        },
        "live_orders": live_orders,
        "live_samples": live_samples,
        "live_medians": live_medians,
        "live_effect": None if live_medians is None else {
            "inference_slowdown_with_coder": (
                live_medians["integrated"]["inference_seconds"]
                / live_medians["inference_only"]["inference_seconds"]
            ),
            "inference_throughput_ratio_with_coder": (
                live_medians["integrated"]["inference_input_symbols_per_second"]
                / live_medians["inference_only"]["inference_input_symbols_per_second"]
            ),
        },
        "frozen_condition_definition": "the integrated v3 batched coder replayed without model inference, including framing and device-to-host transfer",
        "frozen_samples": coder_samples,
        "frozen_medians": coder_medians,
        "archive_sha256": hashlib.sha256(warm_archive).hexdigest(),
        "archive_bytes": len(warm_archive),
        "archive_equality": {
            "frozen_vs_integrated": integrated_archive is not None,
        },
        "gpu": torch.cuda.get_device_name(),
        "torch_version": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "timing_scope": {
            "live_inference": "CUDA events around model inference and v3 target-interval construction at each step",
            "frozen_wall": "synchronized v3 coder call including launch, framing, and device-to-host transfer",
            "frozen_kernel": "CUDA arithmetic-coder kernels only",
            "excluded": "model loading, tokenizer loading, archive file I/O, and the untimed trace-capture pass",
        },
    }
    if cli.output:
        cli.output.parent.mkdir(parents=True, exist_ok=True)
        cli.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
