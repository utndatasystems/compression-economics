"""Lazy-built CUDA target-interval encoder for MSAC v2 archives.

The CUDA kernel owns only arithmetic state and private stream payloads. Archive
framing and CRC32 remain on the host so the resulting bytes are identical to
``MultistreamACEncoder(target_interval=True)``.
"""

from __future__ import annotations

import os
import time
import threading
import zlib
from pathlib import Path

import torch

from src.coding.multistream_ac import MAGIC, _HEADER, _STREAM, _validate_settings


_EXTENSION = None
_EXTENSION_LOCK = threading.Lock()
_ERROR_MESSAGES = {
    1: "symbol count is outside the interval tensor",
    2: "invalid target interval",
    3: "CUDA output workspace overflow",
    4: "invalid arithmetic coder state",
}
_DECODE_ERROR_MESSAGES = {
    4: "invalid arithmetic decoder state",
    5: "stream has no more symbols",
    6: "invalid cumulative frequency row",
    7: "arithmetic code selects a symbol outside the alphabet",
}


def _load_extension():
    """Build and cache the extension on first use, never during module import."""
    global _EXTENSION
    if _EXTENSION is not None:
        return _EXTENSION
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA interval encoding requires an available CUDA device")
    with _EXTENSION_LOCK:
        if _EXTENSION is None:
            from torch.utils.cpp_extension import load

            source_dir = Path(__file__).with_name("csrc")
            _EXTENSION = load(
                name="compression_economics_cuda_ac",
                sources=[
                    str(source_dir / "cuda_ac.cpp"),
                    str(source_dir / "cuda_ac_kernel.cu"),
                ],
                extra_cflags=["-O3"],
                extra_cuda_cflags=["-O3"],
                verbose=os.environ.get("CUDA_AC_VERBOSE_BUILD") == "1",
            )
    return _EXTENSION


def _validate_inputs(
    lows: torch.Tensor,
    highs: torch.Tensor,
    totals: torch.Tensor,
    counts: torch.Tensor,
    nominal_total: int,
    state_bits: int,
) -> tuple[int, int]:
    if lows.ndim != 2 or highs.shape != lows.shape or totals.shape != lows.shape:
        raise ValueError("low, high, and total tensors must share shape [steps, streams]")
    steps, streams = lows.shape
    _validate_settings(streams, state_bits, nominal_total)
    if counts.shape != (streams,):
        raise ValueError("counts must contain one value per stream")
    tensors = {"lows": lows, "highs": highs, "totals": totals, "counts": counts}
    for name, tensor in tensors.items():
        if tensor.device.type != "cuda":
            raise ValueError(f"{name} must be a CUDA tensor")
        if tensor.dtype != torch.int64:
            raise ValueError(f"{name} must have dtype int64")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
        if tensor.device != lows.device:
            raise ValueError("all interval tensors must use the same CUDA device")
    if (1 << state_bits) * nominal_total > (1 << 63) - 1:
        raise ValueError("arithmetic interval product exceeds signed 64-bit arithmetic")
    return steps, streams


def encode_intervals_cuda_raw(
    lows: torch.Tensor,
    highs: torch.Tensor,
    totals: torch.Tensor,
    counts: torch.Tensor,
    *,
    nominal_total: int = 262144,
    state_bits: int = 32,
    workspace_bytes: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Launch the encoder and return device workspace, bit/byte counts, and errors.

    The call is asynchronous on PyTorch's current CUDA stream. Error code 3
    means the caller should relaunch with at least the reported byte count.
    """
    _validate_inputs(lows, highs, totals, counts, nominal_total, state_bits)
    if workspace_bytes < 1:
        raise ValueError("workspace_bytes must be positive")
    extension = _load_extension()
    return tuple(extension.encode_intervals(
        lows, highs, totals, counts, nominal_total, state_bits, workspace_bytes
    ))


def target_intervals_from_probs_cuda(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    total: int = 262144,
    *,
    validate: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fuse floor-count quantization and target-CDF extraction on CUDA.

    The float64 conversion and row sum intentionally remain identical to the
    canonical PyTorch implementation. The fused kernel avoids materializing
    normalized probabilities, integer frequencies, and a complete CDF. This
    makes ``torch`` versus ``cuda_fused`` a single-factor ablation.
    """
    if probabilities.device.type != "cuda":
        raise ValueError("fused target-interval quantization requires CUDA")
    if probabilities.ndim != 2 or not 1 <= probabilities.shape[1] < total:
        raise ValueError("expected a 2D alphabet smaller than the frequency total")
    probabilities64 = probabilities.to(torch.float64).contiguous()
    if validate and (
        not torch.isfinite(probabilities64).all() or (probabilities64 < 0).any()
    ):
        raise ValueError("probabilities must be finite and nonnegative")
    probability_sums = probabilities64.sum(dim=1).contiguous()
    if validate and (probability_sums <= 0).any():
        raise ValueError("probability rows must have positive mass")
    targets = torch.as_tensor(
        targets, device=probabilities.device, dtype=torch.int64
    ).contiguous()
    if targets.ndim != 1 or targets.numel() != probabilities.shape[0]:
        raise ValueError("one target index is required per probability row")
    if validate and (
        (targets < 0).any() or (targets >= probabilities.shape[1]).any()
    ):
        raise ValueError("target outside alphabet")
    return tuple(_load_extension().quantize_target_intervals(
        probabilities64, probability_sums, targets, total
    ))


def decode_cdfs_cuda_raw(
    cdfs: torch.Tensor,
    active: torch.Tensor,
    payload: torch.Tensor,
    bit_counts: torch.Tensor,
    counts: torch.Tensor,
    lows: torch.Tensor,
    highs: torch.Tensor,
    codes: torch.Tensor,
    positions: torch.Tensor,
    decoded: torch.Tensor,
    symbols: torch.Tensor,
    errors: torch.Tensor,
    *,
    nominal_total: int = 262144,
    state_bits: int = 32,
) -> None:
    """Update persistent decoder state from supplied integer CDF rows.

    The launch is asynchronous on PyTorch's current CUDA stream. Device error
    flags are intentionally checked at a caller-controlled boundary instead of
    synchronizing after every autoregressive symbol.
    """
    _load_extension().decode_cdfs(
        cdfs,
        active,
        payload,
        bit_counts,
        counts,
        lows,
        highs,
        codes,
        positions,
        decoded,
        symbols,
        errors,
        nominal_total,
        state_bits,
    )


def decode_error_message(code: int) -> str:
    return _DECODE_ERROR_MESSAGES.get(code, f"unknown CUDA decoder error {code}")


def encode_intervals_cuda(
    lows: torch.Tensor,
    highs: torch.Tensor,
    totals: torch.Tensor,
    counts: torch.Tensor,
    *,
    nominal_total: int = 262144,
    state_bits: int = 32,
    workspace_bytes: int | None = None,
    metrics: dict | None = None,
) -> bytes:
    """Encode exact integer intervals and return a byte-compatible MSAC v2 archive.

    Inputs are contiguous int64 CUDA tensors. Their layout is ``[steps,
    streams]`` and ``counts[stream]`` selects the valid prefix for that stream.
    An undersized workspace is detected and retried once at the exact reported
    size; payloads are never silently truncated. Optional metrics separate the
    kernel, device-to-host transfer, and complete archive-wrapper timings.
    """
    steps, streams = _validate_inputs(
        lows, highs, totals, counts, nominal_total, state_bits
    )
    if workspace_bytes is None:
        workspace_bytes = max(1, (steps * (state_bits + 1) + 8) // 8)
    if workspace_bytes < 1:
        raise ValueError("workspace_bytes must be positive")

    # Compile before recording CUDA events so JIT wall time is never reported
    # as device execution time. Callers should still warm before benchmarking.
    _load_extension()
    total_started = time.perf_counter()
    kernel_seconds = 0.0
    transfer_seconds = 0.0
    transfer_bytes = 0
    retries = 0

    def launch(capacity: int):
        nonlocal kernel_seconds, transfer_seconds, transfer_bytes
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        result = encode_intervals_cuda_raw(
            lows,
            highs,
            totals,
            counts,
            nominal_total=nominal_total,
            state_bits=state_bits,
            workspace_bytes=capacity,
        )
        end_event.record()
        workspace, bit_counts, byte_counts, errors = result
        end_event.synchronize()
        kernel_seconds += start_event.elapsed_time(end_event) / 1000.0
        transfer_started = time.perf_counter()
        host_bit_counts = bit_counts.cpu().tolist()
        host_byte_counts = byte_counts.cpu().tolist()
        host_errors = errors.cpu().tolist()
        transfer_seconds += time.perf_counter() - transfer_started
        transfer_bytes += bit_counts.nbytes + byte_counts.nbytes + errors.nbytes
        return workspace, host_bit_counts, host_byte_counts, host_errors

    workspace, bit_counts, byte_counts, errors = launch(workspace_bytes)
    non_overflow = [code for code in errors if code not in (0, 3)]
    if non_overflow:
        code = non_overflow[0]
        raise ValueError(_ERROR_MESSAGES.get(code, f"unknown CUDA encoder error {code}"))
    if any(code == 3 for code in errors):
        retries += 1
        workspace_bytes = max(byte_counts)
        workspace, bit_counts, byte_counts, errors = launch(workspace_bytes)
        if any(errors):
            code = next(code for code in errors if code)
            raise RuntimeError(_ERROR_MESSAGES.get(code, f"unknown CUDA encoder error {code}"))

    transfer_started = time.perf_counter()
    host_workspace = workspace.cpu().numpy()
    host_counts = counts.cpu().tolist()
    transfer_seconds += time.perf_counter() - transfer_started
    transfer_bytes += workspace.nbytes + counts.nbytes
    descriptors = []
    payloads = []
    for stream in range(streams):
        byte_count = byte_counts[stream]
        bit_count = bit_counts[stream]
        if byte_count != (bit_count + 7) // 8:
            raise RuntimeError("CUDA encoder returned inconsistent payload dimensions")
        data = host_workspace[stream, :byte_count].tobytes()
        descriptors.append(_STREAM.pack(host_counts[stream], bit_count, zlib.crc32(data)))
        payloads.append(data)
    archive = b"".join((
        _HEADER.pack(MAGIC, 2, state_bits, 1, nominal_total, streams),
        *descriptors,
        *payloads,
    ))
    if metrics is not None:
        metrics.update({
            "kernel_seconds": kernel_seconds,
            "device_to_host_seconds": transfer_seconds,
            "device_to_host_bytes": transfer_bytes,
            "workspace_bytes_per_stream": workspace.shape[1],
            "workspace_retries": retries,
            "archive_bytes": len(archive),
            "total_seconds": time.perf_counter() - total_started,
        })
    return archive


def extension_build_directory() -> Path:
    """Return PyTorch's cache location, useful for excluding build time in runs."""
    from torch.utils.cpp_extension import get_default_build_root

    return Path(get_default_build_root()) / "compression_economics_cuda_ac"
