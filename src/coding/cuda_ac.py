"""Lazy-built CUDA target-interval encoder for MSAC v2 archives.

The CUDA kernel owns only arithmetic state and private stream payloads. Archive
framing and CRC32 remain on the host so the resulting bytes are identical to
``MultistreamACEncoder(target_interval=True)``.
"""

from __future__ import annotations

import os
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


def encode_intervals_cuda(
    lows: torch.Tensor,
    highs: torch.Tensor,
    totals: torch.Tensor,
    counts: torch.Tensor,
    *,
    nominal_total: int = 262144,
    state_bits: int = 32,
    workspace_bytes: int | None = None,
) -> bytes:
    """Encode exact integer intervals and return a byte-compatible MSAC v2 archive.

    Inputs are contiguous int64 CUDA tensors. Their layout is ``[steps,
    streams]`` and ``counts[stream]`` selects the valid prefix for that stream.
    An undersized workspace is detected and retried once at the exact reported
    size; payloads are never silently truncated.
    """
    steps, streams = _validate_inputs(
        lows, highs, totals, counts, nominal_total, state_bits
    )
    if workspace_bytes is None:
        # A conservative first allocation. Pathological traces can exceed it;
        # the checked retry below uses the kernel's exact required byte count.
        workspace_bytes = max(1, (steps * (state_bits + 1) + 8) // 8)
    if workspace_bytes < 1:
        raise ValueError("workspace_bytes must be positive")

    def launch(capacity: int):
        result = encode_intervals_cuda_raw(
            lows,
            highs,
            totals,
            counts,
            nominal_total=nominal_total,
            state_bits=state_bits,
            workspace_bytes=capacity,
        )
        workspace, bit_counts, byte_counts, errors = result
        return (
            workspace,
            bit_counts.cpu().tolist(),
            byte_counts.cpu().tolist(),
            errors.cpu().tolist(),
        )

    workspace, bit_counts, byte_counts, errors = launch(workspace_bytes)
    non_overflow = [code for code in errors if code not in (0, 3)]
    if non_overflow:
        code = non_overflow[0]
        raise ValueError(_ERROR_MESSAGES.get(code, f"unknown CUDA encoder error {code}"))
    if any(code == 3 for code in errors):
        workspace_bytes = max(byte_counts)
        workspace, bit_counts, byte_counts, errors = launch(workspace_bytes)
        if any(errors):
            code = next(code for code in errors if code)
            raise RuntimeError(_ERROR_MESSAGES.get(code, f"unknown CUDA encoder error {code}"))

    host_workspace = workspace.cpu().numpy()
    host_counts = counts.cpu().tolist()
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
    return b"".join((
        _HEADER.pack(MAGIC, 2, state_bits, 1, nominal_total, streams),
        *descriptors,
        *payloads,
    ))


def extension_build_directory() -> Path:
    """Return PyTorch's cache location, useful for excluding build time in runs."""
    from torch.utils.cpp_extension import get_default_build_root

    return Path(get_default_build_root()) / "compression_economics_cuda_ac"
