"""Lazy-built CUDA target-interval encoders for MSAC v2/v3 archives.

The CUDA kernels own quantization/interval construction, arithmetic state, and
private stream payloads. Archive framing and CRC32 remain on the host. V1--v3
match the floor-count MSAC-v2 contract; v4 uses the exact fixed-total MSAC-v3
contract and pipelines persistent arithmetic updates with later inference.
"""

from __future__ import annotations

import os
import time
import threading
import zlib
from pathlib import Path

import torch

from src.coding.multistream_ac import (
    EXACT_TARGET_FLAG,
    EXACT_TARGET_VERSION,
    FLOOR_TARGET_FLAG,
    FLOOR_TARGET_VERSION,
    MAGIC,
    _HEADER,
    _STREAM,
    _validate_settings,
)


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


def target_intervals_from_probs_cuda_v3(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    total: int = 262144,
    *,
    validate: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Construct target intervals without materializing float64 probability rows.

    CUDA v3 preserves the v1/v2 floor-count contract.  It casts values while
    reducing row sums and again inside the interval kernel, avoiding v2's
    vocabulary-sized float64 conversion tensor and its extra memory pass.
    """
    if probabilities.device.type != "cuda":
        raise ValueError("CUDA v3 target-interval quantization requires CUDA")
    if probabilities.ndim != 2 or not 1 <= probabilities.shape[1] < total:
        raise ValueError("expected a 2D alphabet smaller than the frequency total")
    probabilities = probabilities.contiguous()
    if validate and (
        not torch.isfinite(probabilities).all() or (probabilities < 0).any()
    ):
        raise ValueError("probabilities must be finite and nonnegative")
    probability_sums = probabilities.sum(dim=1, dtype=torch.float64).contiguous()
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
    return tuple(_load_extension().quantize_target_intervals_compact(
        probabilities, probability_sums, targets, total
    ))


def target_intervals_from_probs_cuda_v4(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    total: int = 262144,
    *,
    validate: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fuse exact fixed-total normalization and target extraction on CUDA."""
    if probabilities.device.type != "cuda":
        raise ValueError("CUDA v4 target-interval quantization requires CUDA")
    if probabilities.ndim != 2 or not 1 < probabilities.shape[1] < total:
        raise ValueError("expected a 2D alphabet smaller than the frequency total")
    probabilities = probabilities.contiguous()
    if validate and (
        not torch.isfinite(probabilities).all() or (probabilities < 0).any()
    ):
        raise ValueError("probabilities must be finite and nonnegative")
    targets = torch.as_tensor(
        targets, device=probabilities.device, dtype=torch.int64
    ).contiguous()
    if targets.ndim != 1 or targets.numel() != probabilities.shape[0]:
        raise ValueError("one target index is required per probability row")
    if validate and (
        (targets < 0).any() or (targets >= probabilities.shape[1]).any()
    ):
        raise ValueError("target outside alphabet")
    return tuple(_load_extension().quantize_target_intervals_exact(
        probabilities, targets, total
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


class CudaPipelinedACEncoder:
    """Persistent v4 encoder whose range updates overlap later inference.

    Target intervals are produced on PyTorch's current stream. Each call records
    that dependency and queues the corresponding arithmetic-state update on a
    dedicated CUDA stream. Stream order preserves each coder lane's sequential
    state while the default stream can advance to the next model invocation.
    """

    def __init__(
        self,
        stream_count: int,
        max_steps: int,
        device: torch.device | str,
        *,
        nominal_total: int = 262144,
        state_bits: int = 32,
    ):
        _validate_settings(stream_count, state_bits, nominal_total)
        if max_steps < 0:
            raise ValueError("max_steps must be nonnegative")
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("pipelined arithmetic encoding requires CUDA")
        if self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.stream_count = stream_count
        self.nominal_total = nominal_total
        self.state_bits = state_bits
        self.workspace_bytes = max(1, (max_steps * (state_bits + 1) + 8) // 8)
        _load_extension()
        self.stream = torch.cuda.Stream(device=self.device)
        int_options = {"dtype": torch.int64, "device": self.device}
        self.lows = torch.zeros(stream_count, **int_options)
        self.highs = torch.full(
            (stream_count,), (1 << state_bits) - 1, **int_options
        )
        self.underflows = torch.zeros(stream_count, **int_options)
        self.workspace = torch.zeros(
            (stream_count, self.workspace_bytes),
            dtype=torch.uint8,
            device=self.device,
        )
        self.bit_counts = torch.zeros(stream_count, **int_options)
        self.byte_counts = torch.zeros(stream_count, **int_options)
        self.current_bytes = torch.zeros(
            stream_count, dtype=torch.uint8, device=self.device
        )
        self.bit_positions = torch.zeros(
            stream_count, dtype=torch.int32, device=self.device
        )
        self.errors = torch.zeros(
            stream_count, dtype=torch.int32, device=self.device
        )
        self._events = []
        self._launches = 0
        self._started = None
        self._finished = False

    def encode_step(
        self,
        lows: torch.Tensor,
        highs: torch.Tensor,
        totals: torch.Tensor,
        active: torch.Tensor,
    ) -> None:
        if self._finished:
            raise RuntimeError("pipelined encoder is already finalized")
        for name, tensor, dtype in (
            ("lows", lows, torch.int64),
            ("highs", highs, torch.int64),
            ("totals", totals, torch.int64),
            ("active", active, torch.bool),
        ):
            if (
                tensor.device != self.device
                or tensor.dtype != dtype
                or tensor.shape != (self.stream_count,)
                or not tensor.is_contiguous()
            ):
                raise ValueError(
                    f"{name} must be contiguous {dtype} on {self.device} "
                    f"with shape ({self.stream_count},)"
                )
        if self._started is None:
            self._started = time.perf_counter()
        producer = torch.cuda.current_stream(self.device)
        ready = torch.cuda.Event()
        ready.record(producer)
        self.stream.wait_event(ready)
        for tensor in (lows, highs, totals, active):
            tensor.record_stream(self.stream)
        with torch.cuda.stream(self.stream):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            _load_extension().encode_intervals_step(
                lows,
                highs,
                totals,
                active,
                self.lows,
                self.highs,
                self.underflows,
                self.workspace,
                self.bit_counts,
                self.byte_counts,
                self.current_bytes,
                self.bit_positions,
                self.errors,
                self.nominal_total,
                self.state_bits,
            )
            end.record()
        self._events.append((start, end))
        self._launches += 1

    def finish(self, counts: torch.Tensor, *, metrics: dict | None = None) -> bytes:
        if self._finished:
            raise RuntimeError("pipelined encoder is already finalized")
        self._finished = True
        finish_started = time.perf_counter()
        with torch.cuda.stream(self.stream):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            _load_extension().finish_intervals(
                self.workspace,
                self.bit_counts,
                self.byte_counts,
                self.current_bytes,
                self.bit_positions,
                self.errors,
            )
            end.record()
        self._events.append((start, end))
        end.synchronize()
        kernel_seconds = sum(
            event_start.elapsed_time(event_end) / 1000.0
            for event_start, event_end in self._events
        )
        transfer_started = time.perf_counter()
        host_bit_counts = self.bit_counts.cpu().tolist()
        host_byte_counts = self.byte_counts.cpu().tolist()
        host_errors = self.errors.cpu().tolist()
        host_workspace = self.workspace.cpu().numpy()
        host_counts = counts.cpu().tolist()
        transfer_seconds = time.perf_counter() - transfer_started
        if any(host_errors):
            code = next(code for code in host_errors if code)
            raise RuntimeError(_ERROR_MESSAGES.get(code, f"unknown CUDA encoder error {code}"))
        descriptors = []
        payloads = []
        for stream in range(self.stream_count):
            byte_count = host_byte_counts[stream]
            bit_count = host_bit_counts[stream]
            if byte_count != (bit_count + 7) // 8:
                raise RuntimeError("CUDA encoder returned inconsistent payload dimensions")
            data = host_workspace[stream, :byte_count].tobytes()
            descriptors.append(
                _STREAM.pack(host_counts[stream], bit_count, zlib.crc32(data))
            )
            payloads.append(data)
        archive = b"".join((
            _HEADER.pack(
                MAGIC,
                EXACT_TARGET_VERSION,
                self.state_bits,
                EXACT_TARGET_FLAG,
                self.nominal_total,
                self.stream_count,
            ),
            *descriptors,
            *payloads,
        ))
        if metrics is not None:
            metrics.update({
                "kernel_seconds": kernel_seconds,
                "device_to_host_seconds": transfer_seconds,
                "device_to_host_bytes": (
                    self.bit_counts.nbytes
                    + self.byte_counts.nbytes
                    + self.errors.nbytes
                    + self.workspace.nbytes
                    + counts.nbytes
                ),
                "workspace_bytes_per_stream": self.workspace_bytes,
                "workspace_retries": 0,
                "archive_bytes": len(archive),
                "total_seconds": time.perf_counter() - finish_started,
                "pipeline_elapsed_seconds": (
                    time.perf_counter() - self._started
                    if self._started is not None
                    else 0.0
                ),
                "pipeline_step_launches": self._launches,
                "pipeline_overlap_enabled": True,
            })
        return archive


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
    exact_fixed: bool = False,
) -> bytes:
    """Encode exact integer intervals and return an MSAC target-interval archive.

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
    version = EXACT_TARGET_VERSION if exact_fixed else FLOOR_TARGET_VERSION
    flags = EXACT_TARGET_FLAG if exact_fixed else FLOOR_TARGET_FLAG
    archive = b"".join((
        _HEADER.pack(MAGIC, version, state_bits, flags, nominal_total, streams),
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
