"""Fast arithmetic-coding helpers.

This module implements a versioned AC_FAST payload with independent arithmetic
streams. The Python implementation is intentionally API-compatible with a future
native core: callers pass compact per-token intervals to row-local range coders.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import os
from typing import Any, Dict, List, Sequence

import numpy as np
import torch

from src.encoding import ArithmeticDecoder, BitInputStream, BitOutputStream

try:
    from numba import njit
except Exception:  # pragma: no cover - exercised when numba is absent
    njit = None


AC_FAST_FORMAT = "AC_FAST_V1"
AC_FAST2_FORMAT = "AC_FAST2_V1"
DEFAULT_TOTAL = 1 << 18


if njit is not None:
    @njit(cache=True, nogil=True)
    def _numba_encode_intervals(
        lows: np.ndarray,
        highs: np.ndarray,
        totals: np.ndarray,
        statesize: int,
    ) -> np.ndarray:
        max_range = 1 << statesize
        mask = max_range - 1
        top_mask = max_range >> 1
        second_mask = top_mask >> 1
        low = 0
        high = mask
        num_underflow = 0
        bits = []

        for i in range(lows.size):
            current_range = high - low + 1
            base_low = low
            total = int(totals[i])
            symlow = int(lows[i])
            symhigh = int(highs[i])

            low = base_low + symlow * current_range // total
            high = base_low + symhigh * current_range // total - 1

            while ((low ^ high) & top_mask) == 0:
                bit = low >> (statesize - 1)
                bits.append(bit)
                for _ in range(num_underflow):
                    bits.append(bit ^ 1)
                num_underflow = 0
                low = (low << 1) & mask
                high = ((high << 1) & mask) | 1

            while (low & ~high & second_mask) != 0:
                num_underflow += 1
                low = (low << 1) & (mask >> 1)
                high = ((high << 1) & (mask >> 1)) | top_mask | 1

        bits.append(1)
        return np.asarray(bits, dtype=np.uint8)

    @njit(cache=True, nogil=True)
    def _numba_encode_intervals_packed(
        lows: np.ndarray,
        highs: np.ndarray,
        totals: np.ndarray,
        statesize: int,
    ) -> tuple[np.ndarray, int]:
        max_range = 1 << statesize
        mask = max_range - 1
        top_mask = max_range >> 1
        second_mask = top_mask >> 1
        low = 0
        high = mask
        num_underflow = 0
        out = []
        current_byte = 0
        bit_pos = 0
        bit_count = 0

        for i in range(lows.size):
            current_range = high - low + 1
            base_low = low
            total = int(totals[i])
            symlow = int(lows[i])
            symhigh = int(highs[i])

            low = base_low + symlow * current_range // total
            high = base_low + symhigh * current_range // total - 1

            while ((low ^ high) & top_mask) == 0:
                bit = low >> (statesize - 1)
                current_byte = (current_byte << 1) | bit
                bit_pos += 1
                bit_count += 1
                if bit_pos == 8:
                    out.append(current_byte)
                    current_byte = 0
                    bit_pos = 0
                inverse = bit ^ 1
                for _ in range(num_underflow):
                    current_byte = (current_byte << 1) | inverse
                    bit_pos += 1
                    bit_count += 1
                    if bit_pos == 8:
                        out.append(current_byte)
                        current_byte = 0
                        bit_pos = 0
                num_underflow = 0
                low = (low << 1) & mask
                high = ((high << 1) & mask) | 1

            while (low & ~high & second_mask) != 0:
                num_underflow += 1
                low = (low << 1) & (mask >> 1)
                high = ((high << 1) & (mask >> 1)) | top_mask | 1

        current_byte = (current_byte << 1) | 1
        bit_pos += 1
        bit_count += 1
        if bit_pos == 8:
            out.append(current_byte)
        else:
            out.append(current_byte << (8 - bit_pos))
        return np.asarray(out, dtype=np.uint8), bit_count
else:
    _numba_encode_intervals = None
    _numba_encode_intervals_packed = None


class _PackedBitInputStream:
    def __init__(self, byte_data: bytes | bytearray, bit_count: int) -> None:
        self._data = byte_data
        self._bit_count = int(bit_count)
        self._index = 0

    def read(self) -> int:
        if self._index >= self._bit_count:
            return -1
        byte = int(self._data[self._index >> 3])
        bit = (byte >> (7 - (self._index & 7))) & 1
        self._index += 1
        return bit


class _IntervalArithmeticEncoder:
    def __init__(self, statesize: int = 32) -> None:
        self.bitout = BitOutputStream()
        self.statesize = statesize
        self.max_range = 1 << statesize
        self.min_range = (self.max_range >> 2) + 2
        self.mask = self.max_range - 1
        self.top_mask = self.max_range >> 1
        self.second_mask = self.top_mask >> 1
        self.low = 0
        self.high = self.mask
        self.num_underflow = 0

    def write_interval(self, symlow: int, symhigh: int, total: int) -> None:
        low = self.low
        high = self.high
        current_range = high - low + 1

        self.low = low + symlow * current_range // total
        self.high = low + symhigh * current_range // total - 1

        while ((self.low ^ self.high) & self.top_mask) == 0:
            self._shift()
            self.low = (self.low << 1) & self.mask
            self.high = ((self.high << 1) & self.mask) | 1

        while (self.low & ~self.high & self.second_mask) != 0:
            self._underflow()
            self.low = (self.low << 1) & (self.mask >> 1)
            self.high = ((self.high << 1) & (self.mask >> 1)) | self.top_mask | 1

    def finish(self) -> List[int]:
        self.bitout.write(1)
        return self.bitout.get_bits()

    def _shift(self) -> None:
        bit = self.low >> (self.statesize - 1)
        self.bitout.write(bit)
        for _ in range(self.num_underflow):
            self.bitout.write(bit ^ 1)
        self.num_underflow = 0

    def _underflow(self) -> None:
        self.num_underflow += 1


@dataclass
class FastACTimings:
    quantize_time: float = 0.0
    range_coder_time: float = 0.0
    transfer_time: float = 0.0

    @property
    def total(self) -> float:
        return self.quantize_time + self.range_coder_time + self.transfer_time


def _normalize_probs_np(probs: np.ndarray) -> np.ndarray:
    probs = np.asarray(probs, dtype=np.float64)
    if probs.ndim != 1:
        raise ValueError("probabilities must be a 1D vector")
    if probs.size == 0:
        raise ValueError("probabilities must not be empty")
    probs = np.clip(probs, 0.0, None)
    prob_sum = float(probs.sum())
    if prob_sum <= 0:
        raise ValueError("at least one probability must be positive")
    return probs / prob_sum


def fast_frequencies_np(probs: np.ndarray, total: int = DEFAULT_TOTAL) -> np.ndarray:
    probs = _normalize_probs_np(probs)
    alphabet_size = probs.size
    if total < alphabet_size:
        raise ValueError("total must be at least the alphabet size")
    return np.floor(probs * (total - alphabet_size)).astype(np.int64) + 1


def fast_cumulative_np(probs: np.ndarray, total: int = DEFAULT_TOTAL) -> np.ndarray:
    freqs = fast_frequencies_np(probs, total=total)
    cumulative = np.empty(freqs.size + 1, dtype=np.int64)
    cumulative[0] = 0
    cumulative[1:] = np.cumsum(freqs)
    return cumulative


def _intervals_from_probs_tensor(
    probs: torch.Tensor,
    targets: torch.Tensor,
    total: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if probs.ndim != 2:
        raise ValueError(f"Expected 2D probabilities, got shape {tuple(probs.shape)}")
    if targets.ndim != 1 or targets.numel() != probs.shape[0]:
        raise ValueError("targets must be a 1D tensor with one item per probability row")

    probs = probs.to(dtype=torch.float64)
    probs = torch.clamp(probs, min=0.0)
    probs = probs / probs.sum(dim=1, keepdim=True).clamp_min(torch.finfo(torch.float64).tiny)
    alphabet_size = probs.shape[1]
    floor_counts = torch.floor(probs * (total - alphabet_size)).to(torch.int64)

    device = probs.device
    row_idx = torch.arange(probs.shape[0], device=device)
    targets = targets.to(device=device, dtype=torch.long)
    before_target = (
        torch.cumsum(floor_counts, dim=1)[row_idx, targets]
        - floor_counts[row_idx, targets]
    )
    lows = targets.to(torch.int64) + before_target
    highs = lows + floor_counts[row_idx, targets] + 1
    totals = floor_counts.sum(dim=1) + alphabet_size
    target_probs = probs[row_idx, targets]

    return (
        lows.detach().cpu().numpy().astype(np.int64),
        highs.detach().cpu().numpy().astype(np.int64),
        totals.detach().cpu().numpy().astype(np.int64),
        target_probs.detach().cpu().numpy().astype(np.float64),
    )


def target_intervals_from_probs_tensor(
    probs: torch.Tensor,
    targets: torch.Tensor,
    total: int = DEFAULT_TOTAL,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute target-only AC intervals without materializing full cumulatives."""
    if probs.ndim != 2:
        raise ValueError(f"Expected 2D probabilities, got shape {tuple(probs.shape)}")
    if targets.ndim != 1 or targets.numel() != probs.shape[0]:
        raise ValueError("targets must be a 1D tensor with one item per probability row")

    probs = probs.to(dtype=torch.float64)
    probs = torch.clamp(probs, min=0.0)
    probs = probs / probs.sum(dim=1, keepdim=True).clamp_min(torch.finfo(torch.float64).tiny)
    alphabet_size = probs.shape[1]
    floor_counts = torch.floor(probs * (total - alphabet_size)).to(torch.int64)

    device = probs.device
    row_idx = torch.arange(probs.shape[0], device=device)
    targets = targets.to(device=device, dtype=torch.long)
    before_target = (
        floor_counts.cumsum(dim=1)[row_idx, targets]
        - floor_counts[row_idx, targets]
    )
    lows = targets.to(torch.int64) + before_target
    highs = lows + floor_counts[row_idx, targets] + 1
    totals = floor_counts.sum(dim=1) + alphabet_size
    target_probs = probs[row_idx, targets]
    return lows, highs, totals, target_probs


class FastACCompressor:
    def __init__(
        self,
        stream_count: int,
        *,
        statesize: int = 32,
        total: int = DEFAULT_TOTAL,
        backend: str = "auto",
        payload_format: str = AC_FAST_FORMAT,
        threads: int | None = None,
    ) -> None:
        if stream_count <= 0:
            raise ValueError("stream_count must be positive")
        if backend not in {"auto", "python", "numba", "numba_threaded", "numba_packed"}:
            raise ValueError("backend must be one of: auto, python, numba, numba_threaded, numba_packed")
        if backend == "auto":
            backend = "numba_packed" if _numba_encode_intervals_packed is not None else "python"
        if backend in {"numba", "numba_threaded", "numba_packed"} and _numba_encode_intervals is None:
            raise RuntimeError("AC_FAST numba backend requested, but numba is not installed")
        if payload_format not in {AC_FAST_FORMAT, AC_FAST2_FORMAT}:
            raise ValueError(f"Unsupported AC_FAST payload format: {payload_format}")
        if backend in {"numba", "numba_threaded"}:
            _numba_encode_intervals(
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.int64),
                statesize,
            )
        if backend == "numba_packed":
            _numba_encode_intervals_packed(
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.int64),
                statesize,
            )

        self.stream_count = stream_count
        self.total = total
        self.statesize = statesize
        self.backend = backend
        self.threads = self._resolve_threads(threads, stream_count, backend)
        self.payload_format = payload_format
        self.encoders = (
            [_IntervalArithmeticEncoder(statesize) for _ in range(stream_count)]
            if backend == "python"
            else None
        )
        self._interval_lows = [[] for _ in range(stream_count)]
        self._interval_highs = [[] for _ in range(stream_count)]
        self._interval_totals = [[] for _ in range(stream_count)]
        self.timings = FastACTimings()
        self.cross_entropy_sum = 0.0
        self.token_count = 0

    @staticmethod
    def _resolve_threads(threads: int | None, stream_count: int, backend: str) -> int:
        if backend not in {"numba_threaded", "numba_packed"}:
            return 1
        if threads is None or threads <= 0:
            cpu_count = os.cpu_count() or 1
            return max(1, min(stream_count, cpu_count))
        return max(1, min(stream_count, int(threads)))

    def encode_intervals_batch(
        self,
        row_ids: Sequence[int],
        lows: Sequence[int] | np.ndarray | torch.Tensor,
        highs: Sequence[int] | np.ndarray | torch.Tensor,
        totals: Sequence[int] | np.ndarray | torch.Tensor,
        target_probs: Sequence[float] | np.ndarray | torch.Tensor,
    ) -> None:
        if not row_ids:
            return
        lows_np = self._to_numpy(lows, np.int64)
        highs_np = self._to_numpy(highs, np.int64)
        totals_np = self._to_numpy(totals, np.int64)
        probs_np = self._to_numpy(target_probs, np.float64)
        if not (len(row_ids) == len(lows_np) == len(highs_np) == len(totals_np) == len(probs_np)):
            raise ValueError("interval arrays must have one item per row_id")

        t0 = time.perf_counter()
        for row_id, low, high, total, prob in zip(row_ids, lows_np, highs_np, totals_np, probs_np):
            row_id = int(row_id)
            low = int(low)
            high = int(high)
            total = int(total)
            if self.backend == "python":
                self.encoders[row_id].write_interval(low, high, total)
            else:
                self._interval_lows[row_id].append(low)
                self._interval_highs[row_id].append(high)
                self._interval_totals[row_id].append(total)
            self.cross_entropy_sum += -float(np.log2(max(float(prob), 1e-300)))
            self.token_count += 1
        self.timings.range_coder_time += time.perf_counter() - t0

    @staticmethod
    def _to_numpy(values, dtype):
        if isinstance(values, torch.Tensor):
            return values.detach().cpu().numpy().astype(dtype, copy=False)
        return np.asarray(values, dtype=dtype)

    def encode_batch(
        self,
        row_ids: Sequence[int],
        target_token_ids: Sequence[int],
        probs: torch.Tensor | np.ndarray,
    ) -> None:
        if len(row_ids) != len(target_token_ids):
            raise ValueError("row_ids and target_token_ids must have equal length")
        if not row_ids:
            return

        t0 = time.perf_counter()
        if not isinstance(probs, torch.Tensor):
            probs = torch.as_tensor(probs)
        targets = torch.as_tensor(target_token_ids, dtype=torch.long, device=probs.device)
        lows, highs, totals, target_probs = _intervals_from_probs_tensor(probs, targets, self.total)
        self.timings.quantize_time += time.perf_counter() - t0

        t0 = time.perf_counter()
        for row_id, low, high, total, prob in zip(row_ids, lows, highs, totals, target_probs):
            row_id = int(row_id)
            low = int(low)
            high = int(high)
            total = int(total)
            if self.backend == "python":
                self.encoders[row_id].write_interval(low, high, total)
            else:
                self._interval_lows[row_id].append(low)
                self._interval_highs[row_id].append(high)
                self._interval_totals[row_id].append(total)
            self.cross_entropy_sum += -float(np.log2(max(float(prob), 1e-300)))
            self.token_count += 1
        self.timings.range_coder_time += time.perf_counter() - t0

    def encode_one(self, row_id: int, target_token_id: int, probs: torch.Tensor | np.ndarray) -> None:
        if isinstance(probs, torch.Tensor):
            probs_2d = probs.reshape(1, -1)
        else:
            probs_2d = np.asarray(probs).reshape(1, -1)
        self.encode_batch([row_id], [target_token_id], probs_2d)

    def finish(self) -> Dict[str, Any]:
        t0 = time.perf_counter()
        bit_counts = None
        if self.backend == "python":
            streams = [encoder.finish() for encoder in self.encoders]
        elif self.backend == "numba_threaded":
            arrays = [
                (
                    np.asarray(lows, dtype=np.int64),
                    np.asarray(highs, dtype=np.int64),
                    np.asarray(totals, dtype=np.int64),
                )
                for lows, highs, totals in zip(
                    self._interval_lows,
                    self._interval_highs,
                    self._interval_totals,
                )
            ]
            with ThreadPoolExecutor(max_workers=self.threads) as executor:
                streams = list(
                    executor.map(
                        lambda arrays_for_stream: _numba_encode_intervals(
                            arrays_for_stream[0],
                            arrays_for_stream[1],
                            arrays_for_stream[2],
                            self.statesize,
                        ).tolist(),
                        arrays,
                    )
                )
        elif self.backend == "numba_packed":
            arrays = [
                (
                    np.asarray(lows, dtype=np.int64),
                    np.asarray(highs, dtype=np.int64),
                    np.asarray(totals, dtype=np.int64),
                )
                for lows, highs, totals in zip(
                    self._interval_lows,
                    self._interval_highs,
                    self._interval_totals,
                )
            ]
            with ThreadPoolExecutor(max_workers=self.threads) as executor:
                encoded = list(
                    executor.map(
                        lambda arrays_for_stream: _numba_encode_intervals_packed(
                            arrays_for_stream[0],
                            arrays_for_stream[1],
                            arrays_for_stream[2],
                            self.statesize,
                        ),
                        arrays,
                    )
                )
            streams = [bytes(byte_arr.tolist()) for byte_arr, _bit_count in encoded]
            bit_counts = [int(bit_count) for _byte_arr, bit_count in encoded]
        else:
            streams = []
            for lows, highs, totals in zip(
                self._interval_lows,
                self._interval_highs,
                self._interval_totals,
            ):
                low_arr = np.asarray(lows, dtype=np.int64)
                high_arr = np.asarray(highs, dtype=np.int64)
                total_arr = np.asarray(totals, dtype=np.int64)
                bits = _numba_encode_intervals(
                    low_arr,
                    high_arr,
                    total_arr,
                    self.statesize,
                )
                streams.append(bits.tolist())
        self.timings.range_coder_time += time.perf_counter() - t0
        stream_mode = "packed_bytes" if self.backend == "numba_packed" else "bits"
        return {
            "format": self.payload_format,
            "stream_count": self.stream_count,
            "statesize": self.statesize,
            "total": self.total,
            "backend": self.backend,
            "threads": self.threads,
            "streams": streams,
            "stream_mode": stream_mode,
            "bit_counts": bit_counts if bit_counts is not None else [len(stream) for stream in streams],
        }


class FastACDecompressor:
    def __init__(self, payload: Dict[str, Any], *, stream_count: int | None = None) -> None:
        if payload.get("format") not in {AC_FAST_FORMAT, AC_FAST2_FORMAT}:
            raise ValueError(f"Unsupported AC_FAST payload format: {payload.get('format')}")
        streams = payload["streams"]
        if stream_count is not None and len(streams) != stream_count:
            raise ValueError(f"Expected {stream_count} AC_FAST streams, got {len(streams)}")
        self.total = int(payload.get("total", DEFAULT_TOTAL))
        statesize = int(payload.get("statesize", 32))
        stream_mode = payload.get("stream_mode", "bits")
        bit_counts = payload.get("bit_counts")
        if stream_mode == "packed_bytes" and bit_counts is None:
            raise ValueError("Packed AC_FAST payload is missing bit_counts")
        self.decoders = [
            ArithmeticDecoder(
                statesize,
                _PackedBitInputStream(stream, bit_counts[idx])
                if stream_mode == "packed_bytes"
                else BitInputStream(stream),
            )
            for idx, stream in enumerate(streams)
        ]

    def decompress(self, row_id: int, probs: torch.Tensor | np.ndarray) -> int:
        if isinstance(probs, torch.Tensor):
            probs = probs.detach().to(dtype=torch.float64, device="cpu").numpy()
        cumulative = fast_cumulative_np(probs, total=self.total)
        return self.decoders[int(row_id)].read(cumulative, len(cumulative) - 1)


def payload_size_bits(payload: Any) -> int:
    if isinstance(payload, dict) and payload.get("format") in {AC_FAST_FORMAT, AC_FAST2_FORMAT}:
        if payload.get("stream_mode") == "packed_bytes":
            return int(sum(payload.get("bit_counts", [])))
        return sum(len(stream) for stream in payload["streams"])
    return len(payload)


def is_fast_ac_payload(payload: Any) -> bool:
    return isinstance(payload, dict) and payload.get("format") in {AC_FAST_FORMAT, AC_FAST2_FORMAT}
