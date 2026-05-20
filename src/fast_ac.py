"""Fast arithmetic-coding helpers.

This module implements a versioned AC_FAST payload with independent arithmetic
streams. The Python implementation is intentionally API-compatible with a future
native core: callers pass compact per-token intervals to row-local range coders.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import torch

from src.encoding import ArithmeticDecoder, BitInputStream, BitOutputStream


AC_FAST_FORMAT = "AC_FAST_V1"
DEFAULT_TOTAL = 1 << 18


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
    freqs = torch.floor(probs * (total - alphabet_size)).to(torch.int64) + 1
    cumulative = torch.nn.functional.pad(torch.cumsum(freqs, dim=1), (1, 0))

    device = probs.device
    row_idx = torch.arange(probs.shape[0], device=device)
    targets = targets.to(device=device, dtype=torch.long)
    lows = cumulative[row_idx, targets]
    highs = cumulative[row_idx, targets + 1]
    totals = cumulative[:, -1]
    target_probs = probs[row_idx, targets]

    return (
        lows.detach().cpu().numpy().astype(np.int64),
        highs.detach().cpu().numpy().astype(np.int64),
        totals.detach().cpu().numpy().astype(np.int64),
        target_probs.detach().cpu().numpy().astype(np.float64),
    )


class FastACCompressor:
    def __init__(self, stream_count: int, *, statesize: int = 32, total: int = DEFAULT_TOTAL) -> None:
        if stream_count <= 0:
            raise ValueError("stream_count must be positive")
        self.stream_count = stream_count
        self.total = total
        self.statesize = statesize
        self.encoders = [_IntervalArithmeticEncoder(statesize) for _ in range(stream_count)]
        self.timings = FastACTimings()
        self.cross_entropy_sum = 0.0
        self.token_count = 0

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
            self.encoders[int(row_id)].write_interval(int(low), int(high), int(total))
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
        streams = [encoder.finish() for encoder in self.encoders]
        return {
            "format": AC_FAST_FORMAT,
            "stream_count": self.stream_count,
            "statesize": self.statesize,
            "total": self.total,
            "streams": streams,
        }


class FastACDecompressor:
    def __init__(self, payload: Dict[str, Any], *, stream_count: int | None = None) -> None:
        if payload.get("format") != AC_FAST_FORMAT:
            raise ValueError(f"Unsupported AC_FAST payload format: {payload.get('format')}")
        streams = payload["streams"]
        if stream_count is not None and len(streams) != stream_count:
            raise ValueError(f"Expected {stream_count} AC_FAST streams, got {len(streams)}")
        self.total = int(payload.get("total", DEFAULT_TOTAL))
        statesize = int(payload.get("statesize", 32))
        self.decoders = [
            ArithmeticDecoder(statesize, BitInputStream(stream))
            for stream in streams
        ]

    def decompress(self, row_id: int, probs: torch.Tensor | np.ndarray) -> int:
        if isinstance(probs, torch.Tensor):
            probs = probs.detach().to(dtype=torch.float64, device="cpu").numpy()
        cumulative = fast_cumulative_np(probs, total=self.total)
        return self.decoders[int(row_id)].read(cumulative, len(cumulative) - 1)


def payload_size_bits(payload: Any) -> int:
    if isinstance(payload, dict) and payload.get("format") == AC_FAST_FORMAT:
        return sum(len(stream) for stream in payload["streams"])
    return len(payload)


def is_fast_ac_payload(payload: Any) -> bool:
    return isinstance(payload, dict) and payload.get("format") == AC_FAST_FORMAT
