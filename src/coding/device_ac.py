"""Batched arithmetic decoding on the model's torch device.

This is an opt-in reference path for MSAC v2 target-interval archives. It keeps
range state, bit cursors, probability quantization, and inverse-CDF lookup on
one device. Only decoded token IDs cross back to the host for prompt assembly.
"""

from __future__ import annotations

import torch

from src.coding.multistream_ac import MultistreamACDecoder
from src.coding.paired_ac import unpack_paired_archive
from src.coding.target_interval import floor_frequencies


class DeviceMultistreamACDecoder:
    def __init__(self, archive: bytes, device: torch.device | str, *, paired: bool = False):
        if paired:
            archive = unpack_paired_archive(archive)
        checked = MultistreamACDecoder(archive)
        if not checked.target_interval:
            raise ValueError("device decoder currently requires MSAC v2 target intervals")
        self.device = torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        if self.device.type == "mps":
            raise ValueError("float64 target-interval decoding requires CPU or CUDA")
        self.stream_count = checked.stream_count
        self.symbol_counts = checked.symbol_counts
        self.total = checked.total
        self.state_bits = checked.state_bits
        if (1 << self.state_bits) * self.total > (1 << 63) - 1:
            raise ValueError("arithmetic state times frequency total exceeds int64")
        self.mask = (1 << self.state_bits) - 1
        self.top_mask = 1 << (self.state_bits - 1)
        self.second_mask = self.top_mask >> 1
        self.low = torch.zeros(self.stream_count, dtype=torch.int64, device=self.device)
        self.high = torch.full_like(self.low, self.mask)
        self.bit_counts = torch.tensor(checked.bit_counts, dtype=torch.int64, device=self.device)
        self.positions = torch.full_like(self.low, self.state_bits)
        self.decoded = torch.zeros_like(self.low)
        self.counts = torch.tensor(self.symbol_counts, dtype=torch.int64, device=self.device)
        initial_codes = []
        for data, bit_count in zip(checked.payload_bytes, checked.bit_counts):
            code = 0
            for position in range(min(self.state_bits, bit_count)):
                code = (code << 1) | ((data[position // 8] >> (7 - position % 8)) & 1)
            initial_codes.append(code << (self.state_bits - min(self.state_bits, bit_count)))
        self.code = torch.tensor(initial_codes, dtype=torch.int64, device=self.device)
        width = max(map(len, checked.payload_bytes))
        padded = [list(data) + [0] * (width - len(data)) for data in checked.payload_bytes]
        self.payload = torch.tensor(padded, dtype=torch.int64, device=self.device)

    def _next_bits(self, active: torch.Tensor) -> torch.Tensor:
        byte_positions = (self.positions // 8).clamp(max=self.payload.shape[1] - 1)
        values = self.payload.gather(1, byte_positions[:, None]).squeeze(1)
        bits = (values >> (7 - self.positions % 8)) & 1
        bits = torch.where(self.positions < self.bit_counts, bits, 0)
        self.positions += active.to(torch.int64)
        return bits

    def decode(self, probabilities: torch.Tensor, active: torch.Tensor | list[bool]) -> torch.Tensor:
        """Return vocabulary column indices; inactive rows keep their coder state."""
        if probabilities.device != self.device:
            raise ValueError("probabilities and arithmetic state must share a device")
        if probabilities.ndim != 2 or probabilities.shape[0] != self.stream_count:
            raise ValueError("one probability row is required per stream")
        active = torch.as_tensor(active, dtype=torch.bool, device=self.device)
        if active.shape != (self.stream_count,):
            raise ValueError("active mask must contain one value per stream")
        if bool((self.decoded + active.to(torch.int64) > self.counts).any()):
            raise ValueError("stream has no more symbols")
        frequencies, _ = floor_frequencies(probabilities, self.total)
        cumulative = torch.cat((torch.zeros((self.stream_count, 1), dtype=torch.int64,
                                             device=self.device), frequencies.cumsum(dim=1)), dim=1)
        totals = cumulative[:, -1]
        current_range = self.high - self.low + 1
        value = ((self.code - self.low + 1) * totals - 1) // current_range
        symbol = torch.searchsorted(cumulative.contiguous(), value[:, None].contiguous(),
                                    right=True).squeeze(1) - 1
        if bool(((symbol < 0) | (symbol >= probabilities.shape[1])).logical_and(active).any()):
            raise ValueError("arithmetic code selects a symbol outside the alphabet")
        symbol = symbol.clamp(0, probabilities.shape[1] - 1)
        lower = cumulative.gather(1, symbol[:, None]).squeeze(1)
        upper = cumulative.gather(1, (symbol + 1)[:, None]).squeeze(1)
        old_low = self.low
        self.low = torch.where(active, old_low + lower * current_range // totals, old_low)
        self.high = torch.where(active, old_low + upper * current_range // totals - 1, self.high)
        while True:
            shifting = active & (((self.low ^ self.high) & self.top_mask) == 0)
            if not bool(shifting.any()):
                break
            bits = self._next_bits(shifting)
            self.code = torch.where(shifting, ((self.code << 1) & self.mask) | bits, self.code)
            self.low = torch.where(shifting, (self.low << 1) & self.mask, self.low)
            self.high = torch.where(shifting, ((self.high << 1) & self.mask) | 1, self.high)
        while True:
            underflow = active & ((self.low & ~self.high & self.second_mask) != 0)
            if not bool(underflow.any()):
                break
            bits = self._next_bits(underflow)
            self.code = torch.where(
                underflow,
                (self.code & self.top_mask) | ((self.code << 1) & (self.mask >> 1)) | bits,
                self.code,
            )
            self.low = torch.where(underflow, (self.low << 1) & (self.mask >> 1), self.low)
            self.high = torch.where(
                underflow, ((self.high << 1) & (self.mask >> 1)) | self.top_mask | 1,
                self.high,
            )
        self.decoded += active.to(torch.int64)
        return symbol

    def assert_complete(self) -> None:
        if self.decoded.cpu().tolist() != self.symbol_counts:
            raise ValueError("not all MSAC stream symbols were decoded")


class CudaCDFMultistreamACDecoder(DeviceMultistreamACDecoder):
    """CUDA v1 decoder: PyTorch CDF construction plus fused AC state update."""

    def __init__(self, archive: bytes, device: torch.device | str = "cuda", *, paired: bool = False):
        super().__init__(archive, device, paired=paired)
        if self.device.type != "cuda":
            raise ValueError("CUDA CDF decoding requires a CUDA device")
        self.symbols = torch.zeros_like(self.low)
        self.errors = torch.zeros(
            self.stream_count, dtype=torch.int32, device=self.device
        )
        self._quantization_events = []
        self._kernel_events = []
        self.metrics = None

    def decode(self, probabilities: torch.Tensor, active: torch.Tensor | list[bool]) -> torch.Tensor:
        """Decode one symbol per active stream without host-visible state loops."""
        from src.coding.cuda_ac import decode_cdfs_cuda_raw

        if probabilities.device != self.device:
            raise ValueError("probabilities and arithmetic state must share a device")
        if probabilities.ndim != 2 or probabilities.shape[0] != self.stream_count:
            raise ValueError("one probability row is required per stream")
        active = torch.as_tensor(
            active, dtype=torch.bool, device=self.device
        ).contiguous()
        if active.shape != (self.stream_count,):
            raise ValueError("active mask must contain one value per stream")

        quantization_start = torch.cuda.Event(enable_timing=True)
        quantization_end = torch.cuda.Event(enable_timing=True)
        kernel_end = torch.cuda.Event(enable_timing=True)
        quantization_start.record()
        frequencies, _ = floor_frequencies(
            probabilities, self.total, validate=False
        )
        cdfs = torch.cat((
            torch.zeros(
                (self.stream_count, 1), dtype=torch.int64, device=self.device
            ),
            frequencies.cumsum(dim=1),
        ), dim=1).contiguous()
        quantization_end.record()
        decode_cdfs_cuda_raw(
            cdfs,
            active,
            self.payload,
            self.bit_counts,
            self.counts,
            self.low,
            self.high,
            self.code,
            self.positions,
            self.decoded,
            self.symbols,
            self.errors,
            nominal_total=self.total,
            state_bits=self.state_bits,
        )
        kernel_end.record()
        self._quantization_events.append((quantization_start, quantization_end))
        self._kernel_events.append((quantization_end, kernel_end))
        return self.symbols

    def assert_complete(self) -> None:
        from src.coding.cuda_ac import decode_error_message

        host_errors = self.errors.cpu().tolist()
        if any(host_errors):
            code = next(code for code in host_errors if code)
            raise ValueError(decode_error_message(code))
        host_decoded = self.decoded.cpu().tolist()
        if host_decoded != self.symbol_counts:
            raise ValueError("not all MSAC stream symbols were decoded")
        self.metrics = {
            "quantization_seconds": sum(
                start.elapsed_time(end) / 1000.0
                for start, end in self._quantization_events
            ),
            "kernel_seconds": sum(
                start.elapsed_time(end) / 1000.0
                for start, end in self._kernel_events
            ),
            "decode_steps": len(self._kernel_events),
        }
