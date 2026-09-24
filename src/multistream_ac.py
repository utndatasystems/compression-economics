"""Portable, independently decodable arithmetic-coded streams.

This is a coder payload, not a text archive: the caller supplies the same
probability vectors, stream assignment, and model context during decoding.
The MSAC v1 format uses network byte order and records every stream's symbol
count, bit count, and CRC32. It deliberately uses the existing ``build_cumul``
quantizer so coder comparisons can hold the coding distribution constant.
"""

from __future__ import annotations

import math
import struct
import zlib
from typing import Sequence

import numpy as np

from src.encoding import ArithmeticDecoder, ArithmeticEncoder, BitInputStream, BitOutputStream
from src.encoding_utils import build_cumul


MAGIC = b"MSAC"
VERSION = 1
_HEADER = struct.Struct(">4sBBHII")  # magic, version, state bits, flags, total, streams
_STREAM = struct.Struct(">QQI")  # symbol count, payload bit count, CRC32


def _validate_settings(stream_count: int, state_bits: int, total: int) -> None:
    if not 1 <= stream_count <= 0xFFFFFFFF:
        raise ValueError("stream_count must fit in a positive uint32")
    if not 16 <= state_bits <= 56:
        raise ValueError("state_bits must be between 16 and 56")
    if not 1 <= total <= min(0xFFFFFFFF, (1 << (state_bits - 2)) + 2):
        raise ValueError("frequency total is outside the arithmetic coder's range")


def _cumulative(probabilities: Sequence[float] | np.ndarray, total: int) -> np.ndarray:
    row = np.asarray(probabilities, dtype=np.float64)
    if row.ndim != 1 or row.size < 2 or row.size >= total:
        raise ValueError("probabilities must be a 1D alphabet smaller than the total")
    if not np.all(np.isfinite(row)) or np.any(row < 0):
        raise ValueError("probabilities must be finite and nonnegative")
    if not math.isclose(float(row.sum()), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("probabilities must sum to one")
    return build_cumul(row, total=total)


class MultistreamACEncoder:
    """Incrementally encode symbols into separate AC states."""

    def __init__(self, stream_count: int, *, total: int = 262144, state_bits: int = 32):
        _validate_settings(stream_count, state_bits, total)
        self.stream_count = stream_count
        self.total = total
        self.state_bits = state_bits
        self._outputs = [BitOutputStream() for _ in range(stream_count)]
        self._encoders = [ArithmeticEncoder(state_bits, out) for out in self._outputs]
        self._counts = [0] * stream_count
        self._finished = False

    def encode(self, stream_id: int, symbol: int, probabilities: Sequence[float] | np.ndarray) -> None:
        if self._finished:
            raise ValueError("encoder has already been finished")
        if not 0 <= stream_id < self.stream_count:
            raise ValueError("stream_id outside archive")
        cumulative = _cumulative(probabilities, self.total)
        if not 0 <= symbol < len(cumulative) - 1:
            raise ValueError("symbol outside alphabet")
        self._encoders[stream_id].write(cumulative, symbol)
        self._counts[stream_id] += 1

    def finish(self) -> bytes:
        if self._finished:
            raise ValueError("encoder has already been finished")
        self._finished = True
        payloads = []
        descriptors = []
        for encoder, output, count in zip(self._encoders, self._outputs, self._counts):
            encoder.finish()
            bits = np.asarray(output.get_bits(), dtype=np.uint8)
            data = np.packbits(bits, bitorder="big").tobytes()
            descriptors.append(_STREAM.pack(count, len(bits), zlib.crc32(data)))
            payloads.append(data)
        return b"".join((
            _HEADER.pack(MAGIC, VERSION, self.state_bits, 0, self.total, self.stream_count),
            *descriptors,
            *payloads,
        ))


class MultistreamACDecoder:
    """Validate an MSAC v1 payload and decode each stream independently."""

    def __init__(self, archive: bytes):
        if len(archive) < _HEADER.size:
            raise ValueError("truncated MSAC header")
        magic, version, state_bits, flags, total, stream_count = _HEADER.unpack_from(archive)
        if magic != MAGIC or version != VERSION or flags != 0:
            raise ValueError("invalid or unsupported MSAC header")
        _validate_settings(stream_count, state_bits, total)
        descriptor_end = _HEADER.size + stream_count * _STREAM.size
        if len(archive) < descriptor_end:
            raise ValueError("truncated MSAC stream directory")

        self.stream_count = stream_count
        self.total = total
        self.state_bits = state_bits
        self.symbol_counts = []
        self.bit_counts = []
        self._decoders = []
        self._decoded = [0] * stream_count
        offset = descriptor_end
        for stream_id in range(stream_count):
            count, bit_count, checksum = _STREAM.unpack_from(
                archive, _HEADER.size + stream_id * _STREAM.size
            )
            byte_count = (bit_count + 7) // 8
            data = archive[offset:offset + byte_count]
            if bit_count < 1 or len(data) != byte_count:
                raise ValueError("truncated MSAC stream payload")
            if zlib.crc32(data) != checksum:
                raise ValueError("MSAC stream checksum mismatch")
            bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8), bitorder="big")
            if np.any(bits[bit_count:]):
                raise ValueError("nonzero MSAC padding bits")
            self._decoders.append(ArithmeticDecoder(state_bits, BitInputStream(bits[:bit_count].tolist())))
            self.symbol_counts.append(count)
            self.bit_counts.append(bit_count)
            offset += byte_count
        if offset != len(archive):
            raise ValueError("trailing bytes in MSAC payload")

    @property
    def payload_bits(self) -> int:
        return sum(self.bit_counts)

    @property
    def framing_bytes(self) -> int:
        return _HEADER.size + self.stream_count * _STREAM.size

    def decode(self, stream_id: int, probabilities: Sequence[float] | np.ndarray) -> int:
        if not 0 <= stream_id < self.stream_count:
            raise ValueError("stream_id outside archive")
        if self._decoded[stream_id] >= self.symbol_counts[stream_id]:
            raise ValueError("stream has no more symbols")
        cumulative = _cumulative(probabilities, self.total)
        symbol = self._decoders[stream_id].read(cumulative, len(cumulative) - 1)
        self._decoded[stream_id] += 1
        return symbol

    def assert_complete(self) -> None:
        if self._decoded != self.symbol_counts:
            raise ValueError("not all MSAC stream symbols were decoded")
