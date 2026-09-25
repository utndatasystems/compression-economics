"""Encode several independent arithmetic streams in one portable byte payload.

Each stream keeps its own arithmetic coder state. Decoding still needs the same
probabilities used during encoding. The calling text archive stores the
initial tokens and model settings. The model remains an external dependency.
MSAC v1 stores stream lengths and
checksums in big-endian fields. It uses the same probability quantizer as the
existing single-stream arithmetic coder.
"""

from __future__ import annotations

import math
import os
import struct
import time
import zlib
from concurrent.futures import ThreadPoolExecutor
from typing import Sequence

import numpy as np

from src.coding.encoding import ArithmeticDecoder, ArithmeticEncoder, BitInputStream, BitOutputStream
from src.coding.encoding_utils import build_cumul, validate_frequency_quantizer
from src.coding.parallel_ac import encode_intervals_packed, prepare_parallel_encoder


MAGIC = b"MSAC"
VERSION = 1
_HEADER = struct.Struct(">4sBBHII")  # magic, version, state bits, flags, total, streams
_STREAM = struct.Struct(">QQI")  # symbol count, payload bit count, CRC32


def _validate_settings(stream_count: int, state_bits: int, total: int) -> None:
    """Check that coder settings can be stored and used by the AC state machine."""
    if not 1 <= stream_count <= 0xFFFFFFFF:
        raise ValueError("stream_count must fit in a positive uint32")
    if not 16 <= state_bits <= 56:
        raise ValueError("state_bits must be between 16 and 56")
    if not 1 <= total <= min(0xFFFFFFFF, (1 << (state_bits - 2)) + 2):
        raise ValueError("frequency total is outside the arithmetic coder's range")


def _cumulative(probabilities: Sequence[float] | np.ndarray, total: int,
                frequency_quantizer: str = "reference") -> np.ndarray:
    """Validate one model distribution before applying the shared quantizer."""
    row = np.asarray(probabilities)
    if not np.issubdtype(row.dtype, np.floating):
        row = row.astype(np.float64)
    if row.ndim != 1 or row.size < 2 or row.size >= total:
        raise ValueError("probabilities must be a 1D alphabet smaller than the total")
    if not np.all(np.isfinite(row)) or np.any(row < 0):
        raise ValueError("probabilities must be finite and nonnegative")
    if not math.isclose(float(row.sum()), 1.0, rel_tol=0.0, abs_tol=1e-5):
        raise ValueError("probabilities must sum to one")
    return build_cumul(row, total=total, method=frequency_quantizer)


class MultistreamACEncoder:
    """Encode independent streams with Python or GIL-free Numba workers."""

    def __init__(self, stream_count: int, *, total: int = 262144, state_bits: int = 32,
                 backend: str = "python", threads: int | None = None,
                 frequency_quantizer: str = "reference"):
        _validate_settings(stream_count, state_bits, total)
        if backend not in {"python", "numba_parallel"}:
            raise ValueError("backend must be python or numba_parallel")
        if threads is not None and threads < 1:
            raise ValueError("threads must be positive")
        if backend == "numba_parallel":
            prepare_parallel_encoder(state_bits, total)
        self.frequency_quantizer = validate_frequency_quantizer(frequency_quantizer)
        self.stream_count = stream_count
        self.total = total
        self.state_bits = state_bits
        self.backend = backend
        self.threads = min(stream_count, threads or os.cpu_count() or 1) if backend == "numba_parallel" else 1
        self._outputs = [BitOutputStream() for _ in range(stream_count)] if backend == "python" else []
        self._encoders = [ArithmeticEncoder(state_bits, out) for out in self._outputs]
        self._lows = [[] for _ in range(stream_count)] if backend == "numba_parallel" else []
        self._highs = [[] for _ in range(stream_count)] if backend == "numba_parallel" else []
        self._counts = [0] * stream_count
        self._finished = False
        self.quantize_seconds = 0.0
        self.range_encode_seconds = 0.0

    def encode(self, stream_id: int, symbol: int, probabilities: Sequence[float] | np.ndarray) -> None:
        if self._finished:
            raise ValueError("encoder has already been finished")
        if not 0 <= stream_id < self.stream_count:
            raise ValueError("stream_id outside archive")
        start = time.perf_counter()
        cumulative = _cumulative(probabilities, self.total, self.frequency_quantizer)
        if not 0 <= symbol < len(cumulative) - 1:
            raise ValueError("symbol outside alphabet")
        self.quantize_seconds += time.perf_counter() - start
        start = time.perf_counter()
        if self.backend == "python":
            self._encoders[stream_id].write(cumulative, symbol)
        else:
            self._lows[stream_id].append(int(cumulative[symbol]))
            self._highs[stream_id].append(int(cumulative[symbol + 1]))
        self.range_encode_seconds += time.perf_counter() - start
        self._counts[stream_id] += 1

    def finish(self) -> bytes:
        """Finalize streams and write the unchanged portable MSAC v1 format."""
        if self._finished:
            raise ValueError("encoder has already been finished")
        self._finished = True
        payloads = []
        descriptors = []
        if self.backend == "python":
            encoded = []
            for encoder, output in zip(self._encoders, self._outputs):
                start = time.perf_counter()
                encoder.finish()
                self.range_encode_seconds += time.perf_counter() - start
                bits = np.asarray(output.get_bits(), dtype=np.uint8)
                encoded.append((np.packbits(bits, bitorder="big").tobytes(), len(bits)))
        else:
            arrays = [
                (np.asarray(lows, dtype=np.int64), np.asarray(highs, dtype=np.int64))
                for lows, highs in zip(self._lows, self._highs)
            ]
            start = time.perf_counter()
            with ThreadPoolExecutor(max_workers=self.threads) as executor:
                results = list(executor.map(
                    lambda pair: encode_intervals_packed(pair[0], pair[1], self.total, self.state_bits),
                    arrays,
                ))
            self.range_encode_seconds += time.perf_counter() - start
            encoded = [(array.tobytes(), int(bits)) for array, bits in results]
        for (data, bit_count), count in zip(encoded, self._counts):
            descriptors.append(_STREAM.pack(count, bit_count, zlib.crc32(data)))
            payloads.append(data)
        return b"".join((
            _HEADER.pack(MAGIC, VERSION, self.state_bits, 0, self.total, self.stream_count),
            *descriptors,
            *payloads,
        ))


class MultistreamACDecoder:
    """Validate an MSAC v1 payload and decode each stream independently."""

    def __init__(self, archive: bytes, *, frequency_quantizer: str = "reference"):
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
        self.frequency_quantizer = validate_frequency_quantizer(frequency_quantizer)
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
        cumulative = _cumulative(probabilities, self.total, self.frequency_quantizer)
        symbol = self._decoders[stream_id].read(cumulative, len(cumulative) - 1)
        self._decoded[stream_id] += 1
        return symbol

    def assert_complete(self) -> None:
        """Reject a decode that did not consume every declared symbol."""
        if self._decoded != self.symbol_counts:
            raise ValueError("not all MSAC stream symbols were decoded")
