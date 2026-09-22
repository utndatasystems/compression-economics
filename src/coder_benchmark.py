"""Matched entropy-coder benchmark over an immutable probability trace.

The predictor is deliberately outside the timed region. Every probability coder
receives the same symbols, floating-point probabilities, and integer quantizer.
Rank codecs are included as transformed baselines and are never described as
coding the original probability distribution.
"""

from __future__ import annotations

from dataclasses import dataclass
import gc
import hashlib
import math
import struct
import time
import tracemalloc
from typing import Callable, Iterable

import numpy as np

from src.encoding import (
    LLMCompressor,
    LLMDecompressor,
    RansBlockDecoder,
    RansBlockEncoder,
    build_huffman_code,
)
from src.encoding_utils import build_cumul, choose_pmatic_r, make_safe_decoder_probs


REFERENCE_BACKEND = "python_reference"
CODERS = ("AC", "ANS", "PMATIC", "HUFFMAN_RANK", "BITPACKED_RANK")
_BITPACK_HEADER = struct.Struct(">4sIBQ")
_HUFFMAN_HEADER = struct.Struct(">4sIIQ")
_HUFFMAN_ENTRY = struct.Struct(">IH")


@dataclass(frozen=True)
class ProbabilityTrace:
    probabilities: np.ndarray
    symbols: np.ndarray

    def __post_init__(self) -> None:
        probabilities = np.asarray(self.probabilities, dtype=np.float64)
        symbols = np.asarray(self.symbols, dtype=np.int64)
        if probabilities.ndim != 2 or probabilities.shape[0] == 0:
            raise ValueError("probabilities must be a non-empty [symbols, alphabet] matrix")
        if symbols.shape != (probabilities.shape[0],):
            raise ValueError("symbols must contain one target per probability row")
        if probabilities.shape[1] < 2:
            raise ValueError("trace alphabet must contain at least two symbols")
        if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0):
            raise ValueError("probabilities must be finite and nonnegative")
        if not np.allclose(probabilities.sum(axis=1), 1.0, rtol=0, atol=1e-9):
            raise ValueError("each probability row must sum to one")
        if np.any(symbols < 0) or np.any(symbols >= probabilities.shape[1]):
            raise ValueError("target symbol outside trace alphabet")
        object.__setattr__(self, "probabilities", np.ascontiguousarray(probabilities))
        object.__setattr__(self, "symbols", np.ascontiguousarray(symbols))

    @property
    def symbol_count(self) -> int:
        return self.probabilities.shape[0]

    @property
    def alphabet_size(self) -> int:
        return self.probabilities.shape[1]

    @property
    def sha256(self) -> str:
        digest = hashlib.sha256()
        digest.update(struct.pack(">II", self.symbol_count, self.alphabet_size))
        digest.update(self.probabilities.astype("<f8", copy=False).tobytes())
        digest.update(self.symbols.astype("<i8", copy=False).tobytes())
        return digest.hexdigest()


@dataclass(frozen=True)
class EncodedStream:
    bits: tuple[int, ...]
    archive: bytes
    payload_bits: int
    framing_bytes: int = 0
    codebook_bytes: int = 0
    padding_bits: int = 0
    helper_symbols: int = 0
    helper_model_bits: float = 0.0

    @property
    def archive_bytes(self) -> int:
        return len(self.archive)


def save_probability_trace(path, trace: ProbabilityTrace) -> None:
    """Persist the exact float64 matrix and target IDs used by all coders."""
    np.savez(path, probabilities=trace.probabilities, symbols=trace.symbols)


def load_probability_trace(path) -> ProbabilityTrace:
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {"probabilities", "symbols"}:
            raise ValueError("trace archive must contain exactly probabilities and symbols")
        return ProbabilityTrace(archive["probabilities"], archive["symbols"])


def synthetic_probability_trace(
    symbol_count: int, alphabet_size: int, *, seed: int = 2027
) -> ProbabilityTrace:
    if symbol_count <= 0 or alphabet_size < 2:
        raise ValueError("synthetic trace dimensions must be positive")
    rng = np.random.default_rng(seed)
    probabilities = rng.dirichlet(np.full(alphabet_size, 0.35), size=symbol_count)
    symbols = np.asarray(
        [rng.choice(alphabet_size, p=row) for row in probabilities], dtype=np.int64
    )
    return ProbabilityTrace(probabilities, symbols)


def ideal_cross_entropy_bits(trace: ProbabilityTrace) -> float:
    selected = trace.probabilities[np.arange(trace.symbol_count), trace.symbols]
    return float(np.sum(-np.log2(np.maximum(selected, 1e-300))))


def quantized_cross_entropy_bits(trace: ProbabilityTrace, total: int) -> float:
    value = 0.0
    for symbol, probabilities in zip(trace.symbols, trace.probabilities):
        cumulative = build_cumul(probabilities, total=total)
        frequency = int(cumulative[symbol + 1] - cumulative[symbol])
        value -= math.log2(frequency / total)
    return value


def _pack_bits(bits: Iterable[int]) -> tuple[bytes, int]:
    values = np.fromiter(bits, dtype=np.uint8)
    padding = (-len(values)) % 8
    return np.packbits(values, bitorder="big").tobytes(), padding


def _unpack_bits(data: bytes, bit_count: int) -> tuple[int, ...]:
    if bit_count > 8 * len(data):
        raise ValueError("declared bit count exceeds payload")
    values = np.unpackbits(np.frombuffer(data, dtype=np.uint8), bitorder="big")[:bit_count]
    return tuple(int(value) for value in values)


def _rank_order(probabilities: np.ndarray) -> np.ndarray:
    # Probability descending, token ID ascending provides deterministic ties.
    return np.lexsort((np.arange(probabilities.size), -probabilities))


def ranks_for_trace(trace: ProbabilityTrace) -> list[int]:
    ranks = []
    for symbol, probabilities in zip(trace.symbols, trace.probabilities):
        order = _rank_order(probabilities)
        ranks.append(int(np.flatnonzero(order == symbol)[0]))
    return ranks


def symbols_from_ranks(ranks: Iterable[int], probabilities: np.ndarray) -> np.ndarray:
    decoded = []
    for rank, row in zip(ranks, probabilities):
        order = _rank_order(row)
        if not 0 <= rank < len(order):
            raise ValueError("decoded rank outside alphabet")
        decoded.append(int(order[rank]))
    return np.asarray(decoded, dtype=np.int64)


def _canonical_codebook(ranks: list[int]) -> dict[int, str]:
    lengths = {symbol: len(code) for symbol, code in build_huffman_code(ranks).items()}
    canonical: dict[int, str] = {}
    code = 0
    previous_length = 0
    for symbol, length in sorted(lengths.items(), key=lambda item: (item[1], item[0])):
        code <<= length - previous_length
        canonical[symbol] = format(code, f"0{length}b")
        code += 1
        previous_length = length
    return canonical


def encode_bitpacked_ranks(trace: ProbabilityTrace) -> EncodedStream:
    ranks = ranks_for_trace(trace)
    width = max(1, max(ranks, default=0).bit_length())
    bits = tuple(bit for rank in ranks for bit in map(int, format(rank, f"0{width}b")))
    packed, padding = _pack_bits(bits)
    header = _BITPACK_HEADER.pack(b"BPR1", len(ranks), width, len(bits))
    return EncodedStream(
        bits=bits, archive=header + packed, payload_bits=len(bits),
        framing_bytes=len(header), padding_bits=padding,
    )


def decode_bitpacked_ranks(stream: EncodedStream, probabilities: np.ndarray) -> np.ndarray:
    magic, count, width, payload_bits = _BITPACK_HEADER.unpack(
        stream.archive[:_BITPACK_HEADER.size]
    )
    if magic != b"BPR1" or payload_bits != count * width:
        raise ValueError("invalid bit-packed rank archive")
    bits = _unpack_bits(stream.archive[_BITPACK_HEADER.size:], payload_bits)
    ranks = [
        int("".join(map(str, bits[offset:offset + width])), 2)
        for offset in range(0, payload_bits, width)
    ]
    return symbols_from_ranks(ranks, probabilities)


def encode_huffman_ranks(trace: ProbabilityTrace) -> EncodedStream:
    ranks = ranks_for_trace(trace)
    codebook = _canonical_codebook(ranks)
    bits = tuple(bit for rank in ranks for bit in map(int, codebook[rank]))
    packed, padding = _pack_bits(bits)
    entries = b"".join(
        _HUFFMAN_ENTRY.pack(symbol, len(code))
        for symbol, code in sorted(codebook.items())
    )
    header = _HUFFMAN_HEADER.pack(b"HFR1", len(ranks), len(codebook), len(bits))
    return EncodedStream(
        bits=bits, archive=header + entries + packed, payload_bits=len(bits),
        framing_bytes=len(header), codebook_bytes=len(entries), padding_bits=padding,
    )


def decode_huffman_ranks(stream: EncodedStream, probabilities: np.ndarray) -> np.ndarray:
    magic, count, entry_count, payload_bits = _HUFFMAN_HEADER.unpack(
        stream.archive[:_HUFFMAN_HEADER.size]
    )
    if magic != b"HFR1":
        raise ValueError("invalid Huffman rank archive")
    offset = _HUFFMAN_HEADER.size
    lengths = {}
    for _ in range(entry_count):
        symbol, length = _HUFFMAN_ENTRY.unpack(stream.archive[offset:offset + _HUFFMAN_ENTRY.size])
        lengths[symbol] = length
        offset += _HUFFMAN_ENTRY.size
    codebook = {}
    code = 0
    previous_length = 0
    for symbol, length in sorted(lengths.items(), key=lambda item: (item[1], item[0])):
        code <<= length - previous_length
        codebook[format(code, f"0{length}b")] = symbol
        code += 1
        previous_length = length
    bits = _unpack_bits(stream.archive[offset:], payload_bits)
    ranks, prefix = [], ""
    for bit in bits:
        prefix += str(bit)
        if prefix in codebook:
            ranks.append(codebook[prefix])
            prefix = ""
    if prefix or len(ranks) != count:
        raise ValueError("Huffman rank payload does not match archive count")
    return symbols_from_ranks(ranks, probabilities)


def _ans_size_breakdown(bits: tuple[int, ...], count: int, block_size: int, lanes: int) -> tuple[int, int]:
    data, padding = _pack_bits(bits)
    if padding or len(data) < 6:
        raise ValueError("invalid byte-aligned rANS stream")
    stored_count, stored_block = struct.unpack(">IH", data[:6])
    if stored_count != count or stored_block != block_size:
        raise ValueError("rANS stream header does not match benchmark settings")
    offset, remaining, framing = 6, count, 6
    while remaining:
        if offset + 4 > len(data):
            raise ValueError("truncated rANS block header")
        (payload_bytes,) = struct.unpack(">I", data[offset:offset + 4])
        block_count = min(block_size, remaining)
        state_bytes = 4 * min(lanes, block_count)
        framing += 4 + state_bytes
        offset += 4 + state_bytes + payload_bytes
        remaining -= block_count
    if offset != len(data):
        raise ValueError("rANS framing parser did not consume the stream")
    return (len(data) - framing) * 8, framing


def _with_ans_lanes(lanes: int, function: Callable[[], object]):
    if not 1 <= lanes <= 32:
        raise ValueError("ans_lanes must be between 1 and 32")
    encoder_lanes, decoder_lanes = RansBlockEncoder.LANES, RansBlockDecoder.LANES
    RansBlockEncoder.LANES = lanes
    RansBlockDecoder.LANES = lanes
    try:
        return function()
    finally:
        RansBlockEncoder.LANES = encoder_lanes
        RansBlockDecoder.LANES = decoder_lanes


def encode_probability_stream(
    trace: ProbabilityTrace, coder: str, *, total: int, ans_block_size: int,
    ans_lanes: int, pmatic_delta: float,
) -> EncodedStream:
    options = {"algorithm": coder, "alphabet_size": trace.alphabet_size, "total": total}
    if coder == "ANS":
        options["ans_block_size"] = ans_block_size
    if coder == "PMATIC":
        options["delta"] = pmatic_delta
        options["r"] = choose_pmatic_r(pmatic_delta)

    def encode():
        compressor = LLMCompressor(**options)
        for symbol, probabilities in zip(trace.symbols, trace.probabilities):
            compressor.next_token(int(symbol), probabilities)
        bits = tuple(compressor.compress())
        return compressor, bits

    compressor, bits = _with_ans_lanes(ans_lanes, encode) if coder == "ANS" else encode()
    packed, padding = _pack_bits(bits)
    if coder == "ANS":
        payload_bits, framing_bytes = _ans_size_breakdown(
            bits, trace.symbol_count, ans_block_size, ans_lanes
        )
    else:
        payload_bits, framing_bytes = len(bits), 0
    helper_symbols = compressor.helper_count if coder == "PMATIC" else 0
    helper_model_bits = 0.0
    if helper_symbols:
        helper_p1 = compressor.delta / compressor.r
        ones = compressor.helper_ones
        helper_model_bits = (
            -ones * math.log2(helper_p1)
            -(helper_symbols - ones) * math.log2(1.0 - helper_p1)
        )
    return EncodedStream(
        bits=bits, archive=packed, payload_bits=payload_bits,
        framing_bytes=framing_bytes, padding_bits=padding,
        helper_symbols=helper_symbols, helper_model_bits=helper_model_bits,
    )


def decode_probability_stream(
    stream: EncodedStream, probabilities: np.ndarray, coder: str, *, total: int,
    alphabet_size: int, ans_lanes: int, pmatic_delta: float,
) -> np.ndarray:
    options = {"algorithm": coder, "alphabet_size": alphabet_size, "total": total}
    if coder == "PMATIC":
        options["delta"] = pmatic_delta
        options["r"] = choose_pmatic_r(pmatic_delta)

    def decode():
        decoder = LLMDecompressor(stream.bits, **options)
        return np.asarray([decoder.decompress(row) for row in probabilities], dtype=np.int64)

    return _with_ans_lanes(ans_lanes, decode) if coder == "ANS" else decode()


def _measure(function: Callable[[], object]) -> tuple[object, float, int]:
    gc.collect()
    tracemalloc.start()
    started = time.perf_counter()
    result = function()
    seconds = time.perf_counter() - started
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, seconds, peak_bytes


def perturb_probabilities(trace: ProbabilityTrace, scale: float, seed: int) -> np.ndarray:
    if scale < 0:
        raise ValueError("perturbation scale must be nonnegative")
    if scale == 0:
        return trace.probabilities.copy()
    rng = np.random.default_rng(seed)
    perturbed = trace.probabilities + rng.normal(0.0, scale, trace.probabilities.shape)
    perturbed = np.clip(perturbed, 1e-300, None)
    return perturbed / perturbed.sum(axis=1, keepdims=True)


def safe_pmatic_probabilities(trace: ProbabilityTrace, delta: float, seed: int) -> np.ndarray:
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        return np.asarray([
            make_safe_decoder_probs(row, trace.alphabet_size, delta)
            for row in trace.probabilities
        ])
    finally:
        np.random.set_state(state)


def _decode_for_coder(
    coder: str, stream: EncodedStream, probabilities: np.ndarray, *, total: int,
    alphabet_size: int, ans_lanes: int, pmatic_delta: float,
) -> np.ndarray:
    if coder == "BITPACKED_RANK":
        return decode_bitpacked_ranks(stream, probabilities)
    if coder == "HUFFMAN_RANK":
        return decode_huffman_ranks(stream, probabilities)
    return decode_probability_stream(
        stream, probabilities, coder, total=total, alphabet_size=alphabet_size,
        ans_lanes=ans_lanes, pmatic_delta=pmatic_delta,
    )


def benchmark_coder(
    trace: ProbabilityTrace, coder: str, *, total: int = 262144,
    ans_block_size: int = 256, ans_lanes: int = 4,
    pmatic_delta: float = 1e-3, perturbation_scales: Iterable[float] = (),
    seed: int = 2027,
) -> dict:
    coder = coder.upper()
    if coder not in CODERS:
        raise ValueError(f"unknown coder: {coder}")
    if total <= trace.alphabet_size or total & (total - 1):
        raise ValueError("frequency total must be a power of two larger than the alphabet")

    if coder == "BITPACKED_RANK":
        encode_function = lambda: encode_bitpacked_ranks(trace)
    elif coder == "HUFFMAN_RANK":
        encode_function = lambda: encode_huffman_ranks(trace)
    else:
        encode_function = lambda: encode_probability_stream(
            trace, coder, total=total, ans_block_size=ans_block_size,
            ans_lanes=ans_lanes, pmatic_delta=pmatic_delta,
        )
    stream, encode_seconds, encode_peak = _measure(encode_function)
    decode_function = lambda: _decode_for_coder(
        coder, stream, trace.probabilities, total=total,
        alphabet_size=trace.alphabet_size, ans_lanes=ans_lanes,
        pmatic_delta=pmatic_delta,
    )
    decoded, decode_seconds, decode_peak = _measure(decode_function)
    exact_roundtrip = bool(np.array_equal(decoded, trace.symbols))
    if not exact_roundtrip:
        raise AssertionError(f"{coder} failed exact round trip")

    perturbations = []
    scenarios = [(f"gaussian_{scale:g}", perturb_probabilities(trace, scale, seed))
                 for scale in perturbation_scales]
    if coder == "PMATIC":
        scenarios.append((
            f"pmatic_safe_delta_{pmatic_delta:g}",
            safe_pmatic_probabilities(trace, pmatic_delta, seed),
        ))
    for name, decoder_probabilities in scenarios:
        try:
            candidate = _decode_for_coder(
                coder, stream, decoder_probabilities, total=total,
                alphabet_size=trace.alphabet_size, ans_lanes=ans_lanes,
                pmatic_delta=pmatic_delta,
            )
            matches = candidate == trace.symbols
            success = bool(np.all(matches))
            mismatch = None if success else int(np.flatnonzero(~matches)[0])
        except (ValueError, IndexError, AssertionError) as error:
            success, mismatch = False, None
            error_name = type(error).__name__
        else:
            error_name = None
        perturbations.append({
            "name": name,
            "max_absolute_probability_delta": float(
                np.max(np.abs(decoder_probabilities - trace.probabilities))
            ),
            "roundtrip_valid": success,
            "first_mismatch": mismatch,
            "decoder_error": error_name,
        })

    probability_semantics = coder in {"AC", "ANS", "PMATIC"}
    parameters = {"frequency_total": total}
    if coder == "ANS":
        parameters.update(block_symbols=ans_block_size, lanes=ans_lanes)
    if coder == "PMATIC":
        parameters.update(delta=pmatic_delta, r=choose_pmatic_r(pmatic_delta))
    return {
        "coder": coder,
        "backend": REFERENCE_BACKEND,
        "implementation_language": "python",
        "throughput_claim_eligible": False,
        "throughput_warning": (
            "Reference-Python result; interpreter and allocation overhead dominate. "
            "Do not use for native-coder throughput claims."
        ),
        "coding_semantics": "probability_distribution" if probability_semantics else "rank_transform",
        "codes_original_probability_distribution": probability_semantics,
        "parameters": parameters,
        "trace_sha256": trace.sha256,
        "symbol_count": trace.symbol_count,
        "alphabet_size": trace.alphabet_size,
        "ideal_cross_entropy_bits": ideal_cross_entropy_bits(trace),
        "quantized_distribution_cross_entropy_bits": quantized_cross_entropy_bits(trace, total),
        "quantized_cross_entropy_is_coder_objective": coder in {"AC", "ANS"},
        "payload_bits": stream.payload_bits,
        "payload_bytes": math.ceil(stream.payload_bits / 8),
        "framing_bytes": stream.framing_bytes,
        "codebook_bytes": stream.codebook_bytes,
        "padding_bits": stream.padding_bits,
        "archive_bytes": stream.archive_bytes,
        "helper_symbols": stream.helper_symbols,
        "helper_model_bits": stream.helper_model_bits,
        "encode_seconds": encode_seconds,
        "decode_seconds": decode_seconds,
        "encode_symbols_per_second": trace.symbol_count / encode_seconds,
        "decode_symbols_per_second": trace.symbol_count / decode_seconds,
        "encode_peak_traced_bytes": encode_peak,
        "decode_peak_traced_bytes": decode_peak,
        "memory_metric": "Python tracemalloc peak; native allocator/device memory excluded",
        "exact_roundtrip_valid": exact_roundtrip,
        "numerical_reproducibility": perturbations,
        "hypothesis": "ANS may approach AC compression while improving throughput; not assumed true",
    }
