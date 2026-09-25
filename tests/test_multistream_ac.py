"""Decoder-verified checks for the portable multistream AC payload."""

import struct

import numpy as np
import pytest

from src.encoding import LLMCompressor
from src.multistream_ac import MultistreamACDecoder, MultistreamACEncoder


def test_independent_streams_roundtrip_from_persisted_bytes(tmp_path):
    probabilities = [
        np.array([0.1, 0.7, 0.2]),
        np.array([0.5, 0.2, 0.3]),
        np.array([0.2, 0.2, 0.6]),
        np.array([0.7, 0.2, 0.1]),
    ]
    encoder = MultistreamACEncoder(2)
    for stream_id, symbol, row in zip((0, 1, 0, 1), (1, 0, 2, 1), probabilities):
        encoder.encode(stream_id, symbol, row)

    path = tmp_path / "multistream.msac"
    path.write_bytes(encoder.finish())
    archive = path.read_bytes()
    assert archive[:4] == b"MSAC"
    assert struct.unpack_from(">I", archive, 12)[0] == 2

    decoder = MultistreamACDecoder(archive)
    assert decoder.symbol_counts == [2, 2]
    assert [decoder.decode(0, probabilities[i]) for i in (0, 2)] == [1, 2]
    assert [decoder.decode(1, probabilities[i]) for i in (1, 3)] == [0, 1]
    decoder.assert_complete()
    assert decoder.framing_bytes + sum((bits + 7) // 8 for bits in decoder.bit_counts) == len(archive)


def test_one_stream_uses_existing_ac_frequency_rule():
    rows = [np.array([0.1, 0.7, 0.2]), np.array([0.3, 0.2, 0.5])]
    symbols = [1, 2]
    reference = LLMCompressor(algorithm="AC")
    encoder = MultistreamACEncoder(1)
    for symbol, row in zip(symbols, rows):
        reference.next_token(symbol, row)
        encoder.encode(0, symbol, row)

    expected_bits = reference.compress()
    archive = encoder.finish()
    decoder = MultistreamACDecoder(archive)
    assert decoder.bit_counts == [len(expected_bits)]
    payload = archive[decoder.framing_bytes:]
    actual_bits = np.unpackbits(np.frombuffer(payload, dtype=np.uint8), bitorder="big")
    assert actual_bits[:len(expected_bits)].tolist() == expected_bits
    assert [decoder.decode(0, row) for row in rows] == symbols
    decoder.assert_complete()


def test_corrupt_or_incomplete_archive_is_rejected():
    encoder = MultistreamACEncoder(1)
    encoder.encode(0, 1, np.array([0.2, 0.8]))
    archive = encoder.finish()
    damaged = bytearray(archive)
    damaged[-1] ^= 1
    for bad in (bytes(damaged), archive[:-1], archive + b"x"):
        with pytest.raises(ValueError):
            MultistreamACDecoder(bad)

    decoder = MultistreamACDecoder(archive)
    with pytest.raises(ValueError, match="not all"):
        decoder.assert_complete()

def test_float32_probabilities_match_single_stream_ac():
    from src.encoding_utils import build_cumul

    rng = np.random.default_rng(17)
    row = None
    for _ in range(1000):
        candidate = rng.random(217).astype(np.float32)
        candidate /= candidate.sum()
        if not np.array_equal(build_cumul(candidate), build_cumul(candidate.astype(np.float64))):
            row = candidate
            break
    assert row is not None
    frequencies = np.diff(build_cumul(row))
    widened_frequencies = np.diff(build_cumul(row.astype(np.float64)))
    symbol = int(np.flatnonzero(frequencies != widened_frequencies)[0])

    reference = LLMCompressor(algorithm="AC")
    multistream = MultistreamACEncoder(1)
    for _ in range(8):
        reference.next_token(symbol, row)
        multistream.encode(0, symbol, row)
    expected = reference.compress()
    archive = multistream.finish()
    decoder = MultistreamACDecoder(archive)
    payload = np.unpackbits(
        np.frombuffer(archive[decoder.framing_bytes:], dtype=np.uint8), bitorder="big"
    )
    assert payload[:len(expected)].tolist() == expected
    assert [decoder.decode(0, row) for _ in range(8)] == [symbol] * 8
    decoder.assert_complete()


@pytest.mark.parametrize("streams,threads", [(1, 1), (4, 1), (4, 4), (7, 3)])
def test_parallel_backend_matches_python_archive_and_persisted_decode(tmp_path, streams, threads):
    pytest.importorskip("numba")
    rng = np.random.default_rng(384)
    rows = rng.dirichlet(np.ones(31), size=173).astype(np.float32)
    targets = rng.integers(0, 31, size=len(rows))
    reference = MultistreamACEncoder(streams)
    parallel = MultistreamACEncoder(streams, backend="numba_parallel", threads=threads)
    for index, (target, row) in enumerate(zip(targets, rows)):
        stream_id = index % streams
        reference.encode(stream_id, int(target), row)
        parallel.encode(stream_id, int(target), row)
    expected = reference.finish()
    path = tmp_path / "parallel.msac"
    path.write_bytes(parallel.finish())
    assert path.read_bytes() == expected
    decoder = MultistreamACDecoder(path.read_bytes())
    for stream_id in range(streams):
        indices = range(stream_id, len(rows), streams)
        assert [decoder.decode(stream_id, rows[i]) for i in indices] == [
            int(targets[i]) for i in indices
        ]
    decoder.assert_complete()
    assert parallel.threads == min(streams, threads)
    assert parallel.range_encode_seconds > 0


def test_parallel_backend_handles_empty_streams_and_rejects_overflow():
    pytest.importorskip("numba")
    python = MultistreamACEncoder(3)
    parallel = MultistreamACEncoder(3, backend="numba_parallel", threads=2)
    row = np.array([0.2, 0.8])
    python.encode(1, 1, row)
    parallel.encode(1, 1, row)
    assert parallel.finish() == python.finish()
    with pytest.raises(ValueError, match="64-bit"):
        MultistreamACEncoder(1, state_bits=56, backend="numba_parallel")
