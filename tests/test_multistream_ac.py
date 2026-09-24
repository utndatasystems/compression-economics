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
