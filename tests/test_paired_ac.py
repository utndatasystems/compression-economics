"""Round-trip and boundary checks for paired MSAC byte packing."""

import numpy as np
import pytest

from src.coding.multistream_ac import MultistreamACDecoder, MultistreamACEncoder
from src.coding.paired_ac import (
    PairedACDecoder, PairedACEncoder, _pack_pair, _unpack_pair,
    pack_standard_archive, unpack_paired_archive,
)


def test_terminal_byte_is_shared_when_fixed_bits_fit():
    packed, shared = _pack_pair(b"\xa0", 3, b"\x60", 3)
    assert shared and packed == b"\xa6"
    assert _unpack_pair(packed, 1, 1, 3, 3, shared) == (b"\xa0", b"\x60")


@pytest.mark.parametrize("streams", [1, 2, 3, 4, 7, 16])
@pytest.mark.parametrize("symbols", [1, 17, 217])
def test_paired_roundtrip_matches_standard(streams, symbols):
    rng = np.random.default_rng(2000 + streams * 1000 + symbols)
    rows = rng.dirichlet(np.ones(13), size=(streams, symbols)).astype(np.float32)
    targets = rng.integers(0, 13, size=(streams, symbols))
    standard = MultistreamACEncoder(streams)
    paired = PairedACEncoder(streams)
    for index in range(symbols):
        for stream_id in range(streams):
            target = int(targets[stream_id, index])
            row = rows[stream_id, index]
            standard.encode(stream_id, target, row)
            paired.encode(stream_id, target, row)
    baseline = standard.finish()
    archive = paired.finish()
    assert archive == pack_standard_archive(baseline)
    assert unpack_paired_archive(archive) == baseline
    decoder = PairedACDecoder(archive)
    for index in range(symbols):
        for stream_id in range(streams):
            assert decoder.decode(stream_id, rows[stream_id, index]) == targets[stream_id, index]
    decoder.assert_complete()
    assert decoder.symbol_counts == MultistreamACDecoder(baseline).symbol_counts
    assert len(archive) == len(baseline) - decoder.shared_pairs


def test_paired_rejects_damaged_payload_and_directory():
    encoder = PairedACEncoder(2)
    for stream_id in range(2):
        encoder.encode(stream_id, 1, np.array([0.25, 0.75]))
    archive = encoder.finish()
    for damaged in (archive[:-1], archive + b"x", archive[:4] + b"\xff" + archive[5:]):
        with pytest.raises(ValueError):
            PairedACDecoder(damaged)
    changed = bytearray(archive)
    changed[-1] ^= 0x80
    with pytest.raises(ValueError, match="checksum|padding"):
        PairedACDecoder(bytes(changed))


def test_target_interval_v2_converts_to_paired():
    encoder = MultistreamACEncoder(2, target_interval=True)
    for _ in range(5):
        encoder.encode_interval(0, 1, 3, 10)
        encoder.encode_interval(1, 5, 9, 10)
    baseline = encoder.finish()
    paired = pack_standard_archive(baseline)
    assert unpack_paired_archive(paired) == baseline
    assert PairedACDecoder(paired).target_interval
