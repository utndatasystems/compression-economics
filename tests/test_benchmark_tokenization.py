import pytest

from src.relational_compression_benchmark.archive import decode_archive, encode_archive
from src.relational_compression_benchmark.serialization import deserialize_table, serialize_table
from src.relational_compression_benchmark.table import Column, LogicalType, Table
from src.relational_compression_benchmark.tokenization import (
    TokenizerAdapter,
    pack_token_ids,
    token_id_width,
    unpack_token_ids,
)


class ByteTokenizer:
    """Small deterministic tokenizer used without external model assets."""

    def __len__(self) -> int:
        return 256

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert not add_special_tokens
        return [ord(character) for character in text]

    def decode(
        self,
        token_ids,
        *,
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ) -> str:
        assert not skip_special_tokens
        assert not clean_up_tokenization_spaces
        return "".join(chr(token_id) for token_id in token_ids)


def _adapter(*, revision: str = "test-revision") -> TokenizerAdapter:
    """Return a lossless byte tokenizer with stable test metadata."""
    return TokenizerAdapter.from_backend(
        ByteTokenizer(), name="test/byte-tokenizer", revision=revision
    )


def _table() -> Table:
    """Return mixed text and binary values for archive round trips."""
    return Table(
        columns=(
            Column("text", LogicalType.UTF8),
            Column("data", LogicalType.BYTES),
            Column("number", LogicalType.INT64),
        ),
        rows=(("München 東京", b"\x00\xff", -7), (None, b"payload", 42)),
    )


@pytest.mark.parametrize(
    ("vocabulary_size", "width"),
    [(1, 1), (256, 1), (257, 2), (151_665, 3)],
)
def test_token_width_is_the_smallest_whole_byte_width(vocabulary_size, width):
    assert token_id_width(vocabulary_size) == width


@pytest.mark.parametrize("width", [1, 2, 3, 4])
def test_fixed_width_token_packing_round_trip(width):
    limit = 1 << (8 * width)
    token_ids = [0, 1, limit // 2, limit - 1]

    packed = pack_token_ids(token_ids, width)

    assert len(packed) == len(token_ids) * width
    assert unpack_token_ids(packed, width) == token_ids


def test_token_unpacking_rejects_a_partial_id():
    with pytest.raises(ValueError, match="partial token ID"):
        unpack_token_ids(b"\x00\x01\x02", width=2)


def test_token_packing_rejects_an_id_that_does_not_fit():
    with pytest.raises(ValueError, match="does not fit"):
        pack_token_ids([256], width=1)


def test_latin1_bridge_round_trips_every_byte_value():
    adapter = _adapter()
    source = bytes(range(256))

    assert adapter.decode_bytes(adapter.encode_bytes(source)) == source


@pytest.mark.parametrize("layout", ["row_major", "column_major"])
@pytest.mark.parametrize("codec", ["identity", "zstd"])
def test_token_archive_round_trip(layout, codec):
    table = _table()
    serialized = serialize_table(table, layout)
    adapter = _adapter()
    encoded = encode_archive(
        serialized,
        target_block_bytes=20,
        representation="token_ids",
        codec=codec,
        compression_level=3 if codec == "zstd" else None,
        tokenizer=adapter,
    )

    decoded = decode_archive(encoded.data, tokenizer=adapter)

    assert encoded.representation == "token_ids"
    assert decoded.representation == "token_ids"
    assert decoded.source_bytes == serialized.source_bytes
    assert deserialize_table(decoded.source_bytes) == table
    assert encoded.accounting.total_stored_bytes == len(encoded.data)
    assert sum(block.token_count for block in encoded.blocks) > 0
    if codec == "identity":
        assert all(
            block.payload_bytes == block.token_count * adapter.id_width
            for block in encoded.blocks
        )


def test_token_archive_rejects_a_different_tokenizer_revision():
    serialized = serialize_table(_table(), "row_major")
    encoded = encode_archive(
        serialized,
        target_block_bytes=64,
        representation="token_ids",
        codec="identity",
        tokenizer=_adapter(),
    )

    with pytest.raises(ValueError, match="does not match"):
        decode_archive(encoded.data, tokenizer=_adapter(revision="other"))



def test_token_archive_requires_an_explicit_tokenizer():
    serialized = serialize_table(_table(), "row_major")

    with pytest.raises(ValueError, match="requires a tokenizer"):
        encode_archive(
            serialized,
            target_block_bytes=64,
            representation="token_ids",
            codec="identity",
        )
