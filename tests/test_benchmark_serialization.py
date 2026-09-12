from dataclasses import replace

import pytest

from src.benchmark.config import SyntheticDatasetConfig
from src.benchmark.serialization import deserialize_table, serialize_table
from src.benchmark.table import Column, LogicalType, Table, generate_synthetic_table


def _mixed_table() -> Table:
    return Table(
        columns=(
            Column("flag", LogicalType.BOOLEAN),
            Column("count", LogicalType.INT64),
            Column("score", LogicalType.FLOAT64),
            Column("name", LogicalType.UTF8),
            Column("payload", LogicalType.BYTES),
        ),
        rows=(
            (True, -7, 1.25, "München", b"\x00\xff"),
            (None, 0, None, "東京", b""),
        ),
    )


@pytest.mark.parametrize("layout", ["row_major", "column_major"])
def test_table_serialization_round_trip(layout):
    table = _mixed_table()
    serialized = serialize_table(table, layout)

    assert deserialize_table(serialized.source_bytes) == table
    assert serialized.source_bytes.startswith(b"CES1")


def test_layout_changes_order_but_not_source_size():
    table = _mixed_table()
    row = serialize_table(table, "row_major").source_bytes
    column = serialize_table(table, "column_major").source_bytes

    assert row != column
    assert len(row) == len(column)


def test_source_decoder_rejects_trailing_bytes():
    data = serialize_table(_mixed_table(), "row_major").source_bytes

    with pytest.raises(ValueError, match="trailing bytes"):
        deserialize_table(data + b"unexpected")


def test_synthetic_generator_is_seeded_and_typed():
    config = SyntheticDatasetConfig(
        kind="synthetic_relational",
        split="evaluation",
        rows=20,
        columns=5,
        generation_seed=17,
        cross_column_correlation=0.8,
        within_column_repetition=0.7,
        cardinality=8,
        null_rate=0.1,
        mean_string_length=12,
        column_kinds=("natural_language", "identifier", "integer"),
    )

    assert generate_synthetic_table(config) == generate_synthetic_table(config)
    assert generate_synthetic_table(config) != generate_synthetic_table(
        replace(config, generation_seed=18)
    )
