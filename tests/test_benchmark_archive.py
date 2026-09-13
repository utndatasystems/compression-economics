from pathlib import Path

import pytest

from src.relational_compression_benchmark.archive import (
    BLOCK_FRAME_BYTES,
    decode_archive,
    encode_archive,
)
from src.relational_compression_benchmark.config import (
    ExecutionConfig,
    PipelineConfig,
    SweepConfig,
    SyntheticDatasetConfig,
)
from src.relational_compression_benchmark.runner import run_sweep
from src.relational_compression_benchmark.serialization import deserialize_table, serialize_table
from src.relational_compression_benchmark.table import Column, LogicalType, Table


def _table() -> Table:
    return Table(
        columns=(
            Column("id", LogicalType.INT64, nullable=False),
            Column("value", LogicalType.UTF8),
        ),
        rows=tuple((index, "repeated-value") for index in range(40)),
    )


@pytest.mark.parametrize(
    ("codec", "level"), [("identity", None), ("zstd", 3)]
)
@pytest.mark.parametrize("layout", ["row_major", "column_major"])
def test_block_archive_round_trip_and_accounting(codec, level, layout):
    table = _table()
    serialized = serialize_table(table, layout)
    encoded = encode_archive(
        serialized,
        target_block_bytes=80,
        codec=codec,
        compression_level=level,
    )
    decoded = decode_archive(encoded.data)

    assert len(encoded.blocks) > 1
    assert decoded.source_bytes == serialized.source_bytes
    assert deserialize_table(decoded.source_bytes) == table
    assert encoded.accounting.total_stored_bytes == len(encoded.data)
    assert encoded.accounting.payload_bytes == sum(
        block.payload_bytes for block in encoded.blocks
    )
    assert all(block.source_bytes > 0 for block in encoded.blocks)


def test_archive_detects_payload_corruption():
    serialized = serialize_table(_table(), "row_major")
    encoded = encode_archive(
        serialized, target_block_bytes=80, codec="identity"
    )
    corrupted = bytearray(encoded.data)
    first_payload = encoded.data.index(b"BLK1") + BLOCK_FRAME_BYTES
    corrupted[first_payload] ^= 1

    with pytest.raises(ValueError, match="checksum mismatch"):
        decode_archive(bytes(corrupted))


def _tiny_sweep() -> SweepConfig:
    return SweepConfig(
        schema_version=1,
        experiment_id="test-sweep",
        execution=ExecutionConfig(
            seed=1,
            repetitions=1,
            warmups=0,
            backend="cpu",
            verify_roundtrip=True,
            result_format="jsonl",
        ),
        dataset=SyntheticDatasetConfig(
            kind="synthetic_relational",
            split="evaluation",
            rows=32,
            columns=3,
            generation_seed=2,
            cross_column_correlation=0.8,
            within_column_repetition=0.5,
            cardinality=8,
            null_rate=0.1,
            mean_string_length=12,
            column_kinds=("natural_language", "identifier", "integer"),
        ),
        layouts=("row_major", "column_major"),
        blocks_bytes=(128,),
        accounting_modes=("shared_model", "self_contained_archive"),
        pipelines=(
            PipelineConfig("raw-identity", "raw_bytes", "identity", None),
            PipelineConfig("raw-zstd-3", "raw_bytes", "zstd", 3),
        ),
        artifact_root=Path("unused"),
        aggregate_results=Path("runs/results.jsonl"),
        block_results=Path("runs/blocks.jsonl"),
        streams=Path("runs/streams"),
    )


def test_runner_writes_valid_long_form_results(tmp_path):
    streams = tmp_path / "runs/streams"
    streams.mkdir(parents=True)
    stale_stream = streams / "obsolete.ceb"
    stale_stream.write_bytes(b"obsolete")

    aggregate, blocks = run_sweep(_tiny_sweep(), output_root=tmp_path)

    assert len(aggregate) == 8
    assert blocks
    assert all(row["roundtrip_valid"] for row in aggregate)
    assert all(
        row["accounting"]["total_stored_bytes"]
        == row["accounting"]["payload_bytes"]
        + row["accounting"]["framing_bytes"]
        + row["accounting"]["index_bytes"]
        for row in aggregate
    )
    assert (tmp_path / "runs/results.jsonl").is_file()
    assert (tmp_path / "runs/blocks.jsonl").is_file()
    assert len(tuple(streams.glob("*.ceb"))) == 4
    assert not stale_stream.exists()
