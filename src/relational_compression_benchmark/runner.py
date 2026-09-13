"""Readable end-to-end runner for the first relational benchmark slice."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import time
from typing import Any, Iterable

from src.relational_compression_benchmark.block_storage_format import (
    BlockMetric,
    EncodedArchive,
    attach_decompression_times,
    decode_archive,
    encode_archive,
)
from src.relational_compression_benchmark.config import (
    DatasetConfig,
    ImdbDatasetConfig,
    PipelineConfig,
    SweepConfig,
)
from src.relational_compression_benchmark.datasets import IMDB_SAMPLING_VERSION, load_dataset
from src.relational_compression_benchmark.serialization import (
    SerializedTable,
    deserialize_table,
    serialize_table,
)
from src.relational_compression_benchmark.table import Table
from src.relational_compression_benchmark.tokenization import TokenizerAdapter, load_tokenizer


METRIC_DEFINITIONS = {
    "source_bytes": "Canonical schema, shape, layout tag, and uncompressed values.",
    "payload_bytes": "Sum of stored codec payloads for all independent blocks.",
    "framing_bytes": "Archive header, schema, and per-block frame bytes.",
    "index_bytes": "Persisted index header and entries.",
    "total_stored_bytes": "Sum of every charged stored-size component.",
    "compression_factor": "source_bytes / total_stored_bytes; larger is better.",
    "bits_per_source_byte": "8 * total_stored_bytes / source_bytes.",
    "source_bytes_per_token": (
        "Canonical cell bytes presented to the tokenizer divided by token count."
    ),
    "compression_mib_per_second": (
        "Source MiB divided by serialization plus archive-encoding wall time; "
        "file I/O excluded and reported separately."
    ),
    "decompression_mib_per_second": (
        "Source MiB divided by archive-decoding plus table-deserialization wall "
        "time; file I/O excluded and reported separately."
    ),
    "block_compression_factor": (
        "Block source bytes divided by payload, block frame, and one index entry; "
        "global schema and index-header bytes excluded."
    ),
}


@dataclass(frozen=True)
class MeasuredRun:
    """Validated products and timings from one benchmark repetition."""

    serialized: SerializedTable
    encoded: EncodedArchive
    blocks: tuple[BlockMetric, ...]
    timings: dict[str, float]


def _repository_state() -> dict[str, Any]:
    """Return the revision and dirty flag when Git is available."""
    root = Path(__file__).resolve().parents[2]
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = None, None
    return {"git_revision": revision, "git_dirty": dirty}


def _environment() -> tuple[dict[str, Any], dict[str, Any]]:
    """Describe the software and CPU host attached to each result."""
    software = {
        **_repository_state(),
        "python": platform.python_version(),
        "zstandard": importlib.metadata.version("zstandard"),
    }
    hardware = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "logical_cpu_count": os.cpu_count(),
        "backend": "cpu",
    }
    return software, hardware


def _config_snapshot(config: SweepConfig) -> dict[str, Any]:
    """Convert a validated configuration into JSON-compatible metadata."""
    return {
        "schema_version": config.schema_version,
        "experiment_id": config.experiment_id,
        "execution": asdict(config.execution),
        "dataset": _dataset_snapshot(config.dataset),
        "matrix": {
            "layouts": list(config.layouts),
            "blocks_bytes": list(config.blocks_bytes),
            "accounting_modes": list(config.accounting_modes),
        },
        "pipelines": [asdict(pipeline) for pipeline in config.pipelines],
        "artifacts": {
            "root": str(config.artifact_root),
            "aggregate_results": str(config.aggregate_results),
            "block_results": str(config.block_results),
            "streams": str(config.streams),
        },
    }


def _dataset_snapshot(dataset: DatasetConfig) -> dict[str, Any]:
    """Convert a typed dataset configuration to JSON-compatible metadata."""
    snapshot = asdict(dataset)
    if isinstance(snapshot.get("path"), Path):
        snapshot["path"] = str(snapshot["path"])
    if isinstance(dataset, ImdbDatasetConfig):
        snapshot["sampling_version"] = IMDB_SAMPLING_VERSION
    return snapshot


def _condition(
    config: SweepConfig,
    pipeline: PipelineConfig,
    layout: str,
    block_bytes: int,
    repetition: int,
) -> dict[str, Any]:
    """Build the immutable inputs that identify one measured run."""
    condition = {
        "experiment_id": config.experiment_id,
        "dataset": _dataset_snapshot(config.dataset),
        "layout": layout,
        "representation": pipeline.representation,
        "pipeline": pipeline.name,
        "codec": pipeline.codec,
        "compression_level": pipeline.compression_level,
        "target_block_bytes": block_bytes,
        "backend": config.execution.backend,
        "repetition": repetition,
        "seed": config.execution.seed,
    }
    if pipeline.representation == "token_ids":
        condition.update(
            tokenizer_name=pipeline.tokenizer_name,
            tokenizer_revision=pipeline.tokenizer_revision,
        )
    return condition


def _identifier(value: Any, length: int = 16) -> str:
    """Return a stable short identifier for JSON-compatible content."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:length]


def _percentile(values: list[float], percentile: float) -> float:
    """Interpolate a percentile from a non-empty sample."""
    ordered = sorted(values)
    if not ordered:
        raise ValueError("Cannot summarize an empty list")
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _bootstrap_factor_interval(
    blocks: tuple[BlockMetric, ...], *, seed: int, samples: int = 2_000
) -> list[float] | None:
    """Estimate a deterministic 95% interval by resampling blocks."""
    if len(blocks) < 8:
        return None
    generator = random.Random(seed)
    factors = []
    for _ in range(samples):
        selected = [generator.choice(blocks) for _ in blocks]
        factors.append(
            sum(block.source_bytes for block in selected)
            / sum(block.stored_bytes for block in selected)
        )
    return [_percentile(factors, 0.025), _percentile(factors, 0.975)]


def _block_summary(
    blocks: tuple[BlockMetric, ...], *, seed: int
) -> dict[str, Any]:
    """Summarize variation in compression factor across blocks."""
    factors = [block.compression_factor for block in blocks]
    return {
        "count": len(factors),
        "minimum": min(factors),
        "median": _percentile(factors, 0.50),
        "p95": _percentile(factors, 0.95),
        "p99": _percentile(factors, 0.99),
        "maximum": max(factors),
        "bootstrap_95_percent_interval": _bootstrap_factor_interval(
            blocks, seed=seed
        ),
    }


def _write_bytes(path: Path, data: bytes) -> float:
    """Atomically persist bytes and return only the write duration."""
    path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(data)
    temporary.replace(path)
    return time.perf_counter() - started


def _write_json(path: Path, value: Any) -> None:
    """Atomically write an indented JSON document."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    """Atomically write compact, deterministic JSON Lines."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")))
            handle.write("\n")
    temporary.replace(path)


def _read_bytes(path: Path) -> tuple[bytes, float]:
    """Read an archive and return the measured I/O duration."""
    started = time.perf_counter()
    data = path.read_bytes()
    return data, time.perf_counter() - started


def _prune_stale_streams(directory: Path, retained: set[Path]) -> None:
    """Remove obsolete benchmark streams from one configured output directory."""
    if not directory.is_dir():
        return
    for path in directory.glob("*.ceb"):
        if path not in retained:
            path.unlink()


def _run_once(
    *,
    table: Table,
    layout: str,
    target_block_bytes: int,
    pipeline: PipelineConfig,
    tokenizer: TokenizerAdapter | None,
    stream_path: Path | None,
) -> MeasuredRun:
    """Execute and validate one in-memory or persisted archive round trip."""
    serialization_started = time.perf_counter()
    serialized = serialize_table(table, layout)
    serialization_seconds = time.perf_counter() - serialization_started

    encoded = encode_archive(
        serialized,
        target_block_bytes=target_block_bytes,
        codec=pipeline.codec,
        compression_level=pipeline.compression_level,
        representation=pipeline.representation,
        tokenizer=tokenizer,
    )
    if stream_path is None:
        stored_data = encoded.data
        io_write_seconds = io_read_seconds = 0.0
    else:
        io_write_seconds = _write_bytes(stream_path, encoded.data)
        stored_data, io_read_seconds = _read_bytes(stream_path)

    decoded = decode_archive(stored_data, tokenizer=tokenizer)
    deserialization_started = time.perf_counter()
    reconstructed = deserialize_table(decoded.source_bytes)
    deserialization_seconds = time.perf_counter() - deserialization_started
    roundtrip_valid = (
        decoded.source_bytes == serialized.source_bytes and reconstructed == table
    )
    if not roundtrip_valid:
        raise RuntimeError("Round-trip validation failed; result is invalid")

    blocks = attach_decompression_times(encoded.blocks, decoded.blocks)
    tokenization_seconds = sum(block.tokenization_seconds for block in blocks)
    token_packing_seconds = sum(block.token_packing_seconds for block in blocks)
    token_unpacking_seconds = sum(block.token_unpacking_seconds for block in blocks)
    detokenization_seconds = sum(block.detokenization_seconds for block in blocks)
    source_mib = encoded.accounting.source_bytes / (1024**2)
    compression_seconds = serialization_seconds + encoded.archive_seconds
    decompression_seconds = decoded.archive_seconds + deserialization_seconds
    return MeasuredRun(
        serialized=serialized,
        encoded=encoded,
        blocks=blocks,
        timings={
            "serialization_seconds": serialization_seconds,
            "tokenization_seconds": tokenization_seconds,
            "token_packing_seconds": token_packing_seconds,
            "codec_compression_seconds": encoded.codec_seconds,
            "framing_compression_seconds": (
                encoded.archive_seconds
                - tokenization_seconds
                - token_packing_seconds
                - encoded.codec_seconds
            ),
            "io_write_seconds": io_write_seconds,
            "io_read_seconds": io_read_seconds,
            "codec_decompression_seconds": decoded.codec_seconds,
            "token_unpacking_seconds": token_unpacking_seconds,
            "detokenization_seconds": detokenization_seconds,
            "framing_decompression_seconds": (
                decoded.archive_seconds
                - decoded.codec_seconds
                - token_unpacking_seconds
                - detokenization_seconds
            ),
            "deserialization_seconds": deserialization_seconds,
            "compression_seconds": compression_seconds,
            "decompression_seconds": decompression_seconds,
            "compression_mib_per_second": source_mib / compression_seconds,
            "decompression_mib_per_second": source_mib / decompression_seconds,
        },
    )


def run_sweep(
    config: SweepConfig, *, output_root: str | Path | None = None
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run every explicit pipeline/layout/block condition and persist results."""
    root = Path(output_root) if output_root is not None else config.artifact_root
    aggregate_path = root / config.aggregate_results
    block_path = root / config.block_results
    streams_path = root / config.streams
    table = load_dataset(config.dataset)
    tokenizer_cache: dict[tuple[str, str], TokenizerAdapter] = {}
    for pipeline in config.pipelines:
        if pipeline.representation != "token_ids":
            continue
        key = (
            pipeline.tokenizer_name or "",
            pipeline.tokenizer_revision or "",
        )
        if key not in tokenizer_cache:
            tokenizer_cache[key] = load_tokenizer(*key)

    software, hardware = _environment()
    config_snapshot = _config_snapshot(config)
    recorded_at = datetime.now(timezone.utc).isoformat()

    row_major = serialize_table(table, "row_major").source_bytes
    dataset_sha256 = hashlib.sha256(row_major).hexdigest()
    _write_json(
        aggregate_path.parent / "resolved_config.json", config_snapshot
    )
    _write_json(
        aggregate_path.parent / "dataset_manifest.json",
        {
            "kind": config.dataset.kind,
            "split": config.dataset.split,
            "rows": len(table.rows),
            "columns": len(table.columns),
            "source": _dataset_snapshot(config.dataset),
            "canonical_row_major_sha256": dataset_sha256,
        },
    )

    aggregate_rows: list[dict[str, Any]] = []
    block_rows: list[dict[str, Any]] = []
    retained_streams: set[Path] = set()
    for layout in config.layouts:
        for pipeline in config.pipelines:
            tokenizer = (
                tokenizer_cache[
                    (
                        pipeline.tokenizer_name or "",
                        pipeline.tokenizer_revision or "",
                    )
                ]
                if pipeline.representation == "token_ids"
                else None
            )
            for target_block_bytes in config.blocks_bytes:
                for _ in range(config.execution.warmups):
                    _run_once(
                        table=table,
                        layout=layout,
                        target_block_bytes=target_block_bytes,
                        pipeline=pipeline,
                        tokenizer=tokenizer,
                        stream_path=None,
                    )
                for repetition in range(config.execution.repetitions):
                    condition = _condition(
                        config,
                        pipeline,
                        layout,
                        target_block_bytes,
                        repetition,
                    )
                    condition_id = _identifier(condition)
                    stream_path = streams_path / f"{condition_id}.ceb"
                    retained_streams.add(stream_path)
                    run = _run_once(
                        table=table,
                        layout=layout,
                        target_block_bytes=target_block_bytes,
                        pipeline=pipeline,
                        tokenizer=tokenizer,
                        stream_path=stream_path,
                    )
                    encoded = run.encoded
                    blocks = run.blocks
                    source_sha256 = hashlib.sha256(
                        run.serialized.source_bytes
                    ).hexdigest()
                    token_count = sum(block.token_count for block in blocks)
                    tokenized_source_bytes = sum(
                        block.source_bytes for block in blocks
                    )
                    tokenizer_metadata = (
                        {
                            **tokenizer.descriptor,
                            "asset_bytes": tokenizer.asset_bytes,
                        }
                        if tokenizer is not None
                        else None
                    )
                    common = {
                        "schema_version": 1,
                        "condition_id": condition_id,
                        "recorded_at_utc": recorded_at,
                        "condition": condition,
                        "resolved_config": config_snapshot,
                        "software": software,
                        "hardware": hardware,
                        "dataset_sha256": dataset_sha256,
                        "source_sha256": source_sha256,
                        "stream": str(stream_path),
                        "roundtrip_valid": True,
                        "tokenizer": tokenizer_metadata,
                        "token_count": token_count,
                        "source_bytes_per_token": (
                            tokenized_source_bytes / token_count
                            if token_count
                            else None
                        ),
                        "metric_definitions": METRIC_DEFINITIONS,
                    }
                    block_summary = _block_summary(
                        blocks,
                        seed=config.execution.seed + int(condition_id[:8], 16),
                    )
                    for accounting_mode in config.accounting_modes:
                        accounting = encoded.accounting
                        if (
                            accounting_mode == "self_contained_archive"
                            and tokenizer is not None
                        ):
                            accounting = replace(
                                accounting, tokenizer_bytes=tokenizer.asset_bytes
                            )
                        aggregate_rows.append(
                            {
                                **common,
                                "record_type": "aggregate",
                                "run_id": _identifier(
                                    {**condition, "accounting_mode": accounting_mode}
                                ),
                                "accounting_mode": accounting_mode,
                                "accounting": accounting.as_dict(),
                                "timings": run.timings,
                                "peak_host_memory_bytes": None,
                                "peak_device_memory_bytes": None,
                                "block_compression_factor_summary": block_summary,
                            }
                        )

                    for block in blocks:
                        block_rows.append(
                            {
                                **common,
                                "record_type": "block",
                                "block_index": block.block_index,
                                "first_cell": block.first_cell,
                                "cell_count": block.cell_count,
                                "source_bytes": block.source_bytes,
                                "payload_bytes": block.payload_bytes,
                                "framing_bytes": block.framing_bytes,
                                "index_bytes": block.index_bytes,
                                "stored_bytes": block.stored_bytes,
                                "compression_factor": block.compression_factor,
                                "compression_seconds": block.compression_seconds,
                                "decompression_seconds": block.decompression_seconds,
                                "token_count": block.token_count,
                                "tokenization_seconds": block.tokenization_seconds,
                                "token_packing_seconds": block.token_packing_seconds,
                                "token_unpacking_seconds": block.token_unpacking_seconds,
                                "detokenization_seconds": block.detokenization_seconds,
                                "accounting_modes": list(config.accounting_modes),
                            }
                        )

    _prune_stale_streams(streams_path, retained_streams)
    _write_jsonl(aggregate_path, aggregate_rows)
    _write_jsonl(block_path, block_rows)
    return aggregate_rows, block_rows
