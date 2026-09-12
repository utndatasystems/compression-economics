"""Typed, strict configuration for relational compression sweeps."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib
from typing import Any


SUPPORTED_LAYOUTS = frozenset({"row_major", "column_major"})
SUPPORTED_CODECS = frozenset({"identity", "zstd"})
SUPPORTED_ACCOUNTING_MODES = frozenset(
    {"shared_model", "self_contained_archive"}
)


@dataclass(frozen=True)
class ExecutionConfig:
    """Settings shared by every measured condition in a sweep."""

    seed: int
    repetitions: int
    warmups: int
    backend: str
    verify_roundtrip: bool
    result_format: str


@dataclass(frozen=True)
class SyntheticDatasetConfig:
    """Controls for the deterministic relational data generator."""

    kind: str
    split: str
    rows: int
    columns: int
    generation_seed: int
    cross_column_correlation: float
    within_column_repetition: float
    cardinality: int
    null_rate: float
    mean_string_length: int
    column_kinds: tuple[str, ...]


@dataclass(frozen=True)
class PipelineConfig:
    """One valid representation and codec combination."""

    name: str
    representation: str
    codec: str
    compression_level: int | None


@dataclass(frozen=True)
class SweepConfig:
    """Fully validated benchmark configuration."""

    schema_version: int
    experiment_id: str
    execution: ExecutionConfig
    dataset: SyntheticDatasetConfig
    layouts: tuple[str, ...]
    blocks_bytes: tuple[int, ...]
    accounting_modes: tuple[str, ...]
    pipelines: tuple[PipelineConfig, ...]
    artifact_root: Path
    aggregate_results: Path
    block_results: Path
    streams: Path


def _expect_keys(
    value: dict[str, Any], *, required: set[str], optional: set[str], where: str
) -> None:
    """Reject missing or unknown fields in one TOML section."""
    missing = required - value.keys()
    unknown = value.keys() - required - optional
    if missing:
        raise ValueError(f"Missing {where} fields: {sorted(missing)}")
    if unknown:
        raise ValueError(f"Unknown {where} fields: {sorted(unknown)}")


def _probability(value: Any, name: str) -> float:
    """Parse and validate a probability-like configuration value."""
    number = float(value)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")
    return number


def _relative_path(value: Any, name: str) -> Path:
    """Validate a path that must stay beneath an artifact root."""
    path = Path(str(value))
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{name} must be a relative path without '..'")
    return path


def load_sweep_config(path: str | Path) -> SweepConfig:
    """Load a TOML sweep and reject ambiguous or unsupported fields."""
    config_path = Path(path)
    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)

    _expect_keys(
        raw,
        required={
            "schema_version",
            "experiment_id",
            "execution",
            "dataset",
            "matrix",
            "pipelines",
            "artifacts",
        },
        optional=set(),
        where="top-level",
    )
    if raw["schema_version"] != 1:
        raise ValueError("Only sweep schema_version = 1 is supported")
    if not str(raw["experiment_id"]).strip():
        raise ValueError("experiment_id cannot be empty")

    execution_raw = raw["execution"]
    _expect_keys(
        execution_raw,
        required={
            "seed",
            "repetitions",
            "warmups",
            "backend",
            "verify_roundtrip",
            "result_format",
        },
        optional=set(),
        where="execution",
    )
    execution = ExecutionConfig(**execution_raw)
    if execution.repetitions < 1 or execution.warmups < 0:
        raise ValueError("repetitions must be positive and warmups non-negative")
    if execution.backend != "cpu":
        raise ValueError("The first benchmark slice supports backend = 'cpu'")
    if not execution.verify_roundtrip:
        raise ValueError("verify_roundtrip must remain enabled")
    if execution.result_format != "jsonl":
        raise ValueError("The first benchmark slice writes JSONL results")

    dataset_raw = raw["dataset"]
    _expect_keys(
        dataset_raw,
        required={
            "kind",
            "split",
            "rows",
            "columns",
            "generation_seed",
            "cross_column_correlation",
            "within_column_repetition",
            "cardinality",
            "null_rate",
            "mean_string_length",
            "column_kinds",
        },
        optional=set(),
        where="dataset",
    )
    dataset = SyntheticDatasetConfig(
        **{
            **dataset_raw,
            "cross_column_correlation": _probability(
                dataset_raw["cross_column_correlation"],
                "cross_column_correlation",
            ),
            "within_column_repetition": _probability(
                dataset_raw["within_column_repetition"],
                "within_column_repetition",
            ),
            "null_rate": _probability(dataset_raw["null_rate"], "null_rate"),
            "column_kinds": tuple(dataset_raw["column_kinds"]),
        }
    )
    if dataset.kind != "synthetic_relational":
        raise ValueError("The first benchmark slice supports synthetic_relational")
    if min(dataset.rows, dataset.columns, dataset.cardinality) < 1:
        raise ValueError("rows, columns, and cardinality must be positive")
    if dataset.mean_string_length < 1 or not dataset.column_kinds:
        raise ValueError("mean_string_length and column_kinds must be non-empty")

    matrix_raw = raw["matrix"]
    _expect_keys(
        matrix_raw,
        required={"layouts", "blocks_bytes", "accounting_modes"},
        optional=set(),
        where="matrix",
    )
    layouts = tuple(matrix_raw["layouts"])
    blocks_bytes = tuple(int(value) for value in matrix_raw["blocks_bytes"])
    accounting_modes = tuple(matrix_raw["accounting_modes"])
    if not layouts or not set(layouts) <= SUPPORTED_LAYOUTS:
        raise ValueError(f"layouts must use {sorted(SUPPORTED_LAYOUTS)}")
    if not blocks_bytes or min(blocks_bytes) < 1:
        raise ValueError("blocks_bytes must contain positive values")
    if (
        not accounting_modes
        or not set(accounting_modes) <= SUPPORTED_ACCOUNTING_MODES
    ):
        raise ValueError(
            f"accounting_modes must use {sorted(SUPPORTED_ACCOUNTING_MODES)}"
        )

    for name, values in (
        ("layouts", layouts),
        ("blocks_bytes", blocks_bytes),
        ("accounting_modes", accounting_modes),
    ):
        if len(values) != len(set(values)):
            raise ValueError(f"{name} cannot contain duplicates")

    pipelines = []
    for index, pipeline_raw in enumerate(raw["pipelines"]):
        _expect_keys(
            pipeline_raw,
            required={"name", "representation", "codec"},
            optional={"compression_level"},
            where=f"pipelines[{index}]",
        )
        pipeline = PipelineConfig(
            name=str(pipeline_raw["name"]),
            representation=str(pipeline_raw["representation"]),
            codec=str(pipeline_raw["codec"]),
            compression_level=pipeline_raw.get("compression_level"),
        )
        if pipeline.representation != "raw_bytes":
            raise ValueError("The first benchmark slice supports raw_bytes only")
        if pipeline.codec not in SUPPORTED_CODECS:
            raise ValueError(f"codec must use {sorted(SUPPORTED_CODECS)}")
        if pipeline.codec == "zstd" and pipeline.compression_level is None:
            raise ValueError("zstd pipelines require compression_level")
        if pipeline.compression_level is not None and type(
            pipeline.compression_level
        ) is not int:
            raise ValueError("compression_level must be an integer")
        if (
            pipeline.codec == "identity"
            and pipeline.compression_level is not None
        ):
            raise ValueError("identity pipelines cannot set compression_level")
        pipelines.append(pipeline)
    if not pipelines or len({pipeline.name for pipeline in pipelines}) != len(
        pipelines
    ):
        raise ValueError("pipelines must be non-empty and have unique names")

    artifacts_raw = raw["artifacts"]
    _expect_keys(
        artifacts_raw,
        required={
            "root",
            "aggregate_results",
            "block_results",
            "streams",
        },
        optional=set(),
        where="artifacts",
    )
    artifact_root = Path(str(artifacts_raw["root"]))
    if artifact_root.is_absolute() or ".." in artifact_root.parts:
        raise ValueError("artifacts.root must be a repository-relative path")

    return SweepConfig(
        schema_version=raw["schema_version"],
        experiment_id=str(raw["experiment_id"]),
        execution=execution,
        dataset=dataset,
        layouts=layouts,
        blocks_bytes=blocks_bytes,
        accounting_modes=accounting_modes,
        pipelines=tuple(pipelines),
        artifact_root=artifact_root,
        aggregate_results=_relative_path(
            artifacts_raw["aggregate_results"], "aggregate_results"
        ),
        block_results=_relative_path(
            artifacts_raw["block_results"], "block_results"
        ),
        streams=_relative_path(artifacts_raw["streams"], "streams"),
    )
