"""Typed, strict configuration for relational compression sweeps."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib
from typing import Any, TypeAlias


SUPPORTED_LAYOUTS = frozenset({"row_major", "column_major"})
SUPPORTED_REPRESENTATIONS = frozenset({"raw_bytes", "token_ids"})
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
class ImdbDatasetConfig:
    """Pinned IMDb title.basics source and deterministic sample controls."""

    kind: str
    split: str
    rows: int
    path: Path
    sha256: str
    split_seed: int
    evaluation_fraction: float


DatasetConfig: TypeAlias = SyntheticDatasetConfig | ImdbDatasetConfig


@dataclass(frozen=True)
class PipelineConfig:
    """One valid representation and codec combination."""

    name: str
    representation: str
    codec: str
    compression_level: int | None
    tokenizer_name: str | None = None
    tokenizer_revision: str | None = None


@dataclass(frozen=True)
class SweepConfig:
    """Fully validated benchmark configuration."""

    schema_version: int
    experiment_id: str
    execution: ExecutionConfig
    dataset: DatasetConfig
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


def _load_dataset_config(raw: dict[str, Any]) -> DatasetConfig:
    """Load one supported dataset configuration with kind-specific fields."""
    kind = raw.get("kind")
    if kind == "synthetic_relational":
        _expect_keys(
            raw,
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
        config = SyntheticDatasetConfig(
            **{
                **raw,
                "cross_column_correlation": _probability(
                    raw["cross_column_correlation"],
                    "cross_column_correlation",
                ),
                "within_column_repetition": _probability(
                    raw["within_column_repetition"],
                    "within_column_repetition",
                ),
                "null_rate": _probability(raw["null_rate"], "null_rate"),
                "column_kinds": tuple(raw["column_kinds"]),
            }
        )
        if min(config.rows, config.columns, config.cardinality) < 1:
            raise ValueError("rows, columns, and cardinality must be positive")
        if config.mean_string_length < 1 or not config.column_kinds:
            raise ValueError("mean_string_length and column_kinds must be non-empty")
        return config

    if kind == "imdb_title_basics":
        _expect_keys(
            raw,
            required={
                "kind",
                "split",
                "rows",
                "path",
                "sha256",
                "split_seed",
                "evaluation_fraction",
            },
            optional=set(),
            where="dataset",
        )
        sha256 = str(raw["sha256"]).lower()
        if len(sha256) != 64 or any(
            character not in "0123456789abcdef" for character in sha256
        ):
            raise ValueError("dataset.sha256 must be a hexadecimal SHA-256")
        config = ImdbDatasetConfig(
            kind=kind,
            split=str(raw["split"]),
            rows=int(raw["rows"]),
            path=_relative_path(raw["path"], "dataset.path"),
            sha256=sha256,
            split_seed=int(raw["split_seed"]),
            evaluation_fraction=_probability(
                raw["evaluation_fraction"], "evaluation_fraction"
            ),
        )
        if config.rows < 1:
            raise ValueError("dataset.rows must be positive")
        if config.split not in {"tuning", "evaluation"}:
            raise ValueError("IMDb split must be 'tuning' or 'evaluation'")
        if config.evaluation_fraction in {0.0, 1.0}:
            raise ValueError("evaluation_fraction must be strictly between 0 and 1")
        return config

    raise ValueError(
        "dataset.kind must be 'synthetic_relational' or 'imdb_title_basics'"
    )


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

    dataset = _load_dataset_config(raw["dataset"])

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
            optional={
                "compression_level",
                "tokenizer_name",
                "tokenizer_revision",
            },
            where=f"pipelines[{index}]",
        )
        pipeline = PipelineConfig(
            name=str(pipeline_raw["name"]),
            representation=str(pipeline_raw["representation"]),
            codec=str(pipeline_raw["codec"]),
            compression_level=pipeline_raw.get("compression_level"),
            tokenizer_name=pipeline_raw.get("tokenizer_name"),
            tokenizer_revision=pipeline_raw.get("tokenizer_revision"),
        )
        if pipeline.representation not in SUPPORTED_REPRESENTATIONS:
            raise ValueError(
                "representation must use "
                f"{sorted(SUPPORTED_REPRESENTATIONS)}"
            )
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
        tokenizer_fields = (
            pipeline.tokenizer_name,
            pipeline.tokenizer_revision,
        )
        if pipeline.representation == "token_ids":
            if not all(
                isinstance(value, str) and value.strip()
                for value in tokenizer_fields
            ):
                raise ValueError(
                    "token_ids pipelines require tokenizer_name and "
                    "tokenizer_revision"
                )
        elif any(value is not None for value in tokenizer_fields):
            raise ValueError("raw_bytes pipelines cannot configure a tokenizer")
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
