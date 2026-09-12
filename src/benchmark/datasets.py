"""Dataset loading and validation for reproducible benchmark inputs."""

from __future__ import annotations

import csv
import gzip
import hashlib
import random
from pathlib import Path

from src.benchmark.config import (
    DatasetConfig,
    ImdbDatasetConfig,
    SyntheticDatasetConfig,
)
from src.benchmark.table import (
    Cell,
    Column,
    LogicalType,
    Table,
    generate_synthetic_table,
)


IMDB_SAMPLING_VERSION = "sha256-split-reservoir-v1"

IMDB_TITLE_BASICS_HEADER = (
    "tconst",
    "titleType",
    "primaryTitle",
    "originalTitle",
    "isAdult",
    "startYear",
    "endYear",
    "runtimeMinutes",
    "genres",
)

IMDB_TITLE_BASICS_COLUMNS = (
    Column("tconst", LogicalType.UTF8, nullable=False),
    Column("titleType", LogicalType.UTF8),
    Column("primaryTitle", LogicalType.UTF8),
    Column("originalTitle", LogicalType.UTF8),
    Column("isAdult", LogicalType.BOOLEAN, nullable=False),
    Column("startYear", LogicalType.INT64),
    Column("endYear", LogicalType.INT64),
    Column("runtimeMinutes", LogicalType.INT64),
    Column("genres", LogicalType.UTF8),
)

_NULL = r"\N"
_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def file_sha256(path: Path) -> str:
    """Return the SHA-256 of a file without loading it all into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_evaluation_row(
    identifier: str, *, seed: int, evaluation_fraction: float
) -> bool:
    """Assign an identifier reproducibly to the evaluation partition."""
    digest = hashlib.sha256(f"{seed}:{identifier}".encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:8], "big")
    return bucket < int(evaluation_fraction * 2**64)


def _nullable_text(value: str) -> str | None:
    """Translate IMDb's null marker while preserving all other text."""
    return None if value == _NULL else value


def _nullable_integer(value: str, *, column: str, line_number: int) -> int | None:
    """Parse an optional IMDb integer with a useful error message."""
    if value == _NULL:
        return None
    try:
        return int(value)
    except ValueError as error:
        raise ValueError(
            f"IMDb line {line_number}: {column} must be an integer or {_NULL!r}"
        ) from error


def _parse_imdb_row(values: list[str], *, line_number: int) -> tuple[Cell, ...]:
    """Convert one title.basics TSV record to canonical Python values."""
    if len(values) != len(IMDB_TITLE_BASICS_HEADER):
        raise ValueError(
            f"IMDb line {line_number}: found {len(values)} fields; "
            f"expected {len(IMDB_TITLE_BASICS_HEADER)}"
        )

    tconst, title_type, primary_title, original_title, is_adult, *tail = values
    if tconst == _NULL:
        raise ValueError(f"IMDb line {line_number}: tconst cannot be null")
    if is_adult not in {"0", "1"}:
        raise ValueError(f"IMDb line {line_number}: isAdult must be 0 or 1")

    start_year, end_year, runtime_minutes, genres = tail
    return (
        tconst,
        _nullable_text(title_type),
        _nullable_text(primary_title),
        _nullable_text(original_title),
        is_adult == "1",
        _nullable_integer(
            start_year, column="startYear", line_number=line_number
        ),
        _nullable_integer(end_year, column="endYear", line_number=line_number),
        _nullable_integer(
            runtime_minutes, column="runtimeMinutes", line_number=line_number
        ),
        _nullable_text(genres),
    )


def load_imdb_title_basics(config: ImdbDatasetConfig) -> Table:
    """Load a checksum-pinned, deterministic sample of IMDb title.basics."""
    path = _REPOSITORY_ROOT / config.path
    if not path.is_file():
        raise FileNotFoundError(
            f"IMDb source not found at {path}; run scripts/prepare_cidr_imdb.py"
        )
    actual_sha256 = file_sha256(path)
    if actual_sha256 != config.sha256:
        raise ValueError(
            f"IMDb source checksum mismatch: expected {config.sha256}, "
            f"found {actual_sha256}"
        )

    sampled_rows: list[tuple[int, tuple[Cell, ...]]] = []
    matching_rows = 0
    randomizer = random.Random(
        f"{IMDB_SAMPLING_VERSION}:{config.split_seed}:{config.split}"
    )
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = tuple(next(reader, ()))
        if header != IMDB_TITLE_BASICS_HEADER:
            raise ValueError(
                "Unexpected IMDb title.basics header: " + repr(header)
            )
        for line_number, values in enumerate(reader, start=2):
            is_evaluation = _is_evaluation_row(
                values[0] if values else "",
                seed=config.split_seed,
                evaluation_fraction=config.evaluation_fraction,
            )
            if is_evaluation != (config.split == "evaluation"):
                continue
            matching_rows += 1
            if len(sampled_rows) < config.rows:
                destination = len(sampled_rows)
            else:
                destination = randomizer.randrange(matching_rows)
                if destination >= config.rows:
                    continue
            row = _parse_imdb_row(values, line_number=line_number)
            entry = (line_number, row)
            if destination == len(sampled_rows):
                sampled_rows.append(entry)
            else:
                sampled_rows[destination] = entry

    if len(sampled_rows) != config.rows:
        raise ValueError(
            f"IMDb {config.split} split contains only {len(sampled_rows)} rows; "
            f"requested {config.rows}"
        )
    sampled_rows.sort(key=lambda entry: entry[0])
    rows = tuple(row for _, row in sampled_rows)
    return Table(columns=IMDB_TITLE_BASICS_COLUMNS, rows=rows)


def load_dataset(config: DatasetConfig) -> Table:
    """Load the configured synthetic or real-world benchmark table."""
    if isinstance(config, SyntheticDatasetConfig):
        return generate_synthetic_table(config)
    if isinstance(config, ImdbDatasetConfig):
        return load_imdb_title_basics(config)
    raise TypeError(f"Unsupported dataset configuration: {type(config).__name__}")
