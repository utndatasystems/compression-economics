"""Small typed table model and deterministic synthetic data generation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import math
import random
from typing import TypeAlias

from src.benchmark.config import SyntheticDatasetConfig


Cell: TypeAlias = None | bool | int | float | str | bytes


class LogicalType(IntEnum):
    """Logical types supported by the canonical pilot serialization."""

    BOOLEAN = 1
    INT64 = 2
    FLOAT64 = 3
    UTF8 = 4
    BYTES = 5


@dataclass(frozen=True)
class Column:
    """A named logical column in the canonical table schema."""

    name: str
    logical_type: LogicalType
    nullable: bool = True

    def __post_init__(self) -> None:
        """Validate constraints imposed by the canonical schema."""
        if not self.name:
            raise ValueError("Column names cannot be empty")
        if len(self.name.encode("utf-8")) > 65_535:
            raise ValueError("Column names must fit in the canonical schema")


@dataclass(frozen=True)
class Table:
    """An immutable typed relation represented as rows."""

    columns: tuple[Column, ...]
    rows: tuple[tuple[Cell, ...], ...]

    def __post_init__(self) -> None:
        """Validate column names, row widths, and cell types."""
        if not self.columns:
            raise ValueError("A table must have at least one column")
        if len({column.name for column in self.columns}) != len(self.columns):
            raise ValueError("Column names must be unique")
        for row_index, row in enumerate(self.rows):
            if len(row) != len(self.columns):
                raise ValueError(
                    f"Row {row_index} has {len(row)} values; "
                    f"expected {len(self.columns)}"
                )
            for column, value in zip(self.columns, row):
                _validate_cell(column, value)


def _validate_cell(column: Column, value: Cell) -> None:
    """Ensure one Python value matches its declared logical type."""
    if value is None:
        if not column.nullable:
            raise ValueError(f"Column {column.name!r} is not nullable")
        return

    expected = {
        LogicalType.BOOLEAN: bool,
        LogicalType.INT64: int,
        LogicalType.FLOAT64: float,
        LogicalType.UTF8: str,
        LogicalType.BYTES: bytes,
    }[column.logical_type]
    if type(value) is not expected:
        raise TypeError(
            f"Column {column.name!r} requires {expected.__name__}, "
            f"got {type(value).__name__}"
        )
    if column.logical_type is LogicalType.INT64 and not -(2**63) <= value < 2**63:
        raise ValueError(f"Column {column.name!r} contains an out-of-range int64")
    if column.logical_type is LogicalType.FLOAT64 and not math.isfinite(value):
        raise ValueError(f"Column {column.name!r} contains a non-finite float")


_KIND_TO_TYPE = {
    "natural_language": LogicalType.UTF8,
    "identifier": LogicalType.UTF8,
    "integer": LogicalType.INT64,
    "float": LogicalType.FLOAT64,
    "boolean": LogicalType.BOOLEAN,
    "bytes": LogicalType.BYTES,
}

_WORDS = (
    "amber",
    "bridge",
    "calm",
    "database",
    "engine",
    "forest",
    "gentle",
    "harbor",
    "index",
    "journey",
    "kind",
    "language",
    "model",
    "narrow",
    "open",
    "predicts",
)


def _natural_text(key: int, mean_length: int) -> str:
    """Create stable pseudo-natural text near the requested length."""
    target = max(1, mean_length + key % 7 - 3)
    words = (
        _WORDS[key % len(_WORDS)],
        _WORDS[(key * 5 + 3) % len(_WORDS)],
        _WORDS[(key * 11 + 7) % len(_WORDS)],
    )
    phrase = " ".join(words)
    repeated = (phrase + " ") * (target // (len(phrase) + 1) + 1)
    return repeated[:target].rstrip() or repeated[0]


def _value_for_kind(kind: str, key: int, mean_string_length: int) -> Cell:
    """Map a latent key to a deterministic value of the requested kind."""
    if kind == "natural_language":
        return _natural_text(key, mean_string_length)
    if kind == "identifier":
        return f"id_{key:08x}"
    if kind == "integer":
        return key
    if kind == "float":
        return key / 10.0
    if kind == "boolean":
        return bool(key % 2)
    if kind == "bytes":
        return f"blob-{key:08x}".encode("ascii")
    raise ValueError(
        f"Unsupported synthetic column kind {kind!r}; "
        f"choose from {sorted(_KIND_TO_TYPE)}"
    )


def generate_synthetic_table(config: SyntheticDatasetConfig) -> Table:
    """Generate a repeatable mixed-type table with tunable dependence.

    A row-level latent key creates cross-column dependence. Independently for
    each cell, the previous non-null value may be repeated to create vertical
    locality. The generator is deliberately simple and auditable; its parameters
    are controls, not claims about real database distributions.
    """
    unknown = set(config.column_kinds) - _KIND_TO_TYPE.keys()
    if unknown:
        raise ValueError(f"Unsupported synthetic column kinds: {sorted(unknown)}")

    kinds = tuple(
        config.column_kinds[index % len(config.column_kinds)]
        for index in range(config.columns)
    )
    columns = tuple(
        Column(
            name=f"{kind}_{index}",
            logical_type=_KIND_TO_TYPE[kind],
            nullable=config.null_rate > 0,
        )
        for index, kind in enumerate(kinds)
    )

    rng = random.Random(config.generation_seed)
    previous: list[Cell] = [None] * config.columns
    rows: list[tuple[Cell, ...]] = []
    for _ in range(config.rows):
        latent_key = rng.randrange(config.cardinality)
        row: list[Cell] = []
        for column_index, kind in enumerate(kinds):
            if rng.random() < config.null_rate:
                value = None
            elif previous[column_index] is not None and (
                rng.random() < config.within_column_repetition
            ):
                value = previous[column_index]
            else:
                key = (
                    latent_key
                    if rng.random() < config.cross_column_correlation
                    else rng.randrange(config.cardinality)
                )
                value = _value_for_kind(kind, key, config.mean_string_length)
            row.append(value)
            if value is not None:
                previous[column_index] = value
        rows.append(tuple(row))

    return Table(columns=columns, rows=tuple(rows))
