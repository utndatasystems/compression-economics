"""Canonical, byte-exact row-major and column-major table serialization.

All integers are big-endian. A source contains a fixed header, one schema entry
per column, and self-delimiting cell values in the selected physical order. The
schema is stored once and cell boundaries are retained for block planning.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from functools import cached_property
import struct

from src.relational_compression_benchmark.table import Cell, Column, LogicalType, Table


SOURCE_MAGIC = b"CES1"
SOURCE_VERSION = 1
_SOURCE_PREFIX = struct.Struct(">4sBBQH")
_COLUMN_PREFIX = struct.Struct(">BBH")
_LENGTH = struct.Struct(">I")


class Layout(IntEnum):
    """Physical order used for the canonical cell stream."""

    ROW_MAJOR = 1
    COLUMN_MAJOR = 2

    @classmethod
    def from_name(cls, name: str) -> "Layout":
        """Translate a configuration name into its stable format ID."""
        try:
            return {
                "row_major": cls.ROW_MAJOR,
                "column_major": cls.COLUMN_MAJOR,
            }[name]
        except KeyError as error:
            raise ValueError(f"Unsupported layout {name!r}") from error



@dataclass(frozen=True)
class SerializedTable:
    """Canonical table bytes split at value boundaries for block planning."""

    layout: Layout
    source_header: bytes
    cells: tuple[bytes, ...]

    @cached_property
    def source_bytes(self) -> bytes:
        """Return the complete canonical source, joining cells only once."""
        return self.source_header + b"".join(self.cells)


def _encode_schema(columns: tuple[Column, ...]) -> bytes:
    """Encode column types, nullability, and UTF-8 names."""
    output = bytearray()
    for column in columns:
        name = column.name.encode("utf-8")
        output.extend(
            _COLUMN_PREFIX.pack(
                int(column.logical_type), int(column.nullable), len(name)
            )
        )
        output.extend(name)
    return bytes(output)


def _encode_cell(column: Column, value: Cell) -> bytes:
    """Encode one typed value with an explicit null marker."""
    if value is None:
        return b"\x00"
    if column.logical_type is LogicalType.BOOLEAN:
        payload = bytes((int(value),))
    elif column.logical_type is LogicalType.INT64:
        payload = struct.pack(">q", value)
    elif column.logical_type is LogicalType.FLOAT64:
        payload = struct.pack(">d", value)
    elif column.logical_type is LogicalType.UTF8:
        encoded = value.encode("utf-8")
        payload = _LENGTH.pack(len(encoded)) + encoded
    elif column.logical_type is LogicalType.BYTES:
        payload = _LENGTH.pack(len(value)) + value
    else:  # pragma: no cover - the enum makes this defensive only
        raise ValueError(f"Unsupported logical type {column.logical_type}")
    return b"\x01" + payload


def serialize_table(table: Table, layout_name: str) -> SerializedTable:
    """Serialize a table without repeating schema or column names."""
    layout = Layout.from_name(layout_name)
    header = _SOURCE_PREFIX.pack(
        SOURCE_MAGIC,
        SOURCE_VERSION,
        int(layout),
        len(table.rows),
        len(table.columns),
    ) + _encode_schema(table.columns)

    cells = []
    if layout is Layout.ROW_MAJOR:
        for row in table.rows:
            cells.extend(
                _encode_cell(column, value)
                for column, value in zip(table.columns, row)
            )
    else:
        for column_index, column in enumerate(table.columns):
            cells.extend(
                _encode_cell(column, row[column_index]) for row in table.rows
            )
    return SerializedTable(layout=layout, source_header=header, cells=tuple(cells))


def _take(
    data: bytes, offset: int, size: int, description: str
) -> tuple[bytes, int]:
    """Read an exact byte range and report its new offset."""
    end = offset + size
    if end > len(data):
        raise ValueError(f"Truncated source while reading {description}")
    return data[offset:end], end


def parse_source_header(data: bytes) -> tuple[Layout, int, tuple[Column, ...], int]:
    """Return layout, row count, columns, and the first data-byte offset."""
    prefix, offset = _take(data, 0, _SOURCE_PREFIX.size, "source header")
    magic, version, layout_id, row_count, column_count = _SOURCE_PREFIX.unpack(
        prefix
    )
    if magic != SOURCE_MAGIC or version != SOURCE_VERSION:
        raise ValueError("Unsupported canonical table format")
    try:
        layout = Layout(layout_id)
    except ValueError as error:
        raise ValueError(f"Unknown physical layout ID {layout_id}") from error

    columns = []
    for _ in range(column_count):
        prefix, offset = _take(
            data, offset, _COLUMN_PREFIX.size, "column schema"
        )
        type_id, nullable, name_length = _COLUMN_PREFIX.unpack(prefix)
        name_bytes, offset = _take(data, offset, name_length, "column name")
        try:
            logical_type = LogicalType(type_id)
        except ValueError as error:
            raise ValueError(f"Unknown logical type ID {type_id}") from error
        if nullable not in (0, 1):
            raise ValueError("Invalid nullable flag")
        try:
            name = name_bytes.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError("Column names must be valid UTF-8") from error
        columns.append(Column(name, logical_type, bool(nullable)))
    return layout, row_count, tuple(columns), offset


def _decode_cell(
    data: bytes, offset: int, column: Column
) -> tuple[Cell, int]:
    """Decode one typed value and return its new offset."""
    marker, offset = _take(data, offset, 1, "null marker")
    if marker == b"\x00":
        if not column.nullable:
            raise ValueError(f"Non-nullable column {column.name!r} contains null")
        return None, offset
    if marker != b"\x01":
        raise ValueError("Invalid null marker")

    if column.logical_type is LogicalType.BOOLEAN:
        payload, offset = _take(data, offset, 1, "boolean")
        if payload[0] not in (0, 1):
            raise ValueError("Invalid boolean value")
        return bool(payload[0]), offset
    if column.logical_type is LogicalType.INT64:
        payload, offset = _take(data, offset, 8, "int64")
        return struct.unpack(">q", payload)[0], offset
    if column.logical_type is LogicalType.FLOAT64:
        payload, offset = _take(data, offset, 8, "float64")
        return struct.unpack(">d", payload)[0], offset
    if column.logical_type in (LogicalType.UTF8, LogicalType.BYTES):
        length_bytes, offset = _take(data, offset, _LENGTH.size, "value length")
        length = _LENGTH.unpack(length_bytes)[0]
        payload, offset = _take(data, offset, length, "variable-length value")
        if column.logical_type is LogicalType.UTF8:
            try:
                return payload.decode("utf-8"), offset
            except UnicodeDecodeError as error:
                raise ValueError("String value is not valid UTF-8") from error
        return payload, offset
    raise ValueError(f"Unsupported logical type {column.logical_type}")


def deserialize_table(data: bytes) -> Table:
    """Decode canonical bytes and reject truncation or trailing data."""
    layout, row_count, columns, offset = parse_source_header(data)
    rows: list[list[Cell]] = [
        [None for _ in columns] for _ in range(row_count)
    ]
    if layout is Layout.ROW_MAJOR:
        for row in rows:
            for column_index, column in enumerate(columns):
                row[column_index], offset = _decode_cell(data, offset, column)
    else:
        for column_index, column in enumerate(columns):
            for row in rows:
                row[column_index], offset = _decode_cell(data, offset, column)
    if offset != len(data):
        raise ValueError(f"Canonical table has {len(data) - offset} trailing bytes")
    return Table(columns=columns, rows=tuple(tuple(row) for row in rows))
