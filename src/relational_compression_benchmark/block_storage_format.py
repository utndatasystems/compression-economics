"""Independent block framing, codecs, indexes, and exact size accounting.

Archives store a fixed prefix, optional representation metadata, the canonical
table header, checksummed block frames and payloads, then a persisted index.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
import struct
import time
import zlib

import zstandard

from src.relational_compression_benchmark.serialization import SerializedTable, parse_source_header
from src.relational_compression_benchmark.tokenization import (
    TokenizerAdapter,
    pack_token_ids,
    unpack_token_ids,
)


ARCHIVE_MAGIC = b"CEB1"
ARCHIVE_VERSION = 2
BLOCK_MAGIC = b"BLK1"
INDEX_MAGIC = b"IDX1"

_ARCHIVE_PREFIX = struct.Struct(">4sBBHIIQ")
_BLOCK_FRAME = struct.Struct(">4sQIIIII")
BLOCK_FRAME_BYTES = _BLOCK_FRAME.size
_INDEX_PREFIX = struct.Struct(">4sI")
_INDEX_ENTRY = struct.Struct(">QIQI")

_PIPELINE_IDS = {
    ("raw_bytes", "identity"): 0,
    ("raw_bytes", "zstd"): 1,
    ("token_ids", "identity"): 2,
    ("token_ids", "zstd"): 3,
}
_PIPELINE_NAMES = {value: key for key, value in _PIPELINE_IDS.items()}


@dataclass(frozen=True)
class Accounting:
    """Non-overlapping byte counts charged to one encoded archive."""

    source_bytes: int
    payload_bytes: int
    framing_bytes: int
    index_bytes: int
    seed_bytes: int = 0
    tokenizer_bytes: int = 0
    adapter_bytes: int = 0
    model_bytes_amortized: int = 0

    @property
    def total_stored_bytes(self) -> int:
        """Return the sum of every persisted component."""
        return (
            self.payload_bytes
            + self.framing_bytes
            + self.index_bytes
            + self.seed_bytes
            + self.tokenizer_bytes
            + self.adapter_bytes
            + self.model_bytes_amortized
        )

    @property
    def bits_per_source_byte(self) -> float:
        """Return total storage normalized to source bytes."""
        return 8.0 * self.total_stored_bytes / self.source_bytes

    @property
    def compression_factor(self) -> float:
        """Return source bytes divided by total stored bytes."""
        return self.source_bytes / self.total_stored_bytes

    def as_dict(self) -> dict[str, int | float]:
        """Expose base and derived metrics for result records."""
        return {
            "source_bytes": self.source_bytes,
            "payload_bytes": self.payload_bytes,
            "framing_bytes": self.framing_bytes,
            "index_bytes": self.index_bytes,
            "seed_bytes": self.seed_bytes,
            "tokenizer_bytes": self.tokenizer_bytes,
            "adapter_bytes": self.adapter_bytes,
            "model_bytes_amortized": self.model_bytes_amortized,
            "total_stored_bytes": self.total_stored_bytes,
            "bits_per_source_byte": self.bits_per_source_byte,
            "compression_factor": self.compression_factor,
        }


@dataclass(frozen=True)
class BlockMetric:
    """Sizes and phase timings for one independently decodable block."""

    block_index: int
    first_cell: int
    cell_count: int
    source_bytes: int
    payload_bytes: int
    framing_bytes: int
    index_bytes: int
    compression_seconds: float
    decompression_seconds: float = 0.0
    token_count: int = 0
    tokenization_seconds: float = 0.0
    token_packing_seconds: float = 0.0
    token_unpacking_seconds: float = 0.0
    detokenization_seconds: float = 0.0

    @property
    def stored_bytes(self) -> int:
        """Return payload plus this block's frame and index entry."""
        return self.payload_bytes + self.framing_bytes + self.index_bytes

    @property
    def compression_factor(self) -> float:
        """Return this block's source-to-storage ratio."""
        return self.source_bytes / self.stored_bytes


@dataclass(frozen=True)
class EncodedArchive:
    """Encoded bytes together with accounting and timing observations."""

    data: bytes
    accounting: Accounting
    blocks: tuple[BlockMetric, ...]
    codec_seconds: float
    archive_seconds: float
    representation: str


@dataclass(frozen=True)
class DecodedArchive:
    """Canonical source bytes recovered from a validated archive."""

    source_bytes: bytes
    blocks: tuple[BlockMetric, ...]
    codec_seconds: float
    archive_seconds: float
    representation: str


def _plan_blocks(
    cells: tuple[bytes, ...], target_bytes: int
) -> list[tuple[int, tuple[bytes, ...]]]:
    """Group complete encoded values into approximately sized blocks."""
    if target_bytes < 1:
        raise ValueError("target_bytes must be positive")
    blocks: list[tuple[int, tuple[bytes, ...]]] = []
    current: list[bytes] = []
    current_size = 0
    first_cell = 0
    for cell_index, cell in enumerate(cells):
        if current and current_size + len(cell) > target_bytes:
            blocks.append((first_cell, tuple(current)))
            current = []
            current_size = 0
            first_cell = cell_index
        current.append(cell)
        current_size += len(cell)
    if current:
        blocks.append((first_cell, tuple(current)))
    return blocks


def encode_archive(
    serialized: SerializedTable,
    *,
    target_block_bytes: int,
    codec: str,
    compression_level: int | None = None,
    representation: str = "raw_bytes",
    tokenizer: TokenizerAdapter | None = None,
) -> EncodedArchive:
    """Transform, compress, and frame canonical cells with an explicit index."""
    pipeline = (representation, codec)
    if pipeline not in _PIPELINE_IDS:
        raise ValueError(f"Unsupported representation/codec pair {pipeline!r}")
    if representation == "token_ids":
        if tokenizer is None:
            raise ValueError("token_ids representation requires a tokenizer")
        metadata = json.dumps(
            tokenizer.descriptor, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    else:
        if tokenizer is not None:
            raise ValueError("raw_bytes representation cannot use a tokenizer")
        metadata = b""
    if len(metadata) > 65_535:
        raise ValueError("archive metadata exceeds its 16-bit length field")

    if codec == "zstd":
        if compression_level is None:
            raise ValueError("zstd requires a compression level")
        compressor = zstandard.ZstdCompressor(level=compression_level)
    else:
        compressor = None

    started = time.perf_counter()
    block_parts: list[bytes] = []
    block_metrics: list[BlockMetric] = []
    index_rows: list[tuple[int, int, int, int]] = []
    offset = (
        _ARCHIVE_PREFIX.size
        + len(metadata)
        + len(serialized.source_header)
    )
    codec_seconds = 0.0

    for block_index, (first_cell, cells) in enumerate(
        _plan_blocks(serialized.cells, target_block_bytes)
    ):
        raw = b"".join(cells)
        token_count = 0
        tokenization_seconds = 0.0
        token_packing_seconds = 0.0
        codec_input = raw
        if tokenizer is not None:
            phase_started = time.perf_counter()
            token_ids = tokenizer.encode_bytes(raw)
            tokenization_seconds = time.perf_counter() - phase_started
            token_count = len(token_ids)

            phase_started = time.perf_counter()
            codec_input = pack_token_ids(token_ids, tokenizer.id_width)
            token_packing_seconds = time.perf_counter() - phase_started

        codec_started = time.perf_counter()
        payload = (
            codec_input
            if compressor is None
            else compressor.compress(codec_input)
        )
        elapsed = time.perf_counter() - codec_started
        codec_seconds += elapsed
        frame = _BLOCK_FRAME.pack(
            BLOCK_MAGIC,
            first_cell,
            len(cells),
            len(raw),
            token_count,
            len(payload),
            zlib.crc32(raw),
        )
        part = frame + payload
        block_parts.append(part)
        index_rows.append((offset, len(part), first_cell, len(cells)))
        offset += len(part)
        block_metrics.append(
            BlockMetric(
                block_index=block_index,
                first_cell=first_cell,
                cell_count=len(cells),
                source_bytes=len(raw),
                payload_bytes=len(payload),
                framing_bytes=len(frame),
                index_bytes=_INDEX_ENTRY.size,
                compression_seconds=elapsed,
                token_count=token_count,
                tokenization_seconds=tokenization_seconds,
                token_packing_seconds=token_packing_seconds,
            )
        )

    index = bytearray(_INDEX_PREFIX.pack(INDEX_MAGIC, len(index_rows)))
    for row in index_rows:
        index.extend(_INDEX_ENTRY.pack(*row))
    prefix = _ARCHIVE_PREFIX.pack(
        ARCHIVE_MAGIC,
        ARCHIVE_VERSION,
        _PIPELINE_IDS[pipeline],
        len(metadata),
        len(serialized.source_header),
        len(block_parts),
        offset,
    )
    archive = (
        prefix
        + metadata
        + serialized.source_header
        + b"".join(block_parts)
        + bytes(index)
    )
    accounting = Accounting(
        source_bytes=len(serialized.source_bytes),
        payload_bytes=sum(metric.payload_bytes for metric in block_metrics),
        framing_bytes=(
            _ARCHIVE_PREFIX.size
            + len(metadata)
            + len(serialized.source_header)
            + len(block_metrics) * _BLOCK_FRAME.size
        ),
        index_bytes=len(index),
    )
    if accounting.total_stored_bytes != len(archive):
        raise AssertionError("Archive accounting does not equal serialized size")
    return EncodedArchive(
        data=archive,
        accounting=accounting,
        blocks=tuple(block_metrics),
        codec_seconds=codec_seconds,
        archive_seconds=time.perf_counter() - started,
        representation=representation,
    )


def decode_archive(
    data: bytes, *, tokenizer: TokenizerAdapter | None = None
) -> DecodedArchive:
    """Decode an archive, validating metadata, index, order, sizes, and CRCs."""
    started = time.perf_counter()
    if len(data) < _ARCHIVE_PREFIX.size:
        raise ValueError("Truncated archive header")
    (
        magic,
        version,
        pipeline_id,
        metadata_length,
        source_header_length,
        block_count,
        index_offset,
    ) = _ARCHIVE_PREFIX.unpack_from(data)
    if magic != ARCHIVE_MAGIC or version != ARCHIVE_VERSION:
        raise ValueError("Unsupported archive format")
    try:
        representation, codec = _PIPELINE_NAMES[pipeline_id]
    except KeyError as error:
        raise ValueError(f"Unknown pipeline ID {pipeline_id}") from error

    metadata_start = _ARCHIVE_PREFIX.size
    metadata_end = metadata_start + metadata_length
    if metadata_end > len(data):
        raise ValueError("Invalid archive metadata length")
    metadata_bytes = data[metadata_start:metadata_end]
    if representation == "token_ids":
        if tokenizer is None:
            raise ValueError("token archive decoding requires a tokenizer")
        try:
            descriptor = json.loads(metadata_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("Invalid tokenizer metadata") from error
        if not isinstance(descriptor, dict):
            raise ValueError("Tokenizer metadata must be an object")
        tokenizer.validate_descriptor(descriptor)
    elif metadata_bytes:
        raise ValueError("Raw-byte archives cannot contain tokenizer metadata")

    decompressor = (
        zstandard.ZstdDecompressor() if codec == "zstd" else None
    )
    source_header_start = metadata_end
    source_header_end = source_header_start + source_header_length
    if source_header_end > len(data) or not source_header_length:
        raise ValueError("Invalid source header length")
    source_header = data[source_header_start:source_header_end]
    _, row_count, columns, parsed_header_end = parse_source_header(source_header)
    if parsed_header_end != len(source_header):
        raise ValueError("Source header contains trailing data")
    expected_cells = row_count * len(columns)

    if not source_header_end <= index_offset <= len(data):
        raise ValueError("Invalid index offset")
    cursor = source_header_end
    expected_first_cell = 0
    raw_blocks: list[bytes] = []
    metrics: list[BlockMetric] = []
    observed_index: list[tuple[int, int, int, int]] = []
    codec_seconds = 0.0
    for block_index in range(block_count):
        frame_offset = cursor
        frame_end = cursor + _BLOCK_FRAME.size
        if frame_end > index_offset:
            raise ValueError("Truncated block frame")
        (
            block_magic,
            first_cell,
            cell_count,
            raw_length,
            token_count,
            payload_length,
            checksum,
        ) = _BLOCK_FRAME.unpack_from(data, cursor)
        if block_magic != BLOCK_MAGIC or first_cell != expected_first_cell:
            raise ValueError("Invalid block order or magic")
        if representation == "raw_bytes" and token_count:
            raise ValueError("Raw-byte block declares token IDs")
        payload_end = frame_end + payload_length
        if payload_end > index_offset:
            raise ValueError("Truncated block payload")
        payload = data[frame_end:payload_end]
        packed_length = (
            raw_length
            if tokenizer is None
            else token_count * tokenizer.id_width
        )

        codec_started = time.perf_counter()
        codec_output = (
            payload
            if decompressor is None
            else decompressor.decompress(
                payload, max_output_size=packed_length
            )
        )
        elapsed = time.perf_counter() - codec_started
        if len(codec_output) != packed_length:
            raise ValueError(
                f"Decoded block has {len(codec_output)} bytes; "
                f"expected {packed_length}"
            )
        codec_seconds += elapsed

        token_unpacking_seconds = 0.0
        detokenization_seconds = 0.0
        raw = codec_output
        if tokenizer is not None:
            phase_started = time.perf_counter()
            token_ids = unpack_token_ids(codec_output, tokenizer.id_width)
            token_unpacking_seconds = time.perf_counter() - phase_started
            if len(token_ids) != token_count:
                raise ValueError("Decoded token count does not match block frame")

            phase_started = time.perf_counter()
            raw = tokenizer.decode_bytes(token_ids)
            detokenization_seconds = time.perf_counter() - phase_started

        if len(raw) != raw_length:
            raise ValueError(
                f"Reconstructed source block has {len(raw)} bytes; "
                f"expected {raw_length}"
            )
        if zlib.crc32(raw) != checksum:
            raise ValueError("Block checksum mismatch")
        raw_blocks.append(raw)
        observed_index.append(
            (
                frame_offset,
                _BLOCK_FRAME.size + payload_length,
                first_cell,
                cell_count,
            )
        )
        metrics.append(
            BlockMetric(
                block_index=block_index,
                first_cell=first_cell,
                cell_count=cell_count,
                source_bytes=raw_length,
                payload_bytes=payload_length,
                framing_bytes=_BLOCK_FRAME.size,
                index_bytes=_INDEX_ENTRY.size,
                compression_seconds=0.0,
                decompression_seconds=elapsed,
                token_count=token_count,
                token_unpacking_seconds=token_unpacking_seconds,
                detokenization_seconds=detokenization_seconds,
            )
        )
        expected_first_cell += cell_count
        cursor = payload_end
    if cursor != index_offset or expected_first_cell != expected_cells:
        raise ValueError("Block region does not match the declared table shape")

    index_header_end = index_offset + _INDEX_PREFIX.size
    if index_header_end > len(data):
        raise ValueError("Truncated archive index")
    index_magic, index_count = _INDEX_PREFIX.unpack_from(data, index_offset)
    if index_magic != INDEX_MAGIC or index_count != block_count:
        raise ValueError("Invalid archive index header")
    cursor = index_header_end
    stored_index = []
    for _ in range(index_count):
        end = cursor + _INDEX_ENTRY.size
        if end > len(data):
            raise ValueError("Truncated archive index entry")
        stored_index.append(_INDEX_ENTRY.unpack_from(data, cursor))
        cursor = end
    if cursor != len(data) or stored_index != observed_index:
        raise ValueError("Archive index does not match block frames")

    return DecodedArchive(
        source_bytes=source_header + b"".join(raw_blocks),
        blocks=tuple(metrics),
        codec_seconds=codec_seconds,
        archive_seconds=time.perf_counter() - started,
        representation=representation,
    )

def attach_decompression_times(
    encoded: tuple[BlockMetric, ...], decoded: tuple[BlockMetric, ...]
) -> tuple[BlockMetric, ...]:
    """Join matching encoder and decoder block measurements."""
    if len(encoded) != len(decoded):
        raise ValueError("Encoder and decoder block counts differ")
    output = []
    for left, right in zip(encoded, decoded):
        if (
            left.block_index,
            left.first_cell,
            left.cell_count,
            left.source_bytes,
            left.payload_bytes,
            left.token_count,
        ) != (
            right.block_index,
            right.first_cell,
            right.cell_count,
            right.source_bytes,
            right.payload_bytes,
            right.token_count,
        ):
            raise ValueError("Encoder and decoder block metadata differ")
        output.append(
            replace(
                left,
                decompression_seconds=right.decompression_seconds,
                token_unpacking_seconds=right.token_unpacking_seconds,
                detokenization_seconds=right.detokenization_seconds,
            )
        )
    return tuple(output)
