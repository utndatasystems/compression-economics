"""Versioned, self-describing containers for Qwen arithmetic streams."""

from __future__ import annotations

import hashlib
import json
import math
import struct
from typing import Any, Mapping, Sequence


MAGIC = b"QACS"
VERSION = 1
PREFIX = struct.Struct(">4sB3xIQ")
REQUIRED_METADATA = {
    "model_name",
    "seed_token_ids",
    "token_count",
    "raw_size_bytes",
    "context_length",
    "retain_tokens",
    "use_kv_cache",
    "statesize",
    "frequency_total",
    "candidate_token_ids",
    "decoded_sha256",
}


def _pack_bits(bits: Sequence[int]) -> bytes:
    output = bytearray(math.ceil(len(bits) / 8))
    for index, raw_bit in enumerate(bits):
        bit = int(raw_bit)
        if bit not in (0, 1):
            raise ValueError("arithmetic payload must contain only bits")
        output[index // 8] |= bit << (7 - index % 8)
    return bytes(output)


def _unpack_bits(payload: bytes, bit_count: int) -> list[int]:
    return [
        (payload[index // 8] >> (7 - index % 8)) & 1
        for index in range(bit_count)
    ]


def decoded_sha256(data: bytes) -> str:
    """Return the byte-level digest stored in a predictive stream."""
    return hashlib.sha256(data).hexdigest()


def serialize_qwen_stream(
    bits: Sequence[int],
    *,
    model_name: str,
    seed_token_ids: Sequence[int],
    token_count: int,
    raw_size_bytes: int,
    context_length: int,
    retain_tokens: int,
    use_kv_cache: bool,
    decoded_bytes: bytes,
    candidate_token_ids: Sequence[int] | None = None,
    statesize: int = 32,
    frequency_total: int = 262144,
) -> bytes:
    """Serialize a complete predictive stream; model weights remain shared."""
    seeds = [int(token_id) for token_id in seed_token_ids]
    candidates = (
        None
        if candidate_token_ids is None
        else sorted(set(int(token_id) for token_id in candidate_token_ids))
    )
    if not model_name or not seeds:
        raise ValueError("model_name and seed_token_ids are required")
    if token_count <= len(seeds):
        raise ValueError("token_count must exceed the seed length")
    if raw_size_bytes != len(decoded_bytes) or raw_size_bytes <= 0:
        raise ValueError("raw_size_bytes must match decoded_bytes")
    if not 0 < retain_tokens <= context_length:
        raise ValueError("invalid context configuration")
    if statesize <= 0 or frequency_total <= 0:
        raise ValueError("invalid arithmetic-coder configuration")

    metadata = {
        "candidate_token_ids": candidates,
        "context_length": int(context_length),
        "decoded_sha256": decoded_sha256(decoded_bytes),
        "frequency_total": int(frequency_total),
        "model_name": model_name,
        "raw_size_bytes": int(raw_size_bytes),
        "retain_tokens": int(retain_tokens),
        "seed_token_ids": seeds,
        "statesize": int(statesize),
        "token_count": int(token_count),
        "use_kv_cache": bool(use_kv_cache),
    }
    encoded_metadata = json.dumps(
        metadata, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    payload = _pack_bits(bits)
    return PREFIX.pack(MAGIC, VERSION, len(encoded_metadata), len(bits)) + encoded_metadata + payload


def parse_qwen_stream(blob: bytes) -> dict[str, Any]:
    """Parse and validate a serialized Qwen arithmetic stream."""
    if len(blob) < PREFIX.size:
        raise ValueError("truncated Qwen stream prefix")
    magic, version, metadata_size, bit_count = PREFIX.unpack(blob[: PREFIX.size])
    if magic != MAGIC or version != VERSION:
        raise ValueError("unsupported Qwen stream")
    metadata_end = PREFIX.size + metadata_size
    if metadata_end > len(blob):
        raise ValueError("truncated Qwen stream metadata")
    try:
        metadata = json.loads(blob[PREFIX.size:metadata_end].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("invalid Qwen stream metadata") from error
    missing = REQUIRED_METADATA - set(metadata)
    if missing:
        raise ValueError(f"Qwen stream metadata is missing {sorted(missing)}")
    payload = blob[metadata_end:]
    if len(payload) != math.ceil(bit_count / 8):
        raise ValueError("Qwen stream payload length is inconsistent")
    if bit_count % 8 and payload and payload[-1] & ((1 << (8 - bit_count % 8)) - 1):
        raise ValueError("Qwen stream padding bits must be zero")
    if not metadata["seed_token_ids"]:
        raise ValueError("Qwen stream has an empty seed")
    if metadata["token_count"] <= len(metadata["seed_token_ids"]):
        raise ValueError("Qwen stream token count is invalid")
    if not 0 < metadata["retain_tokens"] <= metadata["context_length"]:
        raise ValueError("Qwen stream context configuration is invalid")
    return {**metadata, "payload_bit_count": bit_count, "bits": _unpack_bits(payload, bit_count)}


def verify_decoded_bytes(metadata: Mapping[str, Any], data: bytes) -> None:
    """Validate the decoded length and digest recorded by a stream."""
    if len(data) != metadata["raw_size_bytes"]:
        raise AssertionError("decoded byte length does not match the stream")
    if decoded_sha256(data) != metadata["decoded_sha256"]:
        raise AssertionError("decoded bytes do not match the stream digest")
