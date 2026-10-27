"""Shared protocol helpers for external LLM-compression baseline adapters."""

from __future__ import annotations

import hashlib
import json
import platform
import struct
import subprocess
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = 1
LLM_BASELINE_CODECS = frozenset({"finezip", "llmzip", "nacrith"})


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_revision(repository: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def torch_dtype(name: str):
    """Resolve a CLI dtype lazily so non-PyTorch summary tools stay lightweight."""

    import torch

    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def load_text_prefix(path: Path, byte_count: int) -> tuple[bytes, str]:
    if byte_count < 1:
        raise ValueError("input-bytes must be positive")
    source = path.read_bytes()[:byte_count]
    if len(source) != byte_count:
        raise ValueError("input is smaller than the requested prefix")
    try:
        text = source.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("input prefix is not valid UTF-8") from error
    if text.encode("utf-8") != source:
        raise ValueError("input prefix does not round-trip through UTF-8")
    return source, text


def prepare_work_archive(
    output: Path, requested_work_dir: Path | None, archive_name: str
) -> tuple[Path, Path]:
    work_dir = requested_work_dir or output.with_suffix("")
    work_dir.mkdir(parents=True, exist_ok=True)
    archive_path = work_dir / archive_name
    if archive_path.exists():
        raise FileExistsError(f"archive already exists: {archive_path}")
    return work_dir, archive_path


def write_framed_archive(
    path: Path, magic: bytes, metadata: dict[str, Any], payload: bytes
) -> None:
    if len(magic) != 4:
        raise ValueError("archive magic must contain exactly four bytes")
    metadata_bytes = json.dumps(
        metadata, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    with path.open("wb") as output:
        output.write(magic)
        output.write(struct.pack(">I", len(metadata_bytes)))
        output.write(metadata_bytes)
        output.write(payload)


def read_framed_archive(
    path: Path, magic: bytes, *, codec_label: str
) -> tuple[dict[str, Any], bytes]:
    with path.open("rb") as source:
        if source.read(4) != magic:
            raise ValueError(f"not a {magic.decode('ascii', errors='replace')} archive")
        size_bytes = source.read(4)
        if len(size_bytes) != 4:
            raise ValueError(f"{codec_label} archive has a truncated metadata header")
        metadata_size = struct.unpack(">I", size_bytes)[0]
        metadata_bytes = source.read(metadata_size)
        if len(metadata_bytes) != metadata_size:
            raise ValueError(f"{codec_label} archive has truncated metadata")
        metadata = json.loads(metadata_bytes)
        payload = source.read()
    if len(payload) != metadata.get("payload_bytes"):
        raise ValueError(f"{codec_label} payload length disagrees with metadata")
    return metadata, payload


def hugging_face_model_metadata(
    *, name: str, requested_revision: str | None, tokenizer: str, dtype: str
) -> dict[str, Any]:
    return {
        "name": name,
        "requested_revision": requested_revision,
        "tokenizer": tokenizer,
        "dtype": dtype,
        "external_weights_charged": False,
    }


def gpu_name(torch_module=None) -> str | None:
    if torch_module is not None:
        if not torch_module.cuda.is_available():
            return None
        return torch_module.cuda.get_device_name()
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip().splitlines()[0] if result.returncode == 0 else None


def runtime_environment(**details: Any) -> dict[str, Any]:
    return {"python": platform.python_version(), **details}


def torch_environment(
    torch_module,
    *,
    transformers_version: str,
    include_peak_memory: bool = True,
) -> dict[str, Any]:
    environment = runtime_environment(
        torch=torch_module.__version__,
        transformers=transformers_version,
        gpu=gpu_name(torch_module),
    )
    if include_peak_memory:
        environment["peak_torch_bytes"] = torch_module.cuda.max_memory_allocated()
    return environment


def build_baseline_result(
    *,
    codec: str,
    implementation: dict[str, Any],
    input_path: Path,
    source: bytes,
    model: dict[str, Any],
    configuration: dict[str, Any],
    archive_path: Path,
    model_load_seconds: float,
    compression_seconds: float,
    decompression_seconds: float,
    environment: dict[str, Any],
    exact_byte_recovery: bool,
    payload_bytes: int | None = None,
) -> dict[str, Any]:
    archive_bytes = archive_path.stat().st_size
    archive = {
        "path": str(archive_path.resolve()),
        "bytes": archive_bytes,
        "sha256": sha256_file(archive_path),
        "compression_ratio": len(source) / archive_bytes,
        "bits_per_byte": 8 * archive_bytes / len(source),
    }
    if payload_bytes is not None:
        archive["payload_bytes"] = payload_bytes
    return {
        "schema_version": SCHEMA_VERSION,
        "codec": codec,
        "implementation": implementation,
        "input": {
            "path": str(input_path.resolve()),
            "prefix_bytes": len(source),
            "sha256": sha256_bytes(source),
        },
        "model": model,
        "configuration": configuration,
        "archive": archive,
        "timing": {
            "model_load_seconds": model_load_seconds,
            "compression_seconds": compression_seconds,
            "decompression_seconds": decompression_seconds,
            "compression_mib_per_second": len(source) / compression_seconds / 2**20,
            "decompression_mib_per_second": len(source) / decompression_seconds / 2**20,
        },
        "validation": {"exact_byte_recovery": exact_byte_recovery},
        "environment": environment,
    }


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def combine_baseline_results(
    paths: Iterable[Path], *, expected_codecs: frozenset[str] = LLM_BASELINE_CODECS
) -> dict[str, Any]:
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    codecs = [row.get("codec") for row in rows]
    if len(codecs) != len(set(codecs)) or set(codecs) != expected_codecs:
        expected = ", ".join(sorted(expected_codecs))
        raise ValueError(f"inputs must contain exactly one result for each of: {expected}")
    if any(row.get("schema_version") != SCHEMA_VERSION for row in rows):
        raise ValueError(f"baseline results must use schema version {SCHEMA_VERSION}")
    hashes = {row["input"]["sha256"] for row in rows}
    sizes = {row["input"]["prefix_bytes"] for row in rows}
    if len(hashes) != 1 or len(sizes) != 1:
        raise ValueError("baseline inputs do not match")
    if not all(row["validation"]["exact_byte_recovery"] for row in rows):
        raise ValueError("a baseline failed exact byte recovery")
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment": "coding-survey LLM baselines",
        "input": {"bytes": sizes.pop(), "sha256": hashes.pop()},
        "accounting": "complete per-file archive; shared model/tokenizer weights excluded",
        "codecs": rows,
    }
