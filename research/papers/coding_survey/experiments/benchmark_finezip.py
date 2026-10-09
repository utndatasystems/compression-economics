#!/usr/bin/env python3
"""Run the official practical FineZip rank+bzip2 codec reproducibly.

The paper's practical FineZip condition uses dynamic-context token ranks and
bzip2; its arithmetic-coding ablation is much slower. The public implementation
does not provide a working self-describing decoder, so this adapter calls its
ZipModel rank generation/recovery unchanged and wraps the bzip2 payload in a
small FZB1 container. Model weights remain shared external state.
"""

from __future__ import annotations

import argparse
import bz2
import gc
import hashlib
import importlib.util
import json
import platform
import struct
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "artifacts/papers/coding-survey/llm-baselines/finezip-text8-first-10MiB.json"
)
MAGIC = b"FZB1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_revision(repository: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _load_upstream(repository: Path):
    script = repository / "finezip/eval_clean.py"
    if not script.is_file():
        raise FileNotFoundError(f"FineZip implementation not found: {script}")
    spec = importlib.util.spec_from_file_location("finezip_upstream_eval", script)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_archive(path: Path, metadata: dict, payload: bytes) -> None:
    metadata_bytes = json.dumps(
        metadata, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    with path.open("wb") as output:
        output.write(MAGIC)
        output.write(struct.pack(">I", len(metadata_bytes)))
        output.write(metadata_bytes)
        output.write(payload)


def _extract_archive(path: Path) -> tuple[dict, bytes]:
    with path.open("rb") as source:
        if source.read(4) != MAGIC:
            raise ValueError("not an FZB1 archive")
        metadata_size = struct.unpack(">I", source.read(4))[0]
        metadata = json.loads(source.read(metadata_size))
        payload = source.read()
    if len(payload) != metadata["payload_bytes"]:
        raise ValueError("FineZip payload length disagrees with metadata")
    return metadata, payload


def _dtype(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--finezip-repo", type=Path, required=True)
    parser.add_argument("--input", type=Path, default=REPO_ROOT / "data/text8")
    parser.add_argument("--input-bytes", type=int, default=10 * 1024 * 1024)
    parser.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
    parser.add_argument("--tokenizer")
    parser.add_argument("--model-revision")
    parser.add_argument(
        "--dtype",
        choices=("float32", "float16", "bfloat16"),
        default="bfloat16",
    )
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--context-size", type=int, default=512)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("FineZip requires a visible CUDA device")
    if args.input_bytes < 1 or args.batch_size < 1 or args.context_size < 1:
        parser.error("input-bytes, batch-size, and context-size must be positive")

    source = args.input.read_bytes()[: args.input_bytes]
    if len(source) != args.input_bytes:
        parser.error("input is smaller than the requested prefix")
    text = source.decode("utf-8")
    work_dir = args.work_dir or args.output.with_suffix("")
    work_dir.mkdir(parents=True, exist_ok=True)
    archive = work_dir / "finezip.fz"
    if archive.exists():
        parser.error(f"archive already exists: {archive}")

    upstream = _load_upstream(args.finezip_repo.resolve())
    tokenizer_name = args.tokenizer or args.model
    load_started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name, revision=args.model_revision
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.model_revision, dtype=_dtype(args.dtype)
    ).eval().cuda()
    torch.cuda.synchronize()
    model_load_seconds = time.perf_counter() - load_started

    zip_model = upstream.ZipModel(
        args.model,
        tokenizer_name,
        model,
        tokenizer,
        False,
        args.context_size,
        args.batch_size,
    )
    tokens = zip_model.text_to_tokens(text)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    ranks = zip_model.encode_tokens(tokens)
    rank_bytes = ranks.cpu().numpy().astype("<i4", copy=False).tobytes()
    payload = bz2.compress(rank_bytes)
    torch.cuda.synchronize()
    compression_seconds = time.perf_counter() - started

    archive_metadata = {
        "format": "FZB1",
        "algorithm": "FineZip dynamic-context ranks + bzip2",
        "batch_size": args.batch_size,
        "bzip2_level": 9,
        "context_size": args.context_size,
        "dtype": args.dtype,
        "model": args.model,
        "model_revision": args.model_revision,
        "payload_bytes": len(payload),
        "rank_dtype": "little-endian int32",
        "token_count": int(tokens.numel()),
        "tokenizer": tokenizer_name,
    }
    _write_archive(archive, archive_metadata, payload)

    extracted, extracted_payload = _extract_archive(archive)
    decoded_ranks = np.frombuffer(bz2.decompress(extracted_payload), dtype="<i4")
    if decoded_ranks.size != extracted["token_count"]:
        raise ValueError("FineZip decoded rank count disagrees with metadata")
    torch.cuda.synchronize()
    started = time.perf_counter()
    decoded_text = zip_model.decode(torch.from_numpy(decoded_ranks.copy()))
    torch.cuda.synchronize()
    decompression_seconds = time.perf_counter() - started
    recovered = decoded_text.encode("utf-8")
    exact = recovered == source
    if not exact:
        (work_dir / "decoded-mismatch.txt").write_text(
            decoded_text, encoding="utf-8"
        )
        raise AssertionError("FineZip did not recover the exact input bytes")

    archive_bytes = archive.stat().st_size
    result = {
        "schema_version": 1,
        "codec": "finezip",
        "implementation": {
            "repository": "https://github.com/fazalmittu/FineZip",
            "revision": _git_revision(args.finezip_repo.resolve()),
            "adapter_archive": "FZB1 (repository-local accounting wrapper)",
            "upstream_class": "finezip/eval_clean.py:ZipModel",
        },
        "input": {
            "path": str(args.input.resolve()),
            "prefix_bytes": len(source),
            "sha256": hashlib.sha256(source).hexdigest(),
        },
        "model": {
            "name": args.model,
            "requested_revision": args.model_revision,
            "tokenizer": tokenizer_name,
            "dtype": args.dtype,
            "external_weights_charged": False,
        },
        "configuration": {
            "batch_size": args.batch_size,
            "context_size": args.context_size,
            "token_count": int(tokens.numel()),
            "rank_storage": "int32 then bzip2",
        },
        "archive": {
            "path": str(archive.resolve()),
            "bytes": archive_bytes,
            "sha256": _sha256(archive),
            "payload_bytes": len(payload),
            "compression_ratio": len(source) / archive_bytes,
            "bits_per_byte": 8 * archive_bytes / len(source),
        },
        "timing": {
            "model_load_seconds": model_load_seconds,
            "compression_seconds": compression_seconds,
            "decompression_seconds": decompression_seconds,
            "compression_mib_per_second": len(source) / compression_seconds / 2**20,
            "decompression_mib_per_second": len(source) / decompression_seconds / 2**20,
        },
        "validation": {"exact_byte_recovery": exact},
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "gpu": torch.cuda.get_device_name(),
            "peak_torch_bytes": torch.cuda.max_memory_allocated(),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    del zip_model, model
    gc.collect()


if __name__ == "__main__":
    main()
