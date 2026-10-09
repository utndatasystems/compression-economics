#!/usr/bin/env python3
"""Benchmark the official Nacrith compressor on an exact byte prefix."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "artifacts/papers/coding-survey/llm-baselines/nacrith-text8-first-10MiB.json"
)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
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


def _gpu_name() -> str | None:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip().splitlines()[0] if result.returncode == 0 else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nacrith-repo", type=Path, required=True)
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--input", type=Path, default=REPO_ROOT / "data/text8")
    parser.add_argument("--input-bytes", type=int, default=10 * 1024 * 1024)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.input_bytes < 1 or args.workers < 1:
        parser.error("input-bytes and workers must be positive")
    if not args.gguf.is_file():
        parser.error("--gguf must name a GGUF model file")
    upstream = args.nacrith_repo.resolve()
    if not (upstream / "parallel/compressor.py").is_file():
        parser.error("--nacrith-repo is not a Nacrith-GPU checkout")

    source = args.input.read_bytes()[: args.input_bytes]
    if len(source) != args.input_bytes:
        parser.error("input is smaller than the requested prefix")
    text = source.decode("utf-8")
    work_dir = args.work_dir or args.output.with_suffix("")
    work_dir.mkdir(parents=True, exist_ok=True)
    archive_path = work_dir / "nacrith.nc"
    if archive_path.exists():
        parser.error(f"archive already exists: {archive_path}")

    sys.path.insert(0, str(upstream))
    try:
        from parallel import ParallelNeuralCompressor
    finally:
        sys.path.remove(str(upstream))

    load_started = time.perf_counter()
    compressor = ParallelNeuralCompressor(
        n_workers=args.workers,
        gguf_path=str(args.gguf.resolve()),
        verbose=True,
        use_ngram=True,
        use_lzp=False,
        use_adaptive_head=True,
        use_confidence_skip=True,
        ngram_order=4,
        mixer_lr=0.5,
        adaptive_lr=0.001,
        skip_threshold=1.5,
        warmup=100,
        temperature=1.0,
    )
    model_load_seconds = time.perf_counter() - load_started

    started = time.perf_counter()
    archive = compressor.compress(text)
    compression_seconds = time.perf_counter() - started
    archive_path.write_bytes(archive)

    started = time.perf_counter()
    decoded = compressor.decompress(archive)
    decompression_seconds = time.perf_counter() - started
    recovered = decoded if isinstance(decoded, bytes) else decoded.encode("utf-8")
    exact = recovered == source
    if not exact:
        (work_dir / "decoded-mismatch.bin").write_bytes(recovered)
        raise AssertionError("Nacrith did not recover the exact input bytes")

    archive_bytes = len(archive)
    result = {
        "schema_version": 1,
        "codec": "nacrith",
        "implementation": {
            "repository": "https://github.com/robtacconelli/Nacrith-GPU",
            "revision": _git_revision(upstream),
            "format_magic": archive[:4].decode("ascii"),
        },
        "input": {
            "path": str(args.input.resolve()),
            "prefix_bytes": len(source),
            "sha256": _sha256_bytes(source),
        },
        "model": {
            "name": "HuggingFaceTB/SmolLM2-135M",
            "gguf_path": str(args.gguf.resolve()),
            "gguf_bytes": args.gguf.stat().st_size,
            "gguf_sha256": _sha256_file(args.gguf),
            "external_weights_charged": False,
        },
        "configuration": {
            "workers": args.workers,
            "ngram_order": 4,
            "adaptive_lr": 0.001,
            "mixer_lr": 0.5,
            "skip_threshold": 1.5,
            "warmup": 100,
            "temperature": 1.0,
            "lzp": False,
        },
        "archive": {
            "path": str(archive_path.resolve()),
            "bytes": archive_bytes,
            "sha256": _sha256_bytes(archive),
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
            "llama_cpp_python": importlib.metadata.version("llama-cpp-python"),
            "gpu_backend_requested": True,
            "gpu_offload_supported": __import__("llama_cpp").llama_supports_gpu_offload(),
            "gpu": _gpu_name(),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
