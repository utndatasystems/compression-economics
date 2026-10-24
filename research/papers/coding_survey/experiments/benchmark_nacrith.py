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
DEFAULT_MODEL = "HuggingFaceTB/SmolLM2-135M"
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


def _load_local_tokenizer(tokenizer_dir: Path, requested_seed: int | None = None):
    """Load a Nacrith-compatible tokenizer without requiring Transformers."""

    from tokenizers import Tokenizer

    tokenizer_path = tokenizer_dir / "tokenizer.json"
    config_path = tokenizer_dir / "tokenizer_config.json"
    if not tokenizer_path.is_file() or not config_path.is_file():
        raise FileNotFoundError(
            "--tokenizer-dir must contain tokenizer.json and tokenizer_config.json"
        )

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    bos = config.get("bos_token")
    eos = config.get("eos_token")
    if isinstance(bos, dict):
        bos = bos.get("content")
    if isinstance(eos, dict):
        eos = eos.get("content")
    bos_token_id = tokenizer.token_to_id(bos) if bos else None
    eos_token_id = tokenizer.token_to_id(eos) if eos else None
    seed_token_id = requested_seed
    seed_token_source = "explicit"
    if seed_token_id is None and bos_token_id is not None:
        seed_token_id = bos_token_id
        seed_token_source = "bos"
    if seed_token_id is None and eos_token_id is not None:
        seed_token_id = eos_token_id
        seed_token_source = "eos_fallback"
    if seed_token_id is None:
        raise ValueError(
            "tokenizer defines neither BOS nor EOS; pass --seed-token-id explicitly"
        )

    class LocalTokenizer:
        # Nacrith asks for bos_token_id whenever it predicts the first token.
        bos_token_id = seed_token_id

        @staticmethod
        def encode(text: str) -> list[int]:
            return tokenizer.encode(text).ids

        @staticmethod
        def decode(ids: list[int]) -> str:
            return tokenizer.decode(ids)

    return LocalTokenizer(), seed_token_source


def _install_vocab_safe_flag_restore(compressor) -> None:
    """Repair upstream's SmolLM-sized mixer recreation during decode."""

    context_mixer = sys.modules["context_mixer"].ContextMixer
    for _, worker in compressor._workers:
        upstream_apply_flags = worker._apply_flags

        def apply_flags(flags, *, worker=worker, upstream=upstream_apply_flags):
            upstream(flags)
            mixer = worker.mixer
            if mixer is not None and mixer._mix_buf.size != worker.vocab_size:
                worker.mixer = context_mixer(
                    num_models=mixer.num_models,
                    lr=mixer.lr,
                    vocab_size=worker.vocab_size,
                )

        worker._apply_flags = apply_flags


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nacrith-repo", type=Path, required=True)
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument(
        "--tokenizer-dir",
        type=Path,
        help=(
            "tokenizer directory matching the GGUF; required when not using "
            "Nacrith's bundled SmolLM2 tokenizer"
        ),
    )
    parser.add_argument(
        "--seed-token-id",
        type=int,
        help="initial context token (defaults to BOS, then EOS for BOS-less models)",
    )
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
    if args.model_name != DEFAULT_MODEL and args.tokenizer_dir is None:
        parser.error("--tokenizer-dir is required when --model-name is not SmolLM2")
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
        model_name=args.model_name,
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
    _install_vocab_safe_flag_restore(compressor)
    tokenizer_dir = args.tokenizer_dir.resolve() if args.tokenizer_dir else None
    seed_token_source = "bundled_bos"
    if tokenizer_dir is not None:
        try:
            tokenizer, seed_token_source = _load_local_tokenizer(
                tokenizer_dir, args.seed_token_id
            )
        except (FileNotFoundError, ValueError) as error:
            parser.error(str(error))
        for model, _ in compressor._workers:
            model.tokenizer = tokenizer
        compressor._tokenizer = tokenizer
    elif args.seed_token_id is not None:
        for model, _ in compressor._workers:
            model.tokenizer.bos_token_id = args.seed_token_id
        compressor._tokenizer.bos_token_id = args.seed_token_id
        seed_token_source = "explicit"

    token_ids = compressor._tokenizer.encode(text)
    vocabulary_size = compressor._workers[0][0].vocab_size
    seed_token_id = compressor._tokenizer.bos_token_id
    if seed_token_id is None or not 0 <= seed_token_id < vocabulary_size:
        parser.error("initial context token is outside the GGUF vocabulary")
    if token_ids and max(token_ids) >= vocabulary_size:
        parser.error("tokenizer emitted an ID outside the GGUF vocabulary")
    if compressor._tokenizer.decode(token_ids).encode("utf-8") != source:
        parser.error("tokenizer does not round-trip the input bytes")
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
            "adapter_fixes": [
                "recreate decode-time context mixer at the GGUF vocabulary size"
            ],
        },
        "input": {
            "path": str(args.input.resolve()),
            "prefix_bytes": len(source),
            "sha256": _sha256_bytes(source),
        },
        "model": {
            "name": args.model_name,
            "gguf_path": str(args.gguf.resolve()),
            "gguf_bytes": args.gguf.stat().st_size,
            "gguf_sha256": _sha256_file(args.gguf),
            "tokenizer_dir": str(tokenizer_dir) if tokenizer_dir else "bundled",
            "tokenizer_json_sha256": (
                _sha256_file(tokenizer_dir / "tokenizer.json")
                if tokenizer_dir
                else None
            ),
            "vocabulary_size": vocabulary_size,
            "external_weights_charged": False,
        },
        "configuration": {
            "workers": args.workers,
            "seed_token_id": seed_token_id,
            "seed_token_source": seed_token_source,
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
