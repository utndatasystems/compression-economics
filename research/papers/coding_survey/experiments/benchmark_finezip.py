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
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import transformers

from research.papers.coding_survey.experiments.baseline_adapter import (
    build_baseline_result,
    git_revision,
    hugging_face_model_metadata,
    load_text_prefix,
    prepare_work_archive,
    read_framed_archive,
    torch_dtype,
    torch_environment,
    write_framed_archive,
    write_json,
)


DEFAULT_OUTPUT = (
    REPO_ROOT
    / "artifacts/papers/coding-survey/llm-baselines/finezip-text8-first-10MiB.json"
)
MAGIC = b"FZB1"


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

    try:
        source, text = load_text_prefix(args.input, args.input_bytes)
        work_dir, archive = prepare_work_archive(
            args.output, args.work_dir, "finezip.fz"
        )
    except (ValueError, FileExistsError) as error:
        parser.error(str(error))

    upstream = _load_upstream(args.finezip_repo.resolve())
    tokenizer_name = args.tokenizer or args.model
    load_started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name, revision=args.model_revision
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.model_revision, dtype=torch_dtype(args.dtype)
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
    write_framed_archive(archive, MAGIC, archive_metadata, payload)

    extracted, extracted_payload = read_framed_archive(
        archive, MAGIC, codec_label="FineZip"
    )
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

    result = build_baseline_result(
        codec="finezip",
        implementation={
            "repository": "https://github.com/fazalmittu/FineZip",
            "revision": git_revision(args.finezip_repo.resolve()),
            "adapter_archive": "FZB1 (repository-local accounting wrapper)",
            "upstream_class": "finezip/eval_clean.py:ZipModel",
        },
        input_path=args.input,
        source=source,
        model=hugging_face_model_metadata(
            name=args.model,
            requested_revision=args.model_revision,
            tokenizer=tokenizer_name,
            dtype=args.dtype,
        ),
        configuration={
            "batch_size": args.batch_size,
            "context_size": args.context_size,
            "token_count": int(tokens.numel()),
            "rank_storage": "int32 then bzip2",
        },
        archive_path=archive,
        payload_bytes=len(payload),
        model_load_seconds=model_load_seconds,
        compression_seconds=compression_seconds,
        decompression_seconds=decompression_seconds,
        exact_byte_recovery=exact,
        environment=torch_environment(
            torch, transformers_version=transformers.__version__
        ),
    )
    write_json(args.output, result)
    print(json.dumps(result, indent=2))
    del zip_model, model
    gc.collect()


if __name__ == "__main__":
    main()
