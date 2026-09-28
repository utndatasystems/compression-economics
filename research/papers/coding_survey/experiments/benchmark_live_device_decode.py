"""Repeat host/device decoding of one existing Qwen/text8 MSAC v2 archive.

The input archive and excerpt may come from the E02 CPU pilot or a matched
CUDA run. This script loads one model, warms both decode paths, rotates their
order, verifies recovery on every run, and saves raw samples. It never rewrites
the archive. Cross-device probability mismatch is reported as a failure.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import platform
import statistics
import sys
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import transformers
from pyroaring import BitMap
from transformers import AutoModelForCausalLM, AutoTokenizer

import src.global_mask_compressor as pipeline
from src.prediction import TokenPredictor
from src.utils import load_global_mask_file

DEFAULT_PILOT = REPO_ROOT / "artifacts/papers/cidr-2027/target-interval/Qwen2.5-0.5B-text8-1024-cpu"
DEFAULT_OUTPUT = REPO_ROOT / "artifacts/papers/coding-survey/in-engine-decoding/live-cpu.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-dir", type=Path, default=DEFAULT_PILOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--dtype", choices=["float32", "auto"], default="float32")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--threads", type=int, default=8)
    args_cli = parser.parse_args()
    if args_cli.repeats < 1 or args_cli.threads < 1:
        parser.error("repeats and threads must be positive")
    if args_cli.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    device = torch.device(args_cli.device)
    output_path = args_cli.output or (DEFAULT_OUTPUT if args_cli.device == "cpu" else
                                     DEFAULT_OUTPUT.with_name("live-cuda.json"))
    torch.set_num_threads(args_cli.threads)
    torch.set_num_interop_threads(1)
    archive_path = args_cli.pilot_dir / "ac_target_interval.bin"
    excerpt_path = args_cli.pilot_dir / "excerpt.txt"
    if not archive_path.is_file() or not excerpt_path.is_file():
        parser.error("pilot archive and excerpt are required")
    args = SimpleNamespace(input_path=str(archive_path), encoding="AC_TARGET_INTERVAL",
                           reduce_tokens=True, engine="transformer", lora_path=None)
    args, seeds, payload, bitmap = load_global_mask_file(args)
    name = args.model_name
    tokenizer = AutoTokenizer.from_pretrained(name, cache_dir=str(REPO_ROOT / ".cache"),
                                               local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        name, cache_dir=str(REPO_ROOT / ".cache"), local_files_only=True,
        dtype=torch.float32 if args_cli.dtype == "float32" else "auto",
    ).eval().to(device)
    with excerpt_path.open("r", encoding="utf-8", newline="") as handle:
        expected_text = handle.read()
    expected_tokens = tokenizer.encode(expected_text, add_special_tokens=False)
    if len(expected_tokens) != args.first_n_tokens:
        raise ValueError("source and archive token counts disagree")

    def predictor_factory(run_args, bitmap_data):
        predictor = TokenPredictor.__new__(TokenPredictor)
        predictor.args = run_args
        predictor.tokenizer = tokenizer
        predictor.model = model
        predictor.device = device
        predictor.tokens_list = list(BitMap.deserialize(bitmap_data))
        predictor.index_tensor = torch.tensor(predictor.tokens_list, dtype=torch.long, device=device)
        predictor.reduce_tokens = run_args.reduce_tokens
        predictor.reset_kv_cache()
        return predictor

    pipeline._make_token_predictor = predictor_factory
    with torch.inference_mode():
        model(torch.tensor([expected_tokens[:8]] * args.batch_size, device=device), use_cache=True)
    results = {"host": [], "device": []}
    runs = [("host", 0), ("device", 0)]
    for repeat in range(1, args_cli.repeats + 1):
        runs.extend((("host", repeat), ("device", repeat)) if repeat % 2 else
                    (("device", repeat), ("host", repeat)))
    for backend, repeat in runs:
        args.ac_decode_backend = backend
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        with contextlib.redirect_stdout(io.StringIO()), torch.inference_mode():
            decoded, text, stats = pipeline.run_global_mask_decompression(args, seeds, payload, bitmap)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        if decoded != expected_tokens or text != expected_text:
            raise AssertionError(f"{backend} failed exact token/text recovery")
        if repeat:
            sample = {"repeat": repeat, "seconds": stats["decompression_time_sec"],
                      "inference_seconds": stats["inference_time"],
                      "arithmetic_seconds": stats["ac_time"],
                      "softmax_seconds": stats["softmax_time"],
                      "data_copy_seconds": stats["data_copy_time"],
                      "decoded_token_transfer_bytes": stats["decoded_token_transfer_bytes"]}
            results[backend].append(sample)
            print(f"{backend} repeat {repeat}: {sample['seconds']:.3f} s", flush=True)
    output = {
        "source": str(args_cli.pilot_dir), "archive_sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
        "excerpt_sha256": hashlib.sha256(excerpt_path.read_bytes()).hexdigest(),
        "archive_bytes": archive_path.stat().st_size, "source_bytes": excerpt_path.stat().st_size,
        "model": name, "model_revision": getattr(model.config, "_commit_hash", None),
        "dtype": str(model.dtype), "device": args_cli.device,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch_cuda_runtime": torch.version.cuda, "cpu": next((line.split(":", 1)[1].strip()
            for line in Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("model name")),
            platform.processor()), "torch_threads": args_cli.threads,
        "torch_version": torch.__version__, "transformers_version": transformers.__version__,
        "numpy_version": np.__version__, "tokens": len(expected_tokens),
        "batch_size": args.batch_size, "masked_alphabet": len(BitMap.deserialize(bitmap)),
        "context_length": args.context_length, "retain_tokens": args.retain_tokens,
        "kv_cache": args.use_kv_cache, "warmups_per_backend": 1,
        "timing_scope": "decompression function, excludes model loading and archive file I/O; includes prediction, coder, token copies and detokenization",
        "samples": results,
        "medians": {backend: {key: statistics.median(row[key] for row in samples)
                             for key in ("seconds", "inference_seconds", "arithmetic_seconds")}
                    for backend, samples in results.items()},
        "exact_recovery": True,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output["medians"], indent=2))
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
