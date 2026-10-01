"""Compare Python and CUDA target-interval encoding in the live LLM pipeline."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import statistics
import sys
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from src.coding.cuda_ac import encode_intervals_cuda
from src.global_mask_compressor import run_global_mask_compression


def _args(cli, backend: str) -> SimpleNamespace:
    return SimpleNamespace(
        mode="compress",
        input_path=str(cli.input),
        text_input=None,
        output_path=None,
        model_name=cli.model,
        model_revision=None,
        tokenizer_name=None,
        tokenizer_revision=None,
        trust_remote_code=False,
        lora_path=None,
        is_mamba=False,
        is_seq2seq=False,
        engine="transformer",
        encoding="AC_TARGET_INTERVAL",
        frequency_quantizer="reference",
        ac_backend=backend,
        ac_threads=None,
        ac_layout="standard",
        reduce_tokens=True,
        first_n_tokens=cli.tokens,
        batch_size=cli.batch_size,
        context_length=cli.context_length,
        retain_tokens=cli.retain_tokens,
        use_kv_cache=True,
        spec_k=None,
    )


def _warm_extension() -> None:
    lows = torch.zeros((1, 1), dtype=torch.int64, device="cuda")
    highs = torch.ones_like(lows)
    totals = torch.full_like(lows, 2)
    counts = torch.ones((1,), dtype=torch.int64, device="cuda")
    encode_intervals_cuda(lows, highs, totals, counts)


def _sample(stats: dict, payload: bytes) -> dict:
    return {
        "compression_seconds": stats["compression_time"],
        "total_seconds": stats["total_compression_time"],
        "inference_seconds": stats["inference_time"],
        "ac_seconds": stats["ac_time"],
        "total_input_symbols_per_sec": stats["throughput_input_symbols_per_sec"],
        "compression_phase_input_symbols_per_sec": (
            stats["input_symbols_count"] / stats["compression_time"]
        ),
        "model_input_tokens_per_sec": stats["throughput_model_input_tokens_per_sec"],
        "interval_transfer_bytes": stats["interval_transfer_bytes"],
        "archive_bytes": len(payload),
        "archive_sha256": hashlib.sha256(payload).hexdigest(),
        "cuda_ac_metrics": stats["cuda_ac_metrics"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=REPO_ROOT / "data/text8")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--retain-tokens", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--output", type=Path)
    cli = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    if not cli.input.is_file():
        parser.error(f"input does not exist: {cli.input}")
    if cli.tokens <= cli.batch_size or cli.repeats < 1:
        parser.error("tokens must exceed batch size and repeats must be positive")

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    torch.manual_seed(2027)
    _warm_extension()
    warm_args = _args(cli, "cuda")
    warm_args.first_n_tokens = min(cli.tokens, max(cli.batch_size * 2, 32))
    run_global_mask_compression(warm_args)
    gc.collect()
    torch.cuda.empty_cache()
    conditions = {"python": [], "cuda": []}
    for repeat in range(cli.repeats):
        order = ("python", "cuda") if repeat % 2 == 0 else ("cuda", "python")
        matched = {}
        for backend in order:
            seeds, payload, bitmap, stats, _ = run_global_mask_compression(
                _args(cli, backend)
            )
            matched[backend] = (seeds, payload, bitmap)
            conditions[backend].append(_sample(stats, payload))
            gc.collect()
            torch.cuda.empty_cache()
        if matched["python"] != matched["cuda"]:
            raise AssertionError("Python and CUDA pipelines produced different archives")

    median_keys = (
        "compression_seconds",
        "inference_seconds",
        "ac_seconds",
        "total_input_symbols_per_sec",
        "compression_phase_input_symbols_per_sec",
    )
    result = {
        "model": cli.model,
        "input": str(cli.input),
        "input_sha256": hashlib.sha256(cli.input.read_bytes()).hexdigest(),
        "tokens": cli.tokens,
        "batch_size": cli.batch_size,
        "context_length": cli.context_length,
        "retain_tokens": cli.retain_tokens,
        "repeats": cli.repeats,
        "warmup": "one untimed CUDA live-pipeline run",
        "gpu": torch.cuda.get_device_name(),
        "torch_version": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "exact_archive_match": True,
        "samples": conditions,
        "medians": {
            backend: {
                key: statistics.median(sample[key] for sample in samples)
                for key in median_keys
            }
            for backend, samples in conditions.items()
        },
        "timing_scope": (
            "compression_seconds excludes model loading but includes live model "
            "inference, interval capture, arithmetic coding, and final host transfer"
        ),
    }
    if cli.output:
        cli.output.parent.mkdir(parents=True, exist_ok=True)
        cli.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
