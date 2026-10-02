"""Compare Python, CUDA v1, and CUDA v2 on one matched live workload."""

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

from src.coding.cuda_ac import (
    encode_intervals_cuda,
    target_intervals_from_probs_cuda,
)
from src.global_mask_compressor import run_global_mask_compression


_VARIANT_ALIASES = {
    "python": "python",
    "cuda_v1": "cuda_v1",
    "cuda_v2": "cuda_v2",
    # Preserve commands used by the earlier two-condition experiment.
    "cuda_torch": "cuda_v1",
    "cuda_fused": "cuda_v2",
}


def _args(cli, variant: str) -> SimpleNamespace:
    variant = _VARIANT_ALIASES[variant]
    backend = "python" if variant == "python" else "cuda"
    return SimpleNamespace(
        mode="compress",
        input_path=None,
        text_input=cli.benchmark_text,
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
        target_interval_quantizer=(
            "cuda_fused" if variant == "cuda_v2" else "torch"
        ),
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
    probabilities = torch.tensor([[0.25, 0.75]], device="cuda")
    targets = torch.ones((1,), dtype=torch.int64, device="cuda")
    target_intervals_from_probs_cuda(probabilities, targets)


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
        "interval_conversion_seconds": stats["interval_conversion_seconds"],
        "interval_conversion_host_launch_seconds": stats[
            "interval_conversion_host_launch_seconds"
        ],
        "target_interval_quantizer": stats["target_interval_quantizer"],
        "archive_bytes": len(payload),
        "archive_sha256": hashlib.sha256(payload).hexdigest(),
        "cuda_ac_metrics": stats["cuda_ac_metrics"],
    }


def _run_condition(cli, variant: str):
    seeds, payload, bitmap, stats, _ = run_global_mask_compression(
        _args(cli, variant)
    )
    sample = _sample(stats, payload)
    gc.collect()
    torch.cuda.empty_cache()
    return (seeds, payload, bitmap), sample


def _balanced_orders(variants: list[str], repeats: int) -> list[list[str]]:
    """Rotate then reverse conditions to distribute order effects."""
    orders = []
    count = len(variants)
    for repeat in range(repeats):
        offset = repeat % count
        order = variants[offset:] + variants[:offset]
        if (repeat // count) % 2:
            order = list(reversed(order))
        orders.append(order)
    return orders


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=REPO_ROOT / "data/text8")
    parser.add_argument(
        "--input-characters",
        type=int,
        default=1048576,
        help="Read only this deterministic input prefix before tokenization",
    )
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--tokens", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--retain-tokens", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=list(_VARIANT_ALIASES),
        default=["python", "cuda_v1", "cuda_v2"],
        help="Matched conditions; cuda_torch/cuda_fused are legacy aliases",
    )
    parser.add_argument("--output", type=Path)
    cli = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    if not cli.input.is_file():
        parser.error(f"input does not exist: {cli.input}")
    if (
        cli.tokens <= cli.batch_size
        or cli.repeats < 1
        or cli.input_characters < 1
    ):
        parser.error(
            "tokens must exceed batch size; repeats and input-characters must be positive"
        )
    with cli.input.open("r", encoding="utf-8", newline="") as input_file:
        cli.benchmark_text = input_file.read(cli.input_characters)
    if not cli.benchmark_text:
        parser.error("input prefix is empty")

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    torch.manual_seed(2027)
    _warm_extension()
    variants = [_VARIANT_ALIASES[variant] for variant in cli.variants]
    if len(set(variants)) != len(variants):
        parser.error("variants must be unique after resolving legacy aliases")

    # A full-workload warm-up per condition makes initialization behavior
    # visible without mixing it into the reported five-run medians.
    warmups = {}
    warmup_archives = {}
    for variant in variants:
        archive, sample = _run_condition(cli, variant)
        warmup_archives[variant] = archive
        warmups[variant] = sample
    warmup_reference = warmup_archives[variants[0]]
    if any(candidate != warmup_reference for candidate in warmup_archives.values()):
        raise AssertionError("warm-up variants produced different archives")

    conditions = {variant: [] for variant in variants}
    orders = _balanced_orders(variants, cli.repeats)
    for order in orders:
        matched = {}
        for variant in order:
            matched[variant], sample = _run_condition(cli, variant)
            conditions[variant].append(sample)
        reference = matched[variants[0]]
        if any(candidate != reference for candidate in matched.values()):
            raise AssertionError("ablation variants produced different archives")

    median_keys = (
        "compression_seconds",
        "inference_seconds",
        "ac_seconds",
        "interval_conversion_seconds",
        "total_input_symbols_per_sec",
        "compression_phase_input_symbols_per_sec",
    )
    medians = {
        variant: {
            key: statistics.median(sample[key] for sample in samples)
            for key in median_keys
        }
        for variant, samples in conditions.items()
    }
    result = {
        "model": cli.model,
        "input": str(cli.input),
        "input_characters": len(cli.benchmark_text),
        "input_sha256": hashlib.sha256(
            cli.benchmark_text.encode("utf-8")
        ).hexdigest(),
        "tokens": cli.tokens,
        "batch_size": cli.batch_size,
        "context_length": cli.context_length,
        "retain_tokens": cli.retain_tokens,
        "repeats": cli.repeats,
        "variants": variants,
        "variant_definitions": {
            "python": "host Python interval encoder with per-step transfer",
            "cuda_v1": "PyTorch interval construction plus buffered CUDA encoder",
            "cuda_v2": "fused interval construction plus the same CUDA encoder",
        },
        "experimental_factor": "incremental CUDA implementation",
        "warmup_policy": (
            "extension operators, then one full-workload run per condition; "
            "warm-ups are recorded but excluded from medians"
        ),
        "warmups": warmups,
        "timed_condition_orders": orders,
        "gpu": torch.cuda.get_device_name(),
        "torch_version": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "exact_archive_match": True,
        "samples": conditions,
        "medians": medians,
        "timing_scope": (
            "compression_seconds excludes model loading but includes live model "
            "inference, interval capture, arithmetic coding, and final host transfer"
        ),
    }
    if "cuda_v1" in medians and "cuda_v2" in medians:
        baseline = medians["cuda_v1"]
        fused = medians["cuda_v2"]
        result["cuda_v2_vs_cuda_v1"] = {
            "interval_conversion_speedup": (
                baseline["interval_conversion_seconds"]
                / fused["interval_conversion_seconds"]
            ),
            "compression_phase_speedup": (
                baseline["compression_seconds"]
                / fused["compression_seconds"]
            ),
            "compression_phase_throughput_ratio": (
                fused["compression_phase_input_symbols_per_sec"]
                / baseline["compression_phase_input_symbols_per_sec"]
            ),
            "total_throughput_ratio": (
                fused["total_input_symbols_per_sec"]
                / baseline["total_input_symbols_per_sec"]
            ),
        }
    if "python" in medians and "cuda_v1" in medians:
        baseline = medians["python"]
        cuda_v1 = medians["cuda_v1"]
        result["cuda_v1_vs_python"] = {
            "arithmetic_coding_speedup": (
                baseline["ac_seconds"] / cuda_v1["ac_seconds"]
            ),
            "compression_phase_speedup": (
                baseline["compression_seconds"]
                / cuda_v1["compression_seconds"]
            ),
            "compression_phase_throughput_ratio": (
                cuda_v1["compression_phase_input_symbols_per_sec"]
                / baseline["compression_phase_input_symbols_per_sec"]
            ),
            "total_throughput_ratio": (
                cuda_v1["total_input_symbols_per_sec"]
                / baseline["total_input_symbols_per_sec"]
            ),
        }
    if cli.output:
        cli.output.parent.mkdir(parents=True, exist_ok=True)
        cli.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
