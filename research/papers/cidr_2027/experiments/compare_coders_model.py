#!/usr/bin/env python3
"""Compare coder configurations on one frozen GPT/Qwen text8 probability trace."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from pyroaring import BitMap
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.coder_benchmark import ProbabilityTrace, benchmark_coder, save_probability_trace


def configurations(alphabet_size: int) -> list[tuple[str, dict]]:
    """Keep all runs on one trace while varying coder and coding settings."""
    low_total = 1 << max(12, (alphabet_size + 1).bit_length())
    return [
        ("AC_total_low", dict(coder="AC", total=low_total)),
        ("AC_total_262144", dict(coder="AC", total=262144)),
        ("MSAC_python_1", dict(coder="AC_MULTISTREAM", total=262144, ac_streams=1)),
        ("MSAC_python_4", dict(coder="AC_MULTISTREAM", total=262144, ac_streams=4)),
        ("MSAC_numba_1", dict(coder="AC_MULTISTREAM", total=262144, ac_streams=4,
                              ac_backend="numba_parallel", ac_threads=1)),
        ("MSAC_numba_4", dict(coder="AC_MULTISTREAM", total=262144, ac_streams=4,
                              ac_backend="numba_parallel", ac_threads=4)),
        ("ANS_low_block64_lanes1", dict(coder="ANS", total=low_total,
                                        ans_block_size=64, ans_lanes=1)),
        ("ANS_block64_lanes1", dict(coder="ANS", total=262144,
                                    ans_block_size=64, ans_lanes=1)),
        ("ANS_block64_lanes4", dict(coder="ANS", total=262144,
                                    ans_block_size=64, ans_lanes=4)),
        ("ANS_block256_lanes4", dict(coder="ANS", total=262144,
                                     ans_block_size=256, ans_lanes=4)),
        ("PMATIC_delta_0.001", dict(coder="PMATIC", total=262144,
                                    pmatic_delta=0.001)),
        ("HUFFMAN_RANK", dict(coder="HUFFMAN_RANK", total=262144)),
        ("BITPACKED_RANK", dict(coder="BITPACKED_RANK", total=262144)),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--input", type=Path, default=Path("data/text8"))
    parser.add_argument("--target-tokens", type=int, default=512)
    parser.add_argument("--source-bytes", type=int, default=8192)
    parser.add_argument("--torch-threads", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.target_tokens < 4:
        parser.error("target-tokens must be at least four for the MSAC sweep")
    if min(args.source_bytes, args.torch_threads, args.repeats) < 1:
        parser.error("source-bytes, torch-threads, and repeats must be positive")
    if args.output_dir is None:
        slug = args.model.replace("/", "-")
        args.output_dir = Path(
            f"artifacts/papers/cidr-2027/coder-comparison/{slug}-text8-{args.target_tokens}-mask"
        )

    torch.set_num_threads(args.torch_threads)
    torch.manual_seed(2027)
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, cache_dir=".cache", local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, cache_dir=".cache", local_files_only=True,
        dtype=torch.float32).eval()
    source = args.input.read_bytes()[:args.source_bytes]
    ids = tokenizer.encode(source.decode("utf-8"), add_special_tokens=False,
                           truncation=True, max_length=args.target_tokens + 1)
    if len(ids) != args.target_tokens + 1:
        raise ValueError("source prefix contains too few tokens")
    context_limit = getattr(model.config, "max_position_embeddings", None)
    if context_limit is not None and args.target_tokens > context_limit:
        raise ValueError("target token count exceeds model context length")
    input_ids = torch.tensor([ids[:-1]], dtype=torch.long)
    targets = torch.tensor(ids[1:], dtype=torch.long)
    with torch.inference_mode():
        logits = model(input_ids).logits[0]
        target_logits = logits.gather(1, targets[:, None]).squeeze(1)
        full_entropy_bits = float(((torch.logsumexp(logits, dim=-1) - target_logits)
                                   / math.log(2)).sum().item())
        mask_ids = sorted(set(ids))
        masked_logits = logits[:, mask_ids]
        probabilities_fp32 = torch.softmax(masked_logits, dim=-1)
    model_seconds = time.perf_counter() - started
    probabilities = probabilities_fp32.numpy().astype(np.float64)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    symbol_lookup = {token_id: index for index, token_id in enumerate(mask_ids)}
    symbols = np.asarray([symbol_lookup[token_id] for token_id in ids[1:]], dtype=np.int64)
    trace = ProbabilityTrace(probabilities, symbols)
    mask_bytes = BitMap(mask_ids).serialize()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = args.output_dir / "probability-trace.npz"
    save_probability_trace(trace_path, trace)
    (args.output_dir / "token-mask.roaring").write_bytes(mask_bytes)
    results = []
    for name, settings in configurations(trace.alphabet_size):
        print(f"Running {name} on {trace.symbol_count} targets...", flush=True)
        samples = [
            benchmark_coder(
                trace, profile_memory=False, pmatic_safe_scenario=False, **settings)
            for _ in range(args.repeats)
        ]
        if len({(item["archive_bytes"], item["payload_bits"]) for item in samples}) != 1:
            raise AssertionError(f"{name} changed its encoded size between runs")
        row = samples[0]
        row["encode_seconds_samples"] = [item["encode_seconds"] for item in samples]
        row["decode_seconds_samples"] = [item["decode_seconds"] for item in samples]
        row["encode_seconds"] = statistics.median(row["encode_seconds_samples"])
        row["decode_seconds"] = statistics.median(row["decode_seconds_samples"])
        row["encode_symbols_per_second"] = trace.symbol_count / row["encode_seconds"]
        row["decode_symbols_per_second"] = trace.symbol_count / row["decode_seconds"]
        if row["range_encode_seconds"] is not None:
            row["quantize_seconds"] = statistics.median(
                item["quantize_seconds"] for item in samples)
            row["range_encode_seconds"] = statistics.median(
                item["range_encode_seconds"] for item in samples)
            row["range_encode_symbols_per_second"] = (
                trace.symbol_count / row["range_encode_seconds"])
        row["configuration"] = name
        row["mask_bytes_excluded_from_archive"] = len(mask_bytes)
        row["coder_archive_plus_mask_bytes"] = row["archive_bytes"] + len(mask_bytes)
        results.append(row)
        (args.output_dir / f"{name}.json").write_text(
            json.dumps(row, indent=2, sort_keys=True) + "\n")

    model_cache = Path(".cache") / ("models--" + args.model.replace("/", "--"))
    ref = model_cache / "refs" / "main"
    report = {
        "source": str(args.input),
        "source_prefix_bytes": args.source_bytes,
        "model": args.model,
        "model_revision": ref.read_text().strip() if ref.exists() else None,
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "model_seconds_including_load_tokenize_and_inference": model_seconds,
        "device": "cpu",
        "weight_quantization": "none",
        "weight_dtype": str(next(model.parameters()).dtype),
        "logit_dtype": str(logits.dtype),
        "softmax_dtype": str(probabilities_fp32.dtype),
        "persisted_trace_dtype": str(trace.probabilities.dtype),
        "trace_normalization": "float32 masked softmax, widened and renormalized in float64",
        "frequency_quantizer": "build_cumul: floor(p * (total - alphabet)) + 1, then adjust to exact total",
        "frequency_totals": sorted({row["parameters"]["frequency_total"] for row in results}),
        "ac_state_bits": 32,
        "vocabulary_mode": "observed token IDs in the selected source prefix",
        "original_vocabulary_size": model.config.vocab_size,
        "masked_alphabet_size": trace.alphabet_size,
        "mask_format": "Roaring bitmap of original token IDs, sorted order defines trace symbols",
        "mask_bytes_excluded_from_coder_archives": len(mask_bytes),
        "initial_context_token_excluded_from_coder_archives": int(ids[0]),
        "full_vocabulary_cross_entropy_bits": full_entropy_bits,
        "masked_cross_entropy_bits": float(np.sum(-np.log2(
            trace.probabilities[np.arange(trace.symbol_count), trace.symbols]))),
        "target_tokens": trace.symbol_count,
        "timing_repetitions": args.repeats,
        "trace_sha256": trace.sha256,
        "trace_file": str(trace_path),
        "timing_note": "median wall time across repeated exact round trips; excludes model inference and memory tracing",
        "numerical_perturbation_scenarios": "omitted for matched-distribution coder comparison",
        "results": results,
    }
    (args.output_dir / "comparison.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"Saved {len(results)} verified configurations to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
