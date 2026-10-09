"""Benchmark a real text8 prefix with matched CUDA encoding and decoding paths."""

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import io
import json
import os
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
from src.coding.cuda_ac import (
    encode_intervals_cuda,
    target_intervals_from_probs_cuda,
    target_intervals_from_probs_cuda_v3,
    target_intervals_from_probs_cuda_v4,
)
from src.prediction import TokenPredictor


ENCODERS = (
    "python_full_logits",
    "python",
    "cuda_v1",
    "cuda_v2",
    "cuda_v3",
    "cuda_v4",
)
DECODERS = ("host", "device", "cuda")


def _balanced_orders(conditions: tuple[str, ...], repeats: int) -> list[list[str]]:
    orders = []
    for repeat in range(repeats):
        offset = repeat % len(conditions)
        order = list(conditions[offset:] + conditions[:offset])
        if (repeat // len(conditions)) % 2:
            order.reverse()
        orders.append(order)
    return orders


def _compression_args(cli: argparse.Namespace, source: str, token_count: int, variant: str):
    ac_backend = "python" if variant in {"python_full_logits", "python"} else "cuda"
    return SimpleNamespace(
        mode="compress",
        input_path=None,
        text_input=source,
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
        encoding=("AC_MULTISTREAM" if variant == "python_full_logits" else "AC_TARGET_INTERVAL"),
        frequency_quantizer="reference",
        ac_backend=ac_backend,
        target_interval_quantizer={
            "cuda_v2": "cuda_fused",
            "cuda_v3": "cuda_v3",
            "cuda_v4": "cuda_v4",
        }.get(variant, "torch"),
        ac_threads=None,
        ac_layout="standard",
        reduce_tokens=True,
        first_n_tokens=token_count,
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
    target_intervals_from_probs_cuda_v3(probabilities, targets)
    target_intervals_from_probs_cuda_v4(probabilities, targets)


def _encoding_sample(stats: dict, seeds: list[int], payload: bytes, bitmap: bytes) -> dict:
    source_bytes = stats["original_size_bytes"]
    phase_seconds = stats["compression_time"]
    whole_seconds = stats["total_compression_time"]
    representation_bytes = len(payload) + len(bitmap) + 4 * len(seeds)
    return {
        "seconds": phase_seconds,
        "whole_call_seconds": whole_seconds,
        "source_mib_per_second": source_bytes / phase_seconds / (1024 * 1024),
        "whole_call_source_mib_per_second": source_bytes / whole_seconds / (1024 * 1024),
        "symbols_per_second": stats["input_symbols_count"] / phase_seconds,
        "inference_seconds": stats["inference_time"],
        "interval_seconds": stats["interval_conversion_seconds"] or 0.0,
        "data_copy_seconds": stats["data_copy_time"],
        "softmax_seconds": stats["softmax_time"],
        "arithmetic_coding_seconds": stats["ac_time"],
        "quantize_seconds": stats["msac_quantize_seconds"],
        "range_encode_seconds": stats["msac_range_encode_seconds"],
        "payload_bytes": len(payload),
        "bitmap_bytes": len(bitmap),
        "seed_bytes": 4 * len(seeds),
        "representation_bytes": representation_bytes,
        "compression_ratio": source_bytes / representation_bytes,
        "archive_sha256": hashlib.sha256(payload).hexdigest(),
        "target_interval_quantizer": stats["target_interval_quantizer"],
        "cuda_ac_metrics": stats["cuda_ac_metrics"],
        "stage_timing": stats.get("stage_timing"),
    }


def _decode_sample(stats: dict, source_bytes: int) -> dict:
    seconds = stats["decompression_time_sec"]
    return {
        "seconds": seconds,
        "source_mib_per_second": source_bytes / seconds / (1024 * 1024),
        "symbols_per_second": stats["input_symbols_count"] / seconds,
        "inference_seconds": stats["inference_time"],
        "arithmetic_seconds": stats["ac_time"],
        "detokenize_seconds": stats["detokenize_time"],
        "decoded_token_transfer_bytes": stats["decoded_token_transfer_bytes"],
        "cuda_decode_metrics": stats.get("cuda_decode_metrics"),
        "stage_timing": stats.get("stage_timing"),
    }


def _medians(samples: dict[str, list[dict]], keys: tuple[str, ...]) -> dict[str, dict]:
    return {
        condition: {
            key: statistics.median(sample[key] for sample in condition_samples)
            for key in keys
        }
        for condition, condition_samples in samples.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=REPO_ROOT / "data/text8")
    parser.add_argument("--input-bytes", type=int, default=10 * 1024 * 1024)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--batch-size", type=int, default=16384)
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--retain-tokens", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--skip-workload-warmups",
        action="store_true",
        help="Skip the full-input encoder and decoder warm-up passes",
    )
    parser.add_argument(
        "--encoders",
        nargs="+",
        choices=ENCODERS,
        default=list(ENCODERS),
        help="Encoding paths to benchmark",
    )
    parser.add_argument(
        "--decoders",
        nargs="+",
        choices=DECODERS,
        default=list(DECODERS),
        help="Decoding paths to benchmark",
    )
    parser.add_argument(
        "--decoder-archive-encoder",
        choices=ENCODERS,
        help="Use this encoder's archive for all decoder comparisons",
    )
    parser.add_argument("--output", type=Path, required=True)
    cli = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    if not cli.input.is_file() or cli.input_bytes < 1 or cli.repeats < 1:
        parser.error("input must exist; input-bytes and repeats must be positive")
    if cli.retain_tokens >= cli.context_length:
        parser.error("retain-tokens must be smaller than context-length")
    if len(set(cli.encoders)) != len(cli.encoders) or len(set(cli.decoders)) != len(cli.decoders):
        parser.error("encoder and decoder selections must not contain duplicates")
    if cli.decoder_archive_encoder and cli.decoder_archive_encoder not in cli.encoders:
        parser.error("decoder-archive-encoder must also be selected in --encoders")

    encoders = tuple(cli.encoders)
    decoders = tuple(cli.decoders)

    source_raw = cli.input.read_bytes()[: cli.input_bytes]
    if len(source_raw) != cli.input_bytes:
        parser.error("input is smaller than the requested prefix")
    source = source_raw.decode("utf-8")
    if source.encode("utf-8") != source_raw:
        raise ValueError("input prefix does not round-trip through UTF-8")

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    torch.manual_seed(2027)
    tokenizer = AutoTokenizer.from_pretrained(
        cli.model, cache_dir=str(REPO_ROOT / ".cache"), local_files_only=True
    )
    expected_tokens = tokenizer.encode(source, add_special_tokens=False)
    if tokenizer.decode(expected_tokens) != source:
        raise ValueError("tokenizer does not reconstruct the requested source bytes")
    if cli.batch_size > len(expected_tokens):
        parser.error("batch-size exceeds the token count")

    model = AutoModelForCausalLM.from_pretrained(
        cli.model,
        cache_dir=str(REPO_ROOT / ".cache"),
        local_files_only=True,
        dtype="auto",
    ).eval().to("cuda")
    device = torch.device("cuda")

    def predictor_factory(run_args, bitmap_data):
        predictor = TokenPredictor.__new__(TokenPredictor)
        predictor.args = run_args
        predictor.tokenizer = tokenizer
        predictor.model = model
        predictor.device = device
        predictor.tokens_list = list(BitMap.deserialize(bitmap_data))
        predictor.index_tensor = torch.tensor(
            predictor.tokens_list, dtype=torch.long, device=device
        )
        predictor.reduce_tokens = run_args.reduce_tokens
        predictor.base_params = sum(parameter.numel() for parameter in model.parameters())
        predictor.adapter_params = 0
        predictor.base_size_mb = sum(
            parameter.numel() * parameter.element_size() for parameter in model.parameters()
        ) / (1024 * 1024)
        predictor.adapter_size_mb = 0.0
        predictor.reset_kv_cache()
        return predictor

    pipeline._make_token_predictor = predictor_factory
    _warm_extension()

    def run_encoder(variant: str):
        args = _compression_args(cli, source, len(expected_tokens), variant)
        torch.cuda.synchronize(device)
        with contextlib.redirect_stdout(io.StringIO()), torch.inference_mode():
            seeds, payload, bitmap, stats, args = pipeline.run_global_mask_compression(args)
        torch.cuda.synchronize(device)
        if stats["original_size_bytes"] != len(source_raw):
            raise AssertionError("compression source-byte count changed")
        sample = _encoding_sample(stats, seeds, payload, bitmap)
        gc.collect()
        torch.cuda.empty_cache()
        return (seeds, payload, bitmap, args), sample

    encoding_warmups = {}
    warm_archives = {}
    if cli.skip_workload_warmups:
        print("Skipping full-workload encoding warm-ups.", flush=True)
    else:
        print("Running encoding warm-ups...", flush=True)
        for variant in encoders:
            warm_archives[variant], encoding_warmups[variant] = run_encoder(variant)
    matched_encoders = tuple(
        variant
        for variant in encoders
        if variant not in {"python_full_logits", "cuda_v4"}
    )
    reference = (
        warm_archives[matched_encoders[0]][:3]
        if matched_encoders and warm_archives
        else None
    )
    if warm_archives and any(
        warm_archives[variant][:3] != reference for variant in matched_encoders[1:]
    ):
        raise AssertionError("encoding warm-ups produced different archives")

    encoding_samples = {variant: [] for variant in encoders}
    encoding_orders = _balanced_orders(encoders, cli.repeats)
    optimized_archive = (
        warm_archives.get(
            "cuda_v4",
            warm_archives.get(
                "cuda_v3", warm_archives.get("cuda_v2", warm_archives[encoders[-1]])
            ),
        )
        if warm_archives
        else None
    )
    for repeat, order in enumerate(encoding_orders, start=1):
        matched = {}
        timed_archives = {}
        for variant in order:
            archive, sample = run_encoder(variant)
            timed_archives[variant] = archive
            matched[variant] = archive[:3]
            encoding_samples[variant].append(sample)
            print(
                f"encode {variant} repeat {repeat}: {sample['seconds']:.3f} s",
                flush=True,
            )
        optimized_archive = timed_archives.get(
            "cuda_v4",
            timed_archives.get(
                "cuda_v3",
                timed_archives.get(
                    "cuda_v2", timed_archives.get("cuda_v1", optimized_archive)
                ),
            ),
        )
        if matched_encoders and any(
            matched[variant] != matched[matched_encoders[0]]
            for variant in matched_encoders[1:]
        ):
            raise AssertionError("timed encoders produced different archives")

    if cli.decoder_archive_encoder:
        optimized_archive = timed_archives[cli.decoder_archive_encoder]

    seeds, payload, bitmap, decode_args = optimized_archive
    decode_args.mode = "decompress"

    def run_decoder(backend: str):
        decode_args.ac_decode_backend = backend
        torch.cuda.synchronize(device)
        with contextlib.redirect_stdout(io.StringIO()), torch.inference_mode():
            decoded, text, stats = pipeline.run_global_mask_decompression(
                decode_args, seeds, payload, bitmap
            )
        torch.cuda.synchronize(device)
        if decoded != expected_tokens or text.encode("utf-8") != source_raw:
            raise AssertionError(f"{backend} decoder failed exact recovery")
        sample = _decode_sample(stats, len(source_raw))
        gc.collect()
        torch.cuda.empty_cache()
        return sample

    decoding_warmups = {}
    if cli.skip_workload_warmups:
        print("Skipping full-workload decoding warm-ups.", flush=True)
    else:
        print("Running decoding warm-ups...", flush=True)
        for backend in decoders:
            decoding_warmups[backend] = run_decoder(backend)

    decoding_samples = {backend: [] for backend in decoders}
    decoding_orders = _balanced_orders(decoders, cli.repeats)
    for repeat, order in enumerate(decoding_orders, start=1):
        for backend in order:
            sample = run_decoder(backend)
            decoding_samples[backend].append(sample)
            print(
                f"decode {backend} repeat {repeat}: {sample['seconds']:.3f} s",
                flush=True,
            )

    encoding_medians = _medians(
        encoding_samples,
        (
            "seconds",
            "whole_call_seconds",
            "source_mib_per_second",
            "whole_call_source_mib_per_second",
            "symbols_per_second",
            "inference_seconds",
            "interval_seconds",
            "data_copy_seconds",
            "softmax_seconds",
            "arithmetic_coding_seconds",
            "quantize_seconds",
            "range_encode_seconds",
        ),
    )
    decoding_medians = _medians(
        decoding_samples,
        (
            "seconds",
            "source_mib_per_second",
            "symbols_per_second",
            "inference_seconds",
            "arithmetic_seconds",
            "detokenize_seconds",
        ),
    )
    representation_bytes = len(payload) + len(bitmap) + 4 * len(seeds)
    output = {
        "input": str(cli.input),
        "input_prefix_bytes": len(source_raw),
        "input_sha256": hashlib.sha256(source_raw).hexdigest(),
        "tokens": len(expected_tokens),
        "distinct_tokens": len(set(expected_tokens)),
        "model": cli.model,
        "model_revision": getattr(model.config, "_commit_hash", None),
        "dtype": str(model.dtype),
        "batch_size": cli.batch_size,
        "context_length": cli.context_length,
        "retain_tokens": cli.retain_tokens,
        "repeats": cli.repeats,
        "full_workload_warmups_skipped": cli.skip_workload_warmups,
        "gpu": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "transformers_version": transformers.__version__,
        "numpy_version": np.__version__,
        "cpu": next(
            (
                line.split(":", 1)[1].strip()
                for line in Path("/proc/cpuinfo").read_text().splitlines()
                if line.startswith("model name")
            ),
            platform.processor(),
        ),
        "encoding": {
            "conditions": list(encoders),
            "definitions": {
                "python_full_logits": "pre-target-capture host Python path transferring and quantizing the complete reduced-vocabulary probability matrix",
                "python": "pre-CUDA host Python target-interval and arithmetic encoder",
                "cuda_v1": "PyTorch target-interval construction plus buffered CUDA arithmetic encoder",
                "cuda_v2": "fused target-interval construction plus the same CUDA arithmetic encoder",
                "cuda_v3": "compact fused target-interval construction without a full float64 probability tensor, plus the same CUDA arithmetic encoder",
                "cuda_v4": "custom fused exact fixed-total normalization, residual selection, target-only bounds, and persistent arithmetic coding overlapped with subsequent inference",
            },
            "full_logits_baseline_provenance": "preserved AC_MULTISTREAM path from revision 5a91a3f (the parent of c1cc347, which introduced target-interval capture)",
            "python_baseline_provenance": "preserved pre-CUDA target-interval path from revision 009936d (immediately before d1dd294 added the first CUDA interval encoder)",
            "timing_scope": "live inference, interval construction, arithmetic coding, and final device-to-host transfer; excludes tokenization and model loading",
            "warmups": encoding_warmups,
            "orders": encoding_orders,
            "samples": encoding_samples,
            "medians": encoding_medians,
            "cuda_v2_vs_cuda_v1": (
                {
                    "speedup": encoding_medians["cuda_v1"]["seconds"]
                    / encoding_medians["cuda_v2"]["seconds"],
                    "throughput_ratio": encoding_medians["cuda_v2"]["source_mib_per_second"]
                    / encoding_medians["cuda_v1"]["source_mib_per_second"],
                }
                if {"cuda_v1", "cuda_v2"}.issubset(encoding_medians)
                else None
            ),
            "cuda_v2_vs_python": (
                {
                    "speedup": encoding_medians["python"]["seconds"]
                    / encoding_medians["cuda_v2"]["seconds"],
                    "throughput_ratio": encoding_medians["cuda_v2"]["source_mib_per_second"]
                    / encoding_medians["python"]["source_mib_per_second"],
                }
                if {"python", "cuda_v2"}.issubset(encoding_medians)
                else None
            ),
            "cuda_v2_vs_python_full_logits": (
                {
                    "speedup": encoding_medians["python_full_logits"]["seconds"]
                    / encoding_medians["cuda_v2"]["seconds"],
                    "throughput_ratio": encoding_medians["cuda_v2"]["source_mib_per_second"]
                    / encoding_medians["python_full_logits"]["source_mib_per_second"],
                }
                if {"python_full_logits", "cuda_v2"}.issubset(encoding_medians)
                else None
            ),
            "cuda_v3_vs_cuda_v2": (
                {
                    "speedup": encoding_medians["cuda_v2"]["seconds"]
                    / encoding_medians["cuda_v3"]["seconds"],
                    "throughput_ratio": encoding_medians["cuda_v3"]["source_mib_per_second"]
                    / encoding_medians["cuda_v2"]["source_mib_per_second"],
                }
                if {"cuda_v2", "cuda_v3"}.issubset(encoding_medians)
                else None
            ),
            "cuda_v4_vs_cuda_v3": (
                {
                    "speedup": encoding_medians["cuda_v3"]["seconds"]
                    / encoding_medians["cuda_v4"]["seconds"],
                    "throughput_ratio": encoding_medians["cuda_v4"]["source_mib_per_second"]
                    / encoding_medians["cuda_v3"]["source_mib_per_second"],
                }
                if {"cuda_v3", "cuda_v4"}.issubset(encoding_medians)
                else None
            ),
            "exact_archive_match": True,
            "archive_match_scope": list(matched_encoders),
            "note": "The full-logits MSAC-v1 baseline and CUDA-v4 MSAC-v3 use fixed-total normalization; CUDA v1-v3 use the variable-total MSAC-v2 floor-count contract. Cross-version conditions are validated by exact round-trip rather than whole-archive equality.",
        },
        "decoding": {
            "conditions": list(decoders),
            "archive_encoder": cli.decoder_archive_encoder or "best available CUDA encoder",
            "definitions": {
                "host": "pre-CUDA host Python inverse-CDF and arithmetic decoder",
                "device": "PyTorch device-state and inverse-CDF baseline",
                "cuda": "PyTorch CDF construction plus fused CUDA state update",
            },
            "note": "CUDA v2-v4 are encoder quantizer variants; the CUDA decoder performs batched inverse-CDF lookup using the quantizer declared by the MSAC archive",
            "timing_scope": "complete decompression function including inference, arithmetic decoding, token copies, and detokenization; excludes model loading and archive I/O",
            "warmups": decoding_warmups,
            "orders": decoding_orders,
            "samples": decoding_samples,
            "medians": decoding_medians,
            "cuda_vs_device": (
                {
                    "speedup": decoding_medians["device"]["seconds"]
                    / decoding_medians["cuda"]["seconds"],
                    "throughput_ratio": decoding_medians["cuda"]["source_mib_per_second"]
                    / decoding_medians["device"]["source_mib_per_second"],
                }
                if {"device", "cuda"}.issubset(decoding_medians)
                else None
            ),
            "cuda_vs_host": (
                {
                    "speedup": decoding_medians["host"]["seconds"]
                    / decoding_medians["cuda"]["seconds"],
                    "throughput_ratio": decoding_medians["cuda"]["source_mib_per_second"]
                    / decoding_medians["host"]["source_mib_per_second"],
                }
                if {"host", "cuda"}.issubset(decoding_medians)
                else None
            ),
            "exact_recovery": True,
        },
        "representation": {
            "payload_bytes": len(payload),
            "bitmap_bytes": len(bitmap),
            "seed_bytes": 4 * len(seeds),
            "charged_bytes": representation_bytes,
            "compression_ratio": len(source_raw) / representation_bytes,
            "payload_sha256": hashlib.sha256(payload).hexdigest(),
        },
    }
    cli.output.parent.mkdir(parents=True, exist_ok=True)
    cli.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "encoding_medians": encoding_medians,
        "decoding_medians": decoding_medians,
        "representation": output["representation"],
    }, indent=2))
    print(f"Saved {cli.output}")


if __name__ == "__main__":
    main()
