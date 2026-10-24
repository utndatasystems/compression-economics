"""Compare incremental KV decoding with encoder-only teacher-forced scoring.

The benchmark preserves the compressor's periodic context policy exactly.  A
prompt grows until ``context_length`` tokens, is reset to its last
``retain_tokens`` tokens, and then grows again.  Teacher forcing combines all
next-token scores between two such resets into one causal forward pass.  It
also verifies that the two schedules produce identical integer target
intervals under the reference arithmetic-coding quantizer.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import gc
import hashlib
import inspect
import json
from pathlib import Path
import statistics
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.coding.target_interval import DEFAULT_TOTAL, target_intervals_from_probs_tensor


@dataclass(frozen=True)
class ScoringWindow:
    """One forward pass and the current-token positions scored by it."""

    input_start: int
    input_stop: int
    score_start: int
    score_stop: int

    @property
    def score_count(self) -> int:
        return self.score_stop - self.score_start


def teacher_forcing_windows(
    token_count: int, context_length: int, retain_tokens: int,
    max_score_count: int | None = None,
) -> list[ScoringWindow]:
    """Group steps that have the same compressor prompt start.

    A score at current-token position ``j`` predicts token ``j + 1``.  The
    returned half-open score range is in source-token coordinates, while the
    input range includes any retained prefix needed by the first score.
    """
    if token_count < 2:
        raise ValueError("at least two tokens are required")
    if not 0 < retain_tokens < context_length:
        raise ValueError("require 0 < retain_tokens < context_length")
    if max_score_count is not None and max_score_count < 1:
        raise ValueError("max_score_count must be positive")

    windows: list[ScoringWindow] = []
    prompt_start = 0
    group_start = 0
    group_prompt_start = 0
    for current in range(token_count - 1):
        if current - prompt_start + 1 >= context_length:
            prompt_start = current - retain_tokens + 1
        if prompt_start != group_prompt_start:
            windows.append(ScoringWindow(
                input_start=group_prompt_start,
                input_stop=current,
                score_start=group_start,
                score_stop=current,
            ))
            group_start = current
            group_prompt_start = prompt_start
    windows.append(ScoringWindow(
        input_start=group_prompt_start,
        input_stop=token_count - 1,
        score_start=group_start,
        score_stop=token_count - 1,
    ))
    if max_score_count is None:
        return windows
    bounded = []
    for window in windows:
        for score_start in range(
            window.score_start, window.score_stop, max_score_count
        ):
            score_stop = min(score_start + max_score_count, window.score_stop)
            bounded.append(ScoringWindow(
                input_start=window.input_start,
                input_stop=score_stop,
                score_start=score_start,
                score_stop=score_stop,
            ))
    return bounded


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _begin_memory_measurement(device: torch.device):
    if device.type != "cuda":
        return None
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    return {
        "baseline_allocated_bytes": torch.cuda.memory_allocated(device),
        "baseline_reserved_bytes": torch.cuda.memory_reserved(device),
    }


def _finish_memory_measurement(device: torch.device, baseline):
    if baseline is None:
        return {
            "peak_allocated_bytes": None,
            "peak_reserved_bytes": None,
            "incremental_peak_allocated_bytes": None,
            "incremental_peak_reserved_bytes": None,
        }
    peak_allocated = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)
    return {
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "incremental_peak_allocated_bytes": (
            peak_allocated - baseline["baseline_allocated_bytes"]
        ),
        "incremental_peak_reserved_bytes": (
            peak_reserved - baseline["baseline_reserved_bytes"]
        ),
    }


def _model_forward(model, input_ids, *, past_key_values=None, use_cache=False, keep=1):
    kwargs = {
        "input_ids": input_ids,
        "past_key_values": past_key_values,
        "use_cache": use_cache,
    }
    if "logits_to_keep" in inspect.signature(model.forward).parameters:
        kwargs["logits_to_keep"] = keep
    output = model(**kwargs)
    logits = output.logits
    if logits.shape[1] != keep:
        logits = logits[:, -keep:, :]
    return logits, output.past_key_values if use_cache else None


def _quantize(logits, targets, alphabet, frequency_total):
    batch, steps, _ = logits.shape
    reduced = logits.index_select(2, alphabet)
    probabilities = torch.softmax(reduced, dim=-1, dtype=torch.float32)
    flat_probs = probabilities.transpose(0, 1).reshape(batch * steps, -1)
    flat_targets = targets.transpose(0, 1).reshape(-1)
    lows, highs, totals, target_probs = target_intervals_from_probs_tensor(
        flat_probs, flat_targets, total=frequency_total
    )
    return (
        torch.stack((lows, highs, totals), dim=1).detach(),
        target_probs.detach(),
    )


def _incremental(
    model, streams, stream_lengths, dense_targets, alphabet, context_length, retain_tokens,
    frequency_total=DEFAULT_TOTAL,
):
    device = streams.device
    stream_length = streams.shape[1]
    prompt_start = 0
    past = None
    intervals = []
    target_probs = []
    model_input_tokens = 0
    calls = 0
    memory_baseline = _begin_memory_measurement(device)
    _sync(device)
    started = time.perf_counter()
    with torch.inference_mode():
        for current in range(stream_length - 1):
            reset = current - prompt_start + 1 >= context_length
            if reset:
                prompt_start = current - retain_tokens + 1
                past = None
            input_ids = (
                streams[:, prompt_start:current + 1]
                if past is None
                else streams[:, current:current + 1]
            )
            logits, past = _model_forward(
                model, input_ids, past_key_values=past, use_cache=True, keep=1
            )
            step_intervals, step_probs = _quantize(
                logits, dense_targets[:, current + 1:current + 2], alphabet,
                frequency_total,
            )
            valid = stream_lengths > current + 1
            intervals.append(step_intervals[valid])
            target_probs.append(step_probs[valid])
            model_input_tokens += input_ids.numel()
            calls += 1
    _sync(device)
    elapsed = time.perf_counter() - started
    result = {
        "seconds": elapsed,
        "intervals": torch.cat(intervals).cpu(),
        "target_probs": torch.cat(target_probs).cpu(),
        "forward_calls": calls,
        "model_input_tokens": model_input_tokens,
    }
    result.update(_finish_memory_measurement(device, memory_baseline))
    return result


def _teacher_forced(
    model, streams, stream_lengths, dense_targets, alphabet, context_length, retain_tokens,
    frequency_total=DEFAULT_TOTAL, teacher_window_size=None,
):
    device = streams.device
    windows = teacher_forcing_windows(
        streams.shape[1], context_length, retain_tokens, teacher_window_size
    )
    intervals = []
    target_probs = []
    model_input_tokens = 0
    memory_baseline = _begin_memory_measurement(device)
    _sync(device)
    started = time.perf_counter()
    past = None
    active_input_start = None
    consumed_stop = None
    with torch.inference_mode():
        for window in windows:
            if window.input_start != active_input_start:
                past = None
                active_input_start = window.input_start
                consumed_stop = window.input_start
            input_ids = streams[:, consumed_stop:window.input_stop]
            logits, past = _model_forward(
                model, input_ids, past_key_values=past, use_cache=True,
                keep=window.score_count,
            )
            consumed_stop = window.input_stop
            targets = dense_targets[:, window.score_start + 1:window.score_stop + 1]
            step_intervals, step_probs = _quantize(
                logits, targets, alphabet, frequency_total
            )
            positions = torch.arange(
                window.score_start + 1, window.score_stop + 1,
                device=device,
            )
            valid = (positions[:, None] < stream_lengths[None, :]).reshape(-1)
            intervals.append(step_intervals[valid])
            target_probs.append(step_probs[valid])
            model_input_tokens += input_ids.numel()
    _sync(device)
    elapsed = time.perf_counter() - started
    result = {
        "seconds": elapsed,
        "intervals": torch.cat(intervals).cpu(),
        "target_probs": torch.cat(target_probs).cpu(),
        "forward_calls": len(windows),
        "model_input_tokens": model_input_tokens,
        "windows": [asdict(window) for window in windows],
    }
    result.update(_finish_memory_measurement(device, memory_baseline))
    return result


def _dtype(name: str, device: torch.device):
    if name == "auto":
        return "auto"
    value = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[name]
    if device.type == "cpu" and value == torch.float16:
        raise ValueError("float16 is not supported for this CPU benchmark")
    return value


def _summary(samples, symbols):
    seconds = [sample["seconds"] for sample in samples]
    median = statistics.median(seconds)
    first = samples[0]
    frequencies = first["intervals"][:, 1] - first["intervals"][:, 0]
    totals = first["intervals"][:, 2]
    quantized_bits = float((-torch.log2(frequencies / totals)).sum().item())
    return {
        "seconds": seconds,
        "median_seconds": median,
        "median_symbols_per_second": symbols / median,
        "forward_calls": first["forward_calls"],
        "model_input_tokens": first["model_input_tokens"],
        "quantized_payload_bits": quantized_bits,
        "quantized_bits_per_symbol": quantized_bits / symbols,
        "peak_allocated_bytes": [sample["peak_allocated_bytes"] for sample in samples],
        "incremental_peak_allocated_bytes": [
            sample["incremental_peak_allocated_bytes"] for sample in samples
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=REPO_ROOT / "data/text8")
    parser.add_argument("--input-characters", type=int, default=1048576)
    parser.add_argument(
        "--input-bytes", type=int,
        help="read this exact byte prefix; it must be valid round-tripping UTF-8",
    )
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--model-revision")
    parser.add_argument(
        "--tokens", type=int, default=2048,
        help="token limit after reading the prefix; use 0 to keep every token",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--retain-tokens", type=int, default=128)
    parser.add_argument(
        "--teacher-window-size", type=int,
        help="maximum known positions scored per prefill call; default is one reset segment",
    )
    parser.add_argument("--frequency-total", type=int, default=DEFAULT_TOTAL)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--dtype", choices=("auto", "float32", "bfloat16", "float16"), default="auto"
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.repeats < 1 or args.batch_size < 1 or args.tokens < 0:
        parser.error("repeats and batch size must be positive; tokens must be nonnegative")
    if not 0 < args.retain_tokens < args.context_length:
        parser.error("require 0 < retain-tokens < context-length")
    if args.teacher_window_size is not None and args.teacher_window_size < 1:
        parser.error("teacher-window-size must be positive")
    if not args.input.is_file():
        parser.error(f"input does not exist: {args.input}")

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        "cpu" if args.device == "auto" else args.device
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    torch.manual_seed(2027)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, revision=args.model_revision, cache_dir=REPO_ROOT / ".cache"
    )
    if args.input_bytes is not None:
        if args.input_bytes < 1:
            parser.error("input-bytes must be positive")
        source_bytes = args.input.read_bytes()[:args.input_bytes]
        if len(source_bytes) != args.input_bytes:
            parser.error("input is smaller than input-bytes")
        text = source_bytes.decode("utf-8")
        if text.encode("utf-8") != source_bytes:
            parser.error("the requested byte prefix does not round-trip through UTF-8")
    else:
        with args.input.open("r", encoding="utf-8", newline="") as input_file:
            text = input_file.read(args.input_characters)
        source_bytes = text.encode("utf-8")
    encode_options = {"add_special_tokens": False}
    if args.tokens:
        encode_options.update(truncation=True, max_length=args.tokens)
    token_ids = tokenizer.encode(text, **encode_options)
    if len(token_ids) <= args.batch_size:
        parser.error("the tokenized input is too short")
    if not args.tokens and tokenizer.decode(token_ids).encode("utf-8") != source_bytes:
        parser.error("tokenization does not reconstruct the exact input byte prefix")
    base_length, extra = divmod(len(token_ids), args.batch_size)
    lengths = [base_length + (index < extra) for index in range(args.batch_size)]
    stream_length = max(lengths)
    vocabulary = sorted(set(token_ids))
    if args.frequency_total <= len(vocabulary):
        parser.error("frequency-total must exceed the reduced alphabet size")
    dense_columns = {token: index for index, token in enumerate(vocabulary)}
    padded_streams = []
    padded_targets = []
    offset = 0
    for length in lengths:
        row = token_ids[offset:offset + length]
        offset += length
        padding = [vocabulary[0]] * (stream_length - length)
        padded_streams.append(row + padding)
        padded_targets.append([dense_columns[token] for token in row] + [0] * len(padding))
    streams = torch.tensor(padded_streams, dtype=torch.long, device=device)
    stream_lengths = torch.tensor(lengths, dtype=torch.long, device=device)
    dense_targets = torch.tensor(padded_targets, dtype=torch.long, device=device)
    alphabet = torch.tensor(vocabulary, dtype=torch.long, device=device)

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.model_revision,
        cache_dir=REPO_ROOT / ".cache",
        dtype=_dtype(args.dtype, device),
    ).to(device).eval()

    def incremental_runner(*runner_args):
        return _incremental(*runner_args)

    def teacher_runner(*runner_args):
        return _teacher_forced(
            *runner_args, teacher_window_size=args.teacher_window_size
        )

    runners = {"incremental_kv": incremental_runner, "teacher_forced": teacher_runner}
    # Untimed warm-up establishes kernels and allocator state.
    warm_tokens = min(stream_length, max(3, args.retain_tokens + 2))
    for runner in runners.values():
        runner(
            model, streams[:, :warm_tokens], stream_lengths.clamp_max(warm_tokens),
            dense_targets[:, :warm_tokens],
            alphabet, args.context_length, args.retain_tokens, args.frequency_total,
        )

    samples = {name: [] for name in runners}
    for repeat in range(args.repeats):
        order = list(runners)
        if repeat % 2:
            order.reverse()
        for name in order:
            samples[name].append(runners[name](
                model, streams, stream_lengths, dense_targets, alphabet,
                args.context_length, args.retain_tokens, args.frequency_total,
            ))

    incremental = samples["incremental_kv"][0]
    teacher = samples["teacher_forced"][0]
    for name, condition_samples in samples.items():
        reference = condition_samples[0]["intervals"]
        if any(not torch.equal(reference, sample["intervals"])
               for sample in condition_samples[1:]):
            raise RuntimeError(f"{name} produced nondeterministic integer intervals")
    equal_rows = (incremental["intervals"] == teacher["intervals"]).all(dim=1)
    mismatch_count = int((~equal_rows).sum().item())
    interval_delta = (incremental["intervals"] - teacher["intervals"]).abs()
    probability_delta = (
        incremental["target_probs"] - teacher["target_probs"]
    ).abs()
    symbols = sum(length - 1 for length in lengths)
    summaries = {
        name: _summary(condition_samples, symbols)
        for name, condition_samples in samples.items()
    }
    result = {
        "experiment": "teacher_forced_prefill_v1",
        "model": args.model,
        "model_revision": args.model_revision,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "dtype": str(model.dtype),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "torch_num_threads": torch.get_num_threads(),
        "input": str(args.input),
        "input_prefix_bytes": len(source_bytes),
        "input_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "token_sha256": hashlib.sha256(
            torch.tensor(token_ids, dtype=torch.int64).numpy().tobytes()
        ).hexdigest(),
        "tokens": len(token_ids),
        "stream_length_min": min(lengths),
        "stream_length_max": stream_length,
        "batch_size": args.batch_size,
        "alphabet_size": len(vocabulary),
        "context_length": args.context_length,
        "retain_tokens": args.retain_tokens,
        "teacher_window_size": args.teacher_window_size,
        "frequency_total": args.frequency_total,
        "coded_symbols": symbols,
        "repeats": args.repeats,
        "conditions": summaries,
        "comparison": {
            "speedup": (
                summaries["incremental_kv"]["median_seconds"] /
                summaries["teacher_forced"]["median_seconds"]
            ),
            "intervals_exactly_equal": mismatch_count == 0,
            "interval_mismatch_count": mismatch_count,
            "max_integer_interval_absolute_error": int(interval_delta.max().item()),
            "max_target_probability_absolute_error": float(probability_delta.max().item()),
            "mean_target_probability_absolute_error": float(probability_delta.mean().item()),
            "integer_interval_decoder_compatible": mismatch_count == 0,
        },
        "teacher_forcing_windows": teacher["windows"],
    }
    encoded = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
