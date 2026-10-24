"""Full-corpus speculative-decompression feasibility sweep.

This runner strengthens the small 1,024-token pilot in three ways:

* it replays complete byte prefixes and checks every source position;
* it maintains causal online n-gram state incrementally, so long streams are
  practical and no future token can enter the drafter; and
* on CUDA it measures real Qwen verifier calls at several corpus positions.

The source tokens are still the arithmetic decoder's oracle verdict.  Thus the
schedule results are exact control-flow measurements and the verifier timings
are live measurements, but their combination is a projection rather than an
integrated speculative arithmetic-decoder timing.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gc
import hashlib
import inspect
import json
from pathlib import Path
import platform
import statistics
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_OUTPUT = (
    REPO_ROOT
    / "artifacts/papers/coding-survey/speculative-decompression/full-sweep.json"
)


def split_streams(tokens: list[int], stream_count: int) -> list[list[int]]:
    base, extra = divmod(len(tokens), stream_count)
    result = []
    offset = 0
    for index in range(stream_count):
        length = base + (index < extra)
        result.append(tokens[offset : offset + length])
        offset += length
    if offset != len(tokens):
        raise AssertionError("stream split did not consume every token")
    return result


class OnlineNgram:
    """Exact causal n-gram counts with stable frequency/token-id tie breaking."""

    def __init__(self, max_order: int):
        self.max_order = max_order
        self.tables = [None] + [defaultdict(Counter) for _ in range(max_order)]
        self.unigram: Counter[int] = Counter()

    def observe(self, history: list[int], new_tokens: list[int]) -> None:
        """Reveal tokens in order, adding no transition before it is confirmed."""
        for token in new_tokens:
            for order in range(1, min(self.max_order, len(history)) + 1):
                self.tables[order][tuple(history[-order:])][token] += 1
            self.unigram[token] += 1
            history.append(token)

    @staticmethod
    def _mode(counts: Counter[int]) -> int:
        return min(counts, key=lambda token: (-counts[token], token))

    def propose(self, history: list[int], length: int) -> list[int]:
        if not history or not self.unigram:
            return []
        working = history[:]
        result = []
        for _ in range(length):
            choice = None
            for order in range(min(self.max_order, len(working)), 0, -1):
                counts = self.tables[order].get(tuple(working[-order:]))
                if counts:
                    choice = self._mode(counts)
                    break
            if choice is None:
                choice = self._mode(self.unigram)
            result.append(choice)
            working.append(choice)
        return result


def replay(
    streams: list[list[int]], draft_length: int, max_order: int, scope: str
) -> dict:
    if scope not in {"shared", "per_stream"}:
        raise ValueError(f"unknown n-gram scope: {scope}")
    positions = [min(1, len(stream)) for stream in streams]
    histories = [stream[:position] for stream, position in zip(streams, positions)]
    models = (
        [OnlineNgram(max_order)] * len(streams)
        if scope == "shared"
        else [OnlineNgram(max_order) for _ in streams]
    )
    if scope == "shared":
        shared = models[0]
        for history in histories:
            shared.unigram.update(history)
    else:
        for model, history in zip(models, histories):
            model.unigram.update(history)

    baseline_calls = max(max(0, len(stream) - 1) for stream in streams)
    per_stream_calls = [0] * len(streams)
    accepted = rejected = fully_accepted = drafted = 0
    progress_histogram: Counter[int] = Counter()
    synchronous_calls = 0
    started = time.perf_counter()

    while any(position < len(stream) for position, stream in zip(positions, streams)):
        synchronous_calls += 1
        pending: list[tuple[int, list[int]]] = []
        for row, (position, source, history, model) in enumerate(
            zip(positions, streams, histories, models)
        ):
            if position >= len(source):
                continue
            width = min(draft_length, len(source) - position)
            proposal = model.propose(history, width)
            if len(proposal) != width:
                raise AssertionError("drafter returned an incomplete proposal")
            drafted += width
            matched = 0
            while matched < width and proposal[matched] == source[position + matched]:
                matched += 1
            accepted += matched
            if matched < width:
                rejected += 1
                progress = matched + 1
            else:
                fully_accepted += 1
                progress = matched + int(position + matched < len(source))
            progress = min(progress, len(source) - position)
            confirmed = source[position : position + progress]
            pending.append((row, confirmed))
            per_stream_calls[row] += 1
            progress_histogram[progress] += 1

        # Shared proposals within a synchronous round see the same state.  Only
        # after every proposal is fixed do newly confirmed tokens become visible.
        for row, confirmed in pending:
            models[row].observe(histories[row], confirmed)
            positions[row] += len(confirmed)

    if histories != streams or positions != [len(stream) for stream in streams]:
        raise AssertionError("replay did not reconstruct every source token")
    calls = sum(per_stream_calls)
    return {
        "draft_length": draft_length,
        "scope": scope,
        "baseline_synchronous_target_calls": baseline_calls,
        "speculative_synchronous_target_calls": synchronous_calls,
        "target_call_reduction": baseline_calls / max(1, synchronous_calls),
        "baseline_per_stream_target_calls": sum(max(0, len(row) - 1) for row in streams),
        "speculative_per_stream_target_calls": calls,
        "per_stream_target_call_reduction": (
            sum(max(0, len(row) - 1) for row in streams) / max(1, calls)
        ),
        "accepted_draft_tokens": accepted,
        "first_rejection_events": rejected,
        "drafted_tokens": drafted,
        "accepted_over_accepted_plus_rejections": accepted / max(1, accepted + rejected),
        "accepted_over_all_drafted": accepted / max(1, drafted),
        "fully_accepted_blocks": fully_accepted,
        "mean_tokens_advanced_per_stream_call": (
            sum(value * count for value, count in progress_histogram.items()) / max(1, calls)
        ),
        "progress_histogram": {str(key): progress_histogram[key] for key in sorted(progress_histogram)},
        "per_stream_calls_min": min(per_stream_calls, default=0),
        "per_stream_calls_median": statistics.median(per_stream_calls) if per_stream_calls else 0,
        "per_stream_calls_max": max(per_stream_calls, default=0),
        "replay_seconds": time.perf_counter() - started,
        "exact_source_recovery": True,
    }


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def benchmark_target_calls(
    model,
    streams: list[list[int]],
    lengths: list[int],
    context_length: int,
    repeats: int,
    warmups: int,
    sample_positions: int,
    device: torch.device,
) -> dict:
    """Measure cached verifier calls at several positions of the real corpus."""
    usable = min(len(row) for row in streams) - max(lengths) - 1
    context = min(context_length, usable)
    if context < 2:
        raise ValueError("streams are too short for live target-call timing")
    high = min(len(streams[0]) - max(lengths), context * 8)
    offsets = sorted({
        round(context + index * (high - context) / max(1, sample_positions - 1))
        for index in range(sample_positions)
    })
    results = {}
    supports_logits_to_keep = "logits_to_keep" in inspect.signature(model.forward).parameters
    for length in lengths:
        samples = []
        by_offset = {}
        for offset in offsets:
            prompts = torch.tensor(
                [row[offset - context : offset] for row in streams], device=device
            )
            continuation = torch.tensor(
                [row[offset : offset + length] for row in streams], device=device
            )

            def once() -> float:
                with torch.inference_mode():
                    prefill_kwargs = {"input_ids": prompts, "use_cache": True}
                    if supports_logits_to_keep:
                        prefill_kwargs["logits_to_keep"] = 1
                    prefill = model(**prefill_kwargs)
                    torch.cuda.synchronize(device)
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    verify_kwargs = {
                        "input_ids": continuation,
                        "past_key_values": prefill.past_key_values,
                        "use_cache": True,
                    }
                    if supports_logits_to_keep:
                        verify_kwargs["logits_to_keep"] = length
                    model(**verify_kwargs)
                    end.record()
                    end.synchronize()
                    return start.elapsed_time(end) / 1000.0

            for _ in range(warmups):
                once()
            offset_samples = [once() for _ in range(repeats)]
            samples.extend(offset_samples)
            by_offset[str(offset)] = offset_samples
        results[str(length)] = {
            "samples_seconds": samples,
            "samples_by_stream_position": by_offset,
            "median_seconds": statistics.median(samples),
            "min_seconds": min(samples),
            "max_seconds": max(samples),
            "p25_seconds": _percentile(samples, 0.25),
            "p75_seconds": _percentile(samples, 0.75),
        }
    return {
        "context_tokens": context,
        "sampled_stream_positions": offsets,
        "conditions": results,
    }


def _dataset_argument(value: str) -> tuple[Path, int]:
    try:
        path_string, byte_string = value.rsplit(":", 1)
        return Path(path_string), int(byte_string)
    except (ValueError, TypeError) as error:
        raise argparse.ArgumentTypeError("dataset must be PATH:BYTES") from error


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", action="append", type=_dataset_argument,
        default=None, help="repeatable PATH:BYTES input (default: text8 first 10 MiB)",
    )
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--model-revision")
    parser.add_argument("--stream-counts", type=int, nargs="+", default=[256, 1024])
    parser.add_argument("--draft-lengths", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--scopes", choices=["shared", "per_stream"], nargs="+", default=["shared", "per_stream"])
    parser.add_argument("--max-ngram-order", type=int, default=8)
    parser.add_argument("--timing-stream-count", type=int, default=256)
    parser.add_argument("--timing-context", type=int, default=256)
    parser.add_argument("--timing-positions", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--skip-model-timing", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    datasets = args.dataset or [(REPO_ROOT / "data/text8", 10 * 1024 * 1024)]
    if any(value < 1 for value in [*args.stream_counts, *args.draft_lengths, args.max_ngram_order]):
        parser.error("stream counts, draft lengths, and n-gram order must be positive")
    if args.timing_stream_count not in args.stream_counts:
        parser.error("timing-stream-count must also occur in stream-counts")
    if args.repeats < 1 or args.warmups < 0 or args.timing_positions < 1:
        parser.error("invalid timing repetition counts")
    if args.device == "cuda" and not torch.cuda.is_available() and not args.skip_model_timing:
        parser.error("CUDA is unavailable")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, revision=args.model_revision, cache_dir=REPO_ROOT / ".cache",
        local_files_only=True,
    )
    dataset_rows = []
    timing_streams = None
    for dataset_path, byte_count in datasets:
        raw = dataset_path.read_bytes()[:byte_count]
        if len(raw) != byte_count:
            parser.error(f"{dataset_path} is shorter than {byte_count} bytes")
        text = raw.decode("utf-8")
        if text.encode("utf-8") != raw:
            parser.error(f"{dataset_path}:{byte_count} is not round-tripping UTF-8")
        tokens = tokenizer.encode(text, add_special_tokens=False)
        if tokenizer.decode(tokens).encode("utf-8") != raw:
            parser.error(f"tokenizer does not exactly reconstruct {dataset_path}")
        conditions = {}
        for stream_count in args.stream_counts:
            if stream_count >= len(tokens):
                parser.error(f"stream count exceeds tokens for {dataset_path}")
            streams = split_streams(tokens, stream_count)
            if dataset_path == datasets[0][0] and stream_count == args.timing_stream_count:
                timing_streams = streams
            for scope in args.scopes:
                for draft_length in args.draft_lengths:
                    key = f"b{stream_count}-{scope}-k{draft_length}"
                    print(f"replay {dataset_path} {key}", flush=True)
                    conditions[key] = replay(streams, draft_length, args.max_ngram_order, scope)
                    gc.collect()
        dataset_rows.append({
            "path": str(dataset_path),
            "bytes": len(raw),
            "source_sha256": hashlib.sha256(raw).hexdigest(),
            "tokens": len(tokens),
            "token_sha256": hashlib.sha256(
                torch.tensor(tokens, dtype=torch.int64).numpy().tobytes()
            ).hexdigest(),
            "conditions": conditions,
        })

    target_timing = None
    model_revision = args.model_revision
    dtype = None
    gpu = None
    if not args.skip_model_timing:
        device = torch.device(args.device)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, revision=args.model_revision, cache_dir=REPO_ROOT / ".cache",
            local_files_only=True, dtype="auto",
        ).eval().to(device)
        dtype = str(model.dtype)
        model_revision = getattr(model.config, "_commit_hash", model_revision)
        gpu = torch.cuda.get_device_name(device) if device.type == "cuda" else None
        if device.type != "cuda":
            parser.error("live timing currently requires CUDA events")
        target_timing = benchmark_target_calls(
            model, timing_streams, [1, *args.draft_lengths], args.timing_context,
            args.repeats, args.warmups, args.timing_positions, device,
        )
        baseline = target_timing["conditions"]["1"]["median_seconds"]
        for dataset in dataset_rows:
            for condition in dataset["conditions"].values():
                verify = target_timing["conditions"][str(condition["draft_length"])]["median_seconds"]
                condition["live_verifier_call_cost_ratio_vs_one_token"] = verify / baseline
                condition["projected_target_inference_speedup"] = (
                    condition["target_call_reduction"] * baseline / verify
                )

    output = {
        "experiment": "speculative_decompression_full_corpus_replay_v1",
        "scope": "full-corpus causal replay plus repeated live target-call measurements",
        "limitations": [
            "source tokens stand in for arithmetic-decoder verdicts",
            "live target-call samples are combined with replayed schedules as a projection",
            "draft scheduling, arithmetic-state rollback, CDF construction, and ragged-batch execution are not timed end to end",
            "multi-token verifier numerics must be made encoder/decoder compatible before integration",
        ],
        "datasets": dataset_rows,
        "stream_counts": args.stream_counts,
        "draft_lengths": args.draft_lengths,
        "ngram_scopes": args.scopes,
        "max_ngram_order": args.max_ngram_order,
        "causality_contract": "proposals use only tokens confirmed before the current synchronous round",
        "bonus_token_after_fully_accepted_block": True,
        "target_call_timing": target_timing,
        "timing_dataset": str(datasets[0][0]),
        "timing_stream_count": args.timing_stream_count,
        "model": args.model,
        "model_revision": model_revision,
        "dtype": dtype,
        "device": args.device,
        "gpu": gpu,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "python_platform": platform.platform(),
        "repeats": args.repeats,
        "warmups": args.warmups,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
