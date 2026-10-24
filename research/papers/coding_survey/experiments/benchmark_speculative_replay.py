"""Replay exact speculative decompression on a matched text archive.

This is a deliberately small feasibility experiment.  A cheap online n-gram
drafter proposes a block from tokens that have already been confirmed.  The
archive's source tokens stand in for the exact arithmetic-decoder verdict:
the verifier accepts the longest matching prefix, emits the true token at the
first mismatch, and (when the whole block matches) emits one bonus true token.

The replay therefore measures the control-flow opportunity without changing
the archive.  An optional model microbenchmark measures the cost of one target
decode call versus a target verification call over 2, 4, or 8 tokens.  It does
not claim end-to-end speculative-decoder speedup: draft scheduling, CDF
construction, arithmetic-state checkpointing, and rollback are not integrated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_PILOT = (
    REPO_ROOT
    / "artifacts/papers/coding-survey/in-engine-decoding/matched-cuda-qwen-1024"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "artifacts/papers/coding-survey/speculative-decompression/replay-k2-k4-k8.json"
)


def split_streams(tokens: list[int], stream_count: int) -> list[list[int]]:
    base, extra = divmod(len(tokens), stream_count)
    streams: list[list[int]] = []
    offset = 0
    for index in range(stream_count):
        length = base + (index < extra)
        streams.append(tokens[offset : offset + length])
        offset += length
    assert offset == len(tokens)
    return streams


def train_ngram(
    histories: Iterable[list[int]], max_order: int
) -> tuple[dict[int, dict[tuple[int, ...], Counter[int]]], Counter[int]]:
    """Train without introducing transitions across independent streams."""
    tables: dict[int, dict[tuple[int, ...], Counter[int]]] = {}
    histories = list(histories)
    for order in range(1, max_order + 1):
        table: dict[tuple[int, ...], Counter[int]] = defaultdict(Counter)
        for history in histories:
            for end in range(order, len(history)):
                table[tuple(history[end - order : end])][history[end]] += 1
        tables[order] = table
    unigram: Counter[int] = Counter()
    for history in histories:
        unigram.update(history)
    return tables, unigram


def propose_ngram(
    history: list[int],
    length: int,
    max_order: int,
    tables: dict[int, dict[tuple[int, ...], Counter[int]]] | None = None,
    unigram: Counter[int] | None = None,
) -> list[int]:
    """Greedily draft from an n-gram model trained only on confirmed history."""
    if not history or length < 1:
        return []
    if tables is None or unigram is None:
        tables, unigram = train_ngram([history], max_order)
    draft: list[int] = []
    working = history[:]
    for _ in range(length):
        choice = None
        for order in range(min(max_order, len(working)), 0, -1):
            counts = tables.get(order, {}).get(tuple(working[-order:]))
            if counts:
                # Stable tie break: frequency first, then smaller token id.
                choice = min(counts, key=lambda token: (-counts[token], token))
                break
        if choice is None:
            choice = min(unigram, key=lambda token: (-unigram[token], token))
        draft.append(choice)
        working.append(choice)
    return draft


def replay_stream(source: list[int], draft_length: int, max_order: int) -> dict:
    if not source:
        return {
            "source_tokens": 0,
            "baseline_target_calls": 0,
            "speculative_target_calls": 0,
            "accepted_draft_tokens": 0,
            "rejected_draft_tokens": 0,
            "all_accepted_blocks": 0,
            "drafted_tokens": 0,
            "progress_per_call": [],
        }
    position = 1  # The archive stores one seed token per stream.
    history = source[:position]
    accepted = rejected = all_accepted = drafted = 0
    progress_per_call: list[int] = []
    while position < len(source):
        width = min(draft_length, len(source) - position)
        proposal = propose_ngram(history, width, max_order)
        drafted += len(proposal)
        matched = 0
        while matched < width and proposal[matched] == source[position + matched]:
            matched += 1
        accepted += matched
        if matched < width:
            rejected += 1
            progress = matched + 1  # Exact decoder emits the mismatch token.
        else:
            all_accepted += 1
            # The target pass also produces the distribution after the final
            # proposal, so exact arithmetic decoding can emit one bonus token.
            progress = matched + (position + matched < len(source))
        progress = min(progress, len(source) - position)
        history.extend(source[position : position + progress])
        position += progress
        progress_per_call.append(progress)
    return {
        "source_tokens": len(source),
        "baseline_target_calls": max(0, len(source) - 1),
        "speculative_target_calls": len(progress_per_call),
        "accepted_draft_tokens": accepted,
        "rejected_draft_tokens": rejected,
        "all_accepted_blocks": all_accepted,
        "drafted_tokens": drafted,
        "progress_per_call": progress_per_call,
    }


def aggregate_replay(streams: list[list[int]], draft_length: int, max_order: int) -> dict:
    rows = [replay_stream(stream, draft_length, max_order) for stream in streams]
    baseline_steps = max(row["baseline_target_calls"] for row in rows)
    speculative_steps = max(row["speculative_target_calls"] for row in rows)
    accepted = sum(row["accepted_draft_tokens"] for row in rows)
    rejected = sum(row["rejected_draft_tokens"] for row in rows)
    drafted = sum(row["drafted_tokens"] for row in rows)
    progress = [value for row in rows for value in row["progress_per_call"]]
    return {
        "draft_length": draft_length,
        "baseline_synchronous_target_calls": baseline_steps,
        "speculative_synchronous_target_calls": speculative_steps,
        "target_call_reduction": baseline_steps / speculative_steps,
        "accepted_draft_tokens": accepted,
        "rejected_draft_tokens": rejected,
        "drafted_tokens": drafted,
        "token_acceptance_rate": accepted / max(1, accepted + rejected),
        "proposal_efficiency": accepted / max(1, drafted),
        "mean_tokens_advanced_per_stream_call": statistics.mean(progress),
        "median_tokens_advanced_per_stream_call": statistics.median(progress),
        "all_accepted_blocks": sum(row["all_accepted_blocks"] for row in rows),
        "per_stream_speculative_calls": [row["speculative_target_calls"] for row in rows],
        "exact_recovery_by_construction": True,
    }


def aggregate_shared_replay(
    streams: list[list[int]], draft_length: int, max_order: int
) -> dict:
    """Replay lockstep streams with one n-gram trained on all confirmed rows."""
    positions = [min(1, len(stream)) for stream in streams]
    histories = [stream[:position] for stream, position in zip(streams, positions)]
    baseline_steps = max(max(0, len(stream) - 1) for stream in streams)
    per_stream_calls = [0] * len(streams)
    accepted = rejected = all_accepted = drafted = 0
    progress_values: list[int] = []
    synchronous_calls = 0
    while any(position < len(stream) for position, stream in zip(positions, streams)):
        tables, unigram = train_ngram(histories, max_order)
        synchronous_calls += 1
        for row, (position, source) in enumerate(zip(positions, streams)):
            if position >= len(source):
                continue
            width = min(draft_length, len(source) - position)
            proposal = propose_ngram(
                histories[row], width, max_order, tables=tables, unigram=unigram
            )
            drafted += len(proposal)
            matched = 0
            while matched < width and proposal[matched] == source[position + matched]:
                matched += 1
            accepted += matched
            if matched < width:
                rejected += 1
                progress = matched + 1
            else:
                all_accepted += 1
                progress = matched + (position + matched < len(source))
            progress = min(progress, len(source) - position)
            histories[row].extend(source[position : position + progress])
            positions[row] += progress
            per_stream_calls[row] += 1
            progress_values.append(progress)
    return {
        "draft_length": draft_length,
        "baseline_synchronous_target_calls": baseline_steps,
        "speculative_synchronous_target_calls": synchronous_calls,
        "target_call_reduction": baseline_steps / synchronous_calls,
        "accepted_draft_tokens": accepted,
        "rejected_draft_tokens": rejected,
        "drafted_tokens": drafted,
        "token_acceptance_rate": accepted / max(1, accepted + rejected),
        "proposal_efficiency": accepted / max(1, drafted),
        "mean_tokens_advanced_per_stream_call": statistics.mean(progress_values),
        "median_tokens_advanced_per_stream_call": statistics.median(progress_values),
        "all_accepted_blocks": all_accepted,
        "per_stream_speculative_calls": per_stream_calls,
        "exact_recovery_by_construction": True,
    }


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def benchmark_target_calls(
    model,
    streams: list[list[int]],
    lengths: Iterable[int],
    repeats: int,
    warmups: int,
    device: torch.device,
) -> dict[str, dict]:
    # Use equal 32-token prefixes from every stream.  The live archive has at
    # most 64 tokens per stream; this is close to its middle-of-stream state.
    context = min(32, min(len(stream) - 9 for stream in streams))
    if context < 1:
        raise ValueError("streams are too short for the target-call benchmark")
    prompts = torch.tensor([stream[:context] for stream in streams], device=device)
    continuations = torch.tensor(
        [stream[context : context + max(lengths)] for stream in streams], device=device
    )

    def once(length: int) -> float:
        with torch.inference_mode():
            prefill = model(input_ids=prompts, use_cache=True)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                model(
                    input_ids=continuations[:, :length],
                    past_key_values=prefill.past_key_values,
                    use_cache=True,
                )
                end.record()
                end.synchronize()
                return start.elapsed_time(end) / 1000.0
            started = time.perf_counter()
            model(
                input_ids=continuations[:, :length],
                past_key_values=prefill.past_key_values,
                use_cache=True,
            )
            return time.perf_counter() - started

    results: dict[str, dict] = {}
    for length in lengths:
        for _ in range(warmups):
            once(length)
        samples = [once(length) for _ in range(repeats)]
        results[str(length)] = {
            "samples_seconds": samples,
            "median_seconds": statistics.median(samples),
            "min_seconds": min(samples),
            "max_seconds": max(samples),
            "p25_seconds": percentile(samples, 0.25),
            "p75_seconds": percentile(samples, 0.75),
        }
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-dir", type=Path, default=DEFAULT_PILOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--draft-lengths", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--max-ngram-order", type=int, default=8)
    parser.add_argument(
        "--ngram-scope",
        choices=["shared", "per_stream"],
        default="shared",
        help="Train on all already-confirmed streams or on each stream alone",
    )
    parser.add_argument("--streams", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--skip-model-timing", action="store_true")
    args = parser.parse_args()
    if min(args.draft_lengths) < 1 or args.streams < 1 or args.max_ngram_order < 1:
        parser.error("draft lengths, streams, and n-gram order must be positive")
    if args.repeats < 1 or args.warmups < 0 or args.threads < 1:
        parser.error("invalid repeat, warmup, or thread count")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")

    excerpt_path = args.pilot_dir / "excerpt.txt"
    archive_path = args.pilot_dir / "ac_target_interval.bin"
    if not excerpt_path.is_file() or not archive_path.is_file():
        parser.error("pilot directory must contain excerpt.txt and ac_target_interval.bin")

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, cache_dir=str(REPO_ROOT / ".cache"), local_files_only=True
    )
    text = excerpt_path.read_text(encoding="utf-8")
    tokens = tokenizer.encode(text, add_special_tokens=False)
    streams = split_streams(tokens, args.streams)
    replay_function = (
        aggregate_shared_replay if args.ngram_scope == "shared" else aggregate_replay
    )
    replay = {
        str(length): replay_function(streams, length, args.max_ngram_order)
        for length in args.draft_lengths
    }

    target_timing = None
    model_revision = None
    dtype = None
    gpu = None
    if not args.skip_model_timing:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            cache_dir=str(REPO_ROOT / ".cache"),
            local_files_only=True,
            dtype="auto",
        ).eval().to(device)
        dtype = str(model.dtype)
        model_revision = getattr(model.config, "_commit_hash", None)
        gpu = torch.cuda.get_device_name(device) if device.type == "cuda" else None
        target_timing = benchmark_target_calls(
            model,
            streams,
            [1, *args.draft_lengths],
            args.repeats,
            args.warmups,
            device,
        )
        baseline_call = target_timing["1"]["median_seconds"]
        for length in args.draft_lengths:
            row = replay[str(length)]
            verify_call = target_timing[str(length)]["median_seconds"]
            row["target_call_cost_ratio_vs_one_token"] = verify_call / baseline_call
            row["projected_target_inference_speedup"] = (
                row["target_call_reduction"] * baseline_call / verify_call
            )
            speedup = row["projected_target_inference_speedup"]
            row["projected_end_to_end_speedup_at_89pct_inference_share"] = (
                1.0 / (0.11 + 0.89 / speedup)
            )

    output = {
        "scope": "control-flow replay plus isolated target-call microbenchmark",
        "limitations": [
            "source tokens stand in for exact arithmetic-decoder verification",
            "draft scheduling, CDF construction, checkpointing, rollback, and ragged batches are not timed",
            "projected speedups are estimates, not live decoder measurements",
        ],
        "pilot_dir": str(args.pilot_dir),
        "archive_sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
        "excerpt_sha256": hashlib.sha256(excerpt_path.read_bytes()).hexdigest(),
        "source_bytes": excerpt_path.stat().st_size,
        "tokens": len(tokens),
        "streams": args.streams,
        "tokens_per_stream": [len(stream) for stream in streams],
        "drafter": (
            "online longest-context n-gram with unigram fallback; training uses "
            f"{args.ngram_scope.replace('_', ' ')} confirmed histories"
        ),
        "ngram_scope": args.ngram_scope,
        "max_ngram_order": args.max_ngram_order,
        "bonus_token_after_fully_accepted_block": True,
        "replay": replay,
        "target_call_timing": target_timing,
        "model": args.model,
        "model_revision": model_revision,
        "dtype": dtype,
        "device": args.device,
        "gpu": gpu,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "python_platform": platform.platform(),
        "torch_threads": args.threads,
        "repeats": args.repeats,
        "warmups": args.warmups,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"replay": replay, "target_call_timing": target_timing}, indent=2))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
