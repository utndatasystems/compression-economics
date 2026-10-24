"""Audit incremental-versus-block verifier intervals for perfect drafts.

Speculative arithmetic decoding can reuse an ordinary incrementally encoded
archive only if the verifier's multi-token call produces the same integer CDF
intervals as one-token incremental decoding.  This runner gives the block path
the true continuation (a perfect draft), then compares target intervals at
several positions of the paper's text8 workload.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path
import platform
import sys

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.coding.target_interval import DEFAULT_TOTAL, target_intervals_from_probs_tensor
from research.papers.coding_survey.experiments.benchmark_speculative_full import split_streams


def _forward(model, *, keep: int, **kwargs):
    if "logits_to_keep" in inspect.signature(model.forward).parameters:
        kwargs["logits_to_keep"] = keep
    output = model(**kwargs)
    logits = output.logits
    if logits.shape[1] != keep:
        logits = logits[:, -keep:, :]
    return logits, output.past_key_values


def _target_intervals(logits, targets, alphabet, dense_columns, total):
    reduced = logits.index_select(2, alphabet)
    probabilities = torch.softmax(reduced, dim=-1, dtype=torch.float32)
    dense_targets = torch.tensor(
        [[dense_columns[token] for token in row] for row in targets],
        dtype=torch.long,
        device=logits.device,
    )
    flat_probs = probabilities.reshape(-1, probabilities.shape[-1])
    flat_targets = dense_targets.reshape(-1)
    low, high, totals, target_probs = target_intervals_from_probs_tensor(
        flat_probs, flat_targets, total=total
    )
    return torch.stack((low, high, totals), dim=1), target_probs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=REPO_ROOT / "data/text8")
    parser.add_argument("--input-bytes", type=int, default=10 * 1024 * 1024)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--model-revision")
    parser.add_argument("--stream-count", type=int, default=256)
    parser.add_argument("--audit-rows", type=int, default=32)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--positions", type=int, nargs="+", default=[256, 704, 1152, 1600, 2048])
    parser.add_argument("--draft-lengths", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--frequency-total", type=int, default=DEFAULT_TOTAL)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    raw = args.input.read_bytes()[: args.input_bytes]
    if len(raw) != args.input_bytes:
        parser.error("input is shorter than input-bytes")
    text = raw.decode("utf-8")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, revision=args.model_revision, cache_dir=REPO_ROOT / ".cache",
        local_files_only=True,
    )
    tokens = tokenizer.encode(text, add_special_tokens=False)
    streams = split_streams(tokens, args.stream_count)[: args.audit_rows]
    max_k = max(args.draft_lengths)
    if max(args.positions) + max_k > min(map(len, streams)):
        parser.error("an audit position exceeds the available stream length")
    vocabulary = sorted(set(tokens))
    if args.frequency_total <= len(vocabulary):
        parser.error("frequency total must exceed the reduced alphabet")
    dense_columns = {token: index for index, token in enumerate(vocabulary)}
    device = torch.device("cuda")
    alphabet = torch.tensor(vocabulary, dtype=torch.long, device=device)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.model_revision, cache_dir=REPO_ROOT / ".cache",
        local_files_only=True, dtype="auto",
    ).eval().to(device)

    results = {}
    with torch.inference_mode():
        for position in args.positions:
            prompts = torch.tensor(
                [row[position - args.context_length : position] for row in streams],
                dtype=torch.long, device=device,
            )
            targets = [row[position : position + max_k] for row in streams]
            target_tensor = torch.tensor(targets, dtype=torch.long, device=device)

            first_logits, incremental_past = _forward(
                model, keep=1, input_ids=prompts, use_cache=True
            )
            incremental_logits = [first_logits]
            for step in range(1, max_k):
                step_logits, incremental_past = _forward(
                    model,
                    keep=1,
                    input_ids=target_tensor[:, step - 1 : step],
                    past_key_values=incremental_past,
                    use_cache=True,
                )
                incremental_logits.append(step_logits)
            incremental_logits = torch.cat(incremental_logits, dim=1)

            block_first, block_past = _forward(
                model, keep=1, input_ids=prompts, use_cache=True
            )
            block_rest, _ = _forward(
                model,
                keep=max_k - 1,
                input_ids=target_tensor[:, : max_k - 1],
                past_key_values=block_past,
                use_cache=True,
            )
            block_logits = torch.cat((block_first, block_rest), dim=1)

            for draft_length in args.draft_lengths:
                target_slice = [row[:draft_length] for row in targets]
                incremental_intervals, incremental_probs = _target_intervals(
                    incremental_logits[:, :draft_length], target_slice, alphabet,
                    dense_columns, args.frequency_total,
                )
                block_intervals, block_probs = _target_intervals(
                    block_logits[:, :draft_length], target_slice, alphabet,
                    dense_columns, args.frequency_total,
                )
                equal = (incremental_intervals == block_intervals).all(dim=1)
                mismatch = int((~equal).sum().item())
                key = f"position-{position}-k{draft_length}"
                results[key] = {
                    "stream_position": position,
                    "draft_length": draft_length,
                    "compared_target_intervals": equal.numel(),
                    "interval_mismatch_count": mismatch,
                    "interval_match_fraction": float(equal.float().mean().item()),
                    "decoder_compatible": mismatch == 0,
                    "max_integer_interval_absolute_error": int(
                        (incremental_intervals - block_intervals).abs().max().item()
                    ),
                    "max_target_probability_absolute_error": float(
                        (incremental_probs - block_probs).abs().max().item()
                    ),
                    "first_position_intervals_equal": bool(equal.reshape(args.audit_rows, draft_length)[:, 0].all()),
                }

    output = {
        "experiment": "speculative_block_numerics_audit_v1",
        "contract": "perfect true-token drafts; incremental one-token KV calls versus one multi-token verifier call",
        "input": str(args.input),
        "input_prefix_bytes": len(raw),
        "input_sha256": hashlib.sha256(raw).hexdigest(),
        "tokens": len(tokens),
        "model": args.model,
        "model_revision": getattr(model.config, "_commit_hash", args.model_revision),
        "dtype": str(model.dtype),
        "gpu": torch.cuda.get_device_name(device),
        "stream_count": args.stream_count,
        "audit_rows": args.audit_rows,
        "context_length": args.context_length,
        "positions": args.positions,
        "draft_lengths": args.draft_lengths,
        "alphabet_size": len(vocabulary),
        "frequency_total": args.frequency_total,
        "results": results,
        "all_decoder_compatible": all(row["decoder_compatible"] for row in results.values()),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "python_platform": platform.platform(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "all_decoder_compatible": output["all_decoder_compatible"],
        "results": results,
    }, indent=2))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
