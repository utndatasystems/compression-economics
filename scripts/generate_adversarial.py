#!/usr/bin/env python3
"""Generate fixed-length worst-probability inputs for compression experiments."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.adversarial import (
    generate_occurring_token_fixed_point,
    generate_worst_case_sequences,
    top_k_frequent_tokens,
)
from src.prediction import TokenPredictor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-0.5B")
    starts = parser.add_mutually_exclusive_group(required=True)
    starts.add_argument("--start-token-id", type=int, action="append")
    starts.add_argument("--start-text", action="append")
    parser.add_argument("--total-length", type=int, default=1000)
    parser.add_argument(
        "--candidate-mode",
        choices=("full", "occurring", "top-k"),
        action="append",
        help="Repeat for multiple variants; default: full and occurring.",
    )
    parser.add_argument("--reference-path", type=Path)
    parser.add_argument(
        "--reference-max-tokens",
        type=int,
        default=100_000,
        help="Maximum reference tokens used to build occurring/top-k masks.",
    )
    parser.add_argument("--top-k", type=int, default=1000)
    parser.add_argument("--context-length", type=int, default=1000)
    parser.add_argument("--retain-tokens", type=int, default=100)
    parser.add_argument("--max-fixed-point-passes", type=int, default=10)
    parser.add_argument("--include-special-tokens", action="store_true")
    parser.add_argument("--no-kv-cache", dest="use_kv_cache", action="store_false")
    parser.set_defaults(use_kv_cache=True)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/runs/adversarial")
    )
    return parser.parse_args()


def make_predictor_args(args: argparse.Namespace) -> SimpleNamespace:
    model_name = args.model_name.lower()
    if "t5" in model_name:
        raise ValueError("Generation currently requires a causal language model")
    return SimpleNamespace(
        model_name=args.model_name,
        engine="transformer",
        reduce_tokens=False,
        encoding="bitpacked",
        is_seq2seq=False,
        is_mamba="mamba" in model_name,
        lora_path=None,
        spec_k=None,
    )


def encode_starts(args: argparse.Namespace, tokenizer) -> list[list[int]]:
    starts = (
        [[token_id] for token_id in args.start_token_id]
        if args.start_token_id
        else [
            tokenizer.encode(text, add_special_tokens=False)
            for text in args.start_text
        ]
    )
    if any(not start for start in starts):
        raise ValueError("Every starting value must encode to at least one token")
    if len({len(start) for start in starts}) != 1:
        raise ValueError("All starting texts must encode to the same token length")
    return starts


def read_reference_tokens(path: Path, tokenizer, max_tokens: int) -> list[int]:
    """Tokenize a bounded reference prefix without materializing the full corpus."""
    if max_tokens <= 0:
        raise ValueError("--reference-max-tokens must be positive")

    text = ""
    tokens: list[int] = []
    character_chunk_size = max(65_536, max_tokens * 4)
    with path.open(encoding="utf-8") as handle:
        while len(tokens) < max_tokens:
            chunk = handle.read(character_chunk_size)
            if not chunk:
                break
            text += chunk
            tokens = tokenizer.encode(
                text,
                add_special_tokens=False,
                truncation=True,
                max_length=max_tokens,
            )
    return tokens


def make_payload(generation, tokenizer) -> dict:
    runs = []
    for index, (tokens, model_logs, masked_logs, summary) in enumerate(
        zip(
            generation.token_ids,
            generation.model_log_probabilities,
            generation.masked_log_probabilities,
            generation.summary(),
        )
    ):
        decoded = tokenizer.decode(tokens, skip_special_tokens=False)
        runs.append(
            {
                "run_index": index,
                "start_token_ids": tokens[: len(tokens) - len(model_logs)],
                "token_ids": tokens,
                "decoded_text": decoded,
                "decoded_text_round_trips": (
                    tokenizer.encode(decoded, add_special_tokens=False) == tokens
                ),
                "model_log_probabilities": model_logs,
                "masked_log_probabilities": masked_logs,
                "summary": summary,
            }
        )
    return {"runs": runs}


def save_mode(output_dir: Path, mode: str, payload: dict) -> None:
    mode_dir = output_dir / mode
    mode_dir.mkdir(parents=True, exist_ok=True)
    (mode_dir / "results.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8"
    )
    for run in payload["runs"]:
        (mode_dir / f"run_{run['run_index']:02d}.txt").write_text(
            run["decoded_text"], encoding="utf-8"
        )
        (mode_dir / f"run_{run['run_index']:02d}.tokens.json").write_text(
            json.dumps(run["token_ids"]), encoding="utf-8"
        )


def main() -> None:
    args = parse_args()
    modes = args.candidate_mode or ["full", "occurring"]
    if "top-k" in modes and args.reference_path is None:
        raise ValueError("--reference-path is required for top-k mode")

    predictor = TokenPredictor(make_predictor_args(args), bitmap_data=None)
    tokenizer = predictor.tokenizer
    starts = encode_starts(args, tokenizer)
    vocab_size = min(tokenizer.vocab_size, predictor.model.config.vocab_size)
    invalid_starts = [
        token_id for start in starts for token_id in start
        if not 0 <= token_id < vocab_size
    ]
    if invalid_starts:
        raise ValueError(f"Starting token IDs outside vocabulary: {invalid_starts}")
    excluded = set() if args.include_special_tokens else set(tokenizer.all_special_ids)
    full_candidates = [
        token_id for token_id in range(vocab_size) if token_id not in excluded
    ]
    reference_tokens = None
    if args.reference_path:
        reference_tokens = read_reference_tokens(
            args.reference_path, tokenizer, args.reference_max_tokens
        )

    metadata = {
        "model_name": args.model_name,
        "total_length": args.total_length,
        "context_length": args.context_length,
        "retain_tokens": args.retain_tokens,
        "use_kv_cache": args.use_kv_cache,
        "include_special_tokens": args.include_special_tokens,
        "reference_path": str(args.reference_path) if args.reference_path else None,
        "reference_max_tokens": args.reference_max_tokens,
        "reference_tokens_used": len(reference_tokens) if reference_tokens else 0,
        "selection_rule": "lowest finite logit; ties use lowest token ID",
        "top_k_definition": "most frequent reference tokens",
        "occurring_definition": "fixed point of allowed and generated token sets",
    }

    for mode in modes:
        mode_metadata = dict(metadata, candidate_mode=mode)
        if mode == "full":
            generation = generate_worst_case_sequences(
                predictor,
                starts,
                args.total_length,
                [full_candidates for _ in starts],
                context_length=args.context_length,
                retain_tokens=args.retain_tokens,
                use_kv_cache=args.use_kv_cache,
            )
        elif mode == "top-k":
            candidates = top_k_frequent_tokens(reference_tokens, args.top_k)
            generation = generate_worst_case_sequences(
                predictor,
                starts,
                args.total_length,
                [candidates for _ in starts],
                context_length=args.context_length,
                retain_tokens=args.retain_tokens,
                use_kv_cache=args.use_kv_cache,
            )
            mode_metadata["top_k"] = args.top_k
        else:
            initial = (
                sorted(set(reference_tokens))
                if reference_tokens is not None
                else full_candidates
            )
            fixed_point = generate_occurring_token_fixed_point(
                predictor,
                starts,
                args.total_length,
                initial,
                context_length=args.context_length,
                retain_tokens=args.retain_tokens,
                use_kv_cache=args.use_kv_cache,
                max_passes=args.max_fixed_point_passes,
            )
            generation = fixed_point.generation
            mode_metadata.update(
                {
                    "fixed_point_converged": fixed_point.converged,
                    "fixed_point_passes": fixed_point.passes,
                    "candidate_size_history": fixed_point.candidate_size_history,
                }
            )

        payload = make_payload(generation, tokenizer)
        payload["metadata"] = mode_metadata
        save_mode(args.output_dir, mode, payload)
        print(f"Saved {mode} results to {args.output_dir / mode}")


if __name__ == "__main__":
    main()
