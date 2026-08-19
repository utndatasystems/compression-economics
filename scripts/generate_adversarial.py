#!/usr/bin/env python3
"""Generate full-vocabulary worst cases and post-hoc mask rescoring."""

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
    AdversarialGeneration,
    generate_worst_case_sequences,
    rescore_sequences,
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
        "--generation-alphabet",
        choices=("full", "printable-ascii", "ascii-bytes"),
        default="full",
        help=(
            "Candidate alphabet used to construct the sequence. "
            "printable-ascii and ascii-bytes create canonical "
            "one-byte-per-token lossless stress tests."
        ),
    )
    parser.add_argument(
        "--candidate-mode",
        choices=("full", "occurring"),
        action="append",
        help=(
            "Repeat for multiple outputs; default: full generation and "
            "post-hoc occurring-token rescoring."
        ),
    )
    parser.add_argument("--context-length", type=int, default=1000)
    parser.add_argument("--retain-tokens", type=int, default=100)
    parser.add_argument("--include-special-tokens", action="store_true")
    parser.add_argument(
        "--no-kv-cache", dest="use_kv_cache", action="store_false"
    )
    parser.set_defaults(use_kv_cache=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/runs/adversarial"),
    )
    return parser.parse_args()


def make_predictor_args(args: argparse.Namespace) -> SimpleNamespace:
    model_name = args.model_name.lower()
    if "t5" in model_name:
        raise ValueError(
            "Generation currently requires a causal language model"
        )
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


def encode_starts(
    args: argparse.Namespace, tokenizer
) -> list[list[int]]:
    starts = (
        [[token_id] for token_id in args.start_token_id]
        if args.start_token_id
        else [
            tokenizer.encode(text, add_special_tokens=False)
            for text in args.start_text
        ]
    )
    if any(not start for start in starts):
        raise ValueError(
            "Every starting value must encode to at least one token"
        )
    if len({len(start) for start in starts}) != 1:
        raise ValueError(
            "All starting texts must encode to the same token length"
        )
    return starts


def ascii_byte_token_ids(
    tokenizer, *, printable_only: bool
) -> list[int]:
    """Return canonical tokenizer IDs for one-byte ASCII characters."""
    candidates = []
    for token_id in range(tokenizer.vocab_size):
        decoded = tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        if (
            len(decoded) == 1
            and ord(decoded) <= 127
            and (not printable_only or 32 <= ord(decoded) <= 126)
            and tokenizer.encode(
                decoded, add_special_tokens=False
            ) == [token_id]
        ):
            candidates.append(token_id)
    expected = 95 if printable_only else 128
    if len(candidates) != expected:
        qualifier = "printable " if printable_only else ""
        raise ValueError(
            "Tokenizer does not expose exactly one canonical token for "
            f"each {qualifier}ASCII byte; found {len(candidates)}"
        )
    return candidates


def token_ids_round_trip(tokenizer, token_ids) -> bool:
    """Whether decoded text canonically re-encodes to the same token IDs."""
    decoded = tokenizer.decode(
        token_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    return tokenizer.encode(decoded, add_special_tokens=False) == list(token_ids)


def make_round_trip_validator(tokenizer):
    """Build a candidate validator that preserves canonical tokenization."""
    def validator(_row_index, prefix, candidate_token_id):
        return token_ids_round_trip(
            tokenizer, [*prefix, candidate_token_id]
        )

    return validator


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
        decoded = tokenizer.decode(
            tokens,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        round_trips = token_ids_round_trip(tokenizer, tokens)
        if not round_trips:
            raise RuntimeError(
                "Generated adversarial tokens do not round-trip through text"
            )
        runs.append(
            {
                "run_index": index,
                "start_token_ids": tokens[
                    : len(tokens) - len(model_logs)
                ],
                "token_ids": tokens,
                "decoded_text": decoded,
                "decoded_text_round_trips": round_trips,
                "decoded_utf8_size_bytes": len(decoded.encode("utf-8")),
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
        json.dumps(payload, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    for run in payload["runs"]:
        text_path = mode_dir / f"run_{run['run_index']:02d}.txt"
        with text_path.open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            handle.write(run["decoded_text"])
        (
            mode_dir / f"run_{run['run_index']:02d}.tokens.json"
        ).write_text(
            json.dumps(run["token_ids"]), encoding="utf-8"
        )


def main() -> None:
    args = parse_args()
    modes = list(dict.fromkeys(
        args.candidate_mode or ["full", "occurring"]
    ))

    predictor = TokenPredictor(
        make_predictor_args(args), bitmap_data=None
    )
    tokenizer = predictor.tokenizer
    starts = encode_starts(args, tokenizer)
    start_length = len(starts[0])
    vocab_size = min(
        tokenizer.vocab_size, predictor.model.config.vocab_size
    )
    invalid_starts = [
        token_id
        for start in starts
        for token_id in start
        if not 0 <= token_id < vocab_size
    ]
    if invalid_starts:
        raise ValueError(
            f"Starting token IDs outside vocabulary: {invalid_starts}"
        )
    non_round_trip_starts = [
        start for start in starts
        if not token_ids_round_trip(tokenizer, start)
    ]
    if non_round_trip_starts:
        raise ValueError(
            "Every starting sequence must round-trip through decoded text; "
            f"invalid starts: {non_round_trip_starts}"
        )
    round_trip_validator = make_round_trip_validator(tokenizer)

    excluded = (
        set()
        if args.include_special_tokens
        else set(tokenizer.all_special_ids)
    )
    full_candidates = [
        token_id
        for token_id in range(vocab_size)
        if token_id not in excluded
    ]
    if args.generation_alphabet in {
        "printable-ascii", "ascii-bytes"
    }:
        generation_candidates = ascii_byte_token_ids(
            tokenizer,
            printable_only=(
                args.generation_alphabet == "printable-ascii"
            ),
        )
    else:
        generation_candidates = full_candidates
    full_generation = generate_worst_case_sequences(
        predictor,
        starts,
        args.total_length,
        [generation_candidates for _ in starts],
        context_length=args.context_length,
        retain_tokens=args.retain_tokens,
        use_kv_cache=args.use_kv_cache,
        candidate_validator=round_trip_validator,
    )

    if args.generation_alphabet == "full":
        full_scored_generation = full_generation
    else:
        full_scored_generation = AdversarialGeneration(
            token_ids=full_generation.token_ids,
            model_log_probabilities=(
                full_generation.model_log_probabilities
            ),
            masked_log_probabilities=[
                row[:]
                for row in full_generation.model_log_probabilities
            ],
            candidate_sizes=[len(full_candidates) for _ in starts],
        )

    shared_metadata = {
        "model_name": args.model_name,
        "total_length": args.total_length,
        "context_length": args.context_length,
        "retain_tokens": args.retain_tokens,
        "use_kv_cache": args.use_kv_cache,
        "include_special_tokens": args.include_special_tokens,
        "selection_rule": (
            f"lowest finite {args.generation_alphabet} candidate logit "
            "whose extended prefix "
            "round-trips through tokenizer decode/encode; ties use lowest token ID"
        ),
        "sequence_generation_dictionary": args.generation_alphabet,
        "generation_candidate_size": len(generation_candidates),
        "lossless_text_roundtrip_required": True,
        "roundtrip_constraint": "encode(decode(prefix + token)) == prefix + token",
    }

    if "full" in modes:
        payload = make_payload(full_scored_generation, tokenizer)
        payload["metadata"] = dict(
            shared_metadata,
            candidate_mode="full",
            scoring_dictionary="full",
            mask_application=(
                "generation alphabet applied during construction; "
                "full vocabulary used for probability scoring"
            ),
        )
        save_mode(args.output_dir, "full", payload)
        print(
            f"Saved full results to {args.output_dir / 'full'}"
        )

    if "occurring" in modes:
        occurring_candidates = [
            sorted(set(tokens))
            for tokens in full_generation.token_ids
        ]
        occurring_generation = rescore_sequences(
            predictor,
            full_generation.token_ids,
            start_length,
            occurring_candidates,
            context_length=args.context_length,
            retain_tokens=args.retain_tokens,
            use_kv_cache=args.use_kv_cache,
        )
        payload = make_payload(occurring_generation, tokenizer)
        payload["metadata"] = dict(
            shared_metadata,
            candidate_mode="occurring",
            scoring_dictionary=(
                "distinct tokens in each generated "
                "worst-case sequence"
            ),
            mask_application="post-hoc rescoring",
            source_variant="full",
            same_token_ids_as_full=True,
        )
        save_mode(args.output_dir, "occurring", payload)
        print(
            "Saved post-hoc occurring-token results to "
            f"{args.output_dir / 'occurring'}"
        )


if __name__ == "__main__":
    main()
