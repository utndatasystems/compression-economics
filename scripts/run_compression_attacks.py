#!/usr/bin/env python3
"""Run matched token-, byte-, and coder-oriented compression attacks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.generate_adversarial import (
    ascii_byte_token_ids,
    encode_starts,
    make_predictor_args,
    make_round_trip_validator,
    token_ids_round_trip,
)
from src.adversarial import AdversarialGeneration, rescore_sequences
from src.compression_attacks import (
    AttackObjective,
    classical_compression_baselines,
    decoded_utf8_increment,
    generate_beam_search_sequence,
    generate_greedy_sequences,
    generate_random_sequences,
    score_arithmetic_payloads,
    score_full_vocab_sequences,
)
from src.prediction import TokenPredictor


ATTACKS = (
    "random-token",
    "min-probability",
    "surprisal-per-byte",
    "beam-surprisal-per-byte",
    "beam-actual-ratio",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-0.5B")
    starts = parser.add_mutually_exclusive_group(required=True)
    starts.add_argument("--start-token-id", type=int, action="append")
    starts.add_argument("--start-text", action="append")
    parser.add_argument("--total-length", type=int, default=100)
    parser.add_argument(
        "--generation-alphabet",
        choices=("full", "printable-ascii", "ascii-bytes"),
        default="ascii-bytes",
        help="Candidate alphabet. Byte alphabets make denominator control explicit.",
    )
    parser.add_argument("--attack", choices=ATTACKS, action="append")
    parser.add_argument("--context-length", type=int, default=1000)
    parser.add_argument("--retain-tokens", type=int, default=100)
    parser.add_argument("--beam-width", type=int, default=4)
    parser.add_argument("--branch-factor", type=int, default=8)
    parser.add_argument(
        "--fixed-overhead-bits",
        type=int,
        default=0,
        help="Measured header plus bitmap bits included in actual-ratio search.",
    )
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument(
        "--ordinary-text",
        type=Path,
        action="append",
        help="Optional ordinary UTF-8 input (for example data/text8).",
    )
    parser.add_argument(
        "--random-utf8-bytes",
        type=int,
        default=0,
        help="Add a random printable-ASCII byte control of this size.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/runs/compression-attacks"),
    )
    return parser.parse_args()


def _combine(generations: list[AdversarialGeneration]) -> AdversarialGeneration:
    return AdversarialGeneration(
        token_ids=[item.token_ids[0] for item in generations],
        model_log_probabilities=[
            item.model_log_probabilities[0] for item in generations
        ],
        masked_log_probabilities=[
            item.masked_log_probabilities[0] for item in generations
        ],
        candidate_sizes=[item.candidate_sizes[0] for item in generations],
    )


def _decode(tokenizer, tokens: list[int]) -> str:
    return tokenizer.decode(
        tokens,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )


def _run_attack(
    name: str,
    predictor,
    starts: list[list[int]],
    candidates: list[int],
    args: argparse.Namespace,
    byte_increment,
    validator,
) -> AdversarialGeneration:
    candidate_sets = [candidates for _ in starts]
    if name == "random-token":
        sequences = generate_random_sequences(
            starts,
            args.total_length,
            candidate_sets,
            seed=args.random_seed,
            candidate_validator=validator,
        )
        if any(not token_ids_round_trip(predictor.tokenizer, row) for row in sequences):
            raise RuntimeError(
                "Random-token control did not round-trip; use a canonical byte alphabet"
            )
        return rescore_sequences(
            predictor,
            sequences,
            start_length=len(starts[0]),
            candidate_sets=candidate_sets,
            context_length=args.context_length,
            retain_tokens=args.retain_tokens,
            use_kv_cache=False,
        )
    if name in {"min-probability", "surprisal-per-byte"}:
        return generate_greedy_sequences(
            predictor,
            starts,
            args.total_length,
            candidate_sets,
            context_length=args.context_length,
            retain_tokens=args.retain_tokens,
            objective=AttackObjective(name),
            byte_increment=byte_increment,
            candidate_validator=validator,
        )

    objective = (
        AttackObjective.ACTUAL_RATIO
        if name == "beam-actual-ratio"
        else AttackObjective.SURPRISAL_PER_BYTE
    )
    return _combine([
        generate_beam_search_sequence(
            predictor,
            start,
            args.total_length,
            candidates,
            context_length=args.context_length,
            retain_tokens=args.retain_tokens,
            byte_increment=byte_increment,
            initial_decoded_bytes=len(
                _decode(predictor.tokenizer, start).encode("utf-8")
            ),
            objective=objective,
            beam_width=args.beam_width,
            branch_factor=args.branch_factor,
            fixed_overhead_bits=args.fixed_overhead_bits,
            candidate_validator=validator,
        )
        for start in starts
    ])


def _result_rows(
    name: str,
    generation: AdversarialGeneration,
    predictor,
    args: argparse.Namespace,
) -> list[dict]:
    payload_bits = score_arithmetic_payloads(
        predictor,
        generation.token_ids,
        start_length=(
            len(generation.token_ids[0])
            - len(generation.model_log_probabilities[0])
        ),
        context_length=args.context_length,
        retain_tokens=args.retain_tokens,
    )
    rows = []
    for run_index, (tokens, model_logs, bits) in enumerate(zip(
        generation.token_ids,
        generation.model_log_probabilities,
        payload_bits,
    )):
        text = _decode(predictor.tokenizer, tokens)
        raw = text.encode("utf-8")
        entropy_bits = -sum(model_logs) / __import__("math").log(2)
        serialized_bits = bits + args.fixed_overhead_bits
        rows.append({
            "condition": name,
            "run_index": run_index,
            "token_ids": tokens,
            "decoded_text": text,
            "decoded_utf8_size_bytes": len(raw),
            "generated_tokens": len(model_logs),
            "model_entropy_bits": entropy_bits,
            "model_entropy_ratio": entropy_bits / (8 * len(raw)),
            "arithmetic_payload_bits": bits,
            "arithmetic_payload_ratio": bits / (8 * len(raw)),
            "fixed_overhead_bits": args.fixed_overhead_bits,
            "serialized_size_bits": serialized_bits,
            "serialized_compression_ratio": serialized_bits / (8 * len(raw)),
            "classical_baselines": classical_compression_baselines(raw),
        })
    return rows


def _score_text_control(
    name: str,
    text: str,
    predictor,
    args: argparse.Namespace,
) -> list[dict]:
    """Score an ordinary UTF-8 control through the same model and coder."""
    tokens = predictor.tokenizer.encode(text, add_special_tokens=False)
    tokens = tokens[: args.total_length]
    if len(tokens) < 2:
        raise ValueError(f"Control {name!r} encodes to fewer than two tokens")
    generation = score_full_vocab_sequences(
        predictor,
        [tokens],
        start_length=1,
        context_length=args.context_length,
        retain_tokens=args.retain_tokens,
    )
    return _result_rows(name, generation, predictor, args)


def _read_text_prefix(path: Path, tokenizer, token_limit: int) -> str:
    """Read enough source text for a token-limited control without loading all."""
    text = ""
    chunk_size = max(65_536, token_limit * 8)
    with path.open("r", encoding="utf-8", newline="") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                return text
            text += chunk
            token_ids = tokenizer.encode(text, add_special_tokens=False)
            if len(token_ids) >= token_limit:
                return text


def _write_rows(output_dir: Path, rows: list[dict], metadata: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(
        json.dumps({"metadata": metadata, "runs": rows}, indent=2),
        encoding="utf-8",
    )
    for row in rows:
        stem = f"{row['condition']}_run_{row['run_index']:02d}"
        with (output_dir / f"{stem}.txt").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            handle.write(row["decoded_text"])
        (output_dir / f"{stem}.tokens.json").write_text(
            json.dumps(row["token_ids"]), encoding="utf-8"
        )


def main() -> None:
    args = parse_args()
    if args.fixed_overhead_bits < 0:
        raise ValueError("--fixed-overhead-bits must be non-negative")
    attacks = list(dict.fromkeys(args.attack or ATTACKS))
    predictor = TokenPredictor(make_predictor_args(args), bitmap_data=None)
    tokenizer = predictor.tokenizer
    starts = encode_starts(args, tokenizer)
    vocab_size = min(tokenizer.vocab_size, predictor.model.config.vocab_size)
    if args.generation_alphabet == "full":
        excluded = set(tokenizer.all_special_ids)
        candidates = [
            token_id for token_id in range(vocab_size)
            if token_id not in excluded
        ]
    else:
        candidates = ascii_byte_token_ids(
            tokenizer,
            printable_only=args.generation_alphabet == "printable-ascii",
        )
    byte_increment = decoded_utf8_increment(tokenizer)
    validator = make_round_trip_validator(tokenizer)

    rows = []
    for attack in attacks:
        print(f"Running {attack}...")
        generation = _run_attack(
            attack,
            predictor,
            starts,
            candidates,
            args,
            byte_increment,
            validator,
        )
        rows.extend(_result_rows(attack, generation, predictor, args))

    control_names = []
    for path in args.ordinary_text or []:
        text = _read_text_prefix(path, tokenizer, args.total_length)
        name = f"ordinary-{path.stem}"
        rows.extend(_score_text_control(name, text, predictor, args))
        control_names.append(name)
    if args.random_utf8_bytes:
        rng = random.Random(args.random_seed)
        text = "".join(
            chr(rng.randrange(32, 127))
            for _ in range(args.random_utf8_bytes)
        )
        name = "random-printable-utf8"
        rows.extend(_score_text_control(name, text, predictor, args))
        control_names.append(name)

    metadata = {
        "model_name": args.model_name,
        "attacks": attacks,
        "total_length": args.total_length,
        "generation_alphabet": args.generation_alphabet,
        "generation_candidate_size": len(candidates),
        "beam_width": args.beam_width,
        "branch_factor": args.branch_factor,
        "fixed_overhead_bits": args.fixed_overhead_bits,
        "random_seed": args.random_seed,
        "controls": control_names,
        "actual_ratio_definition": (
            "(finalized arithmetic payload bits + fixed overhead bits) / "
            "(8 * decoded UTF-8 bytes)"
        ),
    }
    _write_rows(args.output_dir, rows, metadata)
    print(f"Saved matched attack results to {args.output_dir}")


if __name__ == "__main__":
    main()
