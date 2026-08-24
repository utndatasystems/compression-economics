#!/usr/bin/env python3
"""Generate full-vocabulary worst cases and post-hoc mask rescoring."""

from __future__ import annotations

import argparse
from bisect import bisect_left
from collections import OrderedDict
from dataclasses import dataclass
import json
import os
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
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=250,
        help="Save generation progress after this many additional tokens.",
    )
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Ignore a compatible full/results.json checkpoint.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an incompatible or completed checkpoint.",
    )
    parser.set_defaults(resume=True)
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


def _byte_level_decoder():
    byte_values = [
        *range(ord("!"), ord("~") + 1),
        *range(161, 173),
        *range(174, 256),
    ]
    code_points = list(byte_values)
    for byte_value in range(256):
        if byte_value not in byte_values:
            byte_values.append(byte_value)
            code_points.append(256 + len(code_points) - 188)
    return {
        chr(code_point): byte_value
        for byte_value, code_point in zip(byte_values, code_points)
    }


@dataclass(frozen=True)
class _PrefixState:
    raw: bytes
    text: str
    token_byte_offsets: tuple[int, ...]
    char_byte_offsets: tuple[int, ...]
    pieces: tuple[tuple[str, tuple[int, int]], ...]


class CertifiedRoundTripValidator:
    """Certified tokenizer-local validation with a full-prefix fallback."""

    def __init__(self, tokenizer, *, max_local_bytes=512, cache_size=256):
        self.tokenizer = tokenizer
        self.max_local_bytes = max_local_bytes
        self.cache_size = cache_size
        self.local_checks = 0
        self.fallback_checks = 0
        self._states = OrderedDict()
        self._token_bytes = {}
        self._active_prefixes = OrderedDict()
        self._byte_decoder = _byte_level_decoder()
        backend = getattr(tokenizer, "backend_tokenizer", None)
        self._pre_tokenizer = (
            getattr(backend, "pre_tokenizer", None) if backend else None
        )
        self._special_ids = set(getattr(tokenizer, "all_special_ids", ()))

    def _raw_token(self, token_id):
        if token_id in self._token_bytes:
            return self._token_bytes[token_id]
        raw = None
        if token_id not in self._special_ids:
            token = self.tokenizer.convert_ids_to_tokens(token_id)
            if isinstance(token, str):
                try:
                    raw = bytes(self._byte_decoder[char] for char in token)
                except KeyError:
                    pass
        self._token_bytes[token_id] = raw
        return raw

    def _remember(self, key, state):
        self._states[key] = state
        self._states.move_to_end(key)
        while len(self._states) > self.cache_size:
            self._states.popitem(last=False)

    def _build_state(self, token_ids):
        raw_tokens = [self._raw_token(token_id) for token_id in token_ids]
        if any(raw is None for raw in raw_tokens):
            return None
        offsets = [0]
        for raw_token in raw_tokens:
            offsets.append(offsets[-1] + len(raw_token))
        raw = b"".join(raw_tokens)
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            return None
        decoded = self.tokenizer.decode(
            token_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        if text != decoded or self._pre_tokenizer is None:
            return None
        pieces = tuple(self._pre_tokenizer.pre_tokenize_str(text))
        char_offsets = tuple(self._char_byte_offsets(text))
        return _PrefixState(raw, text, tuple(offsets), char_offsets, pieces)

    def _state(self, token_ids):
        state = self._states.get(token_ids)
        if state is None:
            state = self._build_state(token_ids)
            if state is not None:
                self._remember(token_ids, state)
        else:
            self._states.move_to_end(token_ids)
        return state

    @staticmethod
    def _char_byte_offsets(text):
        offsets = [0]
        for char in text:
            offsets.append(offsets[-1] + len(char.encode("utf-8")))
        return offsets

    def _local_result(self, state, prefix, candidate_token_id):
        candidate_raw = self._raw_token(candidate_token_id)
        if state is None or candidate_raw is None or len(state.pieces) < 2:
            return None
        try:
            candidate_text = candidate_raw.decode("utf-8")
        except UnicodeDecodeError:
            return None

        checkpoint_char = state.pieces[-2][1][0]
        checkpoint_byte = state.char_byte_offsets[checkpoint_char]
        if len(state.raw) - checkpoint_byte > self.max_local_bytes:
            return None
        token_index = bisect_left(state.token_byte_offsets, checkpoint_byte)
        if (
            token_index == len(state.token_byte_offsets)
            or state.token_byte_offsets[token_index] != checkpoint_byte
        ):
            return None

        old_suffix = state.text[checkpoint_char:]
        if not old_suffix or ord(old_suffix[0]) >= 128:
            return None
        new_suffix = old_suffix + candidate_text
        old_pieces = tuple(self._pre_tokenizer.pre_tokenize_str(old_suffix))
        new_pieces = tuple(self._pre_tokenizer.pre_tokenize_str(new_suffix))
        expected_pieces = tuple(
            (piece, (start - checkpoint_char, end - checkpoint_char))
            for piece, (start, end) in state.pieces[-2:]
        )
        if (
            len(old_pieces) < 2
            or old_pieces != expected_pieces
            or not new_pieces
            or old_pieces[0] != new_pieces[0]
        ):
            return None

        expected_old = list(prefix[token_index:])
        if self.tokenizer.encode(
            old_suffix, add_special_tokens=False
        ) != expected_old:
            return None
        expected_new = [*expected_old, candidate_token_id]
        accepted = self.tokenizer.encode(
            new_suffix, add_special_tokens=False
        ) == expected_new
        adjusted_pieces = tuple(
            (piece, (start + checkpoint_char, end + checkpoint_char))
            for piece, (start, end) in new_pieces
        )
        candidate_char_offsets = []
        byte_offset = len(state.raw)
        for char in candidate_text:
            byte_offset += len(char.encode("utf-8"))
            candidate_char_offsets.append(byte_offset)
        child = _PrefixState(
            raw=state.raw + candidate_raw,
            text=state.text + candidate_text,
            token_byte_offsets=(
                *state.token_byte_offsets,
                len(state.raw) + len(candidate_raw),
            ),
            char_byte_offsets=(
                *state.char_byte_offsets,
                *candidate_char_offsets,
            ),
            pieces=(*state.pieces[:-2], *adjusted_pieces),
        )
        return accepted, child

    def __call__(self, _row_index, prefix, candidate_token_id):
        identity = (id(prefix), len(prefix))
        active = self._active_prefixes.get(identity)
        if (
            active is not None
            and active[0] is prefix
            and (active[3] is None or prefix[-1] == active[3])
        ):
            _, prefix_key, state, _ = active
            self._active_prefixes.move_to_end(identity)
        else:
            prefix_key = tuple(prefix)
            state = self._state(prefix_key)
            self._active_prefixes[identity] = (
                prefix, prefix_key, state, None
            )
            while len(self._active_prefixes) > 64:
                self._active_prefixes.popitem(last=False)
        local = self._local_result(state, prefix_key, candidate_token_id)
        if local is not None:
            self.local_checks += 1
            accepted, child = local
            if accepted:
                child_key = (*prefix_key, candidate_token_id)
                self._remember(child_key, child)
                next_identity = (id(prefix), len(prefix) + 1)
                self._active_prefixes[next_identity] = (
                    prefix, child_key, child, candidate_token_id
                )
                while len(self._active_prefixes) > 64:
                    self._active_prefixes.popitem(last=False)
            return accepted

        child_key = (*prefix_key, candidate_token_id)
        self.fallback_checks += 1
        accepted = token_ids_round_trip(self.tokenizer, child_key)
        if accepted:
            child = self._build_state(child_key)
            if child is not None:
                self._remember(child_key, child)
                next_identity = (id(prefix), len(prefix) + 1)
                self._active_prefixes[next_identity] = (
                    prefix, child_key, child, candidate_token_id
                )
                while len(self._active_prefixes) > 64:
                    self._active_prefixes.popitem(last=False)
        return accepted


def make_round_trip_validator(tokenizer, *, max_local_bytes=512):
    """Build a certified local validator with a full-prefix fallback."""
    return CertifiedRoundTripValidator(
        tokenizer, max_local_bytes=max_local_bytes
    )


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
    result_path = mode_dir / "results.json"
    temporary_path = mode_dir / ".results.json.tmp"
    temporary_path.write_text(
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
    os.replace(temporary_path, result_path)


def _generation_from_payload(
    payload: dict, candidate_size: int
) -> AdversarialGeneration:
    return AdversarialGeneration(
        token_ids=[run["token_ids"] for run in payload["runs"]],
        model_log_probabilities=[
            run["model_log_probabilities"] for run in payload["runs"]
        ],
        masked_log_probabilities=[
            run["masked_log_probabilities"] for run in payload["runs"]
        ],
        candidate_sizes=[candidate_size for _ in payload["runs"]],
    )


def _combine_generation(
    previous: AdversarialGeneration,
    extension: AdversarialGeneration,
    candidate_size: int,
) -> AdversarialGeneration:
    return AdversarialGeneration(
        token_ids=extension.token_ids,
        model_log_probabilities=[
            [*old, *new]
            for old, new in zip(
                previous.model_log_probabilities,
                extension.model_log_probabilities,
            )
        ],
        masked_log_probabilities=[
            [*old, *new]
            for old, new in zip(
                previous.masked_log_probabilities,
                extension.masked_log_probabilities,
            )
        ],
        candidate_sizes=[candidate_size for _ in extension.token_ids],
    )


def _as_full_scored(
    generation: AdversarialGeneration,
    *,
    generation_alphabet: str,
    full_candidate_size: int,
) -> AdversarialGeneration:
    if generation_alphabet == "full":
        return generation
    return AdversarialGeneration(
        token_ids=generation.token_ids,
        model_log_probabilities=generation.model_log_probabilities,
        masked_log_probabilities=[
            row[:] for row in generation.model_log_probabilities
        ],
        candidate_sizes=[full_candidate_size for _ in generation.token_ids],
    )


def main() -> None:
    args = parse_args()
    if args.checkpoint_every <= 0:
        raise ValueError("--checkpoint-every must be positive")
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
    shared_metadata = {
        "model_name": args.model_name,
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

    checkpoint_path = args.output_dir / "full" / "results.json"
    full_generation = None
    if checkpoint_path.exists() and args.resume and not args.force:
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        stored = checkpoint.get("metadata", {})
        stable_keys = (
            "model_name",
            "context_length",
            "retain_tokens",
            "use_kv_cache",
            "include_special_tokens",
            "sequence_generation_dictionary",
            "generation_candidate_size",
        )
        mismatches = {
            key: (stored.get(key), shared_metadata.get(key))
            for key in stable_keys
            if stored.get(key) != shared_metadata.get(key)
        }
        stored_starts = [run["start_token_ids"] for run in checkpoint["runs"]]
        if stored_starts != starts:
            mismatches["start_token_ids"] = (stored_starts, starts)
        if mismatches:
            details = ", ".join(
                f"{key}: stored={old!r}, requested={new!r}"
                for key, (old, new) in mismatches.items()
            )
            raise ValueError(
                f"Incompatible checkpoint at {checkpoint_path}: {details}. "
                "Choose another --output-dir or pass --force."
            )
        full_generation = _generation_from_payload(
            checkpoint, len(generation_candidates)
        )
        completed_length = len(full_generation.token_ids[0])
        if completed_length > args.total_length:
            raise ValueError(
                f"Checkpoint length {completed_length} exceeds requested "
                f"length {args.total_length}; choose another output directory"
            )
        print(
            f"Resuming full generation at token {completed_length} "
            f"of {args.total_length}."
        )

    if full_generation is None:
        full_generation = AdversarialGeneration(
            token_ids=[start[:] for start in starts],
            model_log_probabilities=[[] for _ in starts],
            masked_log_probabilities=[[] for _ in starts],
            candidate_sizes=[len(generation_candidates) for _ in starts],
        )

    while len(full_generation.token_ids[0]) < args.total_length:
        completed_length = len(full_generation.token_ids[0])
        checkpoint_length = min(
            args.total_length, completed_length + args.checkpoint_every
        )
        extension = generate_worst_case_sequences(
            predictor,
            full_generation.token_ids,
            checkpoint_length,
            [generation_candidates for _ in starts],
            context_length=args.context_length,
            retain_tokens=args.retain_tokens,
            use_kv_cache=args.use_kv_cache,
            candidate_validator=round_trip_validator,
        )
        full_generation = _combine_generation(
            full_generation, extension, len(generation_candidates)
        )
        full_scored_generation = _as_full_scored(
            full_generation,
            generation_alphabet=args.generation_alphabet,
            full_candidate_size=len(full_candidates),
        )
        payload = make_payload(full_scored_generation, tokenizer)
        payload["metadata"] = dict(
            shared_metadata,
            total_length=checkpoint_length,
            requested_total_length=args.total_length,
            checkpoint_every=args.checkpoint_every,
            candidate_mode="full",
            scoring_dictionary="full",
            mask_application=(
                "generation alphabet applied during construction; "
                "full vocabulary used for probability scoring"
            ),
        )
        save_mode(args.output_dir, "full", payload)
        print(
            f"Checkpointed full generation at {checkpoint_length} tokens."
        )

    full_scored_generation = _as_full_scored(
        full_generation,
        generation_alphabet=args.generation_alphabet,
        full_candidate_size=len(full_candidates),
    )
    if "full" in modes and not checkpoint_path.exists():
        payload = make_payload(full_scored_generation, tokenizer)
        payload["metadata"] = dict(
            shared_metadata,
            total_length=args.total_length,
            requested_total_length=args.total_length,
            checkpoint_every=args.checkpoint_every,
            candidate_mode="full",
            scoring_dictionary="full",
            mask_application=(
                "generation alphabet applied during construction; "
                "full vocabulary used for probability scoring"
            ),
        )
        save_mode(args.output_dir, "full", payload)

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
