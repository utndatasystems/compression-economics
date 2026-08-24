"""Compression-oriented attacks and matched byte-compressor baselines.

Token surprisal is not a compression ratio: tokenizer symbols represent unequal
amounts of source text.  The attacks in this module optimize objectives whose
denominator is the decoded UTF-8 size.  They deliberately live beside the older
minimum-probability generator so existing experiments remain reproducible.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
import gzip
import math
import random
from typing import Callable, Protocol, Sequence

import brotli
import torch
import zstandard
from tqdm.auto import tqdm

from src.adversarial import AdversarialGeneration, normalize_candidate_ids
from src.encoding import LLMCompressor, LLMDecompressor


CandidateValidator = Callable[[int, Sequence[int], int], bool]
DecodedByteIncrement = Callable[[Sequence[int], int], int]


class AttackObjective(str, Enum):
    """Supported sequence-construction objectives."""

    MIN_PROBABILITY = "min-probability"
    SURPRISAL_PER_BYTE = "surprisal-per-byte"
    ACTUAL_RATIO = "actual-ratio"


class BatchedPredictor(Protocol):
    def run_batched_inference(
        self, prompts: list[list[int]], enable_kv_cache: bool = True
    ) -> tuple[list[int], torch.Tensor, float, float]: ...

    def reset_kv_cache(self) -> None: ...


@dataclass
class _Beam:
    token_ids: list[int]
    model_logs: list[float]
    masked_logs: list[float]
    decoded_bytes: int
    compressor: LLMCompressor | None
    realized_bits: int = 0


def decoded_utf8_increment(tokenizer) -> DecodedByteIncrement:
    """Build a context-sensitive decoded-byte measurement function.

    Tokenizers may merge whitespace or emit replacement characters when a token
    is decoded alone.  Measuring the complete prefix before and after extension
    avoids silently assigning the wrong source length to a candidate.
    """

    def increment(prefix: Sequence[int], candidate: int) -> int:
        decode_options = {
            "skip_special_tokens": False,
            "clean_up_tokenization_spaces": False,
        }
        before = tokenizer.decode(list(prefix), **decode_options).encode("utf-8")
        after = tokenizer.decode(
            [*prefix, candidate], **decode_options
        ).encode("utf-8")
        return len(after) - len(before)

    return increment


def _full_vocab_logits(
    score_token_ids: Sequence[int], logits: torch.Tensor
) -> torch.Tensor:
    expected = list(range(len(score_token_ids)))
    if list(score_token_ids) != expected or len(expected) > logits.shape[1]:
        raise ValueError(
            "Compression attacks require full-vocabulary logits in token-ID order"
        )
    return logits[:, : len(expected)]


def _candidate_scores(
    row: torch.Tensor, candidate_ids: Sequence[int]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    candidates = normalize_candidate_ids(candidate_ids, row.shape[0])
    indices = torch.tensor(candidates, dtype=torch.long, device=row.device)
    candidate_logits = row.index_select(0, indices).float()
    finite = torch.isfinite(candidate_logits)
    if not finite.any():
        raise ValueError("No candidate token has a finite model logit")
    finite_ids = indices[finite]
    finite_logits = candidate_logits[finite]
    full_logs = torch.log_softmax(row.float(), dim=0).index_select(0, finite_ids)
    masked_logs = torch.log_softmax(finite_logits, dim=0)
    return finite_ids, full_logs, masked_logs


def rank_candidates(
    row: torch.Tensor,
    candidate_ids: Sequence[int],
    prefix: Sequence[int],
    *,
    objective: AttackObjective | str,
    byte_increment: DecodedByteIncrement | None = None,
) -> list[tuple[int, float, float, int]]:
    """Return candidates best-first with scores needed by the caller.

    Each tuple contains ``(token_id, full_log_probability,
    masked_log_probability, added_bytes)``.  Token ID is the deterministic
    tie-break, making runs reproducible across devices.
    """
    objective = AttackObjective(objective)
    if objective is AttackObjective.ACTUAL_RATIO:
        raise ValueError("actual-ratio requires beam search with coder state")
    token_ids, full_logs, masked_logs = _candidate_scores(row, candidate_ids)
    ranked: list[tuple[float, int, float, float, int]] = []
    for token_id, model_log, masked_log in zip(
        token_ids.tolist(), full_logs.tolist(), masked_logs.tolist()
    ):
        added_bytes = 0
        if objective is AttackObjective.SURPRISAL_PER_BYTE:
            if byte_increment is None:
                raise ValueError(
                    "byte_increment is required for surprisal-per-byte"
                )
            added_bytes = byte_increment(prefix, token_id)
            if added_bytes <= 0:
                continue
            score = -model_log / (math.log(2) * added_bytes)
        else:
            score = -model_log
        ranked.append(
            (-score, token_id, model_log, masked_log, added_bytes)
        )
    return [
        (token_id, model_log, masked_log, added_bytes)
        for _, token_id, model_log, masked_log, added_bytes in sorted(ranked)
    ]


def generate_greedy_sequences(
    predictor: BatchedPredictor,
    start_sequences: Sequence[Sequence[int]],
    total_length: int,
    candidate_sets: Sequence[Sequence[int]],
    *,
    context_length: int,
    retain_tokens: int,
    objective: AttackObjective | str,
    byte_increment: DecodedByteIncrement | None = None,
    candidate_validator: CandidateValidator | None = None,
    use_kv_cache: bool = True,
    progress_desc: str | None = None,
) -> AdversarialGeneration:
    """Generate sequences with a token- or byte-level greedy objective."""
    objective = AttackObjective(objective)
    if objective is AttackObjective.ACTUAL_RATIO:
        raise ValueError("Use generate_beam_search_sequence for actual-ratio")
    sequences = [list(sequence) for sequence in start_sequences]
    if not sequences or any(not sequence for sequence in sequences):
        raise ValueError("At least one non-empty start sequence is required")
    if len({len(sequence) for sequence in sequences}) != 1:
        raise ValueError("All start sequences must have equal length")
    if total_length < len(sequences[0]):
        raise ValueError("total_length must include the start sequence")
    if len(candidate_sets) != len(sequences):
        raise ValueError("One candidate set is required per sequence")
    if not 0 < retain_tokens <= context_length:
        raise ValueError("Require 0 < retain_tokens <= context_length")

    model_logs = [[] for _ in sequences]
    masked_logs = [[] for _ in sequences]
    contexts = [sequence[:] for sequence in sequences]
    predictor.reset_kv_cache()
    progress = tqdm(
        total=total_length - len(sequences[0]),
        desc=progress_desc,
        unit="step",
        dynamic_ncols=True,
        disable=progress_desc is None,
    )
    while len(sequences[0]) < total_length:
        if len(contexts[0]) >= context_length:
            contexts = [context[-retain_tokens:] for context in contexts]
        token_ids, logits, _, _ = predictor.run_batched_inference(
            contexts, enable_kv_cache=use_kv_cache
        )
        logits = _full_vocab_logits(token_ids, logits)
        for row_index, (row, candidates) in enumerate(
            zip(logits, candidate_sets)
        ):
            ranked = rank_candidates(
                row,
                candidates,
                sequences[row_index],
                objective=objective,
                byte_increment=byte_increment,
            )
            chosen = next(
                (
                    item
                    for item in ranked
                    if candidate_validator is None
                    or candidate_validator(
                        row_index, sequences[row_index], item[0]
                    )
                ),
                None,
            )
            if chosen is None:
                raise ValueError("No finite candidate satisfies the constraints")
            token_id, model_log, masked_log, _ = chosen
            sequences[row_index].append(token_id)
            contexts[row_index].append(token_id)
            model_logs[row_index].append(model_log)
            masked_logs[row_index].append(masked_log)
        progress.update(1)

    progress.close()
    return AdversarialGeneration(
        token_ids=sequences,
        model_log_probabilities=model_logs,
        masked_log_probabilities=masked_logs,
        candidate_sizes=[len(set(items)) for items in candidate_sets],
    )


def generate_random_sequences(
    start_sequences: Sequence[Sequence[int]],
    total_length: int,
    candidate_sets: Sequence[Sequence[int]],
    *,
    seed: int = 0,
    candidate_validator: CandidateValidator | None = None,
    progress_desc: str | None = None,
) -> list[list[int]]:
    """Generate a seeded uniform control over valid token extensions."""
    if len(start_sequences) != len(candidate_sets):
        raise ValueError("One candidate set is required per start sequence")
    generator = random.Random(seed)
    results = []
    progress = tqdm(
        total=sum(total_length - len(start) for start in start_sequences),
        desc=progress_desc,
        unit="token",
        dynamic_ncols=True,
        disable=progress_desc is None,
    )
    for row_index, (start, raw_candidates) in enumerate(
        zip(start_sequences, candidate_sets)
    ):
        candidates = sorted(set(raw_candidates))
        if not start or not candidates:
            raise ValueError("Starts and candidate sets must not be empty")
        if total_length < len(start):
            raise ValueError("total_length must include the start sequence")
        sequence = list(start)
        while len(sequence) < total_length:
            # Draw a lazy Fisher-Yates permutation. The old implementation
            # copied and shuffled the complete candidate vocabulary at every
            # position, even though the first valid candidate is usually found
            # after only a few draws. This retains uniform sampling without
            # replacement while doing work proportional to the candidates that
            # are actually inspected.
            swaps: dict[int, int] = {}
            remaining = len(candidates)
            chosen = None
            while remaining:
                offset = generator.randrange(remaining)
                candidate_index = swaps.get(offset, offset)
                remaining -= 1
                swaps[offset] = swaps.get(remaining, remaining)
                token_id = candidates[candidate_index]
                if (
                    candidate_validator is None
                    or candidate_validator(row_index, sequence, token_id)
                ):
                    chosen = token_id
                    break
            if chosen is None:
                raise ValueError("No random candidate satisfies the constraints")
            sequence.append(chosen)
            progress.update(1)
        results.append(sequence)
    progress.close()
    return results


def _finished_code_length(compressor: LLMCompressor) -> int:
    return len(deepcopy(compressor).compress())


def generate_beam_search_sequence(
    predictor: BatchedPredictor,
    start_sequence: Sequence[int],
    total_length: int,
    candidate_ids: Sequence[int],
    *,
    context_length: int,
    retain_tokens: int,
    byte_increment: DecodedByteIncrement,
    initial_decoded_bytes: int,
    objective: AttackObjective | str = AttackObjective.SURPRISAL_PER_BYTE,
    beam_width: int = 4,
    branch_factor: int = 8,
    fixed_overhead_bits: int = 0,
    candidate_validator: CandidateValidator | None = None,
    progress_desc: str | None = None,
) -> AdversarialGeneration:
    """Optimize the sequence-level entropy or realized arithmetic-code ratio.

    The realized objective clones the current arithmetic-coder state for each
    extension and ranks finalized payload lengths.  Fixed header and bitmap costs
    can be included when they are identical for every search branch.  KV caching
    is disabled because beam ancestry changes after every pruning step.
    """
    objective = AttackObjective(objective)
    candidates = sorted(set(candidate_ids))
    if not start_sequence or not candidates:
        raise ValueError("start_sequence and candidate_ids must not be empty")
    if total_length < len(start_sequence):
        raise ValueError("total_length must include the start sequence")
    if not 0 < retain_tokens <= context_length:
        raise ValueError("Require 0 < retain_tokens <= context_length")
    if beam_width <= 0 or branch_factor <= 0:
        raise ValueError("beam_width and branch_factor must be positive")
    if initial_decoded_bytes <= 0 or fixed_overhead_bits < 0:
        raise ValueError("Byte size must be positive and overhead non-negative")

    beams = [_Beam(
        token_ids=list(start_sequence),
        model_logs=[],
        masked_logs=[],
        decoded_bytes=initial_decoded_bytes,
        compressor=(
            LLMCompressor()
            if objective is AttackObjective.ACTUAL_RATIO
            else None
        ),
    )]
    predictor.reset_kv_cache()
    progress = tqdm(
        total=total_length - len(beams[0].token_ids),
        desc=progress_desc,
        unit="step",
        dynamic_ncols=True,
        disable=progress_desc is None,
    )
    while len(beams[0].token_ids) < total_length:
        contexts = [
            beam.token_ids[-retain_tokens:]
            if len(beam.token_ids) >= context_length
            else beam.token_ids
            for beam in beams
        ]
        token_ids, logits, _, _ = predictor.run_batched_inference(
            contexts, enable_kv_cache=False
        )
        logits = _full_vocab_logits(token_ids, logits)
        expanded: list[_Beam] = []
        for row_index, (beam, row) in enumerate(zip(beams, logits)):
            proposal_objective = (
                AttackObjective.SURPRISAL_PER_BYTE
                if objective is AttackObjective.ACTUAL_RATIO
                else objective
            )
            proposals = rank_candidates(
                row,
                candidates,
                beam.token_ids,
                objective=proposal_objective,
                byte_increment=byte_increment,
            )
            accepted = 0
            for token_id, model_log, masked_log, added_bytes in proposals:
                if candidate_validator is not None and not candidate_validator(
                    row_index, beam.token_ids, token_id
                ):
                    continue
                if added_bytes == 0:
                    added_bytes = byte_increment(beam.token_ids, token_id)
                if added_bytes <= 0:
                    continue
                compressor = None
                realized_bits = 0
                if beam.compressor is not None:
                    compressor = deepcopy(beam.compressor)
                    probabilities = torch.softmax(row.float(), dim=0).cpu().numpy()
                    compressor.next_token(token_id, probabilities)
                    realized_bits = _finished_code_length(compressor)
                expanded.append(_Beam(
                    token_ids=[*beam.token_ids, token_id],
                    model_logs=[*beam.model_logs, model_log],
                    masked_logs=[*beam.masked_logs, masked_log],
                    decoded_bytes=beam.decoded_bytes + added_bytes,
                    compressor=compressor,
                    realized_bits=realized_bits,
                ))
                accepted += 1
                if accepted == branch_factor:
                    break
        if not expanded:
            raise ValueError("Beam search found no valid extension")

        def score(beam: _Beam) -> tuple[float, tuple[int, ...]]:
            numerator = (
                fixed_overhead_bits + beam.realized_bits
                if objective is AttackObjective.ACTUAL_RATIO
                else -sum(beam.model_logs) / math.log(2)
            )
            tie_break = tuple(-token_id for token_id in beam.token_ids)
            return numerator / (8 * beam.decoded_bytes), tie_break

        beams = sorted(expanded, key=score, reverse=True)[:beam_width]
        progress.update(1)

    progress.close()
    best = beams[0]
    return AdversarialGeneration(
        token_ids=[best.token_ids],
        model_log_probabilities=[best.model_logs],
        masked_log_probabilities=[best.masked_logs],
        candidate_sizes=[len(candidates)],
    )





def score_full_vocab_sequences(
    predictor: BatchedPredictor,
    sequences: Sequence[Sequence[int]],
    *,
    start_length: int,
    context_length: int,
    retain_tokens: int,
    use_kv_cache: bool = True,
    progress_desc: str | None = None,
) -> AdversarialGeneration:
    """Score fixed sequences without rebuilding a full candidate mask per token."""
    fixed = [list(sequence) for sequence in sequences]
    if not fixed or any(len(sequence) <= start_length for sequence in fixed):
        raise ValueError("Every sequence must contain at least one predicted token")
    if len({len(sequence) for sequence in fixed}) != 1:
        raise ValueError("All sequences must have equal length")
    if start_length <= 0 or not 0 < retain_tokens <= context_length:
        raise ValueError("Invalid start or context length")

    contexts = [sequence[:start_length] for sequence in fixed]
    model_logs = [[] for _ in fixed]
    predictor.reset_kv_cache()
    positions = tqdm(
        range(start_length, len(fixed[0])),
        desc=progress_desc,
        unit="step",
        dynamic_ncols=True,
        disable=progress_desc is None,
    )
    for position in positions:
        if len(contexts[0]) >= context_length:
            contexts = [context[-retain_tokens:] for context in contexts]
        token_ids, logits, _, _ = predictor.run_batched_inference(
            contexts, enable_kv_cache=use_kv_cache
        )
        logits = _full_vocab_logits(token_ids, logits)
        log_probabilities = torch.log_softmax(logits.float(), dim=1)
        for row_index, sequence in enumerate(fixed):
            target = sequence[position]
            model_logs[row_index].append(
                float(log_probabilities[row_index, target].item())
            )
            contexts[row_index].append(target)
    return AdversarialGeneration(
        token_ids=fixed,
        model_log_probabilities=model_logs,
        masked_log_probabilities=[row[:] for row in model_logs],
        candidate_sizes=[logits.shape[1] for _ in fixed],
    )


def encode_arithmetic_payloads(
    predictor: BatchedPredictor,
    sequences: Sequence[Sequence[int]],
    *,
    start_length: int,
    context_length: int,
    retain_tokens: int,
    candidate_sets: Sequence[Sequence[int]] | None = None,
    use_kv_cache: bool = True,
    statesize: int = 32,
    frequency_total: int = 262144,
    progress_desc: str | None = None,
) -> list[list[int]]:
    """Replay fixed sequences and return their arithmetic-coded bitstreams.

    When ``candidate_sets`` is supplied, logits are compacted and renormalized
    over each sequence's dictionary before coding.  This matches the occurring-
    token policy instead of merely zeroing entries in the full vocabulary.
    """
    fixed = [list(sequence) for sequence in sequences]
    if not fixed or any(len(sequence) <= start_length for sequence in fixed):
        raise ValueError("Every sequence must contain at least one predicted token")
    if start_length <= 0 or not 0 < retain_tokens <= context_length:
        raise ValueError("Invalid start or context length")
    if candidate_sets is not None and len(candidate_sets) != len(fixed):
        raise ValueError("One candidate set is required per sequence")

    compact_candidates = (
        None
        if candidate_sets is None
        else [
            sorted(set(candidate_set)) for candidate_set in candidate_sets
        ]
    )
    if compact_candidates is not None:
        for sequence, candidates in zip(fixed, compact_candidates):
            if not candidates or candidates[0] < 0:
                raise ValueError("Candidate token set must contain non-negative IDs")
            missing = sorted(set(sequence[start_length:]) - set(candidates))
            if missing:
                raise ValueError(
                    f"Targets are outside their candidate set: {missing[:5]}"
                )
        compact_target_offsets = [
            {token_id: offset for offset, token_id in enumerate(candidates)}
            for candidates in compact_candidates
        ]
    else:
        compact_target_offsets = None

    contexts = [sequence[:start_length] for sequence in fixed]
    compressors = [
        LLMCompressor(statesize=statesize, total=frequency_total)
        for _ in fixed
    ]
    predictor.reset_kv_cache()
    positions = tqdm(
        range(start_length, max(map(len, fixed))),
        desc=progress_desc,
        unit="step",
        dynamic_ncols=True,
        disable=progress_desc is None,
    )
    for position in positions:
        if len(contexts[0]) >= context_length:
            contexts = [context[-retain_tokens:] for context in contexts]
        token_ids, logits, _, _ = predictor.run_batched_inference(
            contexts, enable_kv_cache=use_kv_cache
        )
        logits = _full_vocab_logits(token_ids, logits)
        for row_index, (sequence, row, compressor) in enumerate(
            zip(fixed, logits, compressors)
        ):
            if position >= len(sequence):
                contexts[row_index].append(sequence[-1])
                continue
            target = sequence[position]
            if compact_candidates is None:
                coder_target = target
                coder_logits = row.float()
            else:
                candidates = compact_candidates[row_index]
                indices = torch.tensor(
                    candidates, dtype=torch.long, device=row.device
                )
                coder_logits = row.index_select(0, indices).float()
                coder_target = compact_target_offsets[row_index][target]
            probabilities = torch.softmax(coder_logits, dim=0).cpu().numpy()
            compressor.next_token(coder_target, probabilities)
            contexts[row_index].append(target)
    return [compressor.compress() for compressor in compressors]


def score_arithmetic_payloads(
    predictor: BatchedPredictor,
    sequences: Sequence[Sequence[int]],
    *,
    start_length: int,
    context_length: int,
    retain_tokens: int,
    candidate_sets: Sequence[Sequence[int]] | None = None,
    use_kv_cache: bool = True,
    statesize: int = 32,
    frequency_total: int = 262144,
    progress_desc: str | None = None,
) -> list[int]:
    """Replay fixed sequences and return exact arithmetic payload lengths."""
    payloads = encode_arithmetic_payloads(
        predictor,
        sequences,
        start_length=start_length,
        context_length=context_length,
        retain_tokens=retain_tokens,
        candidate_sets=candidate_sets,
        use_kv_cache=use_kv_cache,
        statesize=statesize,
        frequency_total=frequency_total,
        progress_desc=progress_desc,
    )
    return [len(payload) for payload in payloads]


def decode_arithmetic_payload(
    predictor: BatchedPredictor,
    payload_bits: Sequence[int],
    *,
    seed_token_ids: Sequence[int],
    token_count: int,
    context_length: int,
    retain_tokens: int,
    candidate_token_ids: Sequence[int] | None = None,
    use_kv_cache: bool = True,
    statesize: int = 32,
    frequency_total: int = 262144,
    progress_desc: str | None = None,
) -> list[int]:
    """Decode one Qwen arithmetic payload by replaying model probabilities."""
    tokens = list(seed_token_ids)
    if not tokens or token_count <= len(tokens):
        raise ValueError("token_count must exceed the non-empty seed length")
    if not 0 < retain_tokens <= context_length:
        raise ValueError("Invalid context configuration")

    candidates = (
        None
        if candidate_token_ids is None
        else sorted(set(int(token_id) for token_id in candidate_token_ids))
    )
    if candidates is not None and not candidates:
        raise ValueError("candidate_token_ids must not be empty")

    context = tokens[:]
    decompressor = LLMDecompressor(
        list(payload_bits), statesize=statesize, total=frequency_total
    )
    predictor.reset_kv_cache()
    positions = tqdm(
        range(len(tokens), token_count),
        desc=progress_desc,
        unit="step",
        dynamic_ncols=True,
        disable=progress_desc is None,
    )
    for _ in positions:
        if len(context) >= context_length:
            context = context[-retain_tokens:]
        score_token_ids, logits, _, _ = predictor.run_batched_inference(
            [context], enable_kv_cache=use_kv_cache
        )
        row = _full_vocab_logits(score_token_ids, logits)[0]
        if candidates is None:
            probabilities = torch.softmax(row.float(), dim=0).cpu().numpy()
            decoded_token = decompressor.decompress(probabilities)
        else:
            indices = torch.tensor(
                candidates, dtype=torch.long, device=row.device
            )
            coder_logits = row.index_select(0, indices).float()
            probabilities = torch.softmax(coder_logits, dim=0).cpu().numpy()
            decoded_offset = decompressor.decompress(probabilities)
            decoded_token = candidates[decoded_offset]
        tokens.append(decoded_token)
        context.append(decoded_token)
    return tokens


def decode_arithmetic_payloads(
    predictor: BatchedPredictor,
    payloads: Sequence[Sequence[int]],
    *,
    seed_token_ids: Sequence[Sequence[int]],
    token_counts: Sequence[int],
    context_length: int,
    retain_tokens: int,
    candidate_sets: Sequence[Sequence[int] | None] | None = None,
    use_kv_cache: bool = True,
    statesize: int = 32,
    frequency_total: int = 262144,
    progress_desc: str | None = None,
) -> list[list[int]]:
    """Decode several streams in one batched model replay."""
    if not payloads or not (
        len(payloads) == len(seed_token_ids) == len(token_counts)
    ):
        raise ValueError("payloads, seeds, and token counts must align")
    tokens = [list(seed) for seed in seed_token_ids]
    if any(not seed for seed in tokens):
        raise ValueError("every stream requires a non-empty seed")
    if len({len(seed) for seed in tokens}) != 1:
        raise ValueError("batched stream seeds must have equal lengths")
    if any(count <= len(seed) for count, seed in zip(token_counts, tokens)):
        raise ValueError("every token count must exceed its seed length")
    if not 0 < retain_tokens <= context_length:
        raise ValueError("invalid context configuration")
    raw_candidates = candidate_sets or [None] * len(payloads)
    if len(raw_candidates) != len(payloads):
        raise ValueError("candidate sets must align with payloads")
    candidates = [
        None if items is None else sorted(set(int(item) for item in items))
        for items in raw_candidates
    ]
    if any(items == [] for items in candidates):
        raise ValueError("candidate sets must not be empty")

    contexts = [seed[:] for seed in tokens]
    decompressors = [
        LLMDecompressor(
            list(bits), statesize=statesize, total=frequency_total
        )
        for bits in payloads
    ]
    predictor.reset_kv_cache()
    start_length = len(tokens[0])
    positions = tqdm(
        range(start_length, max(token_counts)),
        desc=progress_desc,
        unit="step",
        dynamic_ncols=True,
        disable=progress_desc is None,
    )
    for position in positions:
        if len(contexts[0]) >= context_length:
            contexts = [context[-retain_tokens:] for context in contexts]
        score_token_ids, logits, _, _ = predictor.run_batched_inference(
            contexts, enable_kv_cache=use_kv_cache
        )
        logits = _full_vocab_logits(score_token_ids, logits)
        for row_index, (row, decoder, items) in enumerate(
            zip(logits, decompressors, candidates)
        ):
            if position >= token_counts[row_index]:
                contexts[row_index].append(tokens[row_index][-1])
                continue
            if items is None:
                probabilities = torch.softmax(row.float(), dim=0).cpu().numpy()
                decoded_token = decoder.decompress(probabilities)
            else:
                indices = torch.tensor(
                    items, dtype=torch.long, device=row.device
                )
                coder_logits = row.index_select(0, indices).float()
                probabilities = torch.softmax(
                    coder_logits, dim=0
                ).cpu().numpy()
                decoded_token = items[decoder.decompress(probabilities)]
            tokens[row_index].append(decoded_token)
            contexts[row_index].append(decoded_token)
    return tokens


def classical_compression_baselines(
    data: bytes,
) -> dict[str, dict[str, float | int]]:
    """Compress identical bytes with gzip, Zstandard, and Brotli."""
    if not data:
        raise ValueError("Baseline input must not be empty")
    outputs: dict[str, bytes] = {
        "gzip-9": gzip.compress(data, compresslevel=9, mtime=0),
        "zstd-22": zstandard.ZstdCompressor(level=22).compress(data),
        "brotli-11": brotli.compress(data, quality=11),
    }

    results: dict[str, dict[str, float | int]] = {}
    for name, compressed in outputs.items():
        results[name] = {
            "raw_size_bytes": len(data),
            "compressed_size_bytes": len(compressed),
            "compression_ratio": len(compressed) / len(data),
        }
    return results
