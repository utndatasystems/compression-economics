"""Worst-probability sequence generation and post-hoc mask rescoring."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Protocol, Sequence

import torch
from tqdm.auto import tqdm


CandidateValidator = Callable[[int, Sequence[int], int], bool]


class BatchedPredictor(Protocol):
    def run_batched_inference(
        self, prompts: list[list[int]], enable_kv_cache: bool = True
    ) -> tuple[list[int], torch.Tensor, float, float]: ...

    def reset_kv_cache(self) -> None: ...


@dataclass
class AdversarialGeneration:
    token_ids: list[list[int]]
    model_log_probabilities: list[list[float]]
    masked_log_probabilities: list[list[float]]
    candidate_sizes: list[int]

    def summary(self) -> list[dict[str, float | int]]:
        rows = []
        for tokens, model_logs, masked_logs, candidate_size in zip(
            self.token_ids,
            self.model_log_probabilities,
            self.masked_log_probabilities,
            self.candidate_sizes,
        ):
            generated = len(model_logs)
            rows.append(
                {
                    "total_tokens": len(tokens),
                    "generated_tokens": generated,
                    "distinct_tokens": len(set(tokens)),
                    "candidate_size": candidate_size,
                    "model_surprisal_bits": -sum(model_logs) / math.log(2),
                    "masked_surprisal_bits": -sum(masked_logs) / math.log(2),
                    "mean_model_surprisal_bits": (
                        -sum(model_logs) / (generated * math.log(2))
                        if generated
                        else 0.0
                    ),
                    "mean_masked_surprisal_bits": (
                        -sum(masked_logs) / (generated * math.log(2))
                        if generated
                        else 0.0
                    ),
                }
            )
        return rows


def normalize_candidate_ids(
    candidate_ids: Sequence[int], vocab_size: int
) -> list[int]:
    """Return sorted, unique, in-vocabulary candidate IDs."""
    normalized = sorted(set(candidate_ids))
    if not normalized:
        raise ValueError("Candidate token set must not be empty")
    invalid = [
        token_id for token_id in normalized
        if not 0 <= token_id < vocab_size
    ]
    if invalid:
        raise ValueError(
            f"Candidate IDs must be in [0, {vocab_size}); "
            f"invalid IDs: {invalid[:5]}"
        )
    return normalized


def _validate_full_vocab_logits(
    score_token_ids: Sequence[int], logits: torch.Tensor
) -> torch.Tensor:
    if (
        list(score_token_ids) != list(range(len(score_token_ids)))
        or len(score_token_ids) > logits.shape[1]
    ):
        raise ValueError(
            "Adversarial evaluation requires full-vocabulary logits "
            "in token-ID order"
        )
    return logits[:, : len(score_token_ids)]


def select_worst_tokens(
    logits: torch.Tensor,
    candidate_sets: Sequence[Sequence[int]],
    *,
    prefixes: Sequence[Sequence[int]] | None = None,
    candidate_validator: CandidateValidator | None = None,
) -> tuple[list[int], list[float], list[float]]:
    """Select each row's lowest valid finite-logit candidate deterministically."""
    if logits.ndim != 2:
        raise ValueError(
            f"Expected 2-D logits, got shape {tuple(logits.shape)}"
        )
    if logits.shape[0] != len(candidate_sets):
        raise ValueError("One candidate set is required for each logits row")
    if candidate_validator is not None and (
        prefixes is None or len(prefixes) != logits.shape[0]
    ):
        raise ValueError(
            "One prefix is required per logits row when validating candidates"
        )

    token_ids, model_logs, masked_logs = [], [], []
    for row_index, (row, raw_candidates) in enumerate(
        zip(logits, candidate_sets)
    ):
        candidates = normalize_candidate_ids(raw_candidates, row.shape[0])
        indices = torch.tensor(
            candidates, dtype=torch.long, device=row.device
        )
        candidate_logits = row.index_select(0, indices).float()
        finite = torch.isfinite(candidate_logits)
        if not finite.any():
            raise ValueError("No candidate token has a finite model logit")
        finite_indices = indices[finite]
        finite_logits = candidate_logits[finite]
        if candidate_validator is None:
            offset = int(torch.argmin(finite_logits).item())
        else:
            offset = -1
            for candidate_offset in torch.argsort(
                finite_logits, stable=True
            ).tolist():
                token_id = int(finite_indices[candidate_offset].item())
                if candidate_validator(
                    row_index, prefixes[row_index], token_id
                ):
                    offset = candidate_offset
                    break
            if offset < 0:
                raise ValueError(
                    "No finite candidate token satisfies the validation constraint"
                )
        token_id = int(finite_indices[offset].item())
        token_ids.append(token_id)
        model_logs.append(
            float(torch.log_softmax(row.float(), 0)[token_id].item())
        )
        masked_logs.append(
            float(torch.log_softmax(finite_logits, 0)[offset].item())
        )
    return token_ids, model_logs, masked_logs


def score_target_tokens(
    logits: torch.Tensor,
    target_token_ids: Sequence[int],
    candidate_sets: Sequence[Sequence[int]],
) -> tuple[list[float], list[float]]:
    """Score fixed target tokens under full and mask-renormalized logits."""
    if logits.ndim != 2:
        raise ValueError(
            f"Expected 2-D logits, got shape {tuple(logits.shape)}"
        )
    if not (
        logits.shape[0]
        == len(target_token_ids)
        == len(candidate_sets)
    ):
        raise ValueError(
            "One target and candidate set are required for each logits row"
        )

    model_logs, masked_logs = [], []
    for row, target_token_id, raw_candidates in zip(
        logits, target_token_ids, candidate_sets
    ):
        candidates = normalize_candidate_ids(raw_candidates, row.shape[0])
        if target_token_id not in candidates:
            raise ValueError(
                f"Target token {target_token_id} is outside its candidate set"
            )
        indices = torch.tensor(
            candidates, dtype=torch.long, device=row.device
        )
        candidate_logits = row.index_select(0, indices).float()
        finite = torch.isfinite(candidate_logits)
        finite_indices = indices[finite]
        finite_logits = candidate_logits[finite]
        target_offsets = (
            finite_indices == target_token_id
        ).nonzero(as_tuple=False)
        if not len(target_offsets):
            raise ValueError(
                f"Target token {target_token_id} has a non-finite logit"
            )
        offset = int(target_offsets[0].item())
        model_logs.append(
            float(
                torch.log_softmax(row.float(), 0)[target_token_id].item()
            )
        )
        masked_logs.append(
            float(torch.log_softmax(finite_logits, 0)[offset].item())
        )
    return model_logs, masked_logs


def generate_worst_case_sequences(
    predictor: BatchedPredictor,
    start_sequences: Sequence[Sequence[int]],
    total_length: int,
    candidate_sets: Sequence[Sequence[int]],
    *,
    context_length: int,
    retain_tokens: int,
    use_kv_cache: bool = True,
    candidate_validator: CandidateValidator | None = None,
) -> AdversarialGeneration:
    """Generate equally long sequences by repeatedly choosing the worst token."""
    sequences = [list(sequence) for sequence in start_sequences]
    if not sequences or any(not sequence for sequence in sequences):
        raise ValueError("At least one non-empty start sequence is required")
    start_lengths = {len(sequence) for sequence in sequences}
    if len(start_lengths) != 1:
        raise ValueError("All start sequences must have equal length")
    start_length = start_lengths.pop()
    if total_length < start_length:
        raise ValueError(
            "total_length must be at least the start-sequence length"
        )
    if len(candidate_sets) != len(sequences):
        raise ValueError(
            "One candidate set is required per start sequence"
        )
    if not 0 < retain_tokens <= context_length:
        raise ValueError("Require 0 < retain_tokens <= context_length")

    candidates = [list(candidate_set) for candidate_set in candidate_sets]
    # A checkpointed call receives the entire sequence but a fresh predictor
    # cache. Reconstruct the context length that uninterrupted generation would
    # have retained; blindly truncating every resumed sequence to
    # ``retain_tokens`` changes both selection and stored probabilities at each
    # checkpoint boundary.
    def resumed_context(sequence: list[int]) -> list[int]:
        length = len(sequence)
        if length <= context_length:
            return sequence[:]
        cycle = context_length - retain_tokens
        offset = (length - context_length) % cycle
        retained = context_length if offset == 0 else retain_tokens + offset
        return sequence[-retained:]

    contexts = [resumed_context(sequence) for sequence in sequences]
    model_logs = [[] for _ in sequences]
    masked_logs = [[] for _ in sequences]
    predictor.reset_kv_cache()

    while len(sequences[0]) < total_length:
        if len(contexts[0]) >= context_length:
            contexts = [
                context[-retain_tokens:] for context in contexts
            ]
        score_token_ids, logits, _, _ = predictor.run_batched_inference(
            contexts, enable_kv_cache=use_kv_cache
        )
        logits = _validate_full_vocab_logits(score_token_ids, logits)
        next_ids, step_model_logs, step_masked_logs = (
            select_worst_tokens(
                logits,
                candidates,
                prefixes=sequences,
                candidate_validator=candidate_validator,
            )
        )
        for index, token_id in enumerate(next_ids):
            sequences[index].append(token_id)
            contexts[index].append(token_id)
            model_logs[index].append(step_model_logs[index])
            masked_logs[index].append(step_masked_logs[index])

    return AdversarialGeneration(
        token_ids=sequences,
        model_log_probabilities=model_logs,
        masked_log_probabilities=masked_logs,
        candidate_sizes=[
            len(set(candidate_set)) for candidate_set in candidates
        ],
    )


def rescore_sequences(
    predictor: BatchedPredictor,
    sequences: Sequence[Sequence[int]],
    start_length: int,
    candidate_sets: Sequence[Sequence[int]],
    *,
    context_length: int,
    retain_tokens: int,
    use_kv_cache: bool = True,
    progress_desc: str | None = None,
) -> AdversarialGeneration:
    """Replay fixed sequences and score them under post-hoc dictionaries."""
    fixed_sequences = [list(sequence) for sequence in sequences]
    if not fixed_sequences or any(
        len(sequence) <= start_length for sequence in fixed_sequences
    ):
        raise ValueError(
            "Every sequence must contain at least one predicted token"
        )
    if start_length <= 0:
        raise ValueError("start_length must be positive")
    lengths = {len(sequence) for sequence in fixed_sequences}
    if len(lengths) != 1:
        raise ValueError("All fixed sequences must have equal length")
    if len(candidate_sets) != len(fixed_sequences):
        raise ValueError(
            "One candidate set is required per fixed sequence"
        )
    if not 0 < retain_tokens <= context_length:
        raise ValueError("Require 0 < retain_tokens <= context_length")

    candidates = [list(candidate_set) for candidate_set in candidate_sets]
    contexts = [
        sequence[:start_length] for sequence in fixed_sequences
    ]
    model_logs = [[] for _ in fixed_sequences]
    masked_logs = [[] for _ in fixed_sequences]
    predictor.reset_kv_cache()

    sequence_length = lengths.pop()
    positions = tqdm(
        range(start_length, sequence_length),
        desc=progress_desc,
        unit="step",
        dynamic_ncols=True,
        disable=progress_desc is None,
    )
    for token_index in positions:
        if len(contexts[0]) >= context_length:
            contexts = [
                context[-retain_tokens:] for context in contexts
            ]
        score_token_ids, logits, _, _ = predictor.run_batched_inference(
            contexts, enable_kv_cache=use_kv_cache
        )
        logits = _validate_full_vocab_logits(score_token_ids, logits)
        targets = [
            sequence[token_index] for sequence in fixed_sequences
        ]
        step_model_logs, step_masked_logs = score_target_tokens(
            logits, targets, candidates
        )
        for index, token_id in enumerate(targets):
            contexts[index].append(token_id)
            model_logs[index].append(step_model_logs[index])
            masked_logs[index].append(step_masked_logs[index])

    return AdversarialGeneration(
        token_ids=fixed_sequences,
        model_log_probabilities=model_logs,
        masked_log_probabilities=masked_logs,
        candidate_sizes=[
            len(set(candidate_set)) for candidate_set in candidates
        ],
    )
