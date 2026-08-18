"""Worst-probability sequence generation for compression stress tests."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
from typing import Protocol, Sequence

import torch


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
                        if generated else 0.0
                    ),
                    "mean_masked_surprisal_bits": (
                        -sum(masked_logs) / (generated * math.log(2))
                        if generated else 0.0
                    ),
                }
            )
        return rows


@dataclass
class FixedPointGeneration:
    generation: AdversarialGeneration
    converged: bool
    passes: int
    candidate_size_history: list[list[int]]


def normalize_candidate_ids(
    candidate_ids: Sequence[int], vocab_size: int
) -> list[int]:
    """Return sorted, unique, in-vocabulary candidate IDs."""
    normalized = sorted(set(candidate_ids))
    if not normalized:
        raise ValueError("Candidate token set must not be empty")
    invalid = [token_id for token_id in normalized if not 0 <= token_id < vocab_size]
    if invalid:
        raise ValueError(
            f"Candidate IDs must be in [0, {vocab_size}); invalid IDs: {invalid[:5]}"
        )
    return normalized


def top_k_frequent_tokens(reference_tokens: Sequence[int], k: int) -> list[int]:
    """Choose tokens by descending reference frequency, then token ID."""
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    counts = Counter(reference_tokens)
    if not counts:
        raise ValueError("Reference token sequence must not be empty")
    ranked = sorted(counts, key=lambda token_id: (-counts[token_id], token_id))
    return sorted(ranked[:k])


def select_worst_tokens(
    logits: torch.Tensor,
    candidate_sets: Sequence[Sequence[int]],
) -> tuple[list[int], list[float], list[float]]:
    """Select each row's lowest-finite-logit candidate deterministically."""
    if logits.ndim != 2:
        raise ValueError(f"Expected 2-D logits, got shape {tuple(logits.shape)}")
    if logits.shape[0] != len(candidate_sets):
        raise ValueError("One candidate set is required for each logits row")

    token_ids, model_logs, masked_logs = [], [], []
    for row, raw_candidates in zip(logits, candidate_sets):
        candidates = normalize_candidate_ids(raw_candidates, row.shape[0])
        indices = torch.tensor(candidates, dtype=torch.long, device=row.device)
        candidate_logits = row.index_select(0, indices).float()
        finite = torch.isfinite(candidate_logits)
        if not finite.any():
            raise ValueError("No candidate token has a finite model logit")
        finite_indices = indices[finite]
        finite_logits = candidate_logits[finite]
        offset = int(torch.argmin(finite_logits).item())
        token_id = int(finite_indices[offset].item())
        token_ids.append(token_id)
        model_logs.append(float(torch.log_softmax(row.float(), 0)[token_id].item()))
        masked_logs.append(
            float(torch.log_softmax(finite_logits, 0)[offset].item())
        )
    return token_ids, model_logs, masked_logs


def generate_worst_case_sequences(
    predictor: BatchedPredictor,
    start_sequences: Sequence[Sequence[int]],
    total_length: int,
    candidate_sets: Sequence[Sequence[int]],
    *,
    context_length: int,
    retain_tokens: int,
    use_kv_cache: bool = True,
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
        raise ValueError("total_length must be at least the start-sequence length")
    if len(candidate_sets) != len(sequences):
        raise ValueError("One candidate set is required per start sequence")
    if not 0 < retain_tokens <= context_length:
        raise ValueError("Require 0 < retain_tokens <= context_length")

    candidates = [list(candidate_set) for candidate_set in candidate_sets]
    contexts = [sequence[:] for sequence in sequences]
    model_logs = [[] for _ in sequences]
    masked_logs = [[] for _ in sequences]
    predictor.reset_kv_cache()

    while len(sequences[0]) < total_length:
        if len(contexts[0]) >= context_length:
            contexts = [context[-retain_tokens:] for context in contexts]
        score_token_ids, logits, _, _ = predictor.run_batched_inference(
            contexts, enable_kv_cache=use_kv_cache
        )
        if (
            score_token_ids != list(range(len(score_token_ids)))
            or len(score_token_ids) > logits.shape[1]
        ):
            raise ValueError(
                "Generation requires full-vocabulary logits in token-ID order"
            )
        logits = logits[:, : len(score_token_ids)]
        next_ids, step_model_logs, step_masked_logs = select_worst_tokens(
            logits, candidates
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
        candidate_sizes=[len(set(candidate_set)) for candidate_set in candidates],
    )


def generate_occurring_token_fixed_point(
    predictor: BatchedPredictor,
    start_sequences: Sequence[Sequence[int]],
    total_length: int,
    initial_candidates: Sequence[int],
    *,
    context_length: int,
    retain_tokens: int,
    use_kv_cache: bool = True,
    max_passes: int = 10,
) -> FixedPointGeneration:
    """Resolve the occurring-token mask by monotone fixed-point iteration."""
    if max_passes <= 0:
        raise ValueError("max_passes must be positive")
    candidate_sets = [
        sorted(set(initial_candidates).union(start))
        for start in start_sequences
    ]
    history = [[len(candidate_set) for candidate_set in candidate_sets]]

    for pass_index in range(1, max_passes + 1):
        generation = generate_worst_case_sequences(
            predictor,
            start_sequences,
            total_length,
            candidate_sets,
            context_length=context_length,
            retain_tokens=retain_tokens,
            use_kv_cache=use_kv_cache,
        )
        observed_sets = [sorted(set(tokens)) for tokens in generation.token_ids]
        history.append([len(candidate_set) for candidate_set in observed_sets])
        if observed_sets == candidate_sets:
            return FixedPointGeneration(generation, True, pass_index, history)
        candidate_sets = observed_sets

    return FixedPointGeneration(generation, False, max_passes, history)
