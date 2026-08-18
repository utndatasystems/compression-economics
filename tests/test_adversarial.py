import math

import pytest
import torch

from src.adversarial import (
    generate_occurring_token_fixed_point,
    generate_worst_case_sequences,
    select_worst_tokens,
    top_k_frequent_tokens,
)


class FakePredictor:
    def __init__(self, logits):
        self.logits = torch.tensor(logits, dtype=torch.float32)
        self.reset_count = 0
        self.prompt_lengths = []

    def reset_kv_cache(self):
        self.reset_count += 1

    def run_batched_inference(self, prompts, enable_kv_cache=True):
        self.prompt_lengths.append(len(prompts[0]))
        rows = self.logits.repeat(len(prompts), 1)
        return list(range(len(self.logits))), rows, 0.0, 0.0


def test_select_worst_tokens_respects_each_mask():
    logits = torch.tensor([[3.0, -2.0, 0.0], [-5.0, 4.0, 2.0]])
    tokens, model_logs, masked_logs = select_worst_tokens(
        logits, [[0, 1, 2], [1, 2]]
    )
    assert tokens == [1, 2]
    assert model_logs[0] == pytest.approx(
        torch.log_softmax(logits[0], dim=0)[1].item()
    )
    assert masked_logs[1] == pytest.approx(
        torch.log_softmax(logits[1, [1, 2]], dim=0)[1].item()
    )


def test_top_k_frequent_tokens_has_deterministic_ties():
    assert top_k_frequent_tokens([4, 2, 4, 3, 2, 1], 2) == [2, 4]
    assert top_k_frequent_tokens([3, 2, 1], 2) == [1, 2]


def test_generation_has_equal_lengths_and_uses_worst_token():
    predictor = FakePredictor([2.0, -3.0, 0.0])
    result = generate_worst_case_sequences(
        predictor,
        [[0], [2]],
        total_length=5,
        candidate_sets=[[0, 1, 2], [0, 2]],
        context_length=4,
        retain_tokens=2,
    )
    assert result.token_ids == [[0, 1, 1, 1, 1], [2, 2, 2, 2, 2]]
    assert all(len(tokens) == 5 for tokens in result.token_ids)
    assert predictor.reset_count == 1
    assert predictor.prompt_lengths == [1, 2, 3, 2]
    assert all(
        math.isfinite(value)
        for row in result.model_log_probabilities
        for value in row
    )


def test_occurring_mask_converges_to_generated_token_set():
    predictor = FakePredictor([2.0, -3.0, 0.0])
    fixed_point = generate_occurring_token_fixed_point(
        predictor,
        [[0], [2]],
        total_length=4,
        initial_candidates=[0, 1, 2],
        context_length=10,
        retain_tokens=5,
        max_passes=5,
    )
    assert fixed_point.converged is True
    assert fixed_point.passes == 2
    assert fixed_point.candidate_size_history == [[3, 3], [2, 2], [2, 2]]
    for tokens, final_size in zip(
        fixed_point.generation.token_ids,
        fixed_point.candidate_size_history[-1],
    ):
        assert len(set(tokens)) == final_size
