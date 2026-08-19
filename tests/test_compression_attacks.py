import gzip
import math

import pytest
import torch

from src.compression_attacks import (
    AttackObjective,
    classical_compression_baselines,
    decoded_utf8_increment,
    generate_beam_search_sequence,
    generate_greedy_sequences,
    generate_random_sequences,
    rank_candidates,
    score_arithmetic_payloads,
    score_full_vocab_sequences,
)


class FakePredictor:
    def __init__(self, logits_by_prefix):
        self.logits_by_prefix = logits_by_prefix
        self.cache_flags = []
        self.reset_count = 0

    def reset_kv_cache(self):
        self.reset_count += 1

    def run_batched_inference(self, prompts, enable_kv_cache=True):
        self.cache_flags.append(enable_kv_cache)
        rows = [self.logits_by_prefix[tuple(prompt)] for prompt in prompts]
        logits = torch.tensor(rows, dtype=torch.float32)
        return list(range(logits.shape[1])), logits, 0.0, 0.0


class CharacterTokenizer:
    pieces = {0: "A", 1: "x", 2: "long"}

    def decode(self, token_ids, **_kwargs):
        return "".join(self.pieces[token_id] for token_id in token_ids)


def test_surprisal_per_byte_differs_from_minimum_probability():
    row = torch.tensor([0.0, -8.0, -10.0])
    byte_lengths = {0: 1, 1: 1, 2: 4}
    increment = lambda _prefix, token_id: byte_lengths[token_id]

    minimum = rank_candidates(
        row, [1, 2], [0], objective=AttackObjective.MIN_PROBABILITY
    )
    byte_aware = rank_candidates(
        row,
        [1, 2],
        [0],
        objective=AttackObjective.SURPRISAL_PER_BYTE,
        byte_increment=increment,
    )

    assert minimum[0][0] == 2
    assert byte_aware[0][0] == 1


def test_greedy_byte_attack_uses_marginal_decoded_size():
    predictor = FakePredictor({
        (0,): [0.0, -8.0, -10.0],
        (0, 1): [0.0, -8.0, -10.0],
    })
    result = generate_greedy_sequences(
        predictor,
        [[0]],
        total_length=3,
        candidate_sets=[[1, 2]],
        context_length=8,
        retain_tokens=4,
        objective=AttackObjective.SURPRISAL_PER_BYTE,
        byte_increment=decoded_utf8_increment(CharacterTokenizer()),
    )

    assert result.token_ids == [[0, 1, 1]]
    assert result.summary()[0]["generated_tokens"] == 2


def test_beam_search_finds_better_future_than_greedy_choice():
    predictor = FakePredictor({
        (0,): [0.0, -4.0, -3.0],
        (0, 1): [0.0, 0.0, 0.0],
        (0, 2): [0.0, -20.0, 0.0],
    })
    result = generate_beam_search_sequence(
        predictor,
        [0],
        total_length=3,
        candidate_ids=[1, 2],
        context_length=8,
        retain_tokens=4,
        byte_increment=lambda _prefix, _token: 1,
        initial_decoded_bytes=1,
        objective=AttackObjective.SURPRISAL_PER_BYTE,
        beam_width=2,
        branch_factor=2,
    )

    assert result.token_ids == [[0, 2, 1]]
    assert predictor.cache_flags == [False, False]


def test_actual_ratio_beam_uses_real_arithmetic_coder():
    predictor = FakePredictor({
        (0,): [4.0, -3.0, -2.0],
        (0, 1): [4.0, -3.0, -2.0],
        (0, 2): [4.0, -3.0, -2.0],
    })
    result = generate_beam_search_sequence(
        predictor,
        [0],
        total_length=3,
        candidate_ids=[1, 2],
        context_length=8,
        retain_tokens=4,
        byte_increment=lambda _prefix, _token: 1,
        initial_decoded_bytes=1,
        objective=AttackObjective.ACTUAL_RATIO,
        beam_width=2,
        branch_factor=2,
        fixed_overhead_bits=80,
    )

    assert result.token_ids[0][0] == 0
    assert len(result.model_log_probabilities[0]) == 2
    assert all(math.isfinite(x) for x in result.model_log_probabilities[0])


def test_random_control_is_seeded_and_does_not_change_start():
    validator = lambda _row, prefix, token: prefix[-1] != token
    first = generate_random_sequences(
        [[0]], 6, [[1, 2]], seed=7, candidate_validator=validator
    )
    second = generate_random_sequences(
        [[0]], 6, [[1, 2]], seed=7, candidate_validator=validator
    )
    assert first == second
    assert first[0][0] == 0
    assert all(left != right for left, right in zip(first[0], first[0][1:]))


def test_classical_baselines_use_the_exact_input_bytes():
    data = (b"compression ratio " * 40) + bytes(range(32))
    results = classical_compression_baselines(data)

    assert results["gzip-9"]["compressed_size_bytes"] == len(
        gzip.compress(data, compresslevel=9, mtime=0)
    )
    assert results["zstd-22"]["raw_size_bytes"] == len(data)
    assert results["brotli-11"]["raw_size_bytes"] == len(data)
    assert results["zstd-22"]["compression_ratio"] == pytest.approx(
        results["zstd-22"]["compressed_size_bytes"] / len(data)
    )


def test_fixed_sequence_scoring_reports_model_and_realized_costs():
    predictor = FakePredictor({
        (0,): [2.0, -1.0, 0.0],
        (0, 1): [2.0, -1.0, 0.0],
    })
    generation = score_full_vocab_sequences(
        predictor,
        [[0, 1, 2]],
        start_length=1,
        context_length=8,
        retain_tokens=4,
    )
    payload_bits = score_arithmetic_payloads(
        predictor,
        [[0, 1, 2]],
        start_length=1,
        context_length=8,
        retain_tokens=4,
    )

    expected_first = torch.log_softmax(
        torch.tensor([2.0, -1.0, 0.0]), dim=0
    )[1]
    assert generation.model_log_probabilities[0][0] == pytest.approx(
        expected_first.item()
    )
    assert generation.masked_log_probabilities == (
        generation.model_log_probabilities
    )
    assert payload_bits[0] > 0


def test_byte_objective_requires_a_byte_measurement():
    with pytest.raises(ValueError, match="byte_increment"):
        rank_candidates(
            torch.tensor([0.0, -1.0]),
            [1],
            [0],
            objective=AttackObjective.SURPRISAL_PER_BYTE,
        )
