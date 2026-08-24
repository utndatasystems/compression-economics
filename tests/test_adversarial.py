import math

import pytest
import torch

from src.adversarial import (
    AdversarialGeneration,
    generate_worst_case_sequences,
    rescore_sequences,
    score_target_tokens,
    select_worst_tokens,
)
from scripts.generate_adversarial import _combine_generation


class FakePredictor:
    def __init__(self, logits):
        self.logits = torch.tensor(logits, dtype=torch.float32)
        self.reset_count = 0
        self.prompt_lengths = []

    def reset_kv_cache(self):
        self.reset_count += 1

    def run_batched_inference(
        self, prompts, enable_kv_cache=True
    ):
        self.prompt_lengths.append(len(prompts[0]))
        rows = self.logits.repeat(len(prompts), 1)
        return list(range(len(self.logits))), rows, 0.0, 0.0


def test_select_worst_tokens_respects_each_mask():
    logits = torch.tensor(
        [[3.0, -2.0, 0.0], [-5.0, 4.0, 2.0]]
    )
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



def test_select_worst_tokens_skips_candidates_rejected_by_validator():
    logits = torch.tensor([[2.0, -4.0, -2.0, 1.0]])
    checked = []

    def validator(row_index, prefix, candidate):
        checked.append((row_index, list(prefix), candidate))
        return candidate != 1

    tokens, model_logs, masked_logs = select_worst_tokens(
        logits,
        [[0, 1, 2, 3]],
        prefixes=[[0]],
        candidate_validator=validator,
    )

    assert tokens == [2]
    assert [item[2] for item in checked] == [1, 2]
    assert model_logs[0] == pytest.approx(
        torch.log_softmax(logits[0], dim=0)[2].item()
    )
    assert masked_logs[0] == pytest.approx(model_logs[0])


def test_validated_selection_requires_prefixes():
    with pytest.raises(ValueError, match="One prefix"):
        select_worst_tokens(
            torch.tensor([[0.0, 1.0]]),
            [[0, 1]],
            candidate_validator=lambda *_: True,
        )

def test_score_target_tokens_uses_fixed_targets():
    logits = torch.tensor(
        [[3.0, -2.0, 0.0], [-5.0, 4.0, 2.0]]
    )
    model_logs, masked_logs = score_target_tokens(
        logits,
        target_token_ids=[0, 1],
        candidate_sets=[[0, 2], [1, 2]],
    )
    assert model_logs[0] == pytest.approx(
        torch.log_softmax(logits[0], dim=0)[0].item()
    )
    assert masked_logs[0] == pytest.approx(
        torch.log_softmax(logits[0, [0, 2]], dim=0)[0].item()
    )
    assert masked_logs[1] == pytest.approx(
        torch.log_softmax(logits[1, [1, 2]], dim=0)[0].item()
    )


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
    assert result.token_ids == [
        [0, 1, 1, 1, 1],
        [2, 2, 2, 2, 2],
    ]
    assert all(len(tokens) == 5 for tokens in result.token_ids)
    assert predictor.reset_count == 1
    assert predictor.prompt_lengths == [1, 2, 3, 2]
    assert all(
        math.isfinite(value)
        for row in result.model_log_probabilities
        for value in row
    )


def test_occurring_rescore_keeps_full_generation_fixed():
    predictor = FakePredictor([2.0, -3.0, 0.0])
    full = generate_worst_case_sequences(
        predictor,
        [[0], [2]],
        total_length=5,
        candidate_sets=[[0, 1, 2], [0, 1, 2]],
        context_length=4,
        retain_tokens=2,
    )
    candidates = [
        sorted(set(tokens)) for tokens in full.token_ids
    ]
    occurring = rescore_sequences(
        predictor,
        full.token_ids,
        start_length=1,
        candidate_sets=candidates,
        context_length=4,
        retain_tokens=2,
    )

    assert occurring.token_ids == full.token_ids
    assert occurring.candidate_sizes == [2, 2]
    for rescored, original in zip(
        occurring.model_log_probabilities,
        full.model_log_probabilities,
    ):
        assert rescored == pytest.approx(original)
    assert any(
        masked != pytest.approx(model)
        for masked_row, model_row in zip(
            occurring.masked_log_probabilities,
            occurring.model_log_probabilities,
        )
        for masked, model in zip(masked_row, model_row)
    )


def test_rescore_rejects_target_outside_post_hoc_mask():
    predictor = FakePredictor([2.0, -3.0, 0.0])
    with pytest.raises(
        ValueError, match="outside its candidate set"
    ):
        rescore_sequences(
            predictor,
            sequences=[[0, 2]],
            start_length=1,
            candidate_sets=[[0, 1]],
            context_length=4,
            retain_tokens=2,
        )


def test_checkpoint_extension_keeps_prior_scores_once():
    previous = AdversarialGeneration(
        token_ids=[[0, 1]],
        model_log_probabilities=[[-1.0]],
        masked_log_probabilities=[[-0.5]],
        candidate_sizes=[3],
    )
    extension = AdversarialGeneration(
        token_ids=[[0, 1, 2, 1]],
        model_log_probabilities=[[-2.0, -3.0]],
        masked_log_probabilities=[[-1.5, -2.5]],
        candidate_sizes=[3],
    )

    combined = _combine_generation(previous, extension, candidate_size=3)

    assert combined.token_ids == [[0, 1, 2, 1]]
    assert combined.model_log_probabilities == [[-1.0, -2.0, -3.0]]
    assert combined.masked_log_probabilities == [[-0.5, -1.5, -2.5]]


def test_resumed_generation_reconstructs_uninterrupted_context():
    predictor = FakePredictor([2.0, -3.0, 0.0])
    uninterrupted = generate_worst_case_sequences(
        predictor,
        [[0]],
        total_length=8,
        candidate_sets=[[0, 1, 2]],
        context_length=4,
        retain_tokens=2,
    )
    predictor = FakePredictor([2.0, -3.0, 0.0])
    first = generate_worst_case_sequences(
        predictor,
        [[0]],
        total_length=5,
        candidate_sets=[[0, 1, 2]],
        context_length=4,
        retain_tokens=2,
    )
    resumed = generate_worst_case_sequences(
        predictor,
        first.token_ids,
        total_length=8,
        candidate_sets=[[0, 1, 2]],
        context_length=4,
        retain_tokens=2,
    )
    assert resumed.token_ids == uninterrupted.token_ids
    assert predictor.prompt_lengths[-3:] == [3, 2, 3]
    assert resumed.model_log_probabilities[0] == pytest.approx(
        uninterrupted.model_log_probabilities[0][-3:]
    )


def _local_qwen_tokenizer():
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained(
            "Qwen/Qwen2.5-0.5B", local_files_only=True
        )
    except OSError:
        pytest.skip("Qwen2.5 tokenizer is not available locally")


def test_qwen_local_round_trip_validator_matches_full_oracle():
    from scripts.generate_adversarial import (
        ascii_byte_token_ids,
        make_round_trip_validator,
        token_ids_round_trip,
    )

    tokenizer = _local_qwen_tokenizer()
    candidates = ascii_byte_token_ids(tokenizer, printable_only=False)
    validator = make_round_trip_validator(tokenizer)

    prefixes = [
        tokenizer.encode(text, add_special_tokens=False)
        for text in (
            "guard. ordinary tail",
            "guard. can't stop",
            "guard!!!\nnext",
            "guard. " + "x" * 600,
        )
    ]
    prefixes.append([126190, 230])

    for prefix in prefixes:
        for candidate in candidates:
            assert validator(0, prefix, candidate) == token_ids_round_trip(
                tokenizer, [*prefix, candidate]
            )

    sampled_vocabulary = range(0, tokenizer.vocab_size, 997)
    prefix = tokenizer.encode("guard. sampled tail", add_special_tokens=False)
    for candidate in sampled_vocabulary:
        assert validator(0, prefix, candidate) == token_ids_round_trip(
            tokenizer, [*prefix, candidate]
        )

    assert validator.local_checks > 0
    assert validator.fallback_checks > 0


def test_qwen_local_validator_keeps_incremental_state_canonical():
    from scripts.generate_adversarial import (
        ascii_byte_token_ids,
        make_round_trip_validator,
        token_ids_round_trip,
    )

    tokenizer = _local_qwen_tokenizer()
    candidates = ascii_byte_token_ids(tokenizer, printable_only=False)
    validator = make_round_trip_validator(tokenizer)
    prefix = tokenizer.encode("guard. evolving tail", add_special_tokens=False)

    for _ in range(20):
        decisions = [
            validator(0, prefix, candidate) for candidate in candidates
        ]
        oracle = [
            token_ids_round_trip(tokenizer, [*prefix, candidate])
            for candidate in candidates
        ]
        assert decisions == oracle
        candidate = candidates[oracle.index(True)]
        prefix.append(candidate)

    assert token_ids_round_trip(tokenizer, prefix)
    assert validator.local_checks > validator.fallback_checks
