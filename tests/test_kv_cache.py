"""Numerical regression checks for cached next-token inference."""

from types import SimpleNamespace

import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from src.prediction import TokenPredictor


@pytest.fixture(scope="module")
def predictor_factory():
    torch.manual_seed(7)
    model = GPT2LMHeadModel(
        GPT2Config(vocab_size=32, n_positions=32, n_ctx=32, n_embd=16, n_layer=2, n_head=2)
    ).eval()

    def create():
        predictor = TokenPredictor.__new__(TokenPredictor)
        predictor.model = model
        predictor.device = torch.device("cpu")
        predictor.tokenizer = SimpleNamespace(pad_token_id=0)
        predictor.args = SimpleNamespace(engine="transformer", encoding="bitpacked")
        predictor.reduce_tokens = False
        predictor.tokens_list = list(range(32))
        predictor.reset_kv_cache()
        return predictor

    return create


def assert_matches_cachefree(predictor, prompts, predictor_factory):
    cached = predictor.run_batched_inference(prompts, enable_kv_cache=True)[1]
    uncached = predictor_factory().run_batched_inference_cachefree(prompts)[1]
    torch.testing.assert_close(cached, uncached, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize(
    "first, second, expected_work",
    [
        ([[1, 2], [3, 4]], [[1, 2, 5], [3, 4, 6]], 2),
        ([[1, 2]], [[1, 2, 3, 4]], 2),
        ([[1, 2]], [[1, 2]], 2),
        ([[1, 2]], [[1, 9, 3]], 3),
        ([[1, 2, 3]], [[2, 3, 4]], 3),
    ],
)
def test_cached_rectangular_matches_full_forward(
    predictor_factory, first, second, expected_work
):
    predictor = predictor_factory()
    predictor.run_batched_inference(first, enable_kv_cache=True)
    assert_matches_cachefree(predictor, second, predictor_factory)
    assert predictor.last_model_input_tokens == expected_work


def test_cache_free_call_discards_prior_state(predictor_factory):
    predictor = predictor_factory()
    predictor.run_batched_inference([[1, 2]], enable_kv_cache=True)
    predictor.run_batched_inference([[7, 8, 9]], enable_kv_cache=False)
    assert predictor._past_kv is None
    assert predictor._row_past_kv is None
    assert predictor._cached_prompts == []
    assert_matches_cachefree(predictor, [[7, 8, 9, 10]], predictor_factory)
    assert predictor.last_model_input_tokens == 4

    predictor.run_batched_inference_cachefree([[4, 5]])
    assert predictor._past_kv is None
    assert predictor._row_past_kv is None
    assert predictor._cached_prompts == []


def test_uneven_rows_reuse_only_their_own_prefixes(predictor_factory):
    predictor = predictor_factory()
    assert_matches_cachefree(predictor, [[1, 2, 3], [4]], predictor_factory)
    assert predictor.last_model_input_tokens == 4
    assert_matches_cachefree(predictor, [[1, 2, 3, 5], [4, 6, 7]], predictor_factory)
    assert predictor.last_model_input_tokens == 3
    assert_matches_cachefree(predictor, [[1, 2, 8], [4, 6, 7, 9]], predictor_factory)
    assert predictor.last_model_input_tokens == 4


def test_even_uneven_transitions_match_full_forward(predictor_factory):
    predictor = predictor_factory()
    assert_matches_cachefree(predictor, [[1, 2], [3, 4]], predictor_factory)
    assert_matches_cachefree(predictor, [[1, 2, 5], [3, 4]], predictor_factory)
    assert_matches_cachefree(predictor, [[1, 2, 5, 7], [3, 4, 6, 8]], predictor_factory)
    assert_matches_cachefree(predictor, [[1, 2, 5, 7, 9], [3, 4, 6, 8, 10]], predictor_factory)
    assert predictor.last_model_input_tokens == 2
