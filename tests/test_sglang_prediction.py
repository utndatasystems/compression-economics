"""
Tests for SGlang prediction module (src/sglang_prediction.py).
"""

import importlib.util
from types import SimpleNamespace

import pytest
import torch


class _FakeSGLangEngine:
    def __init__(self):
        self.calls = []
        self.closed_sessions = []
        self.opened_sessions = []
        self._session_counter = 0

    def open_session(self, capacity_of_str_len, streaming, timeout):
        session_id = f"session-{self._session_counter}"
        self._session_counter += 1
        self.opened_sessions.append(
            {
                "session_id": session_id,
                "capacity_of_str_len": capacity_of_str_len,
                "streaming": streaming,
                "timeout": timeout,
            }
        )
        return session_id

    def close_session(self, session_id):
        self.closed_sessions.append(session_id)

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        token_ids = kwargs["token_ids_logprob"]
        if token_ids and isinstance(token_ids[0], list):
            return [
                {
                    "meta_info": {
                        "output_token_ids_logprobs": [
                            [(-0.1 * (index + 1), token_id) for index, token_id in enumerate(token_row)]
                        ]
                    }
                }
                for token_row in token_ids
            ]

        return {
            "meta_info": {
                "output_token_ids_logprobs": [
                    [(-0.1 * (index + 1), token_id) for index, token_id in enumerate(token_ids)]
                ]
            }
        }

    def shutdown(self):
        return None


def _make_unit_predictor(use_streaming_sessions=True):
    from src.sglang_prediction import SGLangTokenPredictor

    predictor = SGLangTokenPredictor.__new__(SGLangTokenPredictor)
    predictor.args = SimpleNamespace(
        encoding="bitpacked",
        lora_path=None,
        context_length=8,
        retain_tokens=4,
        sglang_use_streaming_session_kv=use_streaming_sessions,
        sglang_session_timeout=None,
    )
    predictor.engine = _FakeSGLangEngine()
    predictor.device = torch.device("cpu")
    predictor.tokens_list = [11, 22]
    predictor._probe_token_ids = [11, 22]
    predictor._use_streaming_session_kv = use_streaming_sessions
    predictor._session_timeout = None
    predictor._session_capacity = 8
    predictor._session_lanes = []
    predictor._session_batch_size = None
    predictor.base_params = 0
    predictor.base_size_mb = 0.0
    predictor.adapter_params = 0
    predictor.adapter_size_mb = 0.0
    predictor.tokenizer = SimpleNamespace(decode=lambda token_ids: " ".join(map(str, token_ids)))
    return predictor


def test_probe_sglang_ac_support():
    """probe_sglang_ac_support reports availability and a reason consistently."""
    from src.sglang_prediction import probe_sglang_ac_support

    args = SimpleNamespace(model_name="distilbert/distilgpt2")
    supported, reason = probe_sglang_ac_support(args)

    assert isinstance(supported, bool)
    if supported:
        assert reason is None
    else:
        assert isinstance(reason, str) and reason


def test_streaming_session_reuses_delta_and_reopens_on_truncation():
    predictor = _make_unit_predictor(use_streaming_sessions=True)

    predictor.run_batched_inference([[10], [20]], enable_kv_cache=True)
    predictor.run_batched_inference([[10, 11], [20, 21]], enable_kv_cache=True)
    predictor.run_batched_inference([[11], [20, 21, 22]], enable_kv_cache=True)

    first_lane_first = predictor.engine.calls[0]
    second_lane_first = predictor.engine.calls[1]
    first_lane_second = predictor.engine.calls[2]
    second_lane_second = predictor.engine.calls[3]
    first_lane_third = predictor.engine.calls[4]
    second_lane_third = predictor.engine.calls[5]

    assert first_lane_first["input_ids"] == [10]
    assert second_lane_first["input_ids"] == [20]
    assert first_lane_second["input_ids"] == [11]
    assert second_lane_second["input_ids"] == [21]
    assert first_lane_third["input_ids"] == [11]
    assert second_lane_third["input_ids"] == [22]

    assert first_lane_first["session_params"] == {"id": "session-0"}
    assert second_lane_first["session_params"] == {"id": "session-1"}
    assert first_lane_second["session_params"] == {"id": "session-0"}
    assert second_lane_second["session_params"] == {"id": "session-1"}
    assert first_lane_third["session_params"] == {"id": "session-2"}
    assert second_lane_third["session_params"] == {"id": "session-1"}

    assert predictor.engine.closed_sessions == ["session-0"]


def test_streaming_session_feature_flag_falls_back_to_batched_generate():
    predictor = _make_unit_predictor(use_streaming_sessions=False)

    predictor.run_batched_inference([[10], [20]], enable_kv_cache=True)

    assert len(predictor.engine.opened_sessions) == 0
    assert len(predictor.engine.calls) == 1
    assert predictor.engine.calls[0]["input_ids"] == [[10], [20]]
    assert predictor.engine.calls[0]["token_ids_logprob"] == [[11, 22], [11, 22]]


@pytest.fixture(scope="module")
def sglang_predictor():
    """Create an SGLangTokenPredictor with a small model for testing."""
    if importlib.util.find_spec("sglang") is None:
        pytest.skip("sglang not installed")
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    from src.sglang_prediction import SGLangTokenPredictor

    args = SimpleNamespace(
        model_name="distilbert/distilgpt2",
        engine="sglang",
        encoding="AC",
        context_length=256,
        reduce_tokens=True,
        tensor_parallel_size=1,
        sglang_mem_fraction_static=0.3,
        sglang_enable_deterministic_inference=True,
        lora_path=None,
    )
    predictor = SGLangTokenPredictor(args, bitmap_data=None)
    try:
        yield predictor
    finally:
        predictor.cleanup()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")
def test_single_prompt_inference(sglang_predictor):
    """Single-prompt AC inference returns probs of correct shape."""
    predictor = sglang_predictor
    tokenizer = predictor.tokenizer

    text = "Hello world"
    tokens = tokenizer.encode(text)
    prompts = [tokens]

    tokens_list, probs, data_copy_time, softmax_time = (
        predictor.run_batched_inference(prompts, enable_kv_cache=True)
    )

    assert isinstance(probs, torch.Tensor)
    assert probs.shape[0] == 1
    assert probs.shape[1] == len(tokens_list)
    assert data_copy_time >= 0.0
    assert softmax_time >= 0.0
    assert abs(probs.sum().item() - 1.0) < 1e-4


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")
def test_deterministic_logprobs(sglang_predictor):
    """Same prompt returns identical probabilities across two calls."""
    predictor = sglang_predictor
    tokenizer = predictor.tokenizer

    tokens = tokenizer.encode("The quick brown fox")
    prompts = [tokens]

    _, probs1, _, _ = predictor.run_batched_inference(prompts)
    _, probs2, _, _ = predictor.run_batched_inference(prompts)

    assert torch.allclose(probs1, probs2, atol=1e-6), (
        f"Max diff: {(probs1 - probs2).abs().max().item()}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")
def test_batched_prompts_order(sglang_predictor):
    """Multiple prompts return results in the correct order."""
    predictor = sglang_predictor
    tokenizer = predictor.tokenizer

    prompt_a = tokenizer.encode("The cat sat")
    prompt_b = tokenizer.encode("1 + 1 =")

    _, probs_batch, _, _ = predictor.run_batched_inference([prompt_a, prompt_b])
    _, probs_a, _, _ = predictor.run_batched_inference([prompt_a])
    _, probs_b, _, _ = predictor.run_batched_inference([prompt_b])

    assert torch.allclose(probs_batch[0], probs_a[0], atol=1e-5), (
        f"Row 0 mismatch, max diff: {(probs_batch[0] - probs_a[0]).abs().max()}"
    )
    assert torch.allclose(probs_batch[1], probs_b[0], atol=1e-5), (
        f"Row 1 mismatch, max diff: {(probs_batch[1] - probs_b[0]).abs().max()}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")
def test_ac_roundtrip_sglang():
    """Compress and decompress a short text with engine=sglang, encoding=AC."""
    if importlib.util.find_spec("sglang") is None:
        pytest.skip("sglang not installed")

    from src.global_mask_compressor import (
        run_global_mask_compression,
        run_global_mask_decompression,
    )
    from src.prediction import TokenDataPreparer

    args = SimpleNamespace(
        input_path="data/text8",
        text_input=None,
        model_name="distilbert/distilgpt2",
        engine="sglang",
        encoding="AC",
        context_length=256,
        retain_tokens=64,
        first_n_tokens=21,
        use_kv_cache=True,
        reduce_tokens=True,
        batch_size=1,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.3,
        sglang_mem_fraction_static=0.3,
        sglang_enable_deterministic_inference=True,
        lora_path=None,
        output_path="/tmp/test_sglang_roundtrip.bin",
    )

    preparer = TokenDataPreparer(args)
    expected_tokens = preparer.get_data_tokens()
    expected_text = preparer.tokenizer.decode(expected_tokens)

    first_tokens, bit_string, bitmask_data, _, args = run_global_mask_compression(args)
    reconstructed_tokens, detoken_string, _ = run_global_mask_decompression(
        args,
        first_tokens=first_tokens,
        bit_string=bit_string,
        bitmap=bitmask_data,
    )

    assert reconstructed_tokens == expected_tokens
    assert detoken_string == expected_text