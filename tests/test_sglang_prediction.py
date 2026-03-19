"""
Tests for SGlang prediction module (src/sglang_prediction.py).
"""

import importlib.util
from types import SimpleNamespace

import pytest
import torch


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