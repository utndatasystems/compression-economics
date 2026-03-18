"""
Tests for vLLM prediction module (src/vllm_prediction.py).
"""

import pytest
import torch


def test_probe_vllm_ac_support():
    """probe_vllm_ac_support returns (True, None) when vLLM is installed."""
    from types import SimpleNamespace
    from src.vllm_prediction import probe_vllm_ac_support

    args = SimpleNamespace(model_name="distilbert/distilgpt2")
    supported, reason = probe_vllm_ac_support(args)
    # On this machine vLLM + CUDA should be available
    assert supported is True, f"Expected supported, got reason={reason}"


def test_capture_processor_is_not_argmax_invariant():
    """CaptureLogitsProcessor.is_argmax_invariant() must return False."""
    from src.vllm_prediction import CaptureLogitsProcessor

    proc = CaptureLogitsProcessor(vllm_config=None, device="cpu", is_pin_memory=False)
    assert proc.is_argmax_invariant() is False


def test_capture_processor_stores_logits():
    """CaptureLogitsProcessor.apply() writes logits to shared memory."""
    from multiprocessing import shared_memory
    from src.vllm_prediction import CaptureLogitsProcessor, _SHM_NAME, _HEADER_BYTES
    import struct

    vocab_size = 100
    shm_size = _HEADER_BYTES + vocab_size * 4
    # Clean up stale segment if any.
    try:
        old = shared_memory.SharedMemory(name=_SHM_NAME, create=False)
        old.close()
        old.unlink()
    except FileNotFoundError:
        pass

    shm = shared_memory.SharedMemory(name=_SHM_NAME, create=True, size=shm_size)
    try:
        proc = CaptureLogitsProcessor(vllm_config=None, device="cpu", is_pin_memory=False)

        fake_logits = torch.randn(1, vocab_size)
        result = proc.apply(fake_logits)

        # apply must return logits unchanged
        assert torch.equal(result, fake_logits)

        # Read back from shared memory
        rows, cols = struct.unpack_from("II", shm.buf, 0)
        assert rows == 1
        assert cols == vocab_size
    finally:
        shm.close()
        shm.unlink()


@pytest.fixture(scope="module")
def vllm_predictor():
    """Create a VLLMTokenPredictor with a small model for testing."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    from types import SimpleNamespace
    from src.vllm_prediction import VLLMTokenPredictor

    args = SimpleNamespace(
        model_name="distilbert/distilgpt2",
        engine="vllm",
        encoding="AC",
        context_length=256,
        reduce_tokens=False,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.3,
        lora_path=None,
    )
    return VLLMTokenPredictor(args, bitmap_data=None)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")
def test_single_prompt_inference(vllm_predictor):
    """Single-prompt AC inference returns probs of correct shape."""
    predictor = vllm_predictor
    tokenizer = predictor.tokenizer

    text = "Hello world"
    tokens = tokenizer.encode(text)
    prompts = [tokens]

    tokens_list, probs, data_copy_time, softmax_time = (
        predictor.run_batched_inference(prompts, enable_kv_cache=True)
    )

    assert isinstance(probs, torch.Tensor)
    assert probs.shape[0] == 1  # one prompt
    assert probs.shape[1] == len(tokens_list)  # vocab size
    # probs should sum to ~1
    assert abs(probs.sum().item() - 1.0) < 1e-4


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")
def test_deterministic_logits(vllm_predictor):
    """Same prompt returns identical probabilities across two calls."""
    predictor = vllm_predictor
    tokenizer = predictor.tokenizer

    tokens = tokenizer.encode("The quick brown fox")
    prompts = [tokens]

    _, probs1, _, _ = predictor.run_batched_inference(prompts)
    _, probs2, _, _ = predictor.run_batched_inference(prompts)

    assert torch.allclose(probs1, probs2, atol=1e-6), (
        f"Max diff: {(probs1 - probs2).abs().max().item()}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")
def test_batched_prompts_order(vllm_predictor):
    """Multiple prompts return results in the correct order."""
    predictor = vllm_predictor
    tokenizer = predictor.tokenizer

    prompt_a = tokenizer.encode("The cat sat")
    prompt_b = tokenizer.encode("1 + 1 =")

    # Run them as a batch
    _, probs_batch, _, _ = predictor.run_batched_inference(
        [prompt_a, prompt_b]
    )

    # Run individually
    _, probs_a, _, _ = predictor.run_batched_inference([prompt_a])
    _, probs_b, _, _ = predictor.run_batched_inference([prompt_b])

    assert torch.allclose(probs_batch[0], probs_a[0], atol=1e-5), (
        f"Row 0 mismatch, max diff: {(probs_batch[0] - probs_a[0]).abs().max()}"
    )
    assert torch.allclose(probs_batch[1], probs_b[0], atol=1e-5), (
        f"Row 1 mismatch, max diff: {(probs_batch[1] - probs_b[0]).abs().max()}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")
def test_ac_roundtrip_vllm():
    """
    Compress and decompress a short text with engine=vllm, encoding=AC.
    Verify exact token-level reconstruction.
    """
    from types import SimpleNamespace
    from src.global_mask_compressor import (
        run_global_mask_compression,
        run_global_mask_decompression,
    )
    from src.prediction import TokenDataPreparer

    args = SimpleNamespace(
        input_path="data/text8",
        text_input=None,
        model_name="distilbert/distilgpt2",
        engine="vllm",
        encoding="AC",
        context_length=256,
        retain_tokens=64,
        first_n_tokens=21,
        use_kv_cache=True,
        reduce_tokens=True,
        batch_size=1,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.3,
        lora_path=None,
        output_path="/tmp/test_vllm_roundtrip.bin",
    )

    # Get expected tokens
    preparer = TokenDataPreparer(args)
    expected_tokens = preparer.get_data_tokens()
    expected_text = preparer.tokenizer.decode(expected_tokens)

    # Compress
    first_tokens, bit_string, bitmask_data, stats, args = (
        run_global_mask_compression(args)
    )

    # Decompress
    reconstructed_tokens, detoken_string, decomp_stats = (
        run_global_mask_decompression(
            args,
            first_tokens=first_tokens,
            bit_string=bit_string,
            bitmap=bitmask_data,
        )
    )

    assert reconstructed_tokens == expected_tokens, (
        f"Token mismatch!\n"
        f"Expected ({len(expected_tokens)}): {expected_tokens[:30]}...\n"
        f"Got      ({len(reconstructed_tokens)}): {reconstructed_tokens[:30]}..."
    )
    assert detoken_string == expected_text
