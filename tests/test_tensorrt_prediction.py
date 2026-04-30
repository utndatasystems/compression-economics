"""Tests for TensorRT prediction module (src/tensorrt_prediction.py)."""

from types import SimpleNamespace

import pytest
import torch


def test_probe_tensorrt_ac_support_requires_engine_dir():
    """Probe should report a missing engine directory clearly."""
    from src.tensorrt_prediction import probe_tensorrt_ac_support

    args = SimpleNamespace(model_name="distilbert/distilgpt2", tensorrt_engine_dir=None)
    supported, reason = probe_tensorrt_ac_support(args)

    assert supported is False
    assert isinstance(reason, str)
    assert "tensorrt_llm" in reason or "tensorrt_engine_dir" in reason


def test_extract_last_step_logits_prefers_generation_logits():
    """generation_logits should be normalized to a [batch, vocab] tensor."""
    from src.tensorrt_prediction import extract_last_step_logits

    generation_logits = torch.arange(2 * 3 * 5, dtype=torch.float32).reshape(2, 3, 5)
    payload = {"generation_logits": generation_logits}

    logits = extract_last_step_logits(payload, prompt_lengths=[4, 4])

    assert logits.shape == (2, 5)
    assert torch.equal(logits[0], generation_logits[0, -1, :])
    assert torch.equal(logits[1], generation_logits[1, -1, :])


def test_extract_last_step_logits_uses_context_lengths():
    """context_logits fallback should respect per-prompt lengths."""
    from src.tensorrt_prediction import extract_last_step_logits

    context_logits = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3)
    payload = {"context_logits": context_logits}

    logits = extract_last_step_logits(payload, prompt_lengths=[2, 4])

    assert logits.shape == (2, 3)
    assert torch.equal(logits[0], context_logits[0, 1, :])
    assert torch.equal(logits[1], context_logits[1, 3, :])


def test_extract_last_step_logits_rejects_missing_logits():
    """Missing logits fields should raise a helpful runtime error."""
    from src.tensorrt_prediction import extract_last_step_logits

    with pytest.raises(RuntimeError, match="generation_logits or context_logits"):
        extract_last_step_logits({"output_ids": torch.tensor([[1]])})