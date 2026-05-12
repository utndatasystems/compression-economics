"""Tests for TensorRT prediction module (src/tensorrt_prediction.py)."""

import sys
from types import ModuleType
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


def test_build_sampling_config_adds_legacy_beam_width(monkeypatch):
    """SamplingConfig should expose beam_width for ModelRunnerCpp compatibility."""
    from src.tensorrt_prediction import TensorRTTokenPredictor

    class FakeSamplingConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
            self.num_beams = kwargs.get("num_beams", 1)

    fake_runtime = ModuleType("tensorrt_llm.runtime")
    fake_runtime.SamplingConfig = FakeSamplingConfig
    fake_package = ModuleType("tensorrt_llm")
    fake_package.runtime = fake_runtime

    monkeypatch.setitem(sys.modules, "tensorrt_llm", fake_package)
    monkeypatch.setitem(sys.modules, "tensorrt_llm.runtime", fake_runtime)

    predictor = TensorRTTokenPredictor.__new__(TensorRTTokenPredictor)
    predictor.eos_token_id = 2
    predictor.pad_token_id = 1

    sampling_config = predictor._build_sampling_config()

    assert sampling_config is not None
    assert sampling_config.num_beams == 1
    assert sampling_config.beam_width == 1


def test_run_batched_inference_uses_legacy_sampling_kwargs_for_cpp_runner():
    """ModelRunnerCpp should receive kwargs instead of a SamplingConfig object."""
    from src.tensorrt_prediction import TensorRTTokenPredictor

    class FakeRunner:
        def __init__(self):
            self.kwargs = None

        def generate(
            self,
            batch_input_ids,
            return_dict=False,
            output_generation_logits=False,
            streaming=False,
            max_new_tokens=1,
            end_id=None,
            pad_id=None,
            **kwargs,
        ):
            self.kwargs = {
                "batch_input_ids": batch_input_ids,
                "return_dict": return_dict,
                "output_generation_logits": output_generation_logits,
                "streaming": streaming,
                "max_new_tokens": max_new_tokens,
                "end_id": end_id,
                "pad_id": pad_id,
                **kwargs,
            }
            raise RuntimeError("stop")

    predictor = TensorRTTokenPredictor.__new__(TensorRTTokenPredictor)
    predictor.device = "cpu"
    predictor.runner_name = "ModelRunnerCpp"
    predictor.runner = FakeRunner()
    predictor.eos_token_id = 2
    predictor.pad_token_id = 1
    predictor.tokenizer = SimpleNamespace(vocab_size=8)
    predictor.reduce_tokens = False
    predictor.args = SimpleNamespace(encoding="AC")
    predictor._sampling_config = SimpleNamespace(
        num_beams=1,
        beam_width=1,
        top_k=1,
        top_p=0.0,
        temperature=1.0,
        num_return_sequences=None,
        random_seed=None,
    )

    with pytest.raises(RuntimeError, match="stop"):
        predictor.run_batched_inference([[1, 2, 3]])

    assert "sampling_config" not in predictor.runner.kwargs
    assert predictor.runner.kwargs["num_beams"] == 1
    assert predictor.runner.kwargs["top_k"] == 1
    assert predictor.runner.kwargs["top_p"] == 0.0
    assert predictor.runner.kwargs["temperature"] == 1.0