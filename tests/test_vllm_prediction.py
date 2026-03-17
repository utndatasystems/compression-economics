"""
Tests for the vLLM prediction backend.

These tests verify shape/interface contracts of VLLMTokenPredictor.
GPU-dependent tests are automatically skipped when vLLM is not installed.
"""

import os
import inspect

import pytest
import torch
import numpy as np

import src.prediction as prediction_module

try:
    from src import vllm_prediction
    from src.vllm_prediction import (
        MIN_NATIVE_AC_VLLM_VERSION,
        VLLMTokenPredictor,
        VLLM_AVAILABLE,
        VLLMFullLogitsAdapter,
        VLLM_FULL_LOGITS_CAPTURE_ENV,
        VLLM_AC_LOGIT_QUANTIZATION_STEP,
        _maybe_select_vllm_gpus,
        _resolve_gpu_memory_utilization,
        prompt_signature,
        probe_vllm_ac_support,
        stabilize_ac_logits,
    )
except ImportError:
    VLLM_AVAILABLE = False
    stabilize_ac_logits = None
    VLLM_AC_LOGIT_QUANTIZATION_STEP = 0.01

skip_no_vllm = pytest.mark.skipif(
    not VLLM_AVAILABLE,
    reason="vLLM not installed — skipping vLLM tests",
)

skip_no_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available — skipping GPU tests",
)


class MockArgs:
    """Minimal args namespace for VLLMTokenPredictor tests."""

    def __init__(self, **kwargs):
        defaults = {
            "model_name": "Qwen/Qwen3-0.6B",
            "engine": "vllm",
            "encoding": "AC",
            "reduce_tokens": False,
            "tensor_parallel_size": 1,
            "gpu_memory_utilization": None,
            "lora_path": None,
            "use_kv_cache": True,
        }
        defaults.update(kwargs)
        for k, v in defaults.items():
            setattr(self, k, v)


def test_select_vllm_gpus_prefers_freest(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("CUDA_DEVICE_ORDER", raising=False)
    monkeypatch.setattr(
        vllm_prediction,
        "_query_nvidia_smi_gpus",
        lambda: [
            {"index": 0, "total_gib": 40.0, "free_gib": 39.5, "free_fraction": 0.9875},
            {"index": 2, "total_gib": 80.0, "free_gib": 6.5, "free_fraction": 0.08125},
            {"index": 3, "total_gib": 80.0, "free_gib": 79.2, "free_fraction": 0.99},
        ],
    )

    selected = _maybe_select_vllm_gpus(1)

    assert [gpu["index"] for gpu in selected] == [3]
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "3"
    assert os.environ["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"


def test_resolve_gpu_memory_utilization_clamps_to_available_memory():
    args = MockArgs(gpu_memory_utilization=None)

    utilization = _resolve_gpu_memory_utilization(
        args,
        [{"index": 2, "total_gib": 80.0, "free_gib": 6.0, "free_fraction": 0.075}],
    )

    assert utilization == pytest.approx(0.062, abs=1e-3)


def test_resolve_gpu_memory_utilization_respects_lower_explicit_value():
    args = MockArgs(gpu_memory_utilization=0.2)

    utilization = _resolve_gpu_memory_utilization(
        args,
        [{"index": 3, "total_gib": 80.0, "free_gib": 79.0, "free_fraction": 0.9875}],
    )

    assert utilization == pytest.approx(0.2, abs=1e-6)


def test_probe_vllm_ac_support_accepts_verified_version(monkeypatch):
    monkeypatch.setattr(vllm_prediction, "VLLM_AVAILABLE", True)
    monkeypatch.setattr(
        vllm_prediction,
        "_get_vllm_version",
        lambda: MIN_NATIVE_AC_VLLM_VERSION,
    )
    monkeypatch.setattr(
        vllm_prediction.inspect,
        "signature",
        lambda _obj: inspect.Signature(
            parameters=[
                inspect.Parameter(
                    "self",
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                ),
                inspect.Parameter(
                    "model",
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                ),
                inspect.Parameter(
                    "logits_processors",
                    inspect.Parameter.KEYWORD_ONLY,
                    default=None,
                ),
            ]
        ),
    )

    supported, reason = probe_vllm_ac_support(MockArgs())

    assert supported is True
    assert reason == "native-full-logits-adapter-available"


def test_probe_vllm_ac_support_rejects_tensor_parallel(monkeypatch):
    monkeypatch.setattr(vllm_prediction, "VLLM_AVAILABLE", True)
    monkeypatch.setattr(
        vllm_prediction,
        "_get_vllm_version",
        lambda: MIN_NATIVE_AC_VLLM_VERSION,
    )
    monkeypatch.setattr(
        vllm_prediction.inspect,
        "signature",
        lambda _obj: inspect.Signature(
            parameters=[
                inspect.Parameter(
                    "self",
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                ),
                inspect.Parameter(
                    "model",
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                ),
                inspect.Parameter(
                    "logits_processors",
                    inspect.Parameter.KEYWORD_ONLY,
                    default=None,
                ),
            ]
        ),
    )

    supported, reason = probe_vllm_ac_support(MockArgs(tensor_parallel_size=2))

    assert supported is False
    assert "tensor_parallel_size=1" in reason


def test_full_logits_adapter_sets_capture_env(monkeypatch):
    monkeypatch.setattr(vllm_prediction, "VLLM_AVAILABLE", True)
    monkeypatch.setattr(
        vllm_prediction,
        "probe_vllm_ac_support",
        lambda args=None: (True, "native-full-logits-adapter-available"),
    )

    adapter = VLLMFullLogitsAdapter(MockArgs())

    assert os.environ[VLLM_FULL_LOGITS_CAPTURE_ENV] == adapter.capture_path
    assert adapter.get_llm_init_kwargs()["logits_processors"] == [
        "src.vllm_prediction:VLLMFullLogitsCaptureProcessor"
    ]
    adapter.clear_capture()


def test_stabilize_ac_logits_quantizes_small_jitter():
    if stabilize_ac_logits is None:
        pytest.skip("vLLM helpers are not available")

    logits = torch.tensor(
        [[1.2341, -0.9912, 0.5001], [1.2398, -0.9904, 0.4997]],
        dtype=torch.float32,
    )

    stabilized = stabilize_ac_logits(logits)

    expected = torch.round(logits / VLLM_AC_LOGIT_QUANTIZATION_STEP) * VLLM_AC_LOGIT_QUANTIZATION_STEP
    assert torch.equal(stabilized, expected)


def test_reorder_logits_by_prompt_signature_restores_input_order(monkeypatch):
    monkeypatch.setattr(vllm_prediction, "VLLM_AVAILABLE", True)
    monkeypatch.setattr(
        vllm_prediction,
        "probe_vllm_ac_support",
        lambda args=None: (True, "native-full-logits-adapter-available"),
    )

    adapter = VLLMFullLogitsAdapter(MockArgs())
    logits = torch.tensor([[10.0, 0.0], [20.0, 0.0], [30.0, 0.0]], dtype=torch.float32)
    captured_signatures = [
        prompt_signature([3]),
        prompt_signature([1]),
        prompt_signature([2]),
    ]
    expected_signatures = [
        prompt_signature([1]),
        prompt_signature([2]),
        prompt_signature([3]),
    ]

    reordered = adapter._reorder_logits_by_prompt_signature(
        logits,
        captured_signatures,
        expected_signatures,
    )

    assert torch.equal(reordered[:, 0], torch.tensor([20.0, 30.0, 10.0]))


def test_create_predictor_falls_back_for_unsupported_vllm_ac(monkeypatch):
    fallback_predictors = []

    class FakeTokenPredictor:
        def __init__(self, args, bitmap_data):
            fallback_predictors.append((args.engine, bitmap_data))
            self.args = args

    monkeypatch.setattr(prediction_module, "TokenPredictor", FakeTokenPredictor)
    monkeypatch.setattr(
        vllm_prediction,
        "probe_vllm_ac_support",
        lambda args=None: (False, "missing native logits adapter"),
    )

    predictor = prediction_module.create_predictor(
        MockArgs(engine="vllm", encoding="AC"),
        bitmap_data=None,
    )

    assert isinstance(predictor, FakeTokenPredictor)
    assert fallback_predictors == [("transformer", None)]


def test_create_predictor_keeps_vllm_for_supported_ac(monkeypatch):
    class FakeTokenPredictor:
        def __init__(self, args, bitmap_data):
            raise AssertionError("transformer fallback should not be used")

    class FakeVLLMTokenPredictor:
        def __init__(self, args, bitmap_data):
            self.args = args
            self.bitmap_data = bitmap_data

    monkeypatch.setattr(prediction_module, "TokenPredictor", FakeTokenPredictor)
    monkeypatch.setattr(vllm_prediction, "VLLMTokenPredictor", FakeVLLMTokenPredictor)
    monkeypatch.setattr(
        vllm_prediction,
        "probe_vllm_ac_support",
        lambda args=None: (True, "native-full-logits-adapter-available"),
    )

    predictor = prediction_module.create_predictor(
        MockArgs(engine="vllm", encoding="AC"),
        bitmap_data=b"bitmap",
    )

    assert isinstance(predictor, FakeVLLMTokenPredictor)
    assert predictor.bitmap_data == b"bitmap"


@skip_no_vllm
@skip_no_gpu
class TestVLLMTokenPredictor:
    """Integration tests that require a real GPU and vLLM install."""

    @pytest.fixture(scope="class")
    def predictor(self):
        args = MockArgs(encoding="AC")
        return VLLMTokenPredictor(args, bitmap_data=None)

    def test_run_batched_inference_ac_shape(self, predictor):
        """AC mode returns CPU probabilities summing to ~1."""
        prompts = [[1, 2, 3], [4, 5, 6]]
        tokens_list, probs, dc_time, sm_time = predictor.run_batched_inference(prompts)

        assert isinstance(probs, torch.Tensor)
        assert probs.shape[0] == 2  # batch size
        assert probs.shape[1] == len(tokens_list)
        assert probs.device == torch.device("cpu")
        # Each row should sum to ~1
        row_sums = probs.sum(dim=-1)
        np.testing.assert_allclose(row_sums.numpy(), 1.0, atol=1e-4)

    def test_run_batched_inference_rank_shape(self):
        """Rank mode returns logits (not probabilities)."""
        args = MockArgs(encoding="bitpacked")
        predictor = VLLMTokenPredictor(args, bitmap_data=None)
        prompts = [[1, 2, 3]]
        tokens_list, logits, dc_time, sm_time = predictor.run_batched_inference(prompts)

        assert isinstance(logits, torch.Tensor)
        assert logits.shape[0] == 1
        assert logits.shape[1] == len(tokens_list)
        # Logits should NOT sum to 1 (they're raw, not softmax)
        assert logits.sum(dim=-1).item() != pytest.approx(1.0, abs=0.1)

    def test_detokenize(self, predictor):
        """Verify detokenize returns a string."""
        result = predictor.detokenize([1, 2, 3])
        assert isinstance(result, str)

    def test_get_token_by_id(self, predictor):
        """Verify token lookup returns an integer."""
        token = predictor.get_token_by_id(0)
        assert isinstance(token, int)

    def test_lora_rejected(self):
        """Verify that lora_path raises ValueError."""
        args = MockArgs(lora_path="/fake/path")
        with pytest.raises(ValueError, match="LoRA"):
            VLLMTokenPredictor(args, bitmap_data=None)

    def test_ac_distribution_matches_transformer_backend(self):
        from src.prediction import TokenPredictor

        transformer = TokenPredictor(MockArgs(engine="transformer", encoding="AC"), bitmap_data=None)
        predictor = VLLMTokenPredictor(MockArgs(encoding="AC"), bitmap_data=None)
        prompts = [[1, 2, 3, 4], [11, 12, 13, 14]]

        token_ids_vllm, probs_vllm, _, _ = predictor.run_batched_inference(prompts)
        token_ids_hf, probs_hf, _, _ = transformer.run_batched_inference(prompts)

        assert token_ids_vllm == token_ids_hf
        np.testing.assert_allclose(probs_vllm.sum(dim=-1).numpy(), 1.0, atol=1e-4)
        np.testing.assert_allclose(probs_hf.sum(dim=-1).numpy(), 1.0, atol=1e-4)

        top_vllm = probs_vllm.argmax(dim=-1).numpy()
        top_hf = probs_hf.argmax(dim=-1).numpy()
        np.testing.assert_array_equal(top_vllm, top_hf)

        chosen_indices = np.array([top_vllm[0], top_vllm[1]])
        gathered_vllm = probs_vllm.numpy()[np.arange(len(prompts)), chosen_indices]
        gathered_hf = probs_hf.numpy()[np.arange(len(prompts)), chosen_indices]
        np.testing.assert_allclose(gathered_vllm, gathered_hf, rtol=0.15, atol=5e-3)


@skip_no_vllm
@skip_no_gpu
class TestVLLMBulkRank:
    """Tests for the bulk rank inference path."""

    @pytest.fixture(scope="class")
    def predictor(self):
        args = MockArgs(encoding="bitpacked")
        return VLLMTokenPredictor(args, bitmap_data=None)

    def test_bulk_rank_returns_list(self, predictor):
        """Bulk rank returns a list of integers."""
        # Use a small token sequence
        tokens = list(range(10, 30))  # 20 tokens
        rank_list, inf_time = predictor.run_bulk_rank_inference(tokens, batch_size=2)

        assert isinstance(rank_list, list)
        assert all(isinstance(r, int) for r in rank_list)
        assert inf_time > 0
        # Should have ranks for tokens at positions 1..N-1 in each chunk
        # 20 tokens / 2 batches = 10 per chunk, 9 ranks each = 18 total
        assert len(rank_list) == 18

    def test_bulk_rank_values_nonnegative(self, predictor):
        """All ranks should be >= 0."""
        tokens = list(range(10, 30))
        rank_list, _ = predictor.run_bulk_rank_inference(tokens, batch_size=2)
        assert all(r >= 0 for r in rank_list)
