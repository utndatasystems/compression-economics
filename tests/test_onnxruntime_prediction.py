"""Tests for the ONNX Runtime prediction module."""

from types import SimpleNamespace

import torch


def _make_args(**overrides):
    values = {
        "model_name": "distilbert/distilgpt2",
        "engine": "onnxruntime",
        "encoding": "AC",
        "context_length": 256,
        "batch_size": 1,
        "reduce_tokens": True,
        "onnx_model_dir": "/tmp/model-onnx",
        "onnx_tokenizer_source": None,
        "onnx_execution_provider": "CPUExecutionProvider",
        "onnx_intra_op_threads": 4,
        "onnx_inter_op_threads": 1,
        "onnx_graph_optimization_level": "ORT_ENABLE_ALL",
        "lora_path": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_probe_onnxruntime_backend_support_requires_model_dir():
    from src.onnxruntime_prediction import probe_onnxruntime_backend_support

    args = _make_args(onnx_model_dir=None)
    supported, reason = probe_onnxruntime_backend_support(args)

    assert supported is False
    assert "onnx_model_dir" in reason


def test_probe_onnxruntime_backend_support_requires_onnx_artifacts(tmp_path):
    from src.onnxruntime_prediction import probe_onnxruntime_backend_support

    args = _make_args(onnx_model_dir=str(tmp_path))
    supported, reason = probe_onnxruntime_backend_support(args)

    assert supported is False
    assert "No ONNX model files" in reason


def test_get_onnx_tokenizer_source_prefers_local_artifact_dir(tmp_path):
    from src.onnxruntime_prediction import get_onnx_tokenizer_source

    (tmp_path / "tokenizer.json").write_text("{}")
    args = _make_args(onnx_model_dir=str(tmp_path), model_name="fallback/model")

    assert get_onnx_tokenizer_source(args) == str(tmp_path)


def test_build_onnx_session_options_applies_requested_settings():
    from src.onnxruntime_prediction import build_onnx_session_options

    class FakeSessionOptions:
        def __init__(self):
            self.intra_op_num_threads = 0
            self.inter_op_num_threads = 0
            self.graph_optimization_level = None

    class FakeGraphOptimizationLevel:
        ORT_ENABLE_ALL = "all"
        ORT_DISABLE_ALL = "none"

    class FakeOrt:
        SessionOptions = FakeSessionOptions
        GraphOptimizationLevel = FakeGraphOptimizationLevel

    args = _make_args(
        onnx_intra_op_threads=6,
        onnx_inter_op_threads=2,
        onnx_graph_optimization_level="ORT_DISABLE_ALL",
    )

    options = build_onnx_session_options(args, FakeOrt)

    assert options.intra_op_num_threads == 6
    assert options.inter_op_num_threads == 2
    assert options.graph_optimization_level == "none"


def test_create_predictor_routes_onnxruntime_backend(tmp_path, monkeypatch):
    from src.onnxruntime_prediction import ONNXRuntimeTokenPredictor
    from src.prediction import create_predictor

    class FakePredictor(ONNXRuntimeTokenPredictor):
        def __init__(self, args, bitmap_data):
            self.args = args
            self.bitmap_data = bitmap_data

    (tmp_path / "model.onnx").write_bytes(b"onnx")
    args = _make_args(onnx_model_dir=str(tmp_path))

    monkeypatch.setattr(
        "src.onnxruntime_prediction.probe_onnxruntime_backend_support",
        lambda args: (True, None),
    )
    monkeypatch.setattr("src.onnxruntime_prediction.ONNXRuntimeTokenPredictor", FakePredictor)

    predictor = create_predictor(args, bitmap_data=b"bitmap")

    assert isinstance(predictor, FakePredictor)
    assert predictor.bitmap_data == b"bitmap"


def test_run_batched_inference_supplies_attention_mask():
    from src.onnxruntime_prediction import ONNXRuntimeTokenPredictor

    predictor = ONNXRuntimeTokenPredictor.__new__(ONNXRuntimeTokenPredictor)
    predictor.args = _make_args(encoding="AC", reduce_tokens=False)
    predictor.tokens_list = [0, 1, 2]
    predictor.index_tensor = torch.tensor(predictor.tokens_list, dtype=torch.long)
    predictor.reduce_tokens = False
    predictor._past_key_values = None
    predictor._cached_prompt_len = 0
    predictor._cached_prompts = None

    class FakeOutputs:
        def __init__(self):
            self.logits = torch.tensor([[[0.1, 0.2, 0.3]]], dtype=torch.float32)
            self.past_key_values = ((torch.tensor([1.0]), torch.tensor([2.0])),)

    calls = []

    class FakeModel:
        def __call__(self, **kwargs):
            calls.append(kwargs)
            return FakeOutputs()

    predictor.model = FakeModel()

    _, probs, _, _ = predictor.run_batched_inference([[11, 12, 13]], enable_kv_cache=True)

    assert len(calls) == 1
    assert "attention_mask" in calls[0]
    assert torch.equal(calls[0]["attention_mask"], torch.ones((1, 3), dtype=torch.long))
    assert probs.shape == (1, 3)


def test_run_batched_inference_rebuilds_cache_when_prompt_window_slides():
    from src.onnxruntime_prediction import ONNXRuntimeTokenPredictor

    predictor = ONNXRuntimeTokenPredictor.__new__(ONNXRuntimeTokenPredictor)
    predictor.args = _make_args(encoding="AC", reduce_tokens=False)
    predictor.tokens_list = [0, 1, 2]
    predictor.index_tensor = torch.tensor(predictor.tokens_list, dtype=torch.long)
    predictor.reduce_tokens = False
    predictor._past_key_values = None
    predictor._cached_prompt_len = 0
    predictor._cached_prompts = None

    class FakeOutputs:
        def __init__(self):
            self.logits = torch.tensor([[[0.1, 0.2, 0.3]]], dtype=torch.float32)
            self.past_key_values = ((torch.tensor([1.0]), torch.tensor([2.0])),)

    calls = []

    class FakeModel:
        def __call__(self, **kwargs):
            calls.append(kwargs)
            return FakeOutputs()

    predictor.model = FakeModel()

    predictor.run_batched_inference([[1, 2, 3]], enable_kv_cache=True)
    predictor.run_batched_inference([[2, 3, 4]], enable_kv_cache=True)

    assert torch.equal(calls[0]["input_ids"], torch.tensor([[1, 2, 3]], dtype=torch.long))
    assert torch.equal(calls[1]["input_ids"], torch.tensor([[2, 3, 4]], dtype=torch.long))
    assert "past_key_values" not in calls[1]