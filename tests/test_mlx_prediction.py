"""Tests for the MLX prediction module."""

from types import SimpleNamespace

import numpy as np
import torch


def _make_args(**overrides):
    values = {
        "model_name": "mlx-community/Llama-3.2-1B-Instruct-4bit",
        "engine": "mlx",
        "encoding": "AC",
        "context_length": 256,
        "batch_size": 1,
        "reduce_tokens": True,
        "mlx_model_source": None,
        "mlx_tokenizer_source": None,
        "lora_path": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_probe_mlx_backend_support_requires_macos():
    from src.mlx_prediction import probe_mlx_backend_support

    supported, reason = probe_mlx_backend_support(_make_args())

    assert supported is False
    assert "macOS" in reason


def test_get_mlx_tokenizer_source_prefers_explicit_source():
    from src.mlx_prediction import get_mlx_tokenizer_source

    args = _make_args(mlx_tokenizer_source="/tmp/mlx-tokenizer")

    assert get_mlx_tokenizer_source(args) == "/tmp/mlx-tokenizer"


def test_estimate_mlx_parameter_count_reads_local_metadata(tmp_path):
    from src.mlx_prediction import estimate_mlx_parameter_count

    (tmp_path / "model.safetensors.index.json").write_text(
        '{"metadata": {"total_parameters": 123456}}',
        encoding="utf-8",
    )

    args = _make_args(mlx_model_source=str(tmp_path))

    assert estimate_mlx_parameter_count(args) == 123456


def test_create_predictor_routes_mlx_backend(monkeypatch):
    from src.mlx_prediction import MLXTokenPredictor
    from src.prediction import create_predictor

    class FakePredictor(MLXTokenPredictor):
        def __init__(self, args, bitmap_data):
            self.args = args
            self.bitmap_data = bitmap_data

    monkeypatch.setattr(
        "src.mlx_prediction.probe_mlx_backend_support",
        lambda args: (True, None),
    )
    monkeypatch.setattr("src.mlx_prediction.MLXTokenPredictor", FakePredictor)

    predictor = create_predictor(_make_args(), bitmap_data=b"bitmap")

    assert isinstance(predictor, FakePredictor)
    assert predictor.bitmap_data == b"bitmap"


def test_run_batched_inference_advances_single_prompt_cache():
    from src.mlx_prediction import MLXTokenPredictor

    predictor = MLXTokenPredictor.__new__(MLXTokenPredictor)
    predictor.args = _make_args(encoding="AC", reduce_tokens=False)
    predictor.tokens_list = [0, 1, 2, 3, 4]
    predictor.index_tensor = torch.tensor(predictor.tokens_list, dtype=torch.long)
    predictor.reduce_tokens = False
    predictor._prompt_cache = None
    predictor._cached_prompts = None

    class FakeMx:
        uint32 = np.uint32

        @staticmethod
        def array(value, dtype=None):
            return np.array(value, dtype=dtype)

        @staticmethod
        def eval(*_values):
            return None

        @staticmethod
        def clear_cache():
            return None

    class FakeCacheModule:
        def __init__(self):
            self.created = []

        def make_prompt_cache(self, _model):
            cache = {"cache_id": len(self.created)}
            self.created.append(cache)
            return cache

    class FakeModel:
        def __init__(self):
            self.calls = []

        def __call__(self, input_ids, cache=None):
            self.calls.append({"input_ids": np.array(input_ids), "cache": cache})
            batch_size, prompt_len = input_ids.shape
            logits = np.zeros((batch_size, prompt_len, 5), dtype=np.float32)
            logits[..., :] = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)
            return logits

    predictor.mx = FakeMx()
    predictor._mlx_cache = FakeCacheModule()
    predictor.model = FakeModel()

    _, probs, _, _ = predictor.run_batched_inference([[1, 2, 3]], enable_kv_cache=True)
    _, _, _, _ = predictor.run_batched_inference([[1, 2, 3, 4]], enable_kv_cache=True)

    assert len(predictor._mlx_cache.created) == 1
    assert predictor.model.calls[0]["input_ids"].shape == (1, 3)
    assert predictor.model.calls[1]["input_ids"].shape == (1, 1)
    assert predictor.model.calls[0]["cache"] is predictor._mlx_cache.created[0]
    assert predictor.model.calls[1]["cache"] is predictor._mlx_cache.created[0]
    assert probs.shape == (1, 5)


def test_run_batched_inference_rebuilds_cache_when_prompt_window_slides():
    from src.mlx_prediction import MLXTokenPredictor

    predictor = MLXTokenPredictor.__new__(MLXTokenPredictor)
    predictor.args = _make_args(encoding="AC", reduce_tokens=False)
    predictor.tokens_list = [0, 1, 2, 3, 4]
    predictor.index_tensor = torch.tensor(predictor.tokens_list, dtype=torch.long)
    predictor.reduce_tokens = False
    predictor._prompt_cache = None
    predictor._cached_prompts = None

    class FakeMx:
        uint32 = np.uint32

        @staticmethod
        def array(value, dtype=None):
            return np.array(value, dtype=dtype)

        @staticmethod
        def eval(*_values):
            return None

        @staticmethod
        def clear_cache():
            return None

    class FakeCacheModule:
        def __init__(self):
            self.created = []

        def make_prompt_cache(self, _model):
            cache = {"cache_id": len(self.created)}
            self.created.append(cache)
            return cache

    class FakeModel:
        def __init__(self):
            self.calls = []

        def __call__(self, input_ids, cache=None):
            self.calls.append({"input_ids": np.array(input_ids), "cache": cache})
            batch_size, prompt_len = input_ids.shape
            logits = np.zeros((batch_size, prompt_len, 5), dtype=np.float32)
            logits[..., :] = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)
            return logits

    predictor.mx = FakeMx()
    predictor._mlx_cache = FakeCacheModule()
    predictor.model = FakeModel()

    predictor.run_batched_inference([[1, 2, 3]], enable_kv_cache=True)
    predictor.run_batched_inference([[2, 3, 4]], enable_kv_cache=True)

    assert len(predictor._mlx_cache.created) == 2
    assert predictor.model.calls[0]["input_ids"].shape == (1, 3)
    assert predictor.model.calls[1]["input_ids"].shape == (1, 3)