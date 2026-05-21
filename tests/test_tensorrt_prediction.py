import sys
import types
from types import SimpleNamespace

import pytest
import torch

from src.prediction import get_token_predictor
from src.tensorrt_prediction import TensorRTTokenPredictor, extract_last_step_logits


class FakeCompletion:
    def __init__(self, logits):
        self.generation_logits = logits


class FakeRequestOutput:
    def __init__(self, logits):
        self.outputs = [FakeCompletion(logits)]


def make_predictor(encoding="AC", reduce_tokens=False, tokens_list=None):
    predictor = TensorRTTokenPredictor.__new__(TensorRTTokenPredictor)
    predictor.args = SimpleNamespace(encoding=encoding)
    predictor.tokens_list = tokens_list or [0, 1, 2, 3]
    predictor.vocab_size = 4
    predictor.reduce_tokens = reduce_tokens
    predictor.index_tensor = torch.tensor(predictor.tokens_list, dtype=torch.long)
    predictor.device = torch.device("cpu")
    predictor.sampling_params = object()
    predictor._sampling_config = None
    predictor.runner = None
    predictor.runner_name = None
    return predictor


def test_tensorrt_ac_reduced_scores_are_softmaxed_on_cpu():
    predictor = make_predictor(
        encoding="AC",
        reduce_tokens=True,
        tokens_list=[1, 3],
    )

    class FakeLLM:
        def generate(self, prompts, sampling_params, use_tqdm=False):
            del prompts, sampling_params, use_tqdm
            return [
                FakeRequestOutput(torch.tensor([[0.0, 1.0, 2.0, 3.0]])),
                FakeRequestOutput(torch.tensor([[3.0, 2.0, 1.0, 0.0]])),
            ]

    predictor.llm = FakeLLM()

    tokens_list, probs, _, _ = predictor.run_batched_inference([[7], [8]])

    assert tokens_list == [1, 3]
    assert probs.device.type == "cpu"
    assert probs.shape == (2, 2)
    assert torch.allclose(probs.sum(dim=1), torch.ones(2))
    assert probs[0].tolist() == pytest.approx(
        torch.softmax(torch.tensor([1.0, 3.0]), dim=-1).tolist()
    )


def test_tensorrt_rank_encodings_return_raw_scores():
    predictor = make_predictor(
        encoding="bitpacked",
        reduce_tokens=True,
        tokens_list=[0, 2],
    )

    class FakeLLM:
        def generate(self, prompts, sampling_params, use_tqdm=False):
            del prompts, sampling_params, use_tqdm
            return [FakeRequestOutput(torch.tensor([[0.5, 1.5, 2.5, 3.5]]))]

    predictor.llm = FakeLLM()

    tokens_list, scores, _, softmax_time = predictor.run_batched_inference([[7]])

    assert tokens_list == [0, 2]
    assert scores.shape == (1, 2)
    assert scores[0].tolist() == pytest.approx([0.5, 2.5])
    assert softmax_time == 0.0


def test_get_token_predictor_dispatches_to_tensorrt(monkeypatch):
    created = []

    class FakeTensorRTTokenPredictor:
        def __init__(self, args, bitmap_data):
            created.append((args, bitmap_data))

    fake_module = types.SimpleNamespace(
        TensorRTTokenPredictor=FakeTensorRTTokenPredictor
    )
    monkeypatch.setitem(sys.modules, "src.tensorrt_prediction", fake_module)

    args = SimpleNamespace(engine="tensorrt")
    predictor = get_token_predictor(args, bitmap_data=b"bitmap")

    assert isinstance(predictor, FakeTensorRTTokenPredictor)
    assert created == [(args, b"bitmap")]


def test_tensorrt_predictor_reuses_cached_engine(monkeypatch, tmp_path):
    engine_dir = tmp_path / "engine"
    engine_dir.mkdir()
    (engine_dir / "config.json").write_text("{}", encoding="utf-8")
    created = []

    class FakeTokenizer:
        vocab_size = 4
        pad_token_id = None
        eos_token_id = 0

        def decode(self, token_ids):
            return str(token_ids)

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(model_name, cache_dir=None):
            del model_name, cache_dir
            return FakeTokenizer()

    class FakeAutoConfig:
        vocab_size = 4
        hidden_size = 2
        num_hidden_layers = 1

        @staticmethod
        def from_pretrained(model_name, cache_dir=None):
            del model_name, cache_dir
            return FakeAutoConfig()

    class FakeLLM:
        def __init__(self, **kwargs):
            created.append(kwargs)

        def save(self, engine_dir):
            raise AssertionError(f"cached engine should not be rebuilt: {engine_dir}")

    class FakeSamplingParams:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    fake_tllm = types.ModuleType("tensorrt_llm")
    fake_tllm.LLM = FakeLLM
    fake_tllm.SamplingParams = FakeSamplingParams
    fake_trt_engine = types.ModuleType("tensorrt_llm._tensorrt_engine")
    fake_trt_engine.LLM = FakeLLM
    real_tensor = torch.tensor

    def fake_tensor(data, *args, **kwargs):
        kwargs.pop("device", None)
        return real_tensor(data, *args, **kwargs)

    monkeypatch.setattr(
        "src.tensorrt_prediction.probe_tensorrt_backend_support",
        lambda args: (True, None),
    )
    monkeypatch.setattr("src.tensorrt_prediction.AutoTokenizer", FakeAutoTokenizer)
    monkeypatch.setattr("src.tensorrt_prediction.AutoConfig", FakeAutoConfig)
    monkeypatch.setattr("src.tensorrt_prediction.torch.tensor", fake_tensor)
    monkeypatch.setitem(sys.modules, "tensorrt_llm", fake_tllm)
    monkeypatch.setitem(sys.modules, "tensorrt_llm._tensorrt_engine", fake_trt_engine)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    args = SimpleNamespace(
        model_name="Qwen/Qwen2.5-0.5B",
        batch_size=512,
        context_length=256,
        reduce_tokens=False,
        encoding="AC",
        is_seq2seq=False,
        is_mamba=False,
        lora_path=None,
        tensorrt_engine_dir=str(engine_dir),
    )

    predictor = TensorRTTokenPredictor(args, bitmap_data=None)

    assert predictor.engine_dir == engine_dir
    assert created == [
        {
            "model": str(engine_dir),
            "tokenizer": "Qwen/Qwen2.5-0.5B",
            "gather_generation_logits": True,
        }
    ]


def test_tensorrt_model_runner_path_uses_runtime_generate():
    predictor = make_predictor(
        encoding="AC_MULTISTREAM",
        reduce_tokens=True,
        tokens_list=[1, 3],
    )
    predictor.pad_token_id = 0
    predictor.eos_token_id = 0
    predictor._sampling_config = SimpleNamespace(top_k=1, top_p=0.0, temperature=1.0)
    calls = []

    class FakeRunner:
        def generate(self, **kwargs):
            calls.append(kwargs)
            return {
                "generation_logits": torch.tensor(
                    [
                        [[0.0, 1.0, 2.0, 3.0]],
                        [[3.0, 2.0, 1.0, 0.0]],
                    ]
                )
            }

    predictor.runner = FakeRunner()
    predictor.runner_name = "ModelRunnerCpp"
    predictor.llm = None

    tokens_list, probs, _, _ = predictor.run_batched_inference([[7], [8]])

    assert tokens_list == [1, 3]
    assert probs.device.type == "cpu"
    assert probs.shape == (2, 2)
    assert len(calls) == 1
    assert calls[0]["return_dict"] is True
    assert calls[0]["output_generation_logits"] is True
    assert calls[0]["max_new_tokens"] == 1
    assert calls[0]["top_k"] == 1
    assert [tensor.tolist() for tensor in calls[0]["batch_input_ids"]] == [[7], [8]]


def test_extract_last_step_logits_handles_ranked_payloads():
    two_d = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    assert torch.equal(extract_last_step_logits(two_d), two_d)

    three_d = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
        ]
    )
    selected = extract_last_step_logits(
        {"generation_logits": three_d},
        prompt_lengths=[1, 2],
    )

    assert selected.tolist() == [[1.0, 2.0], [7.0, 8.0]]


def test_tensorrt_import_prefers_tensorrt_backend(monkeypatch):
    class TopLevelLLM:
        pass

    class TrtLLM:
        pass

    class FakeSamplingParams:
        pass

    fake_tllm = types.ModuleType("tensorrt_llm")
    fake_tllm.LLM = TopLevelLLM
    fake_tllm.SamplingParams = FakeSamplingParams
    fake_trt_engine = types.ModuleType("tensorrt_llm._tensorrt_engine")
    fake_trt_engine.LLM = TrtLLM
    monkeypatch.setitem(sys.modules, "tensorrt_llm", fake_tllm)
    monkeypatch.setitem(sys.modules, "tensorrt_llm._tensorrt_engine", fake_trt_engine)

    llm_cls, sampling_params_cls = TensorRTTokenPredictor._import_tensorrt_llm_api()

    assert llm_cls is TrtLLM
    assert sampling_params_cls is FakeSamplingParams


def test_tensorrt_cache_built_engine_without_save(tmp_path):
    built_engine_dir = tmp_path / "built"
    built_engine_dir.mkdir()
    (built_engine_dir / "config.json").write_text("{}", encoding="utf-8")
    target_engine_dir = tmp_path / "target"

    predictor = TensorRTTokenPredictor.__new__(TensorRTTokenPredictor)
    predictor.llm = SimpleNamespace(_engine_dir=built_engine_dir)

    predictor._cache_built_engine(target_engine_dir)

    assert (target_engine_dir / "config.json").exists()
