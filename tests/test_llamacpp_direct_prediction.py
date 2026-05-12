"""Tests for the direct llama.cpp prediction module."""

from types import SimpleNamespace

import torch


def _make_args(**overrides):
    values = {
        "model_name": "unused",
        "engine": "llamacpp_direct",
        "encoding": "AC",
        "context_length": 256,
        "batch_size": 1,
        "reduce_tokens": True,
        "llamacpp_model_path": "/tmp/model.gguf",
        "llamacpp_threads": 4,
        "llamacpp_direct_threads_batch": 0,
        "llamacpp_direct_n_batch": 0,
        "llamacpp_direct_n_ubatch": 0,
        "llamacpp_direct_use_mmap": True,
        "llamacpp_direct_use_mlock": False,
        "llamacpp_n_gpu_layers": 12,
        "lora_path": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class FakeLlamaModel:
    def __init__(self):
        self.current_prompt = ()
        self.eval_calls = []
        self.reset_calls = 0
        self.load_state_calls = []
        self.save_state_calls = []
        self.eval_logits = []

    def reset(self):
        self.reset_calls += 1
        self.current_prompt = ()

    def eval(self, tokens):
        tokens = tuple(int(token_id) for token_id in tokens)
        self.eval_calls.append(tokens)
        self.current_prompt = self.current_prompt + tokens
        total = float(sum(self.current_prompt))
        self.eval_logits = [[total + offset for offset in range(4)]]

    def save_state(self):
        state = {"prompt": self.current_prompt}
        self.save_state_calls.append(state["prompt"])
        return state

    def load_state(self, state):
        self.load_state_calls.append(state["prompt"])
        self.current_prompt = tuple(state["prompt"])

    def close(self):
        return None


def test_probe_llamacpp_direct_support_allows_full_vocab(tmp_path, monkeypatch):
    from src.llamacpp_direct_prediction import probe_llamacpp_direct_support

    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"gguf")
    monkeypatch.setattr(
        "src.llamacpp_direct_prediction._load_llamacpp_dependencies",
        lambda: SimpleNamespace(Llama=object),
    )

    supported, reason = probe_llamacpp_direct_support(
        _make_args(llamacpp_model_path=str(model_path), reduce_tokens=False)
    )

    assert supported is True
    assert reason is None


def test_create_predictor_routes_llamacpp_direct_backend(monkeypatch):
    from src.llamacpp_direct_prediction import LlamaCppDirectTokenPredictor
    from src.prediction import create_predictor

    class FakePredictor(LlamaCppDirectTokenPredictor):
        def __init__(self, args, bitmap_data):
            self.args = args
            self.bitmap_data = bitmap_data

    monkeypatch.setattr(
        "src.llamacpp_direct_prediction.probe_llamacpp_direct_support",
        lambda args: (True, None),
    )
    monkeypatch.setattr("src.llamacpp_direct_prediction.LlamaCppDirectTokenPredictor", FakePredictor)

    predictor = create_predictor(_make_args(), bitmap_data=b"bitmap")

    assert isinstance(predictor, FakePredictor)
    assert predictor.bitmap_data == b"bitmap"


def test_build_llamacpp_model_kwargs_applies_direct_tuning(monkeypatch):
    from src.llamacpp_direct_prediction import _build_llamacpp_model_kwargs

    monkeypatch.setattr("src.llamacpp_direct_prediction.os.cpu_count", lambda: 24)

    kwargs = _build_llamacpp_model_kwargs(
        _make_args(
            context_length=2048,
            llamacpp_threads=6,
            llamacpp_direct_threads_batch=18,
            llamacpp_direct_n_batch=1024,
            llamacpp_direct_n_ubatch=256,
            llamacpp_direct_use_mmap=False,
            llamacpp_direct_use_mlock=True,
        ),
        vocab_only=False,
        logits_all=True,
    )

    assert kwargs["n_ctx"] == 2048
    assert kwargs["n_threads"] == 6
    assert kwargs["n_threads_batch"] == 18
    assert kwargs["n_batch"] == 1024
    assert kwargs["n_ubatch"] == 256
    assert kwargs["use_mmap"] is False
    assert kwargs["use_mlock"] is True


def test_run_batched_inference_advances_slot_state():
    from src.llamacpp_direct_prediction import LlamaCppDirectTokenPredictor

    predictor = LlamaCppDirectTokenPredictor.__new__(LlamaCppDirectTokenPredictor)
    predictor.args = _make_args(encoding="AC", reduce_tokens=False)
    predictor.tokens_list = [0, 1, 2, 3]
    predictor.index_tensor = None
    predictor._slot_prompts = {}
    predictor._slot_states = {}
    predictor.model = FakeLlamaModel()
    predictor.tokenizer = SimpleNamespace(detokenize=lambda token_ids: "".join(str(token_id) for token_id in token_ids))

    _, probs_1, _, _ = predictor.run_batched_inference([[1, 2, 3]], enable_kv_cache=True)
    _, probs_2, _, _ = predictor.run_batched_inference([[1, 2, 3, 4]], enable_kv_cache=True)

    assert predictor.model.reset_calls == 1
    assert predictor.model.eval_calls == [(1, 2, 3), (4,)]
    assert predictor.model.load_state_calls == [(1, 2, 3)]
    assert predictor.model.save_state_calls == [(1, 2, 3), (1, 2, 3, 4)]
    assert probs_1.shape == (1, 4)
    assert probs_2.shape == (1, 4)
    assert torch.allclose(probs_1.sum(dim=1), torch.ones(1))
    assert torch.allclose(probs_2.sum(dim=1), torch.ones(1))


def test_run_batched_inference_returns_raw_logits_for_rank_encodings():
    from src.llamacpp_direct_prediction import LlamaCppDirectTokenPredictor

    predictor = LlamaCppDirectTokenPredictor.__new__(LlamaCppDirectTokenPredictor)
    predictor.args = _make_args(encoding="bitpacked", reduce_tokens=True)
    predictor.tokens_list = [1, 3]
    predictor.index_tensor = torch.tensor(predictor.tokens_list, dtype=torch.long)
    predictor._slot_prompts = {}
    predictor._slot_states = {}
    predictor.model = FakeLlamaModel()
    predictor.tokenizer = SimpleNamespace(detokenize=lambda token_ids: "".join(str(token_id) for token_id in token_ids))

    _, logits, _, softmax_time = predictor.run_batched_inference([[1, 2, 3]], enable_kv_cache=True)

    assert torch.equal(logits, torch.tensor([[7.0, 9.0]], dtype=torch.float32))
    assert softmax_time == 0.0


def test_run_batched_inference_rebuilds_state_when_prompt_window_slides():
    from src.llamacpp_direct_prediction import LlamaCppDirectTokenPredictor

    predictor = LlamaCppDirectTokenPredictor.__new__(LlamaCppDirectTokenPredictor)
    predictor.args = _make_args(encoding="AC", reduce_tokens=False)
    predictor.tokens_list = [0, 1, 2, 3]
    predictor.index_tensor = torch.tensor(predictor.tokens_list, dtype=torch.long)
    predictor.reduce_tokens = False
    predictor._slot_prompts = {}
    predictor._slot_states = {}
    predictor.model = FakeLlamaModel()
    predictor.tokenizer = SimpleNamespace(detokenize=lambda token_ids: "".join(str(token_id) for token_id in token_ids))

    predictor.run_batched_inference([[1, 2, 3]], enable_kv_cache=True)
    predictor.run_batched_inference([[2, 3, 4]], enable_kv_cache=True)

    assert predictor.model.reset_calls == 2
    assert predictor.model.eval_calls == [(1, 2, 3), (2, 3, 4)]
    assert predictor.model.load_state_calls == []


def test_run_batched_inference_disables_cache_when_requested():
    from src.llamacpp_direct_prediction import LlamaCppDirectTokenPredictor

    predictor = LlamaCppDirectTokenPredictor.__new__(LlamaCppDirectTokenPredictor)
    predictor.args = _make_args(encoding="AC", reduce_tokens=False)
    predictor.tokens_list = [0, 1, 2, 3]
    predictor.index_tensor = torch.tensor(predictor.tokens_list, dtype=torch.long)
    predictor.reduce_tokens = False
    predictor._slot_prompts = {}
    predictor._slot_states = {}
    predictor.model = FakeLlamaModel()
    predictor.tokenizer = SimpleNamespace(detokenize=lambda token_ids: "".join(str(token_id) for token_id in token_ids))

    predictor.run_batched_inference([[1, 2, 3]], enable_kv_cache=True)
    predictor.run_batched_inference([[1, 2, 3, 4]], enable_kv_cache=False)

    assert predictor.model.reset_calls == 2
    assert predictor.model.eval_calls == [(1, 2, 3), (1, 2, 3, 4)]
    assert predictor.model.load_state_calls == []