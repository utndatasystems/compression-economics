"""Tests for the llama.cpp prediction module."""

from types import SimpleNamespace

import torch


def _make_args(**overrides):
    values = {
        "model_name": "unused",
        "engine": "llamacpp",
        "encoding": "AC",
        "context_length": 256,
        "batch_size": 1,
        "reduce_tokens": True,
        "llamacpp_model_path": "/tmp/model.gguf",
        "llamacpp_binary": "llama-server",
        "llamacpp_host": "127.0.0.1",
        "llamacpp_port": 8080,
        "llamacpp_threads": 4,
        "llamacpp_n_gpu_layers": 12,
        "lora_path": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_probe_llamacpp_ac_support_requires_reduce_tokens(tmp_path, monkeypatch):
    from src.llamacpp_prediction import probe_llamacpp_ac_support

    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"gguf")
    monkeypatch.setattr("src.llamacpp_prediction._resolve_llamacpp_binary", lambda binary: "/usr/bin/llama-server")

    args = _make_args(llamacpp_model_path=str(model_path), reduce_tokens=False)
    supported, reason = probe_llamacpp_ac_support(args)

    assert supported is False
    assert "reduce_tokens" in reason


def test_build_llamacpp_server_command_uses_expected_flags():
    from src.llamacpp_prediction import build_llamacpp_server_command

    args = _make_args(
        llamacpp_model_path="/models/test.gguf",
        llamacpp_binary="/opt/llama-server",
        llamacpp_host="0.0.0.0",
        llamacpp_port=9090,
        llamacpp_threads=8,
        llamacpp_n_gpu_layers=42,
        context_length=2048,
    )

    command = build_llamacpp_server_command(args, parallelism=3)

    assert command == [
        "/opt/llama-server",
        "--model",
        "/models/test.gguf",
        "--host",
        "0.0.0.0",
        "--port",
        "9090",
        "--ctx-size",
        "2048",
        "--parallel",
        "3",
        "--threads",
        "8",
        "--n-gpu-layers",
        "42",
    ]


def test_extract_step_probabilities_orders_requested_token_ids():
    from src.llamacpp_prediction import extract_step_probabilities

    payload = {
        "completion_probabilities": [
            {
                "top_probs": [
                    {"id": 42, "prob": 0.2},
                    {"id": 7, "prob": 0.7},
                    {"id": 9, "prob": 0.1},
                ]
            }
        ]
    }

    probs = extract_step_probabilities(payload, [9, 42, 7])

    assert probs == [0.1, 0.2, 0.7]


def test_slot_reset_behavior_without_real_server():
    from src.llamacpp_prediction import LlamaCppTokenPredictor

    predictor = LlamaCppTokenPredictor.__new__(LlamaCppTokenPredictor)
    predictor.args = _make_args()
    predictor.tokens_list = [11, 13]
    predictor._slot_prompt_lengths = {}
    predictor._slot_prompts = {}
    predictor._disallowed_bias = {}

    reset_calls = []
    request_calls = []

    def fake_reset(slot_id):
        reset_calls.append(slot_id)
        predictor._slot_prompt_lengths.pop(slot_id, None)
        predictor._slot_prompts.pop(slot_id, None)

    def fake_request(prompt_tokens, slot_id, cache_prompt):
        request_calls.append((list(prompt_tokens), slot_id, cache_prompt))
        return {
            "completion_probabilities": [
                {
                    "top_probs": [
                        {"id": 11, "prob": 0.6},
                        {"id": 13, "prob": 0.4},
                    ]
                }
            ]
        }

    predictor._reset_slot = fake_reset
    predictor._request_completion = fake_request

    _, probs_1, _, _ = predictor.run_batched_inference([[1, 2, 3]], enable_kv_cache=True)
    _, probs_2, _, _ = predictor.run_batched_inference([[1, 2]], enable_kv_cache=True)
    _, probs_3, _, _ = predictor.run_batched_inference([[1, 2, 3]], enable_kv_cache=False)

    assert reset_calls == [0, 0]
    assert request_calls == [
        ([1, 2, 3], 0, True),
        ([1, 2], 0, False),
        ([1, 2, 3], 0, False),
    ]
    assert torch.allclose(probs_1, torch.tensor([[0.6, 0.4]], dtype=torch.float32))
    assert torch.allclose(probs_2, torch.tensor([[0.6, 0.4]], dtype=torch.float32))
    assert torch.allclose(probs_3, torch.tensor([[0.6, 0.4]], dtype=torch.float32))


def test_slot_reset_behavior_when_prompt_window_slides():
    from src.llamacpp_prediction import LlamaCppTokenPredictor

    predictor = LlamaCppTokenPredictor.__new__(LlamaCppTokenPredictor)
    predictor.args = _make_args()
    predictor.tokens_list = [11, 13]
    predictor._slot_prompt_lengths = {}
    predictor._slot_prompts = {}
    predictor._disallowed_bias = {}

    reset_calls = []
    request_calls = []

    def fake_reset(slot_id):
        reset_calls.append(slot_id)
        predictor._slot_prompt_lengths.pop(slot_id, None)
        predictor._slot_prompts.pop(slot_id, None)

    def fake_request(prompt_tokens, slot_id, cache_prompt):
        request_calls.append((list(prompt_tokens), slot_id, cache_prompt))
        return {
            "completion_probabilities": [
                {
                    "top_probs": [
                        {"id": 11, "prob": 0.6},
                        {"id": 13, "prob": 0.4},
                    ]
                }
            ]
        }

    predictor._reset_slot = fake_reset
    predictor._request_completion = fake_request

    predictor.run_batched_inference([[1, 2, 3]], enable_kv_cache=True)
    predictor.run_batched_inference([[2, 3, 4]], enable_kv_cache=True)

    assert reset_calls == [0]
    assert request_calls == [
        ([1, 2, 3], 0, True),
        ([2, 3, 4], 0, False),
    ]