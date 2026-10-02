"""Check O3 CLI choices without loading any remote model."""

import sys

import pytest

from src.config import get_main_args, get_adapter_training_args, get_quantize_model_args


@pytest.mark.parametrize("engine", ["transformer", "vllm", "ngram"])
def test_target_interval_cli_engine(engine, monkeypatch, tmp_path):
    arguments = ["main.py", "--mode", "compress", "--engine", engine,
                 "--encoding", "AC_TARGET_INTERVAL"]
    if engine == "ngram":
        training = tmp_path / "training.txt"
        training.write_text("ababa")
        arguments += ["--ngram-model-path", str(tmp_path / "model.pkl"),
                      "--ngram-training-path", str(training)]
    monkeypatch.setattr(sys, "argv", arguments)
    args = get_main_args()
    assert args.engine == engine
    assert args.encoding == "AC_TARGET_INTERVAL"


@pytest.mark.parametrize("options", [
    ["--frequency-quantizer", "vectorized_exact"],
    ["--spec_k", "2"],
])
def test_unsupported_o3_options_rejected(options, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["main.py", "--mode", "compress", "--encoding", "AC_TARGET_INTERVAL", *options])
    with pytest.raises(SystemExit):
        get_main_args()


def test_other_cli_entrypoints_remain_independent(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train_adapter.py"])
    assert get_adapter_training_args().model_id
    monkeypatch.setattr(sys, "argv", ["quantize_model.py", "--model_id",
                                    "Qwen/Qwen2.5-0.5B", "--quantization_bits", "4"])
    assert get_quantize_model_args().quantization_bits == 4


def test_cuda_encoder_cli_contract(monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "main.py", "--mode", "compress", "--engine", "transformer",
        "--encoding", "AC_TARGET_INTERVAL", "--ac-backend", "cuda",
    ])
    assert get_main_args().ac_backend == "cuda"

    monkeypatch.setattr(sys, "argv", [
        "main.py", "--mode", "compress", "--encoding", "AC",
        "--ac-backend", "cuda",
    ])
    with pytest.raises(SystemExit):
        get_main_args()


def test_fused_target_interval_cli_contract(monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "main.py", "--mode", "compress", "--engine", "transformer",
        "--encoding", "AC_TARGET_INTERVAL", "--ac-backend", "cuda",
        "--target-interval-quantizer", "cuda_fused",
    ])
    assert get_main_args().target_interval_quantizer == "cuda_fused"

    monkeypatch.setattr(sys, "argv", [
        "main.py", "--mode", "compress", "--engine", "transformer",
        "--encoding", "AC_TARGET_INTERVAL",
        "--target-interval-quantizer", "cuda_fused",
    ])
    with pytest.raises(SystemExit):
        get_main_args()


@pytest.mark.parametrize("backend", ["device", "cuda"])
def test_device_decoder_cli_contract(monkeypatch, backend):
    monkeypatch.setattr(sys, "argv", [
        "main.py", "--mode", "decompress", "--engine", "transformer",
        "--ac-decode-backend", backend,
    ])
    assert get_main_args().ac_decode_backend == backend

    monkeypatch.setattr(sys, "argv", [
        "main.py", "--mode", "compress", "--engine", "transformer",
        "--ac-decode-backend", backend,
    ])
    with pytest.raises(SystemExit):
        get_main_args()
