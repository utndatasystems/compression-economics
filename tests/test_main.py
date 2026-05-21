from pathlib import Path
from types import SimpleNamespace

import main as main_module

from src.config import get_main_args


def test_get_main_args_accepts_vllm_engine(monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(repo_root)
    monkeypatch.setattr(
        "sys.argv",
        ["prog", "--mode", "compress", "--engine", "vllm", "--model_name", "gpt2"],
    )

    args = get_main_args()

    assert args.engine == "vllm"


def test_get_main_args_accepts_tensorrt_engine(monkeypatch, tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(repo_root)
    monkeypatch.setattr(
        "sys.argv",
        [
            "prog",
            "--mode",
            "compress",
            "--engine",
            "tensorrt",
            "--model_name",
            "gpt2",
            "--tensorrt_engine_dir",
            str(tmp_path / "engine"),
        ],
    )

    args = get_main_args()

    assert args.engine == "tensorrt"
    assert args.tensorrt_engine_dir == str(tmp_path / "engine")


def test_get_main_args_sets_default_tensorrt_engine_dir(monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(repo_root)
    monkeypatch.setattr(
        "sys.argv",
        [
            "prog",
            "--mode",
            "compress",
            "--engine",
            "tensorrt",
            "--model_name",
            "gpt2",
            "--context_length",
            "256",
            "--batch_size",
            "512",
        ],
    )

    args = get_main_args()

    assert args.tensorrt_engine_dir == "trt_engines/gpt2/ctx256_batch512"


def test_main_saves_engine_in_compression_results(monkeypatch):
    saved_results = {}
    args = SimpleNamespace(
        mode="compress",
        input_path="data/text8",
        output_path=None,
        lora_path=None,
        text_input=None,
        model_name="gpt2",
        HF_token=None,
        context_length=16,
        retain_tokens=8,
        first_n_tokens=2,
        use_kv_cache=True,
        batch_size=1,
        reduce_tokens=True,
        engine="vllm",
        tensorrt_engine_dir=None,
        encoding="AC",
        spec_k=None,
        draft_model_name=None,
        print_results=False,
        force=True,
        is_seq2seq=False,
        is_mamba=False,
    )

    class FakeTokenPredictor:
        base_params = 10
        adapter_params = 2
        base_size_mb = 3.0
        adapter_size_mb = 0.5

        def cleanup(self):
            return None

    def fake_save_results(results, results_file):
        saved_results["results"] = results
        saved_results["results_file"] = results_file

    monkeypatch.setattr(main_module, "get_main_args", lambda: args)
    monkeypatch.setattr(main_module, "get_token_predictor", lambda args, bitmap_data=None: FakeTokenPredictor())
    monkeypatch.setattr(
        main_module,
        "run_global_mask_compression",
        lambda args: ([0], "101", b"bitmap", {"args": args.__dict__, "compression_factor": 1.0}, args),
    )
    monkeypatch.setattr(main_module, "load_results", lambda results_file: {})
    monkeypatch.setattr(main_module, "save_results", fake_save_results)
    monkeypatch.setattr(main_module, "save_global_mask_file", lambda *args, **kwargs: None)

    main_module.main()

    result_key = next(iter(saved_results["results"]))
    assert "engine=vllm" in result_key
    assert saved_results["results"][result_key]["compression"]["engine"] == "vllm"
