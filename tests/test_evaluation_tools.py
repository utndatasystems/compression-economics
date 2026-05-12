from pathlib import Path
from types import SimpleNamespace

from evaluation.generate_backend_commands import build_command_groups
from evaluation.plot_cost_vs_speed import (
    build_plot_points,
    resolve_engine_costs,
    resolve_engine_hardware,
)


def _make_command_args(**overrides):
    values = {
        "engines": ["onnxruntime"],
        "models": ["distilbert/distilgpt2"],
        "datasets": ["data/text8"],
        "python_bin": "python",
        "main_script": "main.py",
        "artifacts_dir": "artifacts/backend_benchmarks",
        "results_json_template": "artifacts/backend_benchmarks/{engine}/results.json",
        "context_length": 256,
        "retain_tokens": 100,
        "first_n_tokens": 1000,
        "batch_size": 32,
        "encoding": "AC",
        "reduce_tokens": True,
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": 0.9,
        "sglang_mem_fraction_static": 0.8,
        "sglang_enable_deterministic_inference": True,
        "sglang_use_streaming_session_kv": False,
        "onnx_model_dir_template": "/exports/{model_slug}/onnx",
        "onnx_tokenizer_source_template": None,
        "onnx_execution_provider": "CPUExecutionProvider",
        "onnx_intra_op_threads": 2,
        "onnx_inter_op_threads": 1,
        "onnx_graph_optimization_level": "ORT_ENABLE_ALL",
        "tensorrt_engine_dir_template": "/engines/{model_slug}",
        "llamacpp_model_path_template": "/models/{model_slug}.gguf",
        "llamacpp_binary": "llama-server",
        "llamacpp_host": "127.0.0.1",
        "llamacpp_port": 8080,
        "llamacpp_threads": 8,
        "llamacpp_n_gpu_layers": 0,
        "llamacpp_direct_threads_batch": 0,
        "llamacpp_direct_n_batch": 0,
        "llamacpp_direct_n_ubatch": 0,
        "llamacpp_direct_use_mmap": True,
        "llamacpp_direct_use_mlock": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_build_command_groups_renders_onnx_templates():
    groups = build_command_groups(_make_command_args())

    assert len(groups) == 1
    compress = groups[0]["compress"]
    decompress = groups[0]["decompress"]

    assert "--onnx_model_dir" in compress
    assert "/exports/distilbert_distilgpt2/onnx" in compress
    assert "--results_file" in compress
    assert any(part.endswith("artifacts/backend_benchmarks/onnxruntime/results.json") for part in compress)
    assert any(part.endswith(".bin") for part in decompress)


def test_build_command_groups_renders_sglang_streaming_session_flag():
    groups = build_command_groups(
        _make_command_args(
            engines=["sglang"],
            sglang_use_streaming_session_kv=True,
        )
    )

    assert len(groups) == 1
    compress = groups[0]["compress"]

    assert "--sglang_use_streaming_session_kv" in compress
    assert "--no_sglang_use_streaming_session_kv" not in compress


def test_build_plot_points_derives_total_cost_and_speed(tmp_path):
    results_path = tmp_path / "results.json"
    results_path.write_text(
        """
        {
          "exp": {
            "compression": {
              "args": {"engine": "vllm", "model_name": "Qwen/Qwen2.5-0.5B"},
              "input_tokens_count": 1200,
              "total_compression_time": 12.0,
              "compression_factor": 3.0
            },
            "decompression": {
              "args": {"engine": "vllm", "model_name": "Qwen/Qwen2.5-0.5B"},
              "total_decompression_time": 8.0
            }
          }
        }
        """.strip(),
        encoding="utf-8",
    )

    hardware = resolve_engine_hardware({})
    costs = resolve_engine_costs(hardware, gpu_cost=0.804, cpu_cost=0.019, overrides={})
    points = build_plot_points([results_path], costs, hardware)

    assert len(points) == 1
    assert points[0]["label"] == "vllm:Qwen2.5-0.5B"
    assert points[0]["total_speed_tok_s"] == 60.0
    assert round(points[0]["total_cost_usd"], 6) == round((20.0 * 0.804) / 3600.0, 6)