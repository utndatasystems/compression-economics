#!/usr/bin/env python3
"""Generate separate benchmark commands for Linux-supported inference backends."""

import argparse
import shlex
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.model_registry import BENCHMARK_MODEL_IDS


SUPPORTED_ENGINES = (
    "transformer",
    "vllm",
    "sglang",
    "onnxruntime",
    "tensorrt",
    "llamacpp",
    "llamacpp_direct",
)

ARTIFACT_TEMPLATE_DEFAULTS = {
    "onnx_model_dir_template": "/path/to/onnx/{model_slug}",
    "tensorrt_engine_dir_template": "/path/to/tensorrt/{model_slug}",
    "llamacpp_model_path_template": "/path/to/gguf/{model_slug}.gguf",
}


def slugify(value):
    """Create a filesystem-friendly token from a model or dataset name."""
    return "".join(char if char.isalnum() else "_" for char in value).strip("_")


def render_template(template, engine, model_name, dataset_name):
    """Render a template with standard engine/model placeholders."""
    model_short = model_name.split("/")[-1]
    return template.format(
        engine=engine,
        model_name=model_name,
        model_short=model_short,
        model_slug=slugify(model_name),
        dataset=dataset_name,
        dataset_slug=slugify(dataset_name),
    )


def build_run_paths(args, engine, model_name, dataset):
    """Return output paths and results JSON path for one benchmark run."""
    dataset_name = Path(dataset).name
    run_stem = (
        f"{slugify(dataset_name)}__{engine}__{slugify(model_name)}"
        f"__ctx{args.context_length}__batch{args.batch_size}__{args.encoding.lower()}"
    )
    base_dir = Path(args.artifacts_dir) / engine
    results_json = render_template(
        args.results_json_template,
        engine,
        model_name,
        dataset_name,
    )
    return {
        "output_path": str(base_dir / f"{run_stem}.bin"),
        "reconstruction_output_path": str(base_dir / f"{run_stem}.txt"),
        "results_json": results_json,
    }


def build_compress_command(args, engine, model_name, dataset):
    """Build the compression command for one backend/model pair."""
    run_paths = build_run_paths(args, engine, model_name, dataset)
    command = [
        args.python_bin,
        args.main_script,
        "--mode",
        "compress",
        "--input_path",
        dataset,
        "--output_path",
        run_paths["output_path"],
        "--results_file",
        run_paths["results_json"],
        "--model_name",
        model_name,
        "--context_length",
        str(args.context_length),
        "--retain_tokens",
        str(args.retain_tokens),
        "--first_n_tokens",
        str(args.first_n_tokens),
        "--batch_size",
        str(args.batch_size),
        "--encoding",
        args.encoding,
        "--engine",
        engine,
        "--use_kv_cache",
        "--print_results",
    ]

    if args.reduce_tokens or engine == "llamacpp":
        command.append("--reduce_tokens")
    else:
        command.append("--no_reduce_tokens")

    if engine in {"vllm", "sglang", "tensorrt"}:
        command.extend(["--tensor_parallel_size", str(args.tensor_parallel_size)])
    if engine == "vllm":
        command.extend(["--gpu_memory_utilization", str(args.gpu_memory_utilization)])
    if engine == "sglang":
        command.extend(
            [
                "--sglang_mem_fraction_static",
                str(args.sglang_mem_fraction_static),
            ]
        )
        command.append(
            "--sglang_enable_deterministic_inference"
            if args.sglang_enable_deterministic_inference
            else "--no_sglang_enable_deterministic_inference"
        )
        command.append(
            "--sglang_use_streaming_session_kv"
            if args.sglang_use_streaming_session_kv
            else "--no_sglang_use_streaming_session_kv"
        )
    if engine == "onnxruntime":
        command.extend(
            [
                "--onnx_model_dir",
                render_template(
                    args.onnx_model_dir_template,
                    engine,
                    model_name,
                    Path(dataset).name,
                ),
                "--onnx_execution_provider",
                args.onnx_execution_provider,
                "--onnx_intra_op_threads",
                str(args.onnx_intra_op_threads),
                "--onnx_inter_op_threads",
                str(args.onnx_inter_op_threads),
                "--onnx_graph_optimization_level",
                args.onnx_graph_optimization_level,
            ]
        )
        if args.onnx_tokenizer_source_template:
            command.extend(
                [
                    "--onnx_tokenizer_source",
                    render_template(
                        args.onnx_tokenizer_source_template,
                        engine,
                        model_name,
                        Path(dataset).name,
                    ),
                ]
            )
    if engine == "tensorrt":
        command.extend(
            [
                "--tensorrt_engine_dir",
                render_template(
                    args.tensorrt_engine_dir_template,
                    engine,
                    model_name,
                    Path(dataset).name,
                ),
            ]
        )
    if engine in {"llamacpp", "llamacpp_direct"}:
        command.extend(
            [
                "--llamacpp_model_path",
                render_template(
                    args.llamacpp_model_path_template,
                    engine,
                    model_name,
                    Path(dataset).name,
                ),
                "--llamacpp_threads",
                str(args.llamacpp_threads),
                "--llamacpp_n_gpu_layers",
                str(args.llamacpp_n_gpu_layers),
            ]
        )
    if engine == "llamacpp":
        command.extend(
            [
                "--llamacpp_binary",
                args.llamacpp_binary,
                "--llamacpp_host",
                args.llamacpp_host,
                "--llamacpp_port",
                str(args.llamacpp_port),
            ]
        )
    if engine == "llamacpp_direct":
        command.extend(
            [
                "--llamacpp_direct_threads_batch",
                str(args.llamacpp_direct_threads_batch),
                "--llamacpp_direct_n_batch",
                str(args.llamacpp_direct_n_batch),
                "--llamacpp_direct_n_ubatch",
                str(args.llamacpp_direct_n_ubatch),
            ]
        )
        command.append(
            "--llamacpp_direct_use_mmap"
            if args.llamacpp_direct_use_mmap
            else "--no_llamacpp_direct_use_mmap"
        )
        command.append(
            "--llamacpp_direct_use_mlock"
            if args.llamacpp_direct_use_mlock
            else "--no_llamacpp_direct_use_mlock"
        )

    return command


def build_decompress_command(args, engine, model_name, dataset):
    """Build the decompression command for one backend/model pair."""
    run_paths = build_run_paths(args, engine, model_name, dataset)
    return [
        args.python_bin,
        args.main_script,
        "--mode",
        "decompress",
        "--input_path",
        run_paths["output_path"],
        "--output_path",
        run_paths["reconstruction_output_path"],
        "--results_file",
        run_paths["results_json"],
        "--print_results",
    ]


def build_command_groups(args):
    """Return grouped compress/decompress commands for every requested run."""
    groups = []
    for dataset in args.datasets:
        for engine in args.engines:
            for model_name in args.models:
                groups.append(
                    {
                        "dataset": dataset,
                        "engine": engine,
                        "model_name": model_name,
                        "compress": build_compress_command(args, engine, model_name, dataset),
                        "decompress": build_decompress_command(args, engine, model_name, dataset),
                    }
                )
    return groups


def format_command(command):
    """Return a shell-safe single-line command."""
    return " ".join(shlex.quote(part) for part in command)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate benchmark commands for Linux-supported inference backends"
    )
    parser.add_argument(
        "--engines",
        nargs="+",
        choices=SUPPORTED_ENGINES,
        default=list(SUPPORTED_ENGINES),
        help="Inference backends to include",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(BENCHMARK_MODEL_IDS),
        help="Model names to pair with each backend",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["data/text8"],
        help="Datasets to benchmark",
    )
    parser.add_argument("--python-bin", default="python", help="Python executable to emit in commands")
    parser.add_argument("--main-script", default="main.py", help="Path to the main benchmark entrypoint")
    parser.add_argument("--artifacts-dir", default="artifacts/backend_benchmarks", help="Directory prefix for .bin and reconstruction artifacts")
    parser.add_argument("--results-json-template", default="artifacts/backend_benchmarks/{engine}/results.json", help="Template for results JSON paths. Supports {engine}, {model_slug}, {model_short}, {dataset}, and {dataset_slug}")
    parser.add_argument("--context-length", type=int, default=256, help="Context length to use in generated commands")
    parser.add_argument("--retain-tokens", type=int, default=100, help="Retained tail length when trimming prompts")
    parser.add_argument("--first-n-tokens", type=int, default=100000, help="Token budget for each run")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size to use in generated commands")
    parser.add_argument("--encoding", choices=["AC", "bitpacked", "huffman"], default="AC", help="Encoding method")
    parser.add_argument("--reduce-tokens", dest="reduce_tokens", action="store_true", help="Emit reduced-token commands")
    parser.add_argument("--no-reduce-tokens", dest="reduce_tokens", action="store_false", help="Emit full-vocabulary commands where supported")
    parser.set_defaults(reduce_tokens=True)
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="Tensor parallel size for vLLM, SGlang, and TensorRT")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9, help="GPU memory fraction for vLLM")
    parser.add_argument("--sglang-mem-fraction-static", type=float, default=0.8, help="Reserved memory fraction for SGlang")
    parser.add_argument("--sglang-enable-deterministic-inference", dest="sglang_enable_deterministic_inference", action="store_true", help="Enable deterministic SGlang inference")
    parser.add_argument("--no-sglang-enable-deterministic-inference", dest="sglang_enable_deterministic_inference", action="store_false", help="Disable deterministic SGlang inference")
    parser.set_defaults(sglang_enable_deterministic_inference=True)
    parser.add_argument("--sglang-use-streaming-session-kv", dest="sglang_use_streaming_session_kv", action="store_true", help="Enable SGlang streaming-session KV reuse in generated commands")
    parser.add_argument("--no-sglang-use-streaming-session-kv", dest="sglang_use_streaming_session_kv", action="store_false", help="Disable SGlang streaming-session KV reuse in generated commands")
    parser.set_defaults(sglang_use_streaming_session_kv=False)
    parser.add_argument("--onnx-model-dir-template", default=ARTIFACT_TEMPLATE_DEFAULTS["onnx_model_dir_template"], help="Template for ONNX model directories")
    parser.add_argument("--onnx-tokenizer-source-template", default=None, help="Optional template for ONNX tokenizer sources")
    parser.add_argument("--onnx-execution-provider", choices=["CPUExecutionProvider"], default="CPUExecutionProvider", help="ONNX Runtime execution provider")
    parser.add_argument("--onnx-intra-op-threads", type=int, default=1, help="ONNX Runtime intra-op threads")
    parser.add_argument("--onnx-inter-op-threads", type=int, default=1, help="ONNX Runtime inter-op threads")
    parser.add_argument("--onnx-graph-optimization-level", choices=["ORT_DISABLE_ALL", "ORT_ENABLE_BASIC", "ORT_ENABLE_EXTENDED", "ORT_ENABLE_ALL"], default="ORT_ENABLE_ALL", help="ONNX Runtime graph optimization level")
    parser.add_argument("--tensorrt-engine-dir-template", default=ARTIFACT_TEMPLATE_DEFAULTS["tensorrt_engine_dir_template"], help="Template for TensorRT engine directories")
    parser.add_argument("--llamacpp-model-path-template", default=ARTIFACT_TEMPLATE_DEFAULTS["llamacpp_model_path_template"], help="Template for GGUF model paths")
    parser.add_argument("--llamacpp-binary", default="llama-server", help="Path to llama-server for the managed llama.cpp backend")
    parser.add_argument("--llamacpp-host", default="127.0.0.1", help="Host for the managed llama.cpp backend")
    parser.add_argument("--llamacpp-port", type=int, default=8080, help="Port for the managed llama.cpp backend")
    parser.add_argument("--llamacpp-threads", type=int, default=8, help="CPU threads for llama.cpp backends")
    parser.add_argument("--llamacpp-n-gpu-layers", type=int, default=0, help="Number of llama.cpp layers to offload to GPU")
    parser.add_argument("--llamacpp-direct-threads-batch", type=int, default=0, help="Batch-processing thread count for llama.cpp direct")
    parser.add_argument("--llamacpp-direct-n-batch", type=int, default=0, help="Prompt batch size for llama.cpp direct")
    parser.add_argument("--llamacpp-direct-n-ubatch", type=int, default=0, help="Physical micro-batch size for llama.cpp direct")
    parser.add_argument("--llamacpp-direct-use-mmap", dest="llamacpp_direct_use_mmap", action="store_true", help="Enable mmap-backed GGUF loading for llama.cpp direct")
    parser.add_argument("--no-llamacpp-direct-use-mmap", dest="llamacpp_direct_use_mmap", action="store_false", help="Disable mmap-backed GGUF loading for llama.cpp direct")
    parser.add_argument("--llamacpp-direct-use-mlock", dest="llamacpp_direct_use_mlock", action="store_true", help="Lock GGUF weights in RAM for llama.cpp direct")
    parser.add_argument("--no-llamacpp-direct-use-mlock", dest="llamacpp_direct_use_mlock", action="store_false", help="Do not lock GGUF weights in RAM for llama.cpp direct")
    parser.set_defaults(llamacpp_direct_use_mmap=True, llamacpp_direct_use_mlock=False)
    parser.add_argument("--output-file", default=None, help="Optional file to write the generated shell commands")
    return parser.parse_args()


def main():
    args = parse_args()
    groups = build_command_groups(args)

    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "# Generated benchmark commands for Linux-supported backends.",
        "# MLX is omitted because it requires macOS on Apple Silicon.",
        "",
    ]

    for group in groups:
        lines.extend(
            [
                f"# dataset={group['dataset']} engine={group['engine']} model={group['model_name']}",
                format_command(group["compress"]),
                format_command(group["decompress"]),
                "",
            ]
        )

    output_text = "\n".join(lines).rstrip() + "\n"
    if args.output_file:
        Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_file).write_text(output_text, encoding="utf-8")
        print(f"Wrote {len(groups)} command pairs to {args.output_file}")
        return

    print(output_text, end="")


if __name__ == "__main__":
    main()