import argparse
import json
import os
import subprocess
import sys
from contextlib import nullcontext
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch


DEFAULT_BATCH_SIZES = [4, 16, 64, 128, 256, 512]
DEFAULT_CONTEXT_LENGTHS = [128, 256, 512, 1024]
DEFAULT_NUM_TOKENS = 1_000_000
DEFAULT_RESULTS_FILE = "compression_results_grid_search.json"
DEFAULT_FIGURES_DIR = "figures"
DEFAULT_ARTIFACTS_DIR = "artifacts/grid_search"
REPO_ROOT = Path(__file__).resolve().parent

matplotlib.rcParams.update({'font.size': 11, 'font.family': 'serif', 'axes.titlesize': 'medium', 'figure.titlesize': 'medium', 'text.usetex': True, 'text.latex.preamble': '\\usepackage{amsmath}\\usepackage{amssymb}\\usepackage{siunitx}[=v2]', 'pgf.rcfonts': False, 'pgf.texsystem': 'pdflatex'})


def parse_int_list(value):
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_args():
    parser = argparse.ArgumentParser(description="Run compression grid search experiments and save backend-specific heatmaps.")
    parser.add_argument("--engine", choices=["transformer", "vllm", "tensorrt", "sglang"], default="transformer", help="Inference backend to benchmark")
    parser.add_argument("--input_path", default="data/text8", help="Dataset path to compress")
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B", help="Model name passed through to main.py")
    parser.add_argument("--encoding", choices=["AC", "bitpacked", "huffman"], default="AC", help="Encoding used by the compression run")
    parser.add_argument("--first_n_tokens", type=int, default=DEFAULT_NUM_TOKENS, help="Number of tokens to compress per experiment")
    parser.add_argument("--batch_sizes", type=parse_int_list, default=DEFAULT_BATCH_SIZES, help="Comma-separated batch sizes to test")
    parser.add_argument("--context_lengths", type=parse_int_list, default=DEFAULT_CONTEXT_LENGTHS, help="Comma-separated context lengths to test")
    parser.add_argument("--results_file", default=DEFAULT_RESULTS_FILE, help="JSON file where run metrics are stored")
    parser.add_argument("--figures_dir", default=DEFAULT_FIGURES_DIR, help="Directory where heatmaps are written")
    parser.add_argument("--artifacts_dir", default=DEFAULT_ARTIFACTS_DIR, help="Directory where per-experiment compression files are written")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Tensor parallel size for vLLM")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9, help="GPU memory fraction reserved by vLLM")
    parser.add_argument("--sglang_mem_fraction_static", type=float, default=0.8, help="GPU memory fraction reserved by SGlang")
    parser.add_argument("--max_workers", type=int, default=None, help="Override the number of concurrent workers; defaults to detected GPU count")
    return parser.parse_args()


def validate_backend_or_raise(args):
    if args.engine == "vllm":
        from types import SimpleNamespace
        from src.vllm_prediction import probe_vllm_backend_support

        supported, reason = probe_vllm_backend_support(SimpleNamespace())
        if not supported:
            raise RuntimeError(f"vLLM backend is not available: {reason}")

    if args.engine == "sglang":
        from types import SimpleNamespace
        from src.sglang_prediction import probe_sglang_ac_support

        supported, reason = probe_sglang_ac_support(SimpleNamespace())
        if not supported:
            raise RuntimeError(f"SGlang backend is not available: {reason}")


def plot(metric, title, label, figname, results_file, figures_dir, batch_sizes, context_lengths, engine):
    with open(results_file) as f:
        results = json.load(f)

    rows = []
    for _, result in results.items():
        compression = result.get("compression")
        if compression is None:
            continue
        if compression["args"].get("engine") != engine:
            continue
        if compression["args"]["batch_size"] not in batch_sizes:
            continue
        if compression["args"]["context_length"] not in context_lengths:
            continue

        rows.append({
            "context_len": int(compression["args"]["context_length"]),
            "batch_size": int(compression["args"]["batch_size"]),
            "throughput": metric(result),
        })

    if not rows:
        raise RuntimeError(f"No completed {engine} experiments were found in {results_file}")

    df = (
        pd.DataFrame(rows)
        .groupby(["context_len", "batch_size"])["throughput"]
        .mean()
        .unstack()
    )

    plt.figure(figsize=(4.9, 3.5))
    sns.heatmap(df, annot=True, fmt=".2f", linewidths=.5, cbar_kws={"label": label})
    plt.xlabel("Batch Size")
    plt.ylabel("Context Window Size")
    plt.title(title)

    os.makedirs(figures_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(Path(figures_dir) / f"{figname}.png", dpi=300, bbox_inches='tight', pad_inches=0.01)
    plt.close()


def run_experiment(task):
    batch_size, context_length, gpu_id, config = task
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    output_path = Path(config["artifacts_dir"]) / f"{config['engine']}_b{batch_size}_c{context_length}.bin"
    cmd = [
        sys.executable,
        "-u",
        str(REPO_ROOT / "main.py"),
        "--input_path", config["input_path"],
        "--mode", "compress",
        "--batch_size", str(batch_size),
        "--context_length", str(context_length),
        "--first_n_tokens", str(config["first_n_tokens"]),
        "--use_kv_cache",
        "--engine", config["engine"],
        "--model_name", config["model_name"],
        "--encoding", config["encoding"],
        "--output_path", str(output_path),
        "--results_file", config["results_file"],
        "--tensor_parallel_size", str(config["tensor_parallel_size"]),
        "--gpu_memory_utilization", str(config["gpu_memory_utilization"]),
        "--sglang_mem_fraction_static", str(config["sglang_mem_fraction_static"]),
    ]
    prefix = f"[{config['engine']}:GPU {gpu_id} b={batch_size} c={context_length}]"
    print(f"{prefix} starting", flush=True)

    process = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    stdout_context = process.stdout if process.stdout is not None else nullcontext([])
    with stdout_context as stream:
        if stream is not None:
            for line in stream:
                print(f"{prefix} {line.rstrip()}", flush=True)

    return_code = process.wait()
    if return_code != 0:
        print(f"{prefix} failed with exit code {return_code}", flush=True)
    else:
        print(f"{prefix} done", flush=True)
    return return_code


if __name__ == "__main__":
    args = parse_args()
    validate_backend_or_raise(args)

    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("No GPUs detected on the system.")

    max_workers = args.max_workers or num_gpus
    print(f"Detected {num_gpus} GPU(s). Running up to {max_workers} task(s) for engine={args.engine}.")

    Path(args.figures_dir).mkdir(parents=True, exist_ok=True)
    Path(args.artifacts_dir).mkdir(parents=True, exist_ok=True)

    experiments = []
    for batch_size in args.batch_sizes:
        for context_length in args.context_lengths:
            experiments.append((batch_size, context_length))

    config = {
        "engine": args.engine,
        "input_path": args.input_path,
        "model_name": args.model_name,
        "encoding": args.encoding,
        "first_n_tokens": args.first_n_tokens,
        "results_file": args.results_file,
        "artifacts_dir": args.artifacts_dir,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "sglang_mem_fraction_static": args.sglang_mem_fraction_static,
    }

    tasks = [
        (batch_size, context_length, idx % num_gpus, config)
        for idx, (batch_size, context_length) in enumerate(experiments)
    ]

    if max_workers == 1:
        total_tasks = len(tasks)
        for index, task in enumerate(tasks, start=1):
            batch_size, context_length, gpu_id, _ = task
            print(
                f"Running experiment {index}/{total_tasks}: batch_size={batch_size}, context_length={context_length}, gpu={gpu_id}",
                flush=True,
            )
            rc = run_experiment(task)
            if rc != 0:
                print(f"Experiment batch_size={batch_size}, context_length={context_length} on GPU {gpu_id} exited with code {rc}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(run_experiment, task): task for task in tasks}
            for future in as_completed(futures):
                batch_size, context_length, gpu_id, _ = futures[future]
                try:
                    rc = future.result()
                    if rc != 0:
                        print(f"Experiment batch_size={batch_size}, context_length={context_length} on GPU {gpu_id} exited with code {rc}", flush=True)
                except Exception as exc:
                    print(f"Experiment batch_size={batch_size}, context_length={context_length} on GPU {gpu_id} raised: {exc}", flush=True)

    plot(
        lambda stats: float(stats["compression"]["inference_throughput_kibibytes_per_sec"]),
        "Throughput [KiB/s]",
        "KiB/s",
        "inference_throughput_heatmap",
        args.results_file,
        args.figures_dir,
        args.batch_sizes,
        args.context_lengths,
        args.engine,
    )
    plot(
        lambda stats: float(stats["compression"]["compression_factor"]),
        "Compression Factor",
        "Factor",
        "compression_factor_heatmap",
        args.results_file,
        args.figures_dir,
        args.batch_sizes,
        args.context_lengths,
        args.engine,
    )
    plot(
        lambda stats: float(stats["compression"]["pure_compression_factor"]),
        "Pure Compression Factor",
        "Factor",
        "pure_compression_factor_heatmap",
        args.results_file,
        args.figures_dir,
        args.batch_sizes,
        args.context_lengths,
        args.engine,
    )