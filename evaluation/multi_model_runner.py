#!/usr/bin/env python3
"""
Run all LLM compression experiments + traditional baselines

Usage:
    python run_all_experiments.py [OPTIONS]

Options:
    --skip-llm          Skip LLM compression/decompression experiments
    --skip-baselines    Skip traditional compressor baselines
    --first-n-tokens N  Override token count for LLM experiments (default: full dataset)
    --datasets D [D ..] Datasets to run (default: data/text8)
    --output FILE       Output TSV path (default: experiment_results.tsv)
    --results-json FILE JSON file for LLM results (default: compression_results.json)
    --dry-run           Print commands without executing
    --gpu-label LABEL   Label for the GPU hardware (default: auto-detected)
    --gpu-cost COST     Hourly cost in USD for the GPU instance (default: 0.804)
    --cpu-label LABEL   Label for the CPU hardware (default: auto-detected)
    --cpu-cost COST     Hourly cost in USD for the CPU (default: 0.019)
"""

import argparse
import csv
import json
import os
import platform
import shutil
import subprocess
import sys
import time

LLM_MODELS = [
    "Qwen/Qwen2.5-0.5B",
    "Qwen/Qwen2.5-1.5B",
    "Qwen/Qwen2.5-7B",
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-8B",
    "distilbert/distilgpt2",
]

# Each config is a dict of CLI flags to pass to main.py (beyond model & dataset)
LLM_CONFIGS = [
    {
        "context_length": 256,
        "retain_tokens": 100,
        "batch_size": 128,
        "encoding": "AC",
        "use_kv_cache": True,
        "reduce_tokens": True,
        "first_n_tokens": 1_000_000,
    },
    # Add more configs here to sweep parameters, e.g.:
    # {
    #     "context_length": 2048,
    #     "retain_tokens": 200,
    #     "batch_size": 1,
    #     "encoding": "AC",
    #     "use_kv_cache": True,
    #     "reduce_tokens": True,
    #     "first_n_tokens": 1_000_000,
    # },
]


BASELINE_TOOLS = {
    "gzip-default": {
        "compress": ["gzip", "-k", "-f"],
        "decompress": ["gzip", "-d", "-k", "-f"],
        "ext": ".gz",
    },
    "gzip-9": {
        "compress": ["gzip", "-k", "-9", "-f"],
        "decompress": ["gzip", "-d", "-k", "-f"],
        "ext": ".gz",
    },
    "zstd-1": {
        "compress": ["zstd", "-k", "-1", "-f"],
        "decompress": ["zstd", "-d", "-k", "-f"],
        "ext": ".zst",
    },
    "zstd-3": {
        "compress": ["zstd", "-k", "-3", "-f"],
        "decompress": ["zstd", "-d", "-k", "-f"],
        "ext": ".zst",
    },
    "zstd-default": {
        "compress": ["zstd", "-k", "-f"],
        "decompress": ["zstd", "-d", "-k", "-f"],
        "ext": ".zst",
    },
    "zstd-19": {
        "compress": ["zstd", "-k", "-19", "-f"],
        "decompress": ["zstd", "-d", "-k", "-f"],
        "ext": ".zst",
    },
    "zstd-ultra-22": {
        "compress": ["zstd", "-k", "--ultra", "-22", "-f"],
        "decompress": ["zstd", "-d", "-k", "-f"],
        "ext": ".zst",
    },
    "xz-9e": {
        "compress": ["xz", "-k", "-9e", "-f"],
        "decompress": ["xz", "-d", "-k", "-f"],
        "ext": ".xz",
    },
    "zip": {
        "compress": ["bash", "-c", 'zip "${0}.zip" "${0}"'],
        "decompress": ["bash", "-c", 'unzip -o -p "${0}" > "${0%.zip}"'],
        "ext": ".zip",
    },
}


def detect_gpu_label():
    """Try to detect GPU name via nvidia-smi."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True,
        ).strip().split("\n")[0]
        return out
    except Exception:
        return "Unknown GPU"


def detect_cpu_label():
    """Return a short CPU identifier."""
    try:
        out = subprocess.check_output(["lscpu"], text=True)
        for line in out.split("\n"):
            if "Model name" in line:
                return line.split(":")[1].strip()
    except Exception:
        pass
    return platform.processor() or "Unknown CPU"


def files_are_equal(file1, file2):
    """Byte-by-byte comparison of two files."""
    with open(file1, "rb") as f1, open(file2, "rb") as f2:
        while True:
            b1 = f1.read(8192)
            b2 = f2.read(8192)
            if b1 != b2:
                return False
            if not b1:
                return True


def fmt(val, decimals=4):
    """Format a number nicely, or return empty string for None."""
    if val is None:
        return ""
    if isinstance(val, float):
        return f"{val:.{decimals}f}"
    return str(val)


def run_llm_experiments(datasets, configs, models, results_json, first_n_tokens_override=None, dry_run=False):
    """
    Run main.py compress + decompress for every (dataset, model, config) combo.
    Returns a list of result dicts.
    """
    rows = []

    for dataset in datasets:
        dataset_name = os.path.basename(dataset)
        original_size = os.path.getsize(dataset)

        for config in configs:
            for model in models:
                model_short = model.split("/")[-1]
                config_label = (
                    f"ctx={config['context_length']}"
                    f"|batch={config['batch_size']}"
                    f"|enc={config['encoding']}"
                    f"|kv={config['use_kv_cache']}"
                )

                print(f"\nLLM Compress: {model_short} on {dataset_name}  [{config_label}]")

                # Resolve first_n_tokens: CLI override takes precedence over config value
                first_n_tokens = first_n_tokens_override if first_n_tokens_override is not None else config.get("first_n_tokens")

                # Build compress command
                comp_cmd = [
                    sys.executable, "main.py",
                    "--mode", "compress",
                    "--input_path", dataset,
                    "--model_name", model,
                    "--context_length", str(config["context_length"]),
                    "--retain_tokens", str(config["retain_tokens"]),
                    "--batch_size", str(config["batch_size"]),
                    "--encoding", config["encoding"],
                    "--engine", "transformer",
                    "--print_results",
                ]
                if first_n_tokens is not None:
                    comp_cmd += ["--first_n_tokens", str(first_n_tokens)]
                if config["use_kv_cache"]:
                    comp_cmd.append("--use_kv_cache")
                if config["reduce_tokens"]:
                    comp_cmd.append("--reduce_tokens")
                else:
                    comp_cmd.append("--no_reduce_tokens")

                if dry_run:
                    print(f"  [DRY RUN] {' '.join(comp_cmd)}")
                    continue

                # Run compression
                t_comp_start = time.time()
                result = subprocess.run(comp_cmd, capture_output=True, text=True)
                t_comp_wall = time.time() - t_comp_start
                print(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)
                if result.returncode != 0:
                    print(f"Compression failed: {result.stderr[-300:]}")
                    rows.append({
                        "dataset": dataset_name,
                        "compressor": model_short,
                        "type": "LLM",
                        "config": config_label,
                        "error": result.stderr[-200:],
                    })
                    continue

                # Build decompress command
                decomp_cmd = [
                    sys.executable, "main.py",
                    "--mode", "decompress",
                    "--model_name", model,
                    "--context_length", str(config["context_length"]),
                    "--retain_tokens", str(config["retain_tokens"]),
                    "--batch_size", str(config["batch_size"]),
                    "--encoding", config["encoding"],
                    "--engine", "transformer",
                    "--print_results",
                ]
                if first_n_tokens is not None:
                    decomp_cmd += ["--first_n_tokens", str(first_n_tokens)]
                if config["use_kv_cache"]:
                    decomp_cmd.append("--use_kv_cache")
                if config["reduce_tokens"]:
                    decomp_cmd.append("--reduce_tokens")
                else:
                    decomp_cmd.append("--no_reduce_tokens")

                # Run decompression
                print(f"Decompressing...")
                t_decomp_start = time.time()
                result_d = subprocess.run(decomp_cmd, capture_output=True, text=True)
                t_decomp_wall = time.time() - t_decomp_start
                print(result_d.stdout[-500:] if len(result_d.stdout) > 500 else result_d.stdout)
                if result_d.returncode != 0:
                    print(f"Decompression failed: {result_d.stderr[-300:]}")

                # Now read stats from the results JSON
                comp_stats = {}
                decomp_stats = {}
                try:
                    with open(results_json, "r") as f:
                        all_results = json.load(f)
                    # Find the matching key
                    for key, val in all_results.items():
                        if model in key and dataset_name in key:
                            comp_stats = val.get("compression", {})
                            decomp_stats = val.get("decompression", {})
                            break
                except Exception as e:
                    print(f"Warning: Could not read results JSON: {e}")

                rows.append({
                    "dataset": dataset_name,
                    "compressor": model_short,
                    "type": "LLM",
                    "model_params": model_short,
                    "config": config_label,
                    "original_size_bytes": comp_stats.get("original_size_bytes", original_size),
                    "compressed_size_bytes": comp_stats.get("final_size_bytes", ""),
                    "compression_ratio_pct": (
                        f"{comp_stats['final_size_bytes'] / comp_stats['original_size_bytes'] * 100:.2f}"
                        if comp_stats.get("final_size_bytes") and comp_stats.get("original_size_bytes")
                        else ""
                    ),
                    "compression_factor": comp_stats.get("compression_factor", ""),
                    "pure_compression_factor": comp_stats.get("pure_compression_factor", ""),
                    "comp_total_time_s": comp_stats.get("total_compression_time", t_comp_wall),
                    "comp_inference_time_s": comp_stats.get("inference_time", ""),
                    "comp_ac_time_s": comp_stats.get("ac_time", ""),
                    "comp_tokenize_time_s": comp_stats.get("tokenize_time", ""),
                    "comp_throughput_tok_s": comp_stats.get("throughput_tokens_per_sec", ""),
                    "comp_throughput_kib_s": comp_stats.get("throughput_kibibytes_per_sec", ""),
                    "comp_infer_throughput_tok_s": comp_stats.get("inference_throughput_tokens_per_sec", ""),
                    "decomp_total_time_s": decomp_stats.get("total_decompression_time", t_decomp_wall),
                    "decomp_inference_time_s": decomp_stats.get("inference_time", ""),
                    "decomp_ac_time_s": decomp_stats.get("ac_time", ""),
                    "decomp_throughput_kib_s": decomp_stats.get("throughput_kibibytes_per_sec", ""),
                    "decomp_infer_throughput_kib_s": decomp_stats.get("inference_throughput_kibibytes_per_sec", ""),
                    "entropy": comp_stats.get("entropy", ""),
                    "input_tokens_count": comp_stats.get("input_tokens_count", ""),
                    "context_length": config["context_length"],
                    "batch_size": config["batch_size"],
                    "encoding": config["encoding"],
                    "kv_cache": config["use_kv_cache"],
                    "reduce_tokens": config["reduce_tokens"],
                })

    return rows


def run_baseline_experiments(datasets, dry_run=False):
    """
    Run traditional compression baselines on each dataset.
    Returns a list of result dicts.
    """
    rows = []

    for dataset in datasets:
        dataset_name = os.path.basename(dataset)
        original_size = os.path.getsize(dataset)

        for tool_name, config in BASELINE_TOOLS.items():
            # Check if tool is available
            base_cmd = config["compress"][0]
            if base_cmd == "bash":
                # For bash -c commands, check the actual tool
                inner = config["compress"][2]
                actual_tool = inner.split()[0]
            else:
                actual_tool = base_cmd
            if not shutil.which(actual_tool) and actual_tool not in ("zip", "unzip"):
                print(f"Skipping {tool_name}: '{actual_tool}' not found")
                continue

            print(f"\nBaseline: {tool_name} on {dataset_name}")

            compressed_file = dataset + config["ext"]

            if dry_run:
                print(f"  [DRY RUN] {config['compress']} {dataset}")
                continue

            # Compression
            try:
                comp_cmd = config["compress"] + [dataset]
                t0 = time.perf_counter()
                subprocess.run(comp_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, check=True)
                compress_time = time.perf_counter() - t0
            except Exception as e:
                print(f"Compression failed: {e}")
                rows.append({
                    "dataset": dataset_name,
                    "compressor": tool_name,
                    "type": "Traditional",
                    "error": str(e),
                })
                continue

            if not os.path.exists(compressed_file):
                print(f"Compressed file not found: {compressed_file}")
                continue

            compressed_size = os.path.getsize(compressed_file)
            comp_speed = original_size / compress_time if compress_time > 0 else 0

            # Decompression
            decomp_file = dataset + ".decompressed"
            shutil.copy(compressed_file, decomp_file + config["ext"])

            try:
                decomp_cmd = config["decompress"] + [decomp_file + config["ext"]]
                t0 = time.perf_counter()
                subprocess.run(decomp_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, check=True)
                decompress_time = time.perf_counter() - t0
            except Exception as e:
                print(f"Decompression failed: {e}")
                decompress_time = None

            decomp_speed = original_size / decompress_time if decompress_time else None

            # ── Integrity check ──
            integrity = ""
            if os.path.exists(decomp_file):
                integrity = "PASS" if files_are_equal(dataset, decomp_file) else "FAIL"

            # Clean up temp files
            for f in [decomp_file, decomp_file + config["ext"]]:
                if os.path.exists(f):
                    os.remove(f)
            if os.path.exists(compressed_file):
                os.remove(compressed_file)

            rows.append({
                "dataset": dataset_name,
                "compressor": tool_name,
                "type": "Traditional",
                "model_params": "",
                "config": "",
                "original_size_bytes": original_size,
                "compressed_size_bytes": compressed_size,
                "compression_ratio_pct": f"{compressed_size / original_size * 100:.2f}",
                "compression_factor": f"{original_size / compressed_size:.4f}" if compressed_size > 0 else "",
                "pure_compression_factor": "",
                "comp_total_time_s": compress_time,
                "comp_inference_time_s": "",
                "comp_ac_time_s": "",
                "comp_tokenize_time_s": "",
                "comp_throughput_tok_s": "",
                "comp_throughput_kib_s": f"{comp_speed / 1024:.4f}",
                "comp_infer_throughput_tok_s": "",
                "decomp_total_time_s": decompress_time if decompress_time else "",
                "decomp_inference_time_s": "",
                "decomp_ac_time_s": "",
                "decomp_throughput_kib_s": f"{decomp_speed / 1024:.4f}" if decomp_speed else "",
                "decomp_infer_throughput_kib_s": "",
                "entropy": "",
                "input_tokens_count": "",
                "context_length": "",
                "batch_size": "",
                "encoding": "",
                "kv_cache": "",
                "reduce_tokens": "",
                "integrity": integrity,
            })
            print(f"{compressed_size/original_size*100:.2f}% | "
                  f"comp {compress_time:.2f}s ({comp_speed/1e6:.1f} MB/s) | "
                  f"decomp {decompress_time:.2f}s ({decomp_speed/1e6:.1f} MB/s)" if decompress_time else "")

    return rows



# Column ordering (useful for google sheetst) only columns in this list will be included in the output
OVERVIEW_COLUMNS = [
    "dataset",
    "hardware",
    "type",
    "compressor",
    "model_params",
    "config",
    "original_size_bytes",
    "compressed_size_bytes",
    "compression_ratio_pct",
    "compression_factor",
    "pure_compression_factor",
    "comp_total_time_s",
    "comp_inference_time_s",
    "comp_ac_time_s",
    "comp_tokenize_time_s",
    "comp_throughput_tok_s",
    "comp_throughput_kib_s",
    "comp_infer_throughput_tok_s",
    "decomp_total_time_s",
    "decomp_inference_time_s",
    "decomp_ac_time_s",
    "decomp_throughput_kib_s",
    "decomp_infer_throughput_kib_s",
    "entropy",
    "input_tokens_count",
    "context_length",
    "batch_size",
    "encoding",
    "kv_cache",
    "reduce_tokens",
    "cost_compress_usd",
    "cost_decompress_usd",
]

SHEET_HEADERS = [
    "Dataset",
    "Hardware",
    "Type",
    "Compressor",
    "Model / Params",
    "Config",
    "Original Size (bytes)",
    "Compressed Size (bytes)",
    "Compression Ratio (%)",
    "Compression Factor (x)",
    "Pure Compression Factor (x)",
    "Comp Total Time (s)",
    "Comp Inference Time (s)",
    "Comp AC Time (s)",
    "Comp Tokenize Time (s)",
    "Comp Throughput (tok/s)",
    "Comp Throughput (KiB/s)",
    "Comp Inference Throughput (tok/s)",
    "Decomp Total Time (s)",
    "Decomp Inference Time (s)",
    "Decomp AC Time (s)",
    "Decomp Throughput (KiB/s)",
    "Decomp Inference Throughput (KiB/s)",
    "Entropy",
    "Input Tokens Count",
    "Context Length",
    "Batch Size",
    "Encoding",
    "KV Cache",
    "Reduce Tokens",
    "Cost Compress ($)",
    "Cost Decompress ($)",
]


def write_tsv(rows, output_path, gpu_label, gpu_cost, cpu_label, cpu_cost):
    """Write rows as TSV with cost columns computed."""
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(SHEET_HEADERS)

        for row in rows:
            if row.get("error"):
                # Write error row
                writer.writerow([
                    row.get("dataset", ""),
                    "",
                    row.get("type", ""),
                    row.get("compressor", ""),
                    "",
                    row.get("config", ""),
                    *[""] * (len(SHEET_HEADERS) - 7),
                    f"ERROR: {row['error'][:80]}",
                ])
                continue

            is_llm = row.get("type") == "LLM"
            hw_label = gpu_label if is_llm else cpu_label
            hourly_cost = gpu_cost if is_llm else cpu_cost

            # Compute dollar costs
            comp_time = row.get("comp_total_time_s", "")
            decomp_time = row.get("decomp_total_time_s", "")
            cost_comp = f"{float(comp_time) * hourly_cost / 3600:.6f}" if comp_time != "" else ""
            cost_decomp = f"{float(decomp_time) * hourly_cost / 3600:.6f}" if decomp_time != "" else ""

            row["hardware"] = hw_label
            row["cost_compress_usd"] = cost_comp
            row["cost_decompress_usd"] = cost_decomp

            writer.writerow([row.get(col, "") for col in OVERVIEW_COLUMNS])

    print(f"\nResults written to: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run all compression experiments and output to TSV"
    )
    parser.add_argument("--skip-llm", action="store_true", help="Skip LLM experiments")
    parser.add_argument("--skip-baselines", action="store_true", help="Skip traditional baselines")
    parser.add_argument("--first-n-tokens", type=int, default=None,
                        help="Override first_n_tokens for all LLM configs (default: use per-config value)")
    parser.add_argument("--datasets", nargs="+", default=["data/text8"],
                        help="Dataset file paths")
    parser.add_argument("--output", default="experiment_results.tsv",
                        help="Output TSV file path")
    parser.add_argument("--results-json", default="compression_results.json",
                        help="JSON file for LLM results accumulation")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    parser.add_argument("--gpu-label", default=None, help="GPU hardware label")
    parser.add_argument("--gpu-cost", type=float, default=0.804,
                        help="GPU instance hourly cost in USD")
    parser.add_argument("--cpu-label", default=None, help="CPU hardware label")
    parser.add_argument("--cpu-cost", type=float, default=0.019,
                        help="CPU instance hourly cost in USD")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Detect hardware
    gpu_label = args.gpu_label or detect_gpu_label()
    cpu_label = args.cpu_label or detect_cpu_label()
    
    print("Compression Experiment Runner")
    print(f"GPU: {gpu_label}")
    print(f"CPU: {cpu_label}")
    print(f"GPU cost: ${args.gpu_cost:.3f}/hr")
    print(f"CPU cost: ${args.cpu_cost:.3f}/hr")
    print(f"Datasets: {', '.join(args.datasets)}")
    print(f"Output: {args.output}")
    
    all_rows = []
    
    # Run LLM experiments
    if not args.skip_llm:
        print("\nRunning LLM compression experiments...")
        llm_rows = run_llm_experiments(
            datasets=args.datasets,
            configs=LLM_CONFIGS,
            models=LLM_MODELS,
            results_json=args.results_json,
            first_n_tokens_override=args.first_n_tokens,
            dry_run=args.dry_run
        )
        all_rows.extend(llm_rows)
        print(f"Finished LLM experiments ({len(llm_rows)} rows)")
    
    # Run baseline experiments
    if not args.skip_baselines:
        print("\nRunning baseline compression experiments...")
        baseline_rows = run_baseline_experiments(datasets=args.datasets, dry_run=args.dry_run)
        all_rows.extend(baseline_rows)
        print(f"Finished baseline experiments ({len(baseline_rows)} rows)")
    
    # Write results
    if not args.dry_run and all_rows:
        write_tsv(all_rows, args.output, gpu_label, args.gpu_cost, cpu_label, args.cpu_cost)
        
        # Print summary
        print("\nResults summary:")
        print(f"{'Type':<12} {'Compressor':<20} {'Ratio %':<10} {'Comp Time':<14} {'Decomp Time':<14}")
        for row in all_rows:
            if row.get("error"):
                print(f"{'ERROR':<12} {row['compressor']:<20} {row['error'][:50]}")
                continue
            
            comp_time = str(row.get('comp_total_time_s', ''))[:12]
            decomp_time = str(row.get('decomp_total_time_s', ''))[:12]
            print(f"{row.get('type',''):<12} {row.get('compressor',''):<20} "
                  f"{str(row.get('compression_ratio_pct','')):<10} {comp_time:<14} {decomp_time:<14}")
    
    elif args.dry_run:
        print("\nDry run complete. No results written.")
    else:
        print("\nNo results to write.")


if __name__ == "__main__":
    main()
