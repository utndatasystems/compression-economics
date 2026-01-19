import json
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from collections import defaultdict
import pandas as pd
import json
import seaborn as sns

# --------------------------------------------------------
# (Optional LaTeX config - commented out)
# If LaTeX rendering is needed, uncomment and ensure LaTeX is installed.
# matplotlib.rcParams.update({
#   "pgf.texsystem": "pdflatex",
#   'font.family': 'serif',
#   'text.usetex': True,
#   'pgf.rcfonts': False,
# })
# --------------------------------------------------------

def load_model_results(file_path, selected_datasets=None, selected_n=None):
    """
    Loads model JSON results and flattens into a dictionary of:
        dataset → model → compression/decompression data.
    
    Keeps only the **fastest compression time** entry for each dataset+model.
    Adds ctx and ret to the output.
    Filters results by dataset and n if specified.
    """
    with open(file_path, "r") as f:
        results = json.load(f)

    # Temporary store: dataset → model → best entry (fastest compression time)
    temp_store = defaultdict(lambda: defaultdict(dict))

    for key, value in results.items():
        dataset_name, model_info = key.split(":", 1)
        parts = model_info.split("|")

        model_name = parts[0]
        n_value = None
        ctx_value = None
        ret_value = None
        batch_value = None

        # Extract ctx, ret, n, batch
        for p in parts:
            if p.startswith("n="):
                n_value = int(p.split("=")[1])
            elif p.startswith("ctx="):
                ctx_value = int(p.split("=")[1])
            elif p.startswith("ret="):
                ret_value = int(p.split("=")[1])
            elif p.startswith("batch="):
                batch_value = int(p.split("=")[1])

        # Filter datasets/n values if specified
        if selected_datasets and dataset_name not in selected_datasets:
            continue
        if selected_n and n_value not in selected_n:
            continue

        comp = value.get("compression", {})
        decomp = value.get("decompression", {})

        compression_time = comp.get("total_compression_time_sec", float("inf"))

        # Keep fastest compression time (smallest total_compression_time_sec)
        existing = temp_store[dataset_name].get(model_name)
        if not existing or compression_time < existing["compression_time"]:
            temp_store[dataset_name][model_name] = {
                "batch": batch_value,
                "original_size_bits": comp.get("original_size_bits", 0),
                "ac_bits": comp.get("arithmetic_code_size_bits", 0),
                "bitmap_bits": comp.get("bitmap_size_bits", 0),
                "compressed_size_bits": comp.get("final_size_bits", 0),
                "compression_time": compression_time,
                "decompression_time": decomp.get("decompression_time_sec", 0),
                "ctx": ctx_value,
                "ret": ret_value
            }

    # Convert to final output
    datasets = defaultdict(dict)
    for dataset_name, models in temp_store.items():
        for model_name, info in models.items():
            datasets[dataset_name][model_name] = info
    return datasets


def load_baseline_csv(file_path, dataset_name, selected_compressors=None):
    """
    Reads baseline CSV into dataset → compressor dict.

    Uses the `Uncompressed` row to determine `original_size_bits` for all compressors.
    Filters compressors if a subset is specified.
    """
    df = pd.read_csv(file_path, skip_blank_lines=True)

    # Extract original size from the 'Uncompressed' row
    uncompressed_row = df[df["Compressor"].str.lower() == "uncompressed"]
    if uncompressed_row.empty:
        raise ValueError("CSV must contain an 'Uncompressed' row to determine original size.")
    original_size_bits = int(uncompressed_row.iloc[0]["Size_bytes"]) * 8

    # Filter compressors if user specified a subset
    if selected_compressors:
        df = df[df["Compressor"].isin(selected_compressors)]

    baseline_dict = defaultdict(dict)

    # Store compressor data
    for _, row in df.iterrows():
        if row["Compressor"].lower() == "uncompressed":
            continue  # Skip storing uncompressed in results
        baseline_dict[dataset_name][row["Compressor"]] = {
            "original_size_bits": original_size_bits,
            "ac_bits": 0,  # Baseline compressors have no AC/Bitmap breakdown
            "bitmap_bits": 0,
            "compressed_size_bits": int(row["Size_bytes"]) * 8,
            "compression_time": row.get("Compression_time_s", 0) or 0,
            "decompression_time": row.get("Decompression_time_s", 0) or 0
        }

    return baseline_dict


def merge_results(model_data, baseline_data):
    """
    Merge model results and baseline compressors into one dictionary.
    If both contain the same dataset, entries are combined.
    """
    merged = defaultdict(dict)
    datasets = set(model_data.keys()) | set(baseline_data.keys())
    for ds in datasets:
        merged[ds].update(baseline_data.get(ds, {}))
        merged[ds].update(model_data.get(ds, {}))
    return merged


def results_plot_1(datasets, dataset_order):
    """
    Plot compression ratio breakdown:
    - Model compressors (with AC + Bitmap) as stacked bars.
    - Baseline compressors (no AC/Bitmap split) as single bars.
    """
    model_order = [
        "zip",
        "gzip",
        "zstd -3",
        "zstd -1",
        "zstd -19",
        "distilgpt2",
        "openai-community/gpt2",
        "xz -9e (LZMA2)",
        "Qwen/Qwen2.5-0.5B",
        "Qwen/Qwen2.5-1.5B",
        "Qwen/Qwen2.5-7B",
        "Qwen/Qwen3-0.6B",
        "Qwen/Qwen3-1.7B",
        "Qwen/Qwen3-8B",
        "meta-llama/Llama-3.2-1B",
        "HuggingFaceTB/SmolLM-135M",
        "HuggingFaceTB/SmolLM2-135M"
    ]
    fig, axes = plt.subplots(1, len(dataset_order), figsize=(10, 6))

    if len(dataset_order) == 1:
        axes = [axes]

    for idx, dataset in enumerate(dataset_order):
        if dataset not in datasets:
            continue
        comp_dict = datasets[dataset]

        ax = axes[idx]
        ax.set_title(f"Compression Ratio Breakdown - {dataset}")

        models = list(comp_dict.keys())
        models.sort(key=lambda m: model_order.index(m) if m in model_order else len(model_order))
        x = np.arange(len(models))
        bar_width = 0.6

        # Store values
        ac_ratios, bitmap_ratios, total_ratios, is_model_type = [], [], [], []
        for comp_name in models:
            m = comp_dict[comp_name]
            orig = m["original_size_bits"]
            comp_size = m["compressed_size_bits"]
            total_ratios.append((comp_size / orig) * 100 if orig else 0)

            if m["ac_bits"] > 0 or m["bitmap_bits"] > 0:
                # Model-type compressor
                ac_ratios.append((m["ac_bits"] / orig) * 100 if orig else 0)
                bitmap_ratios.append((m["bitmap_bits"] / orig) * 100 if orig else 0)
                is_model_type.append(True)
            else:
                # Baseline compressor
                ac_ratios.append(0)
                bitmap_ratios.append(0)
                is_model_type.append(False)

        # Labels to avoid duplicate legend entries
        ac_label_added = bitmap_label_added = baseline_label_added = False

        # Plot bars
        for i, comp_name in enumerate(models):
            if is_model_type[i]:
                ax.bar(
                    i, ac_ratios[i], width=bar_width, color="tab:blue", alpha=0.7,
                    label="AC%" if not ac_label_added else None
                )
                ac_label_added = True

                ax.bar(
                    i, bitmap_ratios[i], width=bar_width, bottom=ac_ratios[i], color="tab:orange", alpha=0.7,
                    label="Bitmap%" if not bitmap_label_added else None
                )
                bitmap_label_added = True
            else:
                ax.bar(
                    i, total_ratios[i], width=bar_width, color="gray", alpha=0.7,
                    label="Baseline" if not baseline_label_added else None
                )
                baseline_label_added = True

        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha="right")
        ax.set_ylabel("Compression Ratio (%)")
        ax.legend()

    plt.tight_layout()
    plt.savefig("compression_ratio_stack_plot.png")
    # fig.savefig(
    #     f"compression_ratio_stack_plot.pdf",
    #     format="pdf",
    #     bbox_inches="tight",
    #     pad_inches=0.01,
    #     dpi=300
    # )
    plt.close()
    print("Plots saved as compression_ratio_stack_plot.png")


def results_plot_2(datasets, dataset_order):
    """
    Plot compression & decompression speed vs compression ratio:
    - X-axis: Compression Ratio
    - Y-axis: KB/s (log scale)
    - Color: Compressor model
    """
    # Manual color map for compressors
    color_map = {
        "zip": "gray",
        "gzip": "dimgray",
        "zstd -3": "lightgray",
        "zstd -1": "silver",
        "zstd -19": "darkgray",
        "distilgpt2": "silver",
        "openai-community/gpt2": "lightblue",
        "xz -9e (LZMA2)": "gainsboro",
        "Qwen/Qwen2.5-0.5B": "tab:blue",
        "Qwen/Qwen2.5-1.5B": "tab:orange",
        "Qwen/Qwen2.5-7B": "tab:green",
        "Qwen/Qwen3-0.6B": "tab:purple",
        "Qwen/Qwen3-1.7B": "tab:brown",
        "Qwen/Qwen3-8B": "tab:pink",
        "meta-llama/Llama-3.2-1B": "tab:olive",
        "HuggingFaceTB/SmolLM-135M": "tab:red",
        "HuggingFaceTB/SmolLM2-135M": "tab:cyan"
    }

    symbol_map = {
        "zip": "o",
        "gzip": "o",
        "zstd -3": "o",
        "zstd -1": "o",
        "zstd -19": "o",
        "distilgpt2": "x",
        "openai-community/gpt2": "x",
        "xz -9e (LZMA2)": "o",
        "Qwen/Qwen2.5-0.5B": "*",
        "Qwen/Qwen2.5-1.5B": "*",
        "Qwen/Qwen2.5-7B": "*",
        "Qwen/Qwen2.5-14B": "*",
        "Qwen/Qwen3-0.6B": "s",
        "Qwen/Qwen3-1.7B": "s",
        "Qwen/Qwen3-8B": "s",
        "Qwen/Qwen3-14B": "s",
        "meta-llama/Llama-3.2-1B": "D",
        "HuggingFaceTB/SmolLM-135M": "h",
        "HuggingFaceTB/SmolLM2-135M": "h"
    }

    color_order = list(color_map.keys())

    # ===== Compression Speed Plot =====
    fig_c, axes_c = plt.subplots(1, len(dataset_order), figsize=(10, 6))
    if len(dataset_order) == 1:
        axes_c = [axes_c]

    for idx, dataset in enumerate(dataset_order):
        if dataset not in datasets:
            continue
        comp_dict = datasets[dataset]
        
        ax_c = axes_c[idx]
        ax_c.set_title(f"Compression Speed - {dataset}")

        gzip_compression_ratio = comp_dict.get("gzip", {}).get("original_size_bits", 0) / comp_dict.get("gzip", {}).get("compressed_size_bits", 0)

        for comp_name, m in comp_dict.items():
            orig_bits = m["original_size_bits"]
            comp_bits = m["compressed_size_bits"]
            # ratio_percent = (comp_bits / orig_bits) * 100 if orig_bits else 0
            ratio_percent = (orig_bits / comp_bits) / gzip_compression_ratio
            speed_KBps = (orig_bits / 8 / 1024) / m["compression_time"] if m["compression_time"] > 0 else 0
            color = color_map.get(comp_name, "tab:red")
            ax_c.scatter(ratio_percent, speed_KBps, color=color, label=comp_name, marker=symbol_map.get(comp_name, 'o'))

        ax_c.set_xlabel("Compression Ratio (Normalized to Gzip)")
        ax_c.set_ylabel("Compression Speed (KB/s)")
        ax_c.set_yscale("log")
        handles, labels = ax_c.get_legend_handles_labels()
        sorted_items = sorted(
            zip(labels, handles),
            key=lambda x: color_order.index(x[0]) if x[0] in color_order else len(color_order)
        )

        sorted_labels, sorted_handles = zip(*sorted_items)
        ax_c.legend(sorted_handles, sorted_labels)

    plt.tight_layout()
    plt.savefig("compression_speed_plot.png")
    # plt.savefig(
    #     f"compression_speed_plot.pdf",
    #     format="pdf",
    #     bbox_inches="tight",
    #     pad_inches=0.01,
    #     dpi=300
    # )
    plt.close()

    # ===== Decompression Speed Plot =====
    fig_d, axes_d = plt.subplots(1, len(dataset_order), figsize=(10, 6))
    if len(dataset_order) == 1:
        axes_d = [axes_d]

    for idx, dataset in enumerate(dataset_order):
        if dataset not in datasets:
            continue
        comp_dict = datasets[dataset]

        ax_d = axes_d[idx]
        ax_d.set_title(f"Decompression Speed - {dataset}")

        gzip_compression_ratio = comp_dict.get("gzip", {}).get("original_size_bits", 0) / comp_dict.get("gzip", {}).get("compressed_size_bits", 0)

        for comp_name, m in comp_dict.items():
            orig_bits = m["original_size_bits"]
            comp_bits = m["compressed_size_bits"]
            # ratio_percent = (comp_bits / orig_bits) * 100 if orig_bits else 0
            ratio_percent = (orig_bits / comp_bits) / gzip_compression_ratio
            speed_KBps = (comp_bits / 8 / 1024) / m["decompression_time"] if m["decompression_time"] > 0 else 0
            color = color_map.get(comp_name, "tab:red")
            ax_d.scatter(ratio_percent, speed_KBps, color=color, label=comp_name, marker=symbol_map.get(comp_name, 'o'))

        ax_d.set_xlabel("Compression Ratio (Normalized to Gzip)")
        ax_d.set_ylabel("Decompression Speed (KB/s)")
        ax_d.set_yscale("log")
        handles, labels = ax_d.get_legend_handles_labels()
        sorted_items = sorted(
            zip(labels, handles),
            key=lambda x: color_order.index(x[0]) if x[0] in color_order else len(color_order)
        )

        sorted_labels, sorted_handles = zip(*sorted_items)
        ax_d.legend(sorted_handles, sorted_labels)

    plt.tight_layout()
    plt.savefig("decompression_speed_plot.png")
    # plt.savefig(
    #     f"decompression_speed_plot.pdf",
    #     format="pdf",
    #     bbox_inches="tight",
    #     pad_inches=0.01,
    #     dpi=300
    # )
    plt.close()

    print("Plots saved as compression_speed_plot.png and decompression_speed_plot.png")


def result_plot_3(path):

    BATCH_SIZES = [1, 4, 16, 64, 256]
    CONTEXT_LENGTHS = [128, 256, 512, 1024]

    # Load the JSON data
    with open(path, "r") as f:
        data = json.load(f)

    rows = []

    for batch_size in BATCH_SIZES:
        for context_length in CONTEXT_LENGTHS:
            m = data[f"text8:Qwen/Qwen2.5-0.5B|ctx={context_length}|ret={context_length // 10}|n=100000|kv=True|batch={batch_size}"]
            throughput = float(m["compression"]["original_size_bits"]) / float(m["compression"]["inference_time_sec"]) / 1024 / 8  # Convert to KB/s
            rows.append({
                "context_len": context_length,
                "batch_size": batch_size,
                "throughput": throughput
            })

    df = (
        pd.DataFrame(rows)
        .groupby(["context_len", "batch_size"])["throughput"]
        .mean()
        .unstack()
    )

    plt.figure(figsize=(8, 5))
    sns.heatmap(df, annot=True, fmt=".2f", linewidths=.5, cbar_kws={"label": "kB/s"})
    plt.xlabel("Batch size")
    plt.ylabel("Context length")
    plt.title("Processed kB/s")
    plt.tight_layout()
    # os.makedirs("figures", exist_ok=True)
    plt.savefig(f"kb_heatmap.png")

if __name__ == "__main__":
    # ===== Example Usage =====
    dataset_order = ["text8", "combined_100mb.py"]

    model_data = load_model_results(
        "compression_results.json",
        selected_datasets=dataset_order,
        selected_n=[500000]
    )

    baseline_text8 = load_baseline_csv(
        "text8_baseline.csv",
        dataset_name="text8",
        selected_compressors=["gzip"]
    )

    pytorrent_data = load_baseline_csv(
        "pytorrent_baseline.csv",
        dataset_name="combined_100mb.py",
        selected_compressors=["gzip"]
    )

    print("Loaded model data:", json.dumps(model_data, indent=2))
    # write data to JSON
    with open("compression_results_plot.json", "w") as f:
        json.dump(model_data, f, indent=2)

    # Merge results (model + baselines)
    final_data = merge_results(model_data, baseline_text8)
    final_data = merge_results(final_data, pytorrent_data)

    # Plot results
    results_plot_1(final_data, dataset_order)
    results_plot_2(final_data, dataset_order)
    result_plot_3("compression_results.json")
