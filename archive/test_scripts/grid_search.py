import os
from pathlib import Path
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib


BATCH_SIZES = [4, 16, 64, 128, 256, 512]
CONTEXT_LENGTHS = [128, 256, 512, 1024]
NUM_TOKENS = 1_000_000

matplotlib.rcParams.update({'font.size': 11, 'font.family': 'serif', 'axes.titlesize': 'medium', 'figure.titlesize': 'medium', 'text.usetex': True, 'text.latex.preamble': '\\usepackage{amsmath}\\usepackage{amssymb}\\usepackage{siunitx}[=v2]', 'pgf.rcfonts': False, 'pgf.texsystem': 'pdflatex'})


def plot(metric, title, label, figname):
    with open("compression_results_grid_search.json") as f:
        results = json.load(f)

    rows = []
    for _, result in results.items():
        if result["compression"]["args"]["batch_size"] not in BATCH_SIZES:
            continue
        if result["compression"]["args"]["context_length"] not in CONTEXT_LENGTHS:
            continue

        batch_size = int(result["compression"]["args"]["batch_size"])
        context_length = int(result["compression"]["args"]["context_length"])
        throughput = metric(result)

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

    plt.figure(figsize=(4.9, 3.5))
    sns.heatmap(df, annot=True, fmt=".2f", linewidths=.5, cbar_kws={"label": label})
    plt.xlabel("Batch Size")
    plt.ylabel("Context Window Size")
    plt.title(title)
    
    os.makedirs("figures", exist_ok=True)
    plt.tight_layout()
    plt.savefig(f"figures/{figname}.png", dpi=300, bbox_inches='tight', pad_inches=0.01)


if __name__ == "__main__":
    #for batch_size in BATCH_SIZES:
    #    for context_length in CONTEXT_LENGTHS:
    #        os.system(
    #            f"python main.py --input_path data/text8 --mode compress --batch_size {batch_size} --context_length {context_length} --first_n_tokens {NUM_TOKENS} --use_kv_cache"
    #        )

    plot(
        lambda stats: float(stats["compression"]["inference_throughput_kibibytes_per_sec"]),
        "Throughput [KiB/s]",
        "KiB/s",
        "inference_throughput_heatmap"
    )
    plot(
        lambda stats: float(stats["compression"]["compression_factor"]),
        "Compression Factor",
        "Factor",
        "compression_factor_heatmap"
    )
    plot(
        lambda stats: float(stats["compression"]["pure_compression_factor"]),
        "Pure Compression Factor",
        "Factor",
        "pure_compression_factor_heatmap"
    )