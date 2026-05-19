import os
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib


BATCH_SIZES = [4, 16, 64, 128, 256, 512, 1024, 2048]
CONTEXT_LENGTHS = [128, 256, 512, 1024]
NUM_TOKENS = 1_000_000

matplotlib.rcParams.update({
    "font.size": 11,
    "font.family": "serif",
    "axes.titlesize": "medium",
    "figure.titlesize": "medium",

    # Do NOT require a LaTeX installation
    "text.usetex": False,

    # Use Matplotlib's internal math renderer
    "mathtext.fontset": "dejavuserif",
})


def plot(metric, title, label, figname):
    with open("compression_results_H100.json") as f:
        results = json.load(f)

    rows = []
    for _, result in results.items():
        args = result["compression"]["args"]

        if args["batch_size"] not in BATCH_SIZES:
            continue
        if args["context_length"] not in CONTEXT_LENGTHS:
            continue

        rows.append({
            "context_len": int(args["context_length"]),
            "batch_size": int(args["batch_size"]),
            "throughput": metric(result),
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
    plt.savefig(
        f"other_files/figures/{figname}.png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.01,
    )
    plt.close()


if __name__ == "__main__":
    plot(
        lambda stats: float(stats["compression"]["inference_throughput_kibibytes_per_sec"]),
        "Throughput [KiB/s]",
        "KiB/s",
        "inference_throughput_heatmap_H100",
    )

    plot(
        lambda stats: float(stats["compression"]["compression_factor"]),
        "Compression Factor",
        "Factor",
        "compression_factor_heatmap_H100",
    )

    plot(
        lambda stats: float(stats["compression"]["pure_compression_factor"]),
        "Pure Compression Factor",
        "Factor",
        "pure_compression_factor_heatmap_H100",
    )
    print('Done plotting!')