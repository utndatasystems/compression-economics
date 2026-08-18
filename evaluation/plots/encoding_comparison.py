#!/usr/bin/env python3
"""Plot the checked-in AC, Huffman-rank, and bitpacked-rank results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


SCHEMES = {
    "AC": {
        "label": "Arithmetic coding",
        "color": "#1f77b4",
        "marker": "o",
    },
    "huffman": {
        "label": "Rank + Huffman",
        "color": "#d97706",
        "marker": "s",
    },
    "bitpacked": {
        "label": "Rank + bitpacking",
        "color": "#2fb344",
        "marker": "D",
    },
}


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        type=Path,
        default=repo_root / "artifacts/runs/current/compression_results.json",
        help="Compression-results JSON to plot.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "artifacts/figures/coding_schemes",
        help="Directory for PNG and PDF output.",
    )
    return parser.parse_args()


def load_rows(results_path: Path) -> list[dict]:
    with results_path.open(encoding="utf-8") as handle:
        results = json.load(handle)

    rows = []
    for entry in results.values():
        compression = entry.get("compression", {})
        args = compression.get("args", {})
        encoding = args.get("encoding")
        if (
            args.get("input_path") == "./data/text8"
            and args.get("model_name") == "Qwen/Qwen2.5-0.5B"
            and args.get("first_n_tokens") == 100_000
            and args.get("lora_path") is None
            and args.get("reduce_tokens") is True
            and encoding in SCHEMES
        ):
            rows.append(
                {
                    "batch_size": args["batch_size"],
                    "encoding": encoding,
                    "original_size": compression["original_size_bytes"],
                    "final_size": compression["final_size_bytes"],
                    "factor": compression["compression_factor"],
                }
            )

    expected = len(SCHEMES) * 5
    if len(rows) != expected:
        raise ValueError(
            f"Expected {expected} matching measurements, found {len(rows)}"
        )
    return rows


def add_value_labels(ax, xs, ys, *, fmt, color) -> None:
    for x, y in zip(xs, ys):
        ax.annotate(
            fmt(y),
            (x, y),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#303030",
        )


def plot(rows: list[dict], output_dir: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "axes.titlesize": 15,
            "axes.labelsize": 14,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "axes.labelcolor": "#202020",
            "axes.edgecolor": "#202020",
            "xtick.color": "#202020",
            "ytick.color": "#202020",
        }
    )

    figure, axes = plt.subplots(1, 2, figsize=(15, 7.5), constrained_layout=False)
    figure.patch.set_facecolor("white")

    for ax in axes:
        ax.set_facecolor("white")
        ax.grid(True, color="#D7D7D7", linewidth=1.0, linestyle="--", alpha=0.72)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_linewidth(1.2)
        ax.tick_params(axis="both", which="major", length=5, width=1.1)

    for encoding, style in SCHEMES.items():
        scheme_rows = sorted(
            (row for row in rows if row["encoding"] == encoding),
            key=lambda row: row["batch_size"],
        )
        batches = [row["batch_size"] for row in scheme_rows]
        factors = [row["factor"] for row in scheme_rows]
        size_ratios = [
            row["final_size"] / row["original_size"] for row in scheme_rows
        ]

        line_options = {
            "color": style["color"],
            "marker": style["marker"],
            "markersize": 8,
            "markeredgecolor": "#666666",
            "markeredgewidth": 1.0,
            "linewidth": 1.6,
            "label": style["label"],
        }
        axes[0].plot(batches, factors, **line_options)
        axes[1].plot(batches, size_ratios, **line_options)
        add_value_labels(
            axes[0],
            batches,
            factors,
            fmt=lambda value: f"{value:.2f}×",
            color=style["color"],
        )
        add_value_labels(
            axes[1],
            batches,
            size_ratios,
            fmt=lambda value: f"{value:.1%}",
            color=style["color"],
        )

    batch_sizes = sorted({row["batch_size"] for row in rows})
    axes[0].set(
        title="Compression ratio",
        xlabel="Batch Size",
        ylabel="Compression Ratio (higher is better)",
        xticks=batch_sizes,
    )
    axes[0].set_ylim(2.25, 8.1)

    axes[1].set(
        title="Compressed size relative to input",
        xlabel="Batch Size",
        ylabel="Compressed Size (lower is better)",
        xticks=batch_sizes,
    )
    axes[1].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[1].set_ylim(0.08, 0.40)

    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=3,
        frameon=False,
        fontsize=12,
        handlelength=2.4,
        columnspacing=2.2,
    )

    figure.suptitle(
        "Coding Scheme Impact on text8 (Base model only - no fine-tuning)",
        x=0.5,
        y=0.985,
        ha="center",
        fontsize=20,
        color="#202020",
    )
    figure.text(
        0.5,
        0.905,
        r"$r_{\mathrm{tx}}=100,\ c_{\mathrm{tx}}=1000,\ "
        r"B\in\{16,32,64,128,256\},\ n_{\mathrm{token}}=100000$",
        ha="center",
        fontsize=17,
        color="#202020",
    )
    figure.subplots_adjust(left=0.075, right=0.98, top=0.79, bottom=0.18, wspace=0.23)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output_dir / "coding_scheme_comparison.png",
        dpi=240,
        bbox_inches="tight",
        facecolor=figure.get_facecolor(),
    )
    figure.savefig(
        output_dir / "coding_scheme_comparison.pdf",
        bbox_inches="tight",
        facecolor=figure.get_facecolor(),
    )
    plt.close(figure)


def main() -> None:
    args = parse_args()
    plot(load_rows(args.results), args.output_dir)


if __name__ == "__main__":
    main()
