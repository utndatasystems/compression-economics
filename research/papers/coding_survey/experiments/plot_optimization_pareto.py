#!/usr/bin/env python3
"""Plot compression ratio versus measured LLM encoding throughput."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_DATA = Path(__file__).with_name("optimization_pareto_h100.json")
DEFAULT_PDF = (
    REPO_ROOT
    / "research/papers/coding_survey/manuscript/figures/compression_ratio_throughput.pdf"
)
DEFAULT_PNG = DEFAULT_PDF.with_suffix(".png")

VARIANT_STYLE = {
    "python": {"color": "#9AA0A6", "marker": "o"},
    "cuda_v1": {"color": "#4285F4", "marker": "s"},
    "cuda_v2": {"color": "#34A853", "marker": "D"},
}


def _load(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    representation = data["compressed_representation"]
    charged = (
        representation["payload_bytes"]
        + representation["vocabulary_bitmap_bytes"]
        + representation["seed_token_count"]
        * representation["seed_bytes_per_token"]
    )
    if charged != representation["charged_compressed_bytes"]:
        raise ValueError("charged compressed-size components do not add up")
    if not data["exact_payload_match"]:
        raise ValueError("optimization variants must produce identical payloads")
    variants = data["variants"]
    if [row["id"] for row in variants] != ["python", "cuda_v1", "cuda_v2"]:
        raise ValueError("expected the Python -> CUDA v1 -> CUDA v2 ablation chain")
    if any(len(row["compression_seconds"]) != data["timed_repetitions"] for row in variants):
        raise ValueError("each variant must contain every timed repetition")
    if data["source_bytes"] <= 0 or charged <= 0:
        raise ValueError("source and compressed sizes must be positive")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--png", type=Path, default=DEFAULT_PNG)
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()
    if args.dpi < 300:
        parser.error("--dpi must be at least 300 for a paper figure")

    data = _load(args.data)
    source_bytes = data["source_bytes"]
    compressed_bytes = data["compressed_representation"]["charged_compressed_bytes"]
    compression_ratio = source_bytes / compressed_bytes

    rows = []
    for variant in data["variants"]:
        samples = np.asarray(
            [source_bytes / seconds / 1_000_000 for seconds in variant["compression_seconds"]],
            dtype=np.float64,
        )
        median = statistics.median(samples.tolist())
        rows.append({
            **variant,
            "samples_mb_per_second": samples,
            "median_mb_per_second": median,
            "minimum_mb_per_second": float(samples.min()),
            "maximum_mb_per_second": float(samples.max()),
        })

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "axes.titleweight": "bold",
        "text.color": "#202124",
        "axes.labelcolor": "#202124",
        "xtick.color": "#5F6368",
        "ytick.color": "#5F6368",
        "axes.edgecolor": "#DADCE0",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig, ax = plt.subplots(figsize=(3.45, 2.85))
    medians = np.asarray([row["median_mb_per_second"] for row in rows])

    ax.plot(
        [compression_ratio] * len(rows),
        medians,
        color="#DADCE0",
        linewidth=1.2,
        zorder=1,
    )
    for row in rows:
        median = row["median_mb_per_second"]
        style = VARIANT_STYLE[row["id"]]
        ax.errorbar(
            compression_ratio,
            median,
            yerr=np.asarray([[
                median - row["minimum_mb_per_second"]
            ], [
                row["maximum_mb_per_second"] - median
            ]]),
            fmt=style["marker"],
            markersize=6.2,
            markerfacecolor=style["color"],
            markeredgecolor="white",
            markeredgewidth=0.8,
            ecolor=style["color"],
            elinewidth=1.0,
            capsize=2.5,
            zorder=3,
        )

    annotations = {
        "python": dict(xytext=(-10, -17), ha="right", va="top"),
        "cuda_v1": dict(xytext=(12, -2), ha="left", va="center"),
        "cuda_v2": dict(xytext=(12, 7), ha="left", va="bottom"),
    }
    for row in rows:
        placement = annotations[row["id"]]
        ax.annotate(
            f"{row['label']}  {row['median_mb_per_second']:.5f}",
            xy=(compression_ratio, row["median_mb_per_second"]),
            xytext=placement["xytext"],
            textcoords="offset points",
            ha=placement["ha"],
            va=placement["va"],
            fontsize=7.2,
            color=VARIANT_STYLE[row["id"]]["color"],
            fontweight="bold" if row["id"] == "cuda_v2" else "normal",
        )

    ax.set_yscale("log")
    y_ticks = np.asarray([0.0076, 0.0077, 0.0078, 0.0079, 0.0080])
    ax.yaxis.set_major_locator(FixedLocator(y_ticks))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.4f}"))
    ax.yaxis.set_minor_locator(FixedLocator([]))
    ax.set_ylim(0.00758, 0.00804)
    ax.set_xlim(compression_ratio - 0.22, compression_ratio + 0.22)
    ax.set_xticks([3.2, 3.3, 3.4, 3.5, 3.6])
    ax.set_xlabel("Compression ratio (original / compressed)")
    ax.set_ylabel("Encoding throughput (MB/s)")
    ax.set_title("Same compressed size, higher throughput", loc="left", pad=9)
    ax.text(
        0.98,
        0.05,
        f"Identical charged size: {compressed_bytes:,} B\nExternal model excluded",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.7,
        color="#5F6368",
    )
    ax.text(
        0.98,
        0.96,
        "Higher is better  ↑",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=6.8,
        color="#5F6368",
    )
    ax.text(
        0.02,
        0.96,
        "Log scale",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.8,
        color="#5F6368",
    )
    ax.grid(axis="y", color="#E8EAED", linewidth=0.65)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", length=3, color="#BDC1C6")
    fig.tight_layout(pad=0.7)

    metadata = {
        "Title": "Compression ratio versus LLM encoding throughput",
        "Author": "compression-economics experiment pipeline",
        "Subject": "Matched H100 optimization ablation",
        "CreationDate": datetime(2026, 10, 2, tzinfo=timezone.utc),
        "ModDate": datetime(2026, 10, 2, tzinfo=timezone.utc),
    }
    args.pdf.parent.mkdir(parents=True, exist_ok=True)
    args.png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.pdf, bbox_inches="tight", metadata=metadata)
    fig.savefig(args.png, bbox_inches="tight", dpi=args.dpi, metadata={"Software": metadata["Author"]})
    plt.close(fig)

    print(f"Saved {args.pdf}")
    print(f"Saved {args.png} at {args.dpi} DPI")
    print(f"Compression ratio: {compression_ratio:.6f}x ({source_bytes}/{compressed_bytes} bytes)")
    for row in rows:
        print(
            f"{row['label']}: median {row['median_mb_per_second']:.8f} MB/s; "
            f"range {row['minimum_mb_per_second']:.8f}--"
            f"{row['maximum_mb_per_second']:.8f}"
        )


if __name__ == "__main__":
    main()
