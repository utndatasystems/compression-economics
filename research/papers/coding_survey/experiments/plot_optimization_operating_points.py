#!/usr/bin/env python3
"""Plot measured throughput/compression trade-offs for complete 10 MiB runs."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[4]
ARTIFACT_DIR = REPO_ROOT / "artifacts/papers/coding-survey/cuda-ac"
DEFAULT_INPUTS = (
    ARTIFACT_DIR / "text8-first-10MiB-roundtrip-b256-c256-r100.json",
    ARTIFACT_DIR / "text8-first-10MiB-roundtrip-b16384.json",
)
DEFAULT_PDF = (
    REPO_ROOT
    / "research/papers/coding_survey/manuscript/figures/"
    "optimization_operating_points.pdf"
)


def _load(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data["input_prefix_bytes"] != 10 * 1024 * 1024:
        raise ValueError(f"{path} is not a complete 10 MiB run")
    if data["repeats"] != 3:
        raise ValueError(f"{path} does not contain the expected three runs")
    if not data["encoding"]["exact_archive_match"]:
        raise ValueError(f"{path} has non-identical target-interval archives")
    if not data["decoding"]["exact_recovery"]:
        raise ValueError(f"{path} contains a failed decoder")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs=2, type=Path, default=DEFAULT_INPUTS)
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--png", type=Path, default=DEFAULT_PDF.with_suffix(".png"))
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()

    rows = [_load(path) for path in args.inputs]
    rows.sort(key=lambda row: row["batch_size"])

    ratios = [row["representation"]["compression_ratio"] for row in rows]
    encoding = [
        row["encoding"]["medians"]["cuda_v2"]["source_mib_per_second"]
        for row in rows
    ]
    decoding = [
        row["decoding"]["medians"]["cuda"]["source_mib_per_second"]
        for row in rows
    ]

    plt.rcParams.update(
        {
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
        }
    )
    fig, ax = plt.subplots(figsize=(4.7, 3.2))
    styles = (
        ("CUDA v2 encoding", encoding, "#4477AA", "o"),
        ("CUDA-CDF decoding", decoding, "#228833", "s"),
    )
    for label, values, color, marker in styles:
        ax.plot(
            ratios,
            values,
            color=color,
            marker=marker,
            markersize=6.5,
            linewidth=1.5,
            markeredgecolor="white",
            markeredgewidth=0.8,
            label=label,
            zorder=3,
        )

    for index, row in enumerate(rows):
        batch = row["batch_size"]
        context = row["context_length"]
        offset = (8, 10) if index == 0 else (8, -10)
        ax.annotate(
            f"batch {batch:,}, context {context}\n{ratios[index]:.3f}× ratio",
            xy=(ratios[index], max(encoding[index], decoding[index])),
            xytext=offset,
            textcoords="offset points",
            ha="left",
            va="bottom" if index == 0 else "top",
            fontsize=7.1,
            color="#202124",
        )

    encode_gain = encoding[1] / encoding[0]
    decode_gain = decoding[1] / decoding[0]
    ratio_loss = 1.0 - ratios[1] / ratios[0]
    ax.text(
        0.03,
        0.05,
        f"More streams: {encode_gain:.2f}× encode, {decode_gain:.2f}× decode\n"
        f"but {ratio_loss:.1%} lower compression ratio",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.2,
        bbox={
            "boxstyle": "round,pad=0.3",
            "facecolor": "white",
            "edgecolor": "#DADCE0",
            "linewidth": 0.7,
        },
    )
    ax.set_xlabel("Compression ratio (original / charged archive)")
    ax.set_ylabel("Complete source throughput (MiB/s)")
    ax.set_title("Parallelism buys speed by spending context and seeds", loc="left", pad=8)
    ax.set_xlim(5.15, 8.03)
    ax.set_ylim(0.0, 0.64)
    ax.grid(color="#E8EAED", linewidth=0.65)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper right", frameon=False, fontsize=7.2)
    fig.text(
        0.10,
        0.01,
        "H100, Qwen2.5-0.5B, text8 10 MiB; three-run medians; exact round trips",
        ha="left",
        va="bottom",
        fontsize=7.0,
        color="#5F6368",
    )
    fig.tight_layout(rect=(0.0, 0.06, 1.0, 1.0), pad=0.8)

    metadata = {
        "Title": "Measured throughput and compression-ratio operating points",
        "Author": "compression-economics experiment pipeline",
        "Subject": "Complete matched 10 MiB text8 round-trip experiments",
        "CreationDate": datetime(2026, 10, 6, tzinfo=timezone.utc),
        "ModDate": datetime(2026, 10, 6, tzinfo=timezone.utc),
    }
    args.pdf.parent.mkdir(parents=True, exist_ok=True)
    args.png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.pdf, bbox_inches="tight", metadata=metadata)
    fig.savefig(
        args.png,
        bbox_inches="tight",
        dpi=args.dpi,
        metadata={"Software": metadata["Author"]},
    )
    plt.close(fig)
    print(f"Saved {args.pdf}")
    print(f"Saved {args.png} at {args.dpi} DPI")
    print(f"Encoding throughput gain: {encode_gain:.6f}x")
    print(f"Decoding throughput gain: {decode_gain:.6f}x")
    print(f"Compression-ratio reduction: {ratio_loss:.6%}")


if __name__ == "__main__":
    main()
