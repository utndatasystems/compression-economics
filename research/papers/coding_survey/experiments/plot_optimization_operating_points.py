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
DEFAULT_BASELINES = (
    REPO_ROOT
    / "artifacts/papers/coding-survey/traditional-codecs/text8-first-10MiB.json"
)
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


def _load_baselines(path: Path, input_sha256: str) -> dict:
    data = _load_json(path)
    if data["source"]["bytes"] != 10 * 1024 * 1024:
        raise ValueError(f"{path} is not a complete 10 MiB run")
    if data["source"]["sha256"] != input_sha256:
        raise ValueError(f"{path} does not use the matched source bytes")
    if data["protocol"]["timed_repetitions"] < 5:
        raise ValueError(f"{path} has fewer than five timed repetitions")
    if not all(row["exact_recovery"] for row in data["codecs"]):
        raise ValueError(f"{path} contains a failed decoder")
    return data


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs=2, type=Path, default=DEFAULT_INPUTS)
    parser.add_argument("--baselines", type=Path, default=DEFAULT_BASELINES)
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--png", type=Path, default=DEFAULT_PDF.with_suffix(".png"))
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()

    rows = [_load(path) for path in args.inputs]
    rows.sort(key=lambda row: row["batch_size"])
    if rows[0]["input_sha256"] != rows[1]["input_sha256"]:
        raise ValueError("LLM operating points use different source bytes")
    baselines = _load_baselines(args.baselines, rows[0]["input_sha256"])

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
    baseline_ratios = [row["compression_ratio"] for row in baselines["codecs"]]
    baseline_encoding = [
        row["compression_summary"]["median"]["source_mib_per_second"]
        for row in baselines["codecs"]
    ]
    baseline_decoding = [
        row["decompression_summary"]["median"]["source_mib_per_second"]
        for row in baselines["codecs"]
    ]
    short_labels = {
        "fsst8": "FSST8",
        "lz4_default": "LZ4",
        "gzip_6": "gzip-6",
        "zstd_3": "zstd-3",
        "zstd_19": "zstd-19",
        "brotli_11": "Brotli-11",
    }
    r_star = baselines["rate_constraint"]["r_star"]
    encode_gain = encoding[1] / encoding[0]
    decode_gain = decoding[1] / decoding[0]
    ratio_loss = 1.0 - ratios[1] / ratios[0]

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.25), sharex=True)
    panel_specs = (
        {
            "ax": axes[0],
            "title": "(a) Compression",
            "llm_label": "LLM CUDA v2",
            "llm_values": encoding,
            "baseline_values": baseline_encoding,
            "color": "#4477AA",
            "llm_marker": "o",
            "baseline_marker": "^",
            "ylabel": "Compression throughput (MiB/s)",
            "ylim": (0.06, 2000),
            "label_offsets": {
                "fsst8": (5, -7),
                "lz4_default": (5, 5),
                "gzip_6": (-5, -7),
                "zstd_3": (5, 5),
                "zstd_19": (5, 5),
                "brotli_11": (-5, -8),
            },
        },
        {
            "ax": axes[1],
            "title": "(b) Decompression",
            "llm_label": "LLM CUDA-CDF",
            "llm_values": decoding,
            "baseline_values": baseline_decoding,
            "color": "#228833",
            "llm_marker": "s",
            "baseline_marker": "v",
            "ylabel": "Decompression throughput (MiB/s)",
            "ylim": (0.06, 10000),
            "label_offsets": {
                "fsst8": (5, 5),
                "lz4_default": (-5, -7),
                "gzip_6": (-5, -7),
                "zstd_3": (5, 5),
                "zstd_19": (5, 5),
                "brotli_11": (-5, -7),
            },
        },
    )
    for spec in panel_specs:
        ax = spec["ax"]
        ax.plot(
            ratios,
            spec["llm_values"],
            color=spec["color"],
            marker=spec["llm_marker"],
            markersize=6.5,
            linewidth=1.5,
            markeredgecolor="white",
            markeredgewidth=0.8,
            label=spec["llm_label"],
            zorder=3,
        )
        ax.scatter(
            baseline_ratios,
            spec["baseline_values"],
            s=34,
            marker=spec["baseline_marker"],
            facecolors="white",
            edgecolors=spec["color"],
            linewidths=1.2,
            label="Traditional CPU",
            zorder=4,
        )
        for row, ratio, rate in zip(
            baselines["codecs"], baseline_ratios, spec["baseline_values"]
        ):
            offset = spec["label_offsets"][row["id"]]
            ax.annotate(
                short_labels[row["id"]],
                xy=(ratio, rate),
                xytext=offset,
                textcoords="offset points",
                ha="left" if offset[0] > 0 else "right",
                va="bottom" if offset[1] > 0 else "top",
                fontsize=6.7,
                color="#3C4043",
            )
        ax.axvline(
            r_star,
            color="#B06000",
            linewidth=1.0,
            linestyle="--",
            label=f"$R^*={r_star:.3f}\\times$",
            zorder=2,
        )
        ax.axvspan(r_star, 8.1, color="#FFF4E5", alpha=0.35, zorder=0)
        for index, row in enumerate(rows):
            batch = row["batch_size"]
            context = row["context_length"]
            offset = (-6, 31) if batch == 256 else (6, -9)
            ax.annotate(
                f"batch {batch:,}, ctx {context}\n{ratios[index]:.3f}× ratio",
                xy=(ratios[index], spec["llm_values"][index]),
                xytext=offset,
                textcoords="offset points",
                ha="right" if batch == 256 else "left",
                va="bottom" if batch == 256 else "top",
                fontsize=6.5,
                color="#202124",
                arrowprops={
                    "arrowstyle": "-",
                    "color": "#9AA0A6",
                    "linewidth": 0.6,
                },
            )
        ax.set_title(spec["title"], loc="left", pad=6)
        ax.set_ylabel(spec["ylabel"])
        ax.set_xlim(1.55, 8.10)
        ax.set_yscale("log")
        ax.set_ylim(*spec["ylim"])
        ax.grid(color="#E8EAED", linewidth=0.65, which="both")
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(loc="upper right", frameon=False, fontsize=6.4)

    fig.suptitle(
        "Rate-constrained frontier on the same 10 MiB source",
        x=0.08,
        ha="left",
        fontsize=10,
        fontweight="bold",
    )
    fig.supxlabel("Compression ratio (original / charged archive)", y=0.06)
    fig.text(
        0.08,
        0.01,
        "Same text8 10 MiB bytes; LLM on H100 (3 runs), traditional codecs on CPU (7 runs); exact round trips",
        ha="left",
        va="bottom",
        fontsize=7.0,
        color="#5F6368",
    )
    fig.tight_layout(rect=(0.0, 0.11, 1.0, 0.94), pad=0.8, w_pad=1.2)

    metadata = {
        "Title": "Rate-constrained LLM and traditional-codec operating points",
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
