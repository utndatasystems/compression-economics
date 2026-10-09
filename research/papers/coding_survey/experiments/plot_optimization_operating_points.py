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
DEFAULT_EXTERNAL_BASELINES = (
    REPO_ROOT
    / "artifacts/papers/coding-survey/llm-baselines/text8-first-10MiB.json"
)
DEFAULT_INPUTS = (
    ARTIFACT_DIR / "text8-first-10MiB-roundtrip-b1024-c512-r256.json",
    ARTIFACT_DIR / "text8-first-10MiB-roundtrip-b8192-c128-r64.json",
    ARTIFACT_DIR / "text8-first-10MiB-roundtrip-b16384.json",
)
DEFAULT_PDF = (
    REPO_ROOT
    / "research/papers/coding_survey/manuscript/figures/"
    "optimization_operating_points.pdf"
)
LLMZIP_REPORTED_RATIO = 8 / 0.7101
LLMZIP_PROJECTED_ENCODING_MIB_S = 0.000356
X_MAX = 12.0


def _load(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data["input_prefix_bytes"] != 10 * 1024 * 1024:
        raise ValueError(f"{path} is not a complete 10 MiB run")
    if data["repeats"] < 3:
        raise ValueError(f"{path} contains fewer than three runs")
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


def _load_external_baselines(path: Path, input_sha256: str) -> dict:
    data = _load_json(path)
    if data["input"]["bytes"] != 10 * 1024 * 1024:
        raise ValueError(f"{path} is not a complete 10 MiB run")
    if data["input"]["sha256"] != input_sha256:
        raise ValueError(f"{path} does not use the matched source bytes")
    if {row["codec"] for row in data["codecs"]} != {"finezip", "nacrith"}:
        raise ValueError(f"{path} must contain FineZip and Nacrith")
    if not all(row["validation"]["exact_byte_recovery"] for row in data["codecs"]):
        raise ValueError(f"{path} contains a failed decoder")
    return data


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", type=Path, default=DEFAULT_INPUTS)
    parser.add_argument("--baselines", type=Path, default=DEFAULT_BASELINES)
    parser.add_argument(
        "--external-baselines",
        type=Path,
        default=DEFAULT_EXTERNAL_BASELINES,
    )
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    args = parser.parse_args()

    rows = [_load(path) for path in args.inputs]
    rows.sort(key=lambda row: row["batch_size"])
    if len({row["input_sha256"] for row in rows}) != 1:
        raise ValueError("LLM operating points use different source bytes")
    baselines = _load_baselines(args.baselines, rows[0]["input_sha256"])
    external_baselines = _load_external_baselines(
        args.external_baselines, rows[0]["input_sha256"]
    )

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
            "axes.titlesize": 8,
            "axes.labelsize": 8,
            "axes.titleweight": "normal",
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
    external_rows = external_baselines["codecs"]
    external_ratios = [row["archive"]["compression_ratio"] for row in external_rows]
    external_encoding = [
        row["timing"]["compression_mib_per_second"] for row in external_rows
    ]
    external_decoding = [
        row["timing"]["decompression_mib_per_second"] for row in external_rows
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
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0), sharex=True)
    panel_specs = (
        {
            "ax": axes[0],
            "title": "(a) Compression",
            "llm_label": "Ours (CUDA)",
            "llm_values": encoding,
            "baseline_values": baseline_encoding,
            "external_values": external_encoding,
            "color": "#4477AA",
            "llm_marker": "o",
            "baseline_marker": "^",
            "ylabel": "Compression throughput (MiB/s)",
            "ylim": (0.0002, 2000),
            "show_projected_llmzip": True,
            "llm_label_offsets": {
                1024: (-3, 5),
                8192: (3, 5),
                16384: (-3, 5),
            },
            "label_offsets": {
                "fsst8": (3, -4),
                "lz4_default": (3, 3),
                "gzip_6": (-3, -4),
                "zstd_3": (3, 3),
                "zstd_19": (3, 3),
                "brotli_11": (-3, -5),
            },
        },
        {
            "ax": axes[1],
            "title": "(b) Decompression",
            "llm_label": "Ours (CUDA)",
            "llm_values": decoding,
            "baseline_values": baseline_decoding,
            "external_values": external_decoding,
            "color": "#4477AA",
            "llm_marker": "o",
            "baseline_marker": "^",
            "ylabel": "Decompression throughput (MiB/s)",
            "ylim": (0.006, 10000),
            "show_projected_llmzip": False,
            "llm_label_offsets": {
                1024: (-3, -6),
                8192: (3, 6),
                16384: (-3, 5),
            },
            "label_offsets": {
                "fsst8": (3, 3),
                "lz4_default": (-3, -4),
                "gzip_6": (-3, -4),
                "zstd_3": (3, 3),
                "zstd_19": (3, 3),
                "brotli_11": (-3, -4),
            },
        },
    )
    for spec in panel_specs:
        ax = spec["ax"]
        ax.scatter(
            ratios,
            spec["llm_values"],
            s=42,
            facecolors=spec["color"],
            marker=spec["llm_marker"],
            edgecolors="white",
            linewidths=0.8,
            label=spec["llm_label"],
            zorder=3,
        )
        winner_indices = [
            index for index, row in enumerate(rows)
            if row["batch_size"] in {1024, 16384}
        ]
        ax.scatter(
            [ratios[index] for index in winner_indices],
            [spec["llm_values"][index] for index in winner_indices],
            s=78,
            marker="o",
            facecolors="none",
            edgecolors="#B8860B",
            linewidths=1.1,
            zorder=5,
        )
        external_styles = {
            "finezip": {"label": "FineZip", "marker": "D", "color": "#CC6677"},
            "nacrith": {"label": "Nacrith", "marker": "X", "color": "#EE7733"},
        }
        for row, ratio, rate in zip(
            external_rows, external_ratios, spec["external_values"]
        ):
            style = external_styles[row["codec"]]
            ax.scatter(
                [ratio],
                [rate],
                s=42,
                marker=style["marker"],
                facecolors=style["color"],
                edgecolors="white",
                linewidths=0.8,
                label=style["label"],
                zorder=4,
            )
        if spec["show_projected_llmzip"]:
            ax.scatter(
                [LLMZIP_REPORTED_RATIO],
                [LLMZIP_PROJECTED_ENCODING_MIB_S],
                s=58,
                marker="*",
                facecolors="#AA4499",
                edgecolors="white",
                linewidths=0.7,
                label="LLMZip (projected)",
                zorder=5,
            )
            ax.annotate(
                "LLMZip projected",
                xy=(LLMZIP_REPORTED_RATIO, LLMZIP_PROJECTED_ENCODING_MIB_S),
                xytext=(-3, 5),
                textcoords="offset points",
                ha="right",
                va="bottom",
                fontsize=6.7,
                color="#3C4043",
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
        for row, ratio, rate in zip(
            external_rows, external_ratios, spec["external_values"]
        ):
            offset = (-3, -5) if row["codec"] == "finezip" else (3, 3)
            ax.annotate(
                "FineZip" if row["codec"] == "finezip" else "Nacrith",
                xy=(ratio, rate),
                xytext=offset,
                textcoords="offset points",
                ha="right" if offset[0] < 0 else "left",
                va="top" if offset[1] < 0 else "bottom",
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
        ax.axvspan(r_star, X_MAX, color="#FFF4E5", alpha=0.35, zorder=0)
        for index, row in enumerate(rows):
            batch = row["batch_size"]
            label = {
                1024: "B1K · C512",
                8192: "B8K · C128",
                16384: "B16K · C128",
            }[batch]
            offset = spec["llm_label_offsets"][batch]
            ax.annotate(
                label,
                xy=(ratios[index], spec["llm_values"][index]),
                xytext=offset,
                textcoords="offset points",
                ha="right" if offset[0] < 0 else "left",
                va="bottom" if offset[1] > 0 else "top",
                fontsize=6.5,
                color="#202124",
            )
        ax.set_title(spec["title"], loc="center", pad=8)
        ax.set_ylabel(spec["ylabel"])
        ax.set_xlim(0, X_MAX)
        ax.set_xticks(range(0, 13, 2), [f"{value}×" for value in range(0, 13, 2)])
        ax.set_yscale("log")
        ax.set_ylim(*spec["ylim"])
        ax.grid(color="#E8EAED", linewidth=0.65, which="both")
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.annotate(
            "Better",
            xy=(0.90, 0.84),
            xytext=(0.73, 0.68),
            xycoords="axes fraction",
            textcoords="axes fraction",
            ha="center",
            va="center",
            rotation=38,
            fontsize=7.2,
            fontweight="bold",
            color="#37474F",
            arrowprops={
                "arrowstyle": "-|>",
                "color": "#37474F",
                "linewidth": 1.0,
                "mutation_scale": 8,
            },
        )

    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.06),
        ncol=6,
        frameon=False,
        fontsize=6.4,
    )
    fig.supxlabel(
        "Compression ratio",
        y=0.17,
        fontsize=plt.rcParams["axes.labelsize"],
        fontweight="normal",
        color=plt.rcParams["axes.labelcolor"],
    )
    fig.tight_layout(rect=(0.0, 0.16, 1.0, 1.0), pad=0.8, w_pad=1.2)

    metadata = {
        "Title": "Rate-constrained LLM and traditional-codec operating points",
        "Author": "compression-economics experiment pipeline",
        "Subject": "Complete matched 10 MiB text8 round-trip experiments",
        "CreationDate": datetime(2026, 10, 6, tzinfo=timezone.utc),
        "ModDate": datetime(2026, 10, 6, tzinfo=timezone.utc),
    }
    args.pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.pdf, bbox_inches="tight", metadata=metadata)
    plt.close(fig)
    print(f"Saved {args.pdf}")


if __name__ == "__main__":
    main()
