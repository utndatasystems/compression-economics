#!/usr/bin/env python3
"""Plot measured CUDA-v3 encoding and matched decoding operating points."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullLocator


REPO_ROOT = Path(__file__).resolve().parents[4]
ARTIFACT_DIR = REPO_ROOT / "artifacts/papers/coding-survey/cuda-ac"
V3_SWEEP_DIR = ARTIFACT_DIR / "text8-v3-heatmap-lower-triangle"
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
DEFAULT_ENCODING_INPUTS = (
    V3_SWEEP_DIR / "text8-first-10MiB-v3-b1024-c512-r256.json",
    V3_SWEEP_DIR / "text8-first-10MiB-v3-b8192-c128-r64.json",
    V3_SWEEP_DIR / "text8-first-10MiB-v3-b16384-c128-r64.json",
)
DEFAULT_PDF = (
    REPO_ROOT
    / "research/papers/coding_survey/manuscript/figures/"
    "optimization_operating_points.pdf"
)
DEFAULT_ZOOMED_PDF = DEFAULT_PDF.with_name(
    "optimization_operating_points_llm_zoom.pdf"
)
X_MAX = 8.4
Y_LIMITS = (0.0002, 10000)
Y_TICKS = (0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000)
OURS_COLOR = "#0072B2"
TRADITIONAL_COLOR = "#7A8793"


def _load_result(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data["input_prefix_bytes"] != 10 * 1024 * 1024:
        raise ValueError(f"{path} is not a complete 10 MiB run")
    if data["repeats"] < 3:
        raise ValueError(f"{path} contains fewer than three runs")
    if not data["encoding"]["exact_archive_match"]:
        raise ValueError(f"{path} has non-identical target-interval archives")
    return data


def _load_encoding(path: Path) -> dict:
    data = _load_result(path)
    if data["encoding"]["conditions"] != ["cuda_v3"]:
        raise ValueError(f"{path} is not a CUDA-v3-only encoding result")
    if "cuda_v3" not in data["encoding"]["medians"]:
        raise ValueError(f"{path} has no CUDA-v3 encoding median")
    return data


def _load_roundtrip(path: Path) -> dict:
    data = _load_result(path)
    if not data["decoding"]["exact_recovery"]:
        raise ValueError(f"{path} contains a failed decoder")
    if "cuda" not in data["decoding"]["medians"]:
        raise ValueError(f"{path} has no fused-CUDA decoding median")
    return data


def _coordinate(row: dict) -> tuple[int, int, int]:
    return row["batch_size"], row["context_length"], row["retain_tokens"]


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
    if {row["codec"] for row in data["codecs"]} != {
        "finezip",
        "llmzip",
        "nacrith",
    }:
        raise ValueError(f"{path} must contain FineZip, LLMZip, and Nacrith")
    if not all(row["validation"]["exact_byte_recovery"] for row in data["codecs"]):
        raise ValueError(f"{path} contains a failed decoder")
    return data


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--encoding-inputs",
        nargs="+",
        type=Path,
        default=DEFAULT_ENCODING_INPUTS,
        help="CUDA-v3 encoding-only sweep results",
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        type=Path,
        default=DEFAULT_INPUTS,
        help="matched round-trip results supplying fused-decoder rates",
    )
    parser.add_argument("--baselines", type=Path, default=DEFAULT_BASELINES)
    parser.add_argument(
        "--external-baselines",
        type=Path,
        default=DEFAULT_EXTERNAL_BASELINES,
    )
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--zoomed-pdf", type=Path, default=DEFAULT_ZOOMED_PDF)
    parser.add_argument("--preview-png", type=Path)
    parser.add_argument("--zoomed-preview-png", type=Path)
    args = parser.parse_args()

    rows = [_load_encoding(path) for path in args.encoding_inputs]
    rows.sort(key=lambda row: row["batch_size"])
    decoder_rows = [_load_roundtrip(path) for path in args.inputs]
    decoder_by_coordinate = {_coordinate(row): row for row in decoder_rows}
    if len(decoder_by_coordinate) != len(decoder_rows):
        raise ValueError("duplicate decoder operating-point coordinates")
    if set(decoder_by_coordinate) != {_coordinate(row) for row in rows}:
        raise ValueError("encoding and decoding operating-point coordinates differ")
    all_rows = rows + decoder_rows
    if len({row["input_sha256"] for row in all_rows}) != 1:
        raise ValueError("LLM operating points use different source bytes")
    if len({(row["model"], row["model_revision"]) for row in all_rows}) != 1:
        raise ValueError("LLM operating points use different model revisions")
    baselines = _load_baselines(args.baselines, rows[0]["input_sha256"])
    external_baselines = _load_external_baselines(
        args.external_baselines, rows[0]["input_sha256"]
    )

    ratios = [row["representation"]["compression_ratio"] for row in rows]
    encoding = [
        row["encoding"]["medians"]["cuda_v3"]["source_mib_per_second"]
        for row in rows
    ]
    decoding = [
        decoder_by_coordinate[_coordinate(row)]["decoding"]["medians"]["cuda"]
        ["source_mib_per_second"]
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
    traditional_artists = []
    threshold_artists = []
    direction_artists = []
    llm_annotations = {}
    external_annotations = {}
    panel_specs = (
        {
            "ax": axes[0],
            "title": "(a) Compression",
            "llm_label": "Ours (CUDA)",
            "llm_values": encoding,
            "baseline_values": baseline_encoding,
            "external_values": external_encoding,
            "color": OURS_COLOR,
            "llm_marker": "o",
            "baseline_marker": "^",
            "ylabel": "Compression throughput (MiB/s)",
            "llm_label_offsets": {
                1024: (-3, 8),
                8192: (3, 8),
                16384: (3, 10),
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
            "color": OURS_COLOR,
            "llm_marker": "o",
            "baseline_marker": "^",
            "ylabel": "Decompression throughput (MiB/s)",
            "llm_label_offsets": {
                1024: (-3, 8),
                8192: (3, 8),
                16384: (3, 10),
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
    for panel_index, spec in enumerate(panel_specs):
        ax = spec["ax"]
        ax.plot(
            ratios,
            spec["llm_values"],
            color=spec["color"],
            linewidth=1.2,
            alpha=0.75,
            zorder=2,
        )
        ax.scatter(
            ratios,
            spec["llm_values"],
            s=58,
            facecolors=spec["color"],
            marker=spec["llm_marker"],
            edgecolors="white",
            linewidths=1.0,
            label=spec["llm_label"],
            zorder=5,
        )
        external_styles = {
            "finezip": {
                "label": "FineZip", "marker": "D", "color": "#CC6677", "size": 42
            },
            "llmzip": {
                "label": "LLMZip", "marker": "*", "color": "#AA4499", "size": 60
            },
            "nacrith": {
                "label": "Nacrith", "marker": "X", "color": "#EE7733", "size": 48
            },
        }
        for row, ratio, rate in zip(
            external_rows, external_ratios, spec["external_values"]
        ):
            style = external_styles[row["codec"]]
            ax.scatter(
                [ratio],
                [rate],
                s=style["size"],
                marker=style["marker"],
                facecolors=style["color"],
                edgecolors="white",
                linewidths=0.8,
                label=style["label"],
                zorder=4,
            )
        traditional_artists.append(ax.scatter(
            baseline_ratios,
            spec["baseline_values"],
            s=30,
            marker=spec["baseline_marker"],
            facecolors="white",
            edgecolors=TRADITIONAL_COLOR,
            linewidths=1.0,
            label="Traditional CPU",
            zorder=3,
        ))
        for row, ratio, rate in zip(
            baselines["codecs"], baseline_ratios, spec["baseline_values"]
        ):
            offset = spec["label_offsets"][row["id"]]
            traditional_artists.append(ax.annotate(
                short_labels[row["id"]],
                xy=(ratio, rate),
                xytext=offset,
                textcoords="offset points",
                ha="left" if offset[0] > 0 else "right",
                va="bottom" if offset[1] > 0 else "top",
                fontsize=6.7,
                color="#3C4043",
            ))
        for row, ratio, rate in zip(
            external_rows, external_ratios, spec["external_values"]
        ):
            offsets = {
                "finezip": (-3, -5),
                "llmzip": (-3, 5),
                "nacrith": (3, 3),
            }
            offset = offsets[row["codec"]]
            external_annotations[(panel_index, row["codec"])] = ax.annotate(
                external_styles[row["codec"]]["label"],
                xy=(ratio, rate),
                xytext=offset,
                textcoords="offset points",
                ha="right" if offset[0] < 0 else "left",
                va="top" if offset[1] < 0 else "bottom",
                fontsize=6.7,
                color="#3C4043",
            )
        threshold_artists.append(ax.axvline(
            r_star,
            color="#B06000",
            linewidth=1.0,
            linestyle="--",
            label=f"$R^*={r_star:.3f}\\times$",
            zorder=2,
        ))
        threshold_artists.append(
            ax.axvspan(r_star, X_MAX, color="#FFF4E5", alpha=0.35, zorder=0)
        )
        for index, row in enumerate(rows):
            batch = row["batch_size"]
            label = {
                1024: "B1K",
                8192: "B8K",
                16384: "B16K",
            }[batch]
            offset = spec["llm_label_offsets"][batch]
            llm_annotations[(panel_index, batch)] = ax.annotate(
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
        ax.set_xticks(range(0, 9), [f"{value}×" for value in range(0, 9)])
        ax.set_yscale("log")
        ax.set_ylim(*Y_LIMITS)
        ax.yaxis.set_major_locator(FixedLocator(Y_TICKS))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.grid(color="#E8EAED", linewidth=0.65, which="major")
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        direction_artists.append(ax.annotate(
            "Better",
            xy=(0.97, 0.95),
            xytext=(0.82, 0.80),
            xycoords="axes fraction",
            textcoords="axes fraction",
            ha="center",
            va="center",
            rotation=45,
            transform_rotates_text=True,
            fontsize=7.2,
            fontweight="bold",
            color="#37474F",
            arrowprops={
                "arrowstyle": "-|>",
                "color": "#37474F",
                "linewidth": 1.0,
                "mutation_scale": 8,
            },
        ))

    handles, legend_labels = axes[0].get_legend_handles_labels()
    full_legend = fig.legend(
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
        "Subject": "CUDA-v3 encoding and matched decoding on 10 MiB text8",
        "CreationDate": datetime(2026, 10, 14, tzinfo=timezone.utc),
        "ModDate": datetime(2026, 10, 14, tzinfo=timezone.utc),
    }
    args.pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.pdf, bbox_inches="tight", pad_inches=0.2, metadata=metadata)
    if args.preview_png is not None:
        args.preview_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.preview_png, bbox_inches="tight", pad_inches=0.2, dpi=200)

    zoom_fig, zoom_axes = plt.subplots(1, 2, figsize=(7.4, 3.0), sharex=True)
    for zoom_ax, spec in zip(zoom_axes, panel_specs):
        zoom_ax.plot(
            ratios,
            spec["llm_values"],
            color=spec["color"],
            linewidth=1.2,
            alpha=0.75,
            zorder=2,
        )
        zoom_ax.scatter(
            ratios,
            spec["llm_values"],
            s=58,
            facecolors=spec["color"],
            marker=spec["llm_marker"],
            edgecolors="white",
            linewidths=1.0,
            label=spec["llm_label"],
            zorder=5,
        )
        for row, ratio, rate in zip(
            external_rows, external_ratios, spec["external_values"]
        ):
            style = external_styles[row["codec"]]
            zoom_ax.scatter(
                [ratio],
                [rate],
                s=style["size"],
                marker=style["marker"],
                facecolors=style["color"],
                edgecolors="white",
                linewidths=0.8,
                label=style["label"],
                zorder=4,
            )
            zoom_ax.annotate(
                style["label"],
                xy=(ratio, rate),
                xytext={
                    "finezip": (0, -10),
                    "llmzip": (-3, 5),
                    "nacrith": (3, 3),
                }[row["codec"]],
                textcoords="offset points",
                ha="center" if row["codec"] == "finezip" else (
                    "right" if row["codec"] == "llmzip" else "left"
                ),
                va="top" if row["codec"] == "finezip" else "bottom",
                fontsize=6.7,
                color="#3C4043",
            )
        for index, row in enumerate(rows):
            batch = row["batch_size"]
            offset = {
                1024: (-5, 10),
                8192: (3, 10),
                16384: (6, 10),
            }[batch]
            zoom_ax.annotate(
                {
                    1024: "B1K · C512",
                    8192: "B8K · C128",
                    16384: "B16K · C128",
                }[batch],
                xy=(ratios[index], spec["llm_values"][index]),
                xytext=offset,
                textcoords="offset points",
                ha="right" if offset[0] < 0 else "left",
                va="bottom",
                fontsize=6.5,
                color="#202124",
            )
        zoom_ax.set_title(spec["title"], loc="center", pad=8)
        zoom_ax.set_ylabel(spec["ylabel"])
        zoom_ax.set_xlim(5.2, 8.35)
        zoom_ax.set_xticks(
            (5.5, 6.0, 6.5, 7.0, 7.5, 8.0),
            ("5.5×", "6×", "6.5×", "7×", "7.5×", "8×"),
        )
        zoom_ax.set_yscale("log")
        zoom_ax.set_ylim(0.0002, 1.2)
        zoom_ax.yaxis.set_major_locator(FixedLocator((0.001, 0.01, 0.1, 1.0)))
        zoom_ax.yaxis.set_minor_locator(NullLocator())
        zoom_ax.grid(color="#E8EAED", linewidth=0.65, which="major")
        zoom_ax.set_axisbelow(True)
        zoom_ax.spines[["top", "right"]].set_visible(False)
    zoom_handles, zoom_labels = zoom_axes[0].get_legend_handles_labels()
    unique_zoom_legend = dict(zip(zoom_labels, zoom_handles))
    zoom_fig.legend(
        unique_zoom_legend.values(),
        unique_zoom_legend.keys(),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.06),
        ncol=4,
        frameon=False,
        fontsize=6.4,
    )
    zoom_fig.supxlabel(
        "Compression ratio",
        y=0.17,
        fontsize=plt.rcParams["axes.labelsize"],
        color=plt.rcParams["axes.labelcolor"],
    )
    zoom_fig.tight_layout(rect=(0.0, 0.16, 1.0, 1.0), pad=0.8, w_pad=1.2)
    args.zoomed_pdf.parent.mkdir(parents=True, exist_ok=True)
    zoom_fig.savefig(
        args.zoomed_pdf,
        bbox_inches="tight",
        pad_inches=0.2,
        metadata={
            **metadata,
            "Title": "LLM-only compression-rate and throughput operating points",
            "Subject": "Zoomed matched 10 MiB text8 LLM round-trip experiments",
        },
    )
    if args.zoomed_preview_png is not None:
        args.zoomed_preview_png.parent.mkdir(parents=True, exist_ok=True)
        zoom_fig.savefig(
            args.zoomed_preview_png,
            bbox_inches="tight",
            pad_inches=0.2,
            dpi=200,
        )
    plt.close(zoom_fig)
    plt.close(fig)

    print(f"Saved {args.pdf}")
    print(f"Saved {args.zoomed_pdf}")


if __name__ == "__main__":
    main()
