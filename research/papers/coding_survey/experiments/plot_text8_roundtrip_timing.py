#!/usr/bin/env python3
"""Plot matched text8 encoding and decoding time as horizontal stacked bars."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_DATA = (
    REPO_ROOT
    / "artifacts/papers/coding-survey/cuda-ac/"
    "text8-first-10MiB-roundtrip-b256-c256-r100.json"
)
DEFAULT_PDF = (
    REPO_ROOT
    / "research/papers/coding_survey/manuscript/figures/"
    "text8_roundtrip_timing.pdf"
)
DEFAULT_PNG = DEFAULT_PDF.with_suffix(".png")

LABELS = {
    "python_full_logits": "Python full logits",
    "python": "Python target intervals",
    "cuda_v1": "CUDA v1",
    "cuda_v2": "CUDA v2",
    "cuda_v3": "CUDA v3",
    "host": "Python host",
    "device": "PyTorch device",
    "cuda": "CUDA-CDF",
}
COMPONENTS = (
    ("model", "Model inference", "#4477AA", ""),
    ("interval", "Interval construction", "#66CCEE", "//"),
    ("arithmetic", "Arithmetic coder/decoder", "#228833", ""),
    ("detokenize", "Detokenization", "#CCBB44", "\\\\"),
    ("other", "Other pipeline work", "#BBBBBB", ".."),
)


def _load(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not data["encoding"]["exact_archive_match"]:
        raise ValueError("encoding archives are not identical")
    if not data["decoding"]["exact_recovery"]:
        raise ValueError("a decoder failed exact recovery")
    encoder_conditions = data["encoding"]["conditions"]
    expected_v2 = ["python", "cuda_v1", "cuda_v2"]
    expected_v3 = [*expected_v2, "cuda_v3"]
    accepted = (
        expected_v2,
        expected_v3,
        ["python_full_logits", *expected_v2],
        ["python_full_logits", *expected_v3],
    )
    if encoder_conditions not in accepted:
        raise ValueError(
            "expected Python target intervals, CUDA v1/v2, optional CUDA v3, "
            "and optional Python full logits"
        )
    if data["decoding"]["conditions"] != ["host", "device", "cuda"]:
        raise ValueError("expected host, device, and CUDA decoders")
    return data


def _encoding_parts(row: dict) -> dict[str, float]:
    # Interval construction is measured inside the inference scope, so subtract
    # it before stacking to avoid double counting nested timers.
    model = row["inference_seconds"] - row["interval_seconds"]
    arithmetic = row["arithmetic_coding_seconds"]
    other = row["seconds"] - row["inference_seconds"] - arithmetic
    parts = {
        "model": model,
        "interval": row["interval_seconds"],
        "arithmetic": arithmetic,
        "detokenize": 0.0,
        "other": other,
    }
    return _validated_parts(parts, row["seconds"])


def _decoding_parts(row: dict) -> dict[str, float]:
    parts = {
        "model": row["inference_seconds"],
        "interval": 0.0,
        "arithmetic": row["arithmetic_seconds"],
        "detokenize": row["detokenize_seconds"],
        "other": row["seconds"]
        - row["inference_seconds"]
        - row["arithmetic_seconds"]
        - row["detokenize_seconds"],
    }
    return _validated_parts(parts, row["seconds"])


def _validated_parts(parts: dict[str, float], total: float) -> dict[str, float]:
    if any(value < -1e-6 for value in parts.values()):
        raise ValueError(f"negative non-overlapping timing component: {parts}")
    parts = {key: max(0.0, value) for key, value in parts.items()}
    if not np.isclose(sum(parts.values()), total, atol=1e-6):
        raise ValueError("timing components do not sum to total time")
    return parts


def _panel(
    ax,
    data: dict,
    section: str,
    conditions: list[str],
    part_builder,
    title: str,
) -> None:
    medians = data[section]["medians"]
    samples = data[section]["samples"]
    positions = np.arange(len(conditions))[::-1]
    left = np.zeros(len(conditions), dtype=np.float64)

    for key, label, color, hatch in COMPONENTS:
        values = np.asarray(
            [part_builder(medians[condition])[key] for condition in conditions]
        )
        if np.allclose(values, 0.0):
            continue
        ax.barh(
            positions,
            values,
            left=left,
            height=0.58,
            color=color,
            edgecolor="white",
            linewidth=0.45,
            hatch=hatch,
            label=label,
            zorder=2,
        )
        left += values

    totals = np.asarray([medians[condition]["seconds"] for condition in conditions])
    minima = np.asarray(
        [min(sample["seconds"] for sample in samples[condition]) for condition in conditions]
    )
    maxima = np.asarray(
        [max(sample["seconds"] for sample in samples[condition]) for condition in conditions]
    )
    ax.errorbar(
        totals,
        positions,
        xerr=np.vstack((totals - minima, maxima - totals)),
        fmt="none",
        ecolor="#202124",
        elinewidth=0.8,
        capsize=2.1,
        capthick=0.8,
        zorder=4,
    )

    source_mib = data["input_prefix_bytes"] / (1024 * 1024)
    x_limit = float(maxima.max() * 1.21)
    for position, condition, total in zip(positions, conditions, totals, strict=True):
        throughput = source_mib / total
        ax.text(
            total + x_limit * 0.015,
            position,
            f"{total:.1f} s\n{throughput:.3f} MiB/s",
            va="center",
            ha="left",
            fontsize=7.0,
            color="#202124",
            linespacing=1.15,
        )

    ax.set_xlim(0.0, x_limit)
    ax.set_yticks(positions, [LABELS[condition] for condition in conditions])
    ax.set_xlabel("Elapsed time (s; lower is better)")
    ax.set_title(title, loc="left", pad=7)
    ax.grid(axis="x", color="#E8EAED", linewidth=0.65)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", length=3, color="#BDC1C6")


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
            "ytick.color": "#202124",
            "axes.edgecolor": "#DADCE0",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.15))
    _panel(
        axes[0],
        data,
        "encoding",
        data["encoding"]["conditions"],
        _encoding_parts,
        "(a) Encoding",
    )
    _panel(
        axes[1],
        data,
        "decoding",
        data["decoding"]["conditions"],
        _decoding_parts,
        "(b) Decoding",
    )

    handles_by_label = {}
    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        handles_by_label.update(zip(labels, handles, strict=True))
    legend_order = [label for _, label, _, _ in COMPONENTS if label in handles_by_label]
    fig.legend(
        [handles_by_label[label] for label in legend_order],
        legend_order,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.005),
        ncol=5,
        frameon=False,
        fontsize=7.1,
        handlelength=1.5,
        columnspacing=1.2,
    )
    fig.suptitle(
        "text8 (10 MiB), Qwen2.5-0.5B, batch 256, context 256/retain 100, "
        f"MSAC-v2 ratio {data['representation']['compression_ratio']:.3f}×; median of "
        f"{data['repeats']} runs",
        x=0.075,
        y=1.01,
        ha="left",
        fontsize=8.2,
        fontweight="normal",
        color="#5F6368",
    )
    fig.tight_layout(rect=(0.0, 0.12, 1.0, 0.96), w_pad=2.0)

    metadata = {
        "Title": "Matched text8 round-trip timing decomposition",
        "Author": "compression-economics experiment pipeline",
        "Subject": "Python full logits, Python target intervals, and CUDA v1/v2/v3 comparison",
        "CreationDate": datetime(2026, 10, 3, tzinfo=timezone.utc),
        "ModDate": datetime(2026, 10, 3, tzinfo=timezone.utc),
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


if __name__ == "__main__":
    main()
