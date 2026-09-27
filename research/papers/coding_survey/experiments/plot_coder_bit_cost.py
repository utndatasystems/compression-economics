#!/usr/bin/env python3
"""Plot matched coder archive sizes against their relevant entropy references."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.coding.trace_benchmark import load_probability_trace, ranks_for_trace


LABELS = {
    "AC": "AC",
    "AC_MULTISTREAM": "Multistream AC",
    "ANS": "rANS",
    "PMATIC": "PMATIC",
    "HUFFMAN_RANK": "Huffman rank",
    "BITPACKED_RANK": "Bit-packed rank",
}
PROBABILITY_CODERS = ("AC", "AC_MULTISTREAM", "ANS", "PMATIC")
RANK_CODERS = ("HUFFMAN_RANK", "BITPACKED_RANK")


def empirical_rank_entropy(trace) -> float:
    counts = np.bincount(ranks_for_trace(trace), minlength=trace.alphabet_size)
    probabilities = counts[counts > 0] / trace.symbol_count
    return float(-np.sum(probabilities * np.log2(probabilities)))


def mean_source_entropy(trace) -> float:
    probabilities = trace.probabilities
    return float(-np.mean(np.sum(
        probabilities * np.log2(np.maximum(probabilities, 1e-300)), axis=1
    )))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, help="results.json from evaluate_coders.py")
    parser.add_argument("trace", type=Path, help="NPZ trace used for those results")
    parser.add_argument("output", type=Path, help="Output PDF path")
    parser.add_argument(
        "--synthetic-source", action="store_true",
        help="The trace probabilities generated its symbols; show their expected entropy bound.",
    )
    args = parser.parse_args()

    data = json.loads(args.results.read_text(encoding="utf-8"))
    trace = load_probability_trace(args.trace)
    if data["trace_sha256"] != trace.sha256:
        raise ValueError("result and trace hashes differ")
    rows = {row["coder"]: row for row in data["results"]}
    if set(rows) != set(LABELS):
        raise ValueError(f"expected exactly {sorted(LABELS)}, got {sorted(rows)}")
    if any(row["trace_sha256"] != trace.sha256 or
           row["symbol_count"] != trace.symbol_count or
           not row["exact_roundtrip_valid"] for row in rows.values()):
        raise ValueError("all rows must refer to this trace and round-trip exactly")

    model_reference = rows["AC"]["quantized_distribution_cross_entropy_bits"] / trace.symbol_count
    for coder in PROBABILITY_CODERS[:3]:
        candidate = rows[coder]["quantized_distribution_cross_entropy_bits"] / trace.symbol_count
        if not math.isclose(candidate, model_reference, rel_tol=0, abs_tol=1e-10):
            raise ValueError("AC, multistream AC, and rANS have different quantized references")
    rank_reference = empirical_rank_entropy(trace)
    source_bound = mean_source_entropy(trace) if args.synthetic_source else None

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "text.color": "#202124",
        "axes.labelcolor": "#202124",
        "xtick.color": "#5f6368",
        "ytick.color": "#202124",
        "pdf.fonttype": 42,
    })
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.05),
                             gridspec_kw={"width_ratios": [1.16, 1]})
    for ax, family, title, reference, reference_name in zip(
        axes,
        (PROBABILITY_CODERS, RANK_CODERS),
        ("(a) Token-probability coders", "(b) Rank coders"),
        (model_reference, rank_reference),
        ("Quantized model log-loss\nAC/rANS target",
         "Empirical rank entropy\nGlobal rank-code target"),
    ):
        ordered = sorted(family, key=lambda coder: rows[coder]["archive_bytes"])
        positions = np.arange(len(ordered))
        payload = np.array([rows[coder]["payload_bits"] / trace.symbol_count
                            for coder in ordered])
        overhead = np.array([
            (8 * rows[coder]["archive_bytes"] - rows[coder]["payload_bits"])
            / trace.symbol_count for coder in ordered
        ])
        totals = payload + overhead
        ax.barh(positions, payload, height=0.63, color="#1a73e8")
        ax.barh(positions, overhead, left=payload, height=0.63,
                color="#9aa0a6")
        ax.axvline(reference, color="#d56b00", linestyle=(0, (5, 2)),
                   linewidth=1.1, zorder=4)
        for y, total in zip(positions, totals):
            ax.text(total + max(totals) * 0.015, y, f"{total:.2f}",
                    va="center", ha="left", fontsize=7.5)
        ax.set_yticks(positions, [LABELS[coder] for coder in ordered])
        ax.invert_yaxis()
        if "BITPACKED_RANK" in ordered:
            ax.set_ylim(2.05, -0.5)  # Room for the bit-packed payload callout.
        ax.set_xlim(0, max(totals) * 1.18)
        ax.set_title(title, loc="left", fontweight="bold", pad=30)
        ax.text(0, 1.025, f"{reference_name}: {reference:.3f} bits/token",
                transform=ax.transAxes, va="bottom", fontsize=7,
                color="#9a5200")
        ax.set_xlabel("Archive bits per token")
        ax.grid(axis="x", color="#e8eaed", linewidth=0.55)
        ax.set_axisbelow(True)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0, pad=5)

        arrow_style = dict(arrowstyle="-|>", color="#174ea6", lw=0.9,
                           mutation_scale=8, shrinkA=2, shrinkB=2)
        if "PMATIC" in ordered:
            y = ordered.index("PMATIC")
            ax.annotate(
                "Helper bits + robust bins",
                xy=(reference + 0.55 * (payload[y] - reference), y),
                xytext=(reference + 2.15, y - 0.82),
                fontsize=7, color="#174ea6", ha="left", va="center",
                arrowprops=arrow_style,
            )
        if "BITPACKED_RANK" in ordered:
            y = ordered.index("BITPACKED_RANK")
            ax.annotate(
                "Fixed six-bit ranks\nignore rank skew",
                xy=(reference + 0.55 * (payload[y] - reference), y),
                xytext=(0.8, y + 0.76),
                fontsize=7, color="#174ea6", ha="left", va="center",
                arrowprops=arrow_style,
            )

    legend_handles = [
        Patch(facecolor="#1a73e8", label="Coder bitstream"),
        Patch(facecolor="#9aa0a6", label="Archive extras (headers, codebook, padding)"),
        Line2D([0], [0], color="#d56b00", linestyle=(0, (5, 2)),
               label="Coder target"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3,
               frameon=False, fontsize=7, bbox_to_anchor=(0.5, 0.01))
    fig.tight_layout(rect=(0, 0.18, 1, 0.87), w_pad=2.2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {args.output}")
    print(f"Quantized model log-loss: {model_reference:.3f} bits/token")
    print(f"Empirical rank entropy: {rank_reference:.3f} bits/token")
    if source_bound is not None:
        print(f"Expected Shannon lower bound: {source_bound:.3f} bits/token")


if __name__ == "__main__":
    main()
