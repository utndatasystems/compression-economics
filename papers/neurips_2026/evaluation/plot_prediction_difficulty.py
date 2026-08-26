#!/usr/bin/env python3
"""Generate the appendix prediction-difficulty figure from paper-table values."""

from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

from papers.neurips_2026.evaluation.crucial_figures import (
    find_repo_root,
    plot_paper_prediction_difficulty,
)


TEXT8_BITS_PER_TOKEN = 5.11
PAPER_VALUES = pd.DataFrame(
    [
        {
            "condition": "MinProb",
            "prediction_bits_per_token": 29.04,
            "std_bits_per_token": 0.04,
            "single_run": False,
            "tokens": 10_000,
            "runs": 3,
        },
        {
            "condition": "MaxSurprisal/Byte",
            "prediction_bits_per_token": 15.52,
            "std_bits_per_token": pd.NA,
            "single_run": True,
            "tokens": 1_024,
            "runs": 1,
        },
        {
            "condition": "One-byte UTF-8 ablation",
            "prediction_bits_per_token": 8.33,
            "std_bits_per_token": 0.00,
            "single_run": False,
            "tokens": 10_000,
            "runs": 3,
        },
    ]
)


def main() -> None:
    root = find_repo_root(Path(__file__))
    output_dir = root / "papers/neurips_2026/manuscript/plots"
    artifact_dir = root / "artifacts/papers/neurips-2026/derived/figure-data"
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    table = PAPER_VALUES.copy()
    table["text8_bits_per_token"] = TEXT8_BITS_PER_TOKEN
    table["relative_to_text8"] = (
        table["prediction_bits_per_token"] / TEXT8_BITS_PER_TOKEN
    )
    table.to_csv(artifact_dir / "paper_prediction_difficulty.csv", index=False)

    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "font.family": "DejaVu Sans",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
        }
    )
    fig, _ = plot_paper_prediction_difficulty(
        PAPER_VALUES,
        text8_bits_per_token=TEXT8_BITS_PER_TOKEN,
    )
    try:
        fig.savefig(output_dir / "paper_prediction_difficulty.pdf")
    finally:
        plt.close(fig)


if __name__ == "__main__":
    main()
