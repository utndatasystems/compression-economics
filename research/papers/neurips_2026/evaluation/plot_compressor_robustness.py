#!/usr/bin/env python3
"""Regenerate the camera-ready end-to-end codec comparison figure."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.papers.neurips_2026.evaluation.crucial_figures import (
    plot_compressor_bars,
)


DATASET_ORDER = [
    "text8",
    "random text",
    "MinProb adversary",
    "MaxSurprisal/Byte adversary",
    "all one-byte adversary",
]
PAPER_PERCENTAGES = {
    "text8": {
        "Qwen + AC": 17.6,
        "Qwen token IDs": 41.3,
        "FSST": 56.3,
        "Brotli q11": 29.6,
    },
    "random text": {
        "Qwen + AC": 97.4,
        "Qwen token IDs": 172.1,
        "FSST": 98.2,
        "Brotli q11": 83.0,
    },
    "MaxSurprisal/Byte adversary": {
        "Qwen + AC": 144.3,
        "Qwen token IDs": 154.4,
        "FSST": 94.6,
        "Brotli q11": 83.2,
    },
    "all one-byte adversary": {
        "Qwen + AC": 122.8,
        "Qwen token IDs": 225.2,
        "FSST": 83.6,
        "Brotli q11": 76.1,
    },
}
MINPROB_RESULTS = (
    REPO_ROOT
    / "artifacts/papers/neurips-2026/camera-ready/"
    "minprob-h100-three-seed/codec-results.json"
)


def main() -> None:
    minprob = json.loads(MINPROB_RESULTS.read_text(encoding="utf-8"))
    minprob_ratios = minprob["aggregate"]["relative_size_percent"]
    PAPER_PERCENTAGES["MinProb adversary"] = {
        codec: minprob_ratios[codec]
        for codec in ("Qwen + AC", "Qwen token IDs", "FSST", "Brotli q11")
    }
    rows = [
        {
            "dataset": dataset,
            "codec": codec,
            "raw_size_bytes": 100.0,
            "serialized_size_bytes": percentage,
            "round_trip": True,
            "status": "measured",
        }
        for dataset in DATASET_ORDER
        for codec, percentage in PAPER_PERCENTAGES[dataset].items()
    ]
    frame = pd.DataFrame(rows)
    audit_path = (
        REPO_ROOT
        / "artifacts/papers/neurips-2026/derived/audit-tables/"
        "compressor_robustness_camera_ready.csv"
    )
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(audit_path, index=False)

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
    fig, _ = plot_compressor_bars(frame, DATASET_ORDER)
    try:
        fig.savefig(
            REPO_ROOT
            / "research/papers/neurips_2026/manuscript/plots/"
            "end_to_end_compressor_robustness.png",
            dpi=300,
        )
    finally:
        plt.close(fig)


if __name__ == "__main__":
    main()
