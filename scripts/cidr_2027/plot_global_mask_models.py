#!/usr/bin/env python3
"""Plot payload compression versus total description length for CIDR models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


MODEL_COLORS = {
    "token-bigram": "#0072B2",
    "token-trigram": "#56B4E9",
    "token-nplm-w8": "#D55E00",
    "token-nplm-w32": "#E69F00",
    "token-nplm-w128": "#CC79A7",
    "token-tiny-transformer-w128": "#7B61A8",
    "token-tiny-gru-w128": "#009E73",
}
MODE_MARKERS = {"disjoint": "o", "same_data": "s"}


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, action="append")
    parser.add_argument("--results-dir", type=Path, action="append")
    parser.add_argument(
        "--output", type=Path,
        default=Path("artifacts/papers/cidr-2027/model-survey/description-length-tradeoff.png"),
    )
    return parser.parse_args()


def load_rows(paths: list[Path]) -> list[dict]:
    rows = []
    required = {
        "model", "training_mode", "source_bytes", "payload_bytes",
        "stored_stream_bytes", "model_state_bytes", "roundtrip_valid",
        "accounting_mode",
    }
    for path in paths:
        payload = json.loads(path.read_text())
        for row in payload["results"]:
            missing = required - row.keys()
            if missing:
                raise ValueError(f"{path} row missing {sorted(missing)}")
            if row["accounting_mode"] != "global_bitmap_framed" or not row["roundtrip_valid"]:
                raise ValueError(f"{path} contains an invalid global-mask measurement")
            rows.append(dict(row))
    if not rows:
        raise ValueError("no normalized result rows found")
    if len({row["source_bytes"] for row in rows}) != 1:
        raise ValueError("all plotted rows must have identical source_bytes")
    return rows


def label_offset(index: int) -> tuple[int, int]:
    offsets = ((8, 10), (8, -14), (-75, 10), (-75, -14), (12, 24), (-80, 24))
    return offsets[index % len(offsets)]


def main() -> None:
    options = args()
    paths = list(options.results or [])
    for directory in options.results_dir or []:
        if not directory.is_dir():
            raise FileNotFoundError(f"results directory not found: {directory}")
        paths.extend(sorted(directory.rglob("results.json")))
    paths = list(dict.fromkeys(paths))
    if not paths:
        raise SystemExit("Provide --results or --results-dir")
    print(f"Combining {len(paths)} result files", flush=True)
    rows = load_rows(paths)

    for row in rows:
        source = row["source_bytes"]
        row["payload_only_compression_factor"] = source / row["payload_bytes"]
        row["total_description_length_ratio"] = (
            row["stored_stream_bytes"] + row["model_state_bytes"]
        ) / source

    fig, axis = plt.subplots(figsize=(8.0, 5.5))
    sorted_rows = sorted(
        rows,
        key=lambda row: (
            row["payload_only_compression_factor"],
            row["total_description_length_ratio"],
            row["model"],
            row["training_mode"],
        ),
    )
    for index, row in enumerate(sorted_rows):
        color = MODEL_COLORS.get(row["model"], "#6B7280")
        mode = row["training_mode"]
        x, y = row["payload_only_compression_factor"], row["total_description_length_ratio"]
        axis.scatter(
            x, y, color=color, marker=MODE_MARKERS[mode], s=88,
            edgecolor="white", linewidth=0.9, zorder=3,
        )
        axis.annotate(
            f'{row["model"]} ({mode.replace("_", " ")})',
            (x, y), xytext=label_offset(index), textcoords="offset points",
            fontsize=7, arrowprops={"arrowstyle": "-", "color": "#9CA3AF", "lw": 0.55},
            zorder=4,
        )

    x_min, x_max = axis.get_xlim()
    diagonal_x = np.geomspace(max(x_min, 1e-6), x_max, 100)
    axis.plot(
        diagonal_x, 1 / diagonal_x, color="#6B7280", linestyle=(0, (3, 2)),
        linewidth=0.85, label="Payload-only limit", zorder=1,
    )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set(
        xlabel="Payload-only compression factor (raw / arithmetic payload; higher is better)",
        ylabel="Total description length ratio ((payload + bitmap + framing + model) / raw; lower is better)",
        title=f'text8: {rows[0]["source_bytes"]:,} bytes; global bitmap and main.py framing',
    )
    axis.grid(color="#E5E7EB", linewidth=0.6, zorder=0, which="both")
    axis.spines[["top", "right"]].set_visible(False)
    mode_handles = [
        Line2D([], [], marker=MODE_MARKERS[mode], linestyle="None", color="#4B5563",
               markerfacecolor="#4B5563", markersize=7,
               label={"disjoint": "Disjoint training", "same_data": "Same-data training"}[mode])
        for mode in ("disjoint", "same_data")
    ]
    axis.legend(
        handles=[*mode_handles, axis.lines[0]], frameon=False, fontsize=8,
        ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.01),
    )
    fig.subplots_adjust(left=0.14, right=0.98, top=0.84, bottom=0.16)
    options.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(options.output, dpi=180)
    print(f"Wrote {options.output}")


if __name__ == "__main__":
    main()
