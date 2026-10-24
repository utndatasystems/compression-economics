#!/usr/bin/env python3
"""Plot CUDA-v3 text8 throughput and compression-ratio sweep heatmaps."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import seaborn as sns


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_INPUT_DIR = (
    REPO_ROOT
    / "artifacts/papers/coding-survey/cuda-ac/text8-v3-heatmap-lower-triangle"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "research/papers/coding_survey/manuscript/figures"
DEFAULT_SPEED_PDF = DEFAULT_OUTPUT_DIR / "text8_v3_compression_speed_heatmap.pdf"
DEFAULT_RATIO_PDF = DEFAULT_OUTPUT_DIR / "text8_v3_compression_ratio_heatmap.pdf"
EXPECTED_INPUT_BYTES = 10 * 1024 * 1024
EXPECTED_ENCODER = "cuda_v3"


def _load_results(input_dir: Path) -> list[dict]:
    paths = sorted(input_dir.glob("*.json"))
    if not paths:
        raise FileNotFoundError(f"no JSON results found in {input_dir}")

    rows = []
    source_hashes = set()
    models = set()
    coordinates = set()
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data["input_prefix_bytes"] != EXPECTED_INPUT_BYTES:
            raise ValueError(f"{path} is not a complete 10 MiB run")
        if data["repeats"] < 3:
            raise ValueError(f"{path} contains fewer than three timed runs")
        if data["encoding"]["conditions"] != [EXPECTED_ENCODER]:
            raise ValueError(f"{path} is not a CUDA-v3-only result")
        if not data["encoding"]["exact_archive_match"]:
            raise ValueError(f"{path} failed archive validation")

        coordinate = (data["context_length"], data["batch_size"])
        if coordinate in coordinates:
            raise ValueError(f"duplicate context/batch coordinate: {coordinate}")
        coordinates.add(coordinate)
        source_hashes.add(data["input_sha256"])
        models.add((data["model"], data["model_revision"]))
        rows.append(data)

    if len(source_hashes) != 1:
        raise ValueError("sweep results use different source bytes")
    if len(models) != 1:
        raise ValueError("sweep results use different model revisions")
    return rows


def _matrix(
    rows: list[dict],
    contexts: list[int],
    batches: list[int],
    value,
) -> np.ndarray:
    values = np.full((len(contexts), len(batches)), np.nan, dtype=np.float64)
    context_index = {context: index for index, context in enumerate(contexts)}
    batch_index = {batch: index for index, batch in enumerate(batches)}
    for row in rows:
        values[
            context_index[row["context_length"]], batch_index[row["batch_size"]]
        ] = value(row)
    return values


def _annotation_color(cmap, norm: Normalize, value: float) -> str:
    """Choose dark or light text from the rendered cell luminance."""
    red, green, blue, _ = cmap(norm(value))
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    return "#202020" if luminance >= 0.58 else "white"


def _plot_heatmap(
    values: np.ndarray,
    contexts: list[int],
    batches: list[int],
    *,
    title: str,
    colorbar_label: str,
    annotation_format: str,
    output_pdf: Path,
    output_png: Path,
    dpi: int,
) -> None:
    valid = values[np.isfinite(values)]
    if valid.size == 0:
        raise ValueError("heatmap contains no measured cells")

    # Match the compact heatmap styling of Kipf et al., CIDR 2026, Fig. 3.
    cmap = sns.color_palette("rocket", as_cmap=True).copy()
    cmap.set_bad("#E6E6E6")
    norm = Normalize(vmin=float(valid.min()), vmax=float(valid.max()))

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "font.size": 8,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.titleweight": "normal",
            "text.color": "#202020",
            "axes.labelcolor": "#202020",
            "xtick.color": "#202020",
            "ytick.color": "#202020",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(3.35, 2.2))
    image = ax.imshow(values, cmap=cmap, norm=norm, aspect="auto")

    for row_index in range(values.shape[0]):
        for column_index in range(values.shape[1]):
            measured = values[row_index, column_index]
            if np.isfinite(measured):
                label = format(measured, annotation_format)
                color = _annotation_color(cmap, norm, float(measured))
                weight = "normal"
            else:
                label = "—"
                color = "#777777"
                weight = "normal"
            ax.text(
                column_index,
                row_index,
                label,
                ha="center",
                va="center",
                color=color,
                fontsize=8.4,
                fontweight=weight,
            )

    batch_labels = [
        f"{batch // 1024}K" if batch >= 1024 else str(batch) for batch in batches
    ]
    ax.set_xticks(np.arange(len(batches)), batch_labels)
    ax.set_yticks(np.arange(len(contexts)), [str(context) for context in contexts])
    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Context Size")
    ax.set_title(title, loc="center", pad=5)
    ax.tick_params(which="major", length=0)
    ax.set_xticks(np.arange(-0.5, len(batches), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(contexts), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.0)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.spines[:].set_visible(False)
    fig.subplots_adjust(left=0.16, right=0.84, bottom=0.22, top=0.86)

    # Resolve the equal-aspect axes first, then place the color bar from that
    # final figure-space rectangle. This keeps it outside the grid and makes
    # its top and bottom exactly flush with the heatmap in both PNG and PDF.
    fig.canvas.draw()
    heatmap_box = ax.get_position()
    colorbar_ax = fig.add_axes(
        [
            heatmap_box.x1 + heatmap_box.width * 0.025,
            heatmap_box.y0,
            heatmap_box.width * 0.035,
            heatmap_box.height,
        ]
    )
    colorbar = fig.colorbar(image, cax=colorbar_ax)
    if colorbar_label:
        colorbar.set_label(colorbar_label)
    colorbar.outline.set_visible(False)
    colorbar.ax.tick_params(length=2.5, width=0.5, color="#202020", labelsize=7)

    metadata = {
        "Title": title,
        "Author": "compression-economics experiment pipeline",
        "Subject": "CUDA-v3 batch-size and context-length sweep on text8",
        "CreationDate": datetime(2026, 10, 14, tzinfo=timezone.utc),
        "ModDate": datetime(2026, 10, 14, tzinfo=timezone.utc),
    }
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf, metadata=metadata)
    fig.savefig(
        output_png,
        dpi=dpi,
        metadata={"Software": metadata["Author"]},
    )
    plt.close(fig)
    print(f"Saved {output_pdf}")
    print(f"Saved {output_png} at {dpi} DPI")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--speed-pdf", type=Path, default=DEFAULT_SPEED_PDF)
    parser.add_argument("--ratio-pdf", type=Path, default=DEFAULT_RATIO_PDF)
    parser.add_argument("--speed-png", type=Path)
    parser.add_argument("--ratio-png", type=Path)
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()
    if args.dpi < 300:
        parser.error("--dpi must be at least 300 for a paper figure")

    speed_png = args.speed_png or args.speed_pdf.with_suffix(".png")
    ratio_png = args.ratio_png or args.ratio_pdf.with_suffix(".png")
    rows = _load_results(args.input_dir)
    batches = sorted({row["batch_size"] for row in rows})
    # imshow places the first row at the top, so descending storage order makes
    # the plotted y-axis increase conventionally from bottom to top.
    contexts = sorted({row["context_length"] for row in rows}, reverse=True)

    speed = _matrix(
        rows,
        contexts,
        batches,
        lambda row: 1024
        * row["encoding"]["medians"][EXPECTED_ENCODER]["source_mib_per_second"],
    )
    ratio = _matrix(
        rows,
        contexts,
        batches,
        lambda row: row["representation"]["compression_ratio"],
    )

    _plot_heatmap(
        speed,
        contexts,
        batches,
        title="Throughput [KiB/s]",
        colorbar_label="",
        annotation_format=".0f",
        output_pdf=args.speed_pdf,
        output_png=speed_png,
        dpi=args.dpi,
    )
    _plot_heatmap(
        ratio,
        contexts,
        batches,
        title="Compression Ratio",
        colorbar_label="",
        annotation_format=".2f",
        output_pdf=args.ratio_pdf,
        output_png=ratio_png,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
