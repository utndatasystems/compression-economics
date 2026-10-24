#!/usr/bin/env python3
"""Plot exclusive encoding/decoding timing shares from a round-trip benchmark."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


LABELS = {
    "model_inference": "Model inference",
    "framework_residual": "Framework/orchestration residual",
    "vocabulary_selection": "Vocabulary selection",
    "softmax": "Softmax",
    "score_or_interval_transfer": "Scores/intervals D2H",
    "score_transfer": "Scores D2H",
    "frequency_or_interval_construction": "Frequency/interval construction",
    "frequency_or_cdf_construction": "Frequency/CDF construction",
    "range_coding": "Range coding",
    "inverse_cdf_and_range_update": "Inverse CDF + range update",
    "arithmetic_decoding": "Frequency/CDF construction + range decoding",
    "decoded_token_transfer": "Decoded tokens D2H",
    "detokenization": "Detokenization",
}

COLORS = {
    "model_inference": "#4C78A8",
    "framework_residual": "#BAB0AC",
    "vocabulary_selection": "#72B7B2",
    "softmax": "#54A24B",
    "score_or_interval_transfer": "#E45756",
    "score_transfer": "#E45756",
    "frequency_or_interval_construction": "#F2CF5B",
    "frequency_or_cdf_construction": "#F2CF5B",
    "range_coding": "#B279A2",
    "inverse_cdf_and_range_update": "#B279A2",
    "arithmetic_decoding": "#B279A2",
    "decoded_token_transfer": "#FF9DA6",
    "detokenization": "#9D755D",
}

LEGEND_ENTRIES = (
    ("model_inference", "Model inference"),
    ("vocabulary_selection", "Vocabulary selection"),
    ("softmax", "Softmax"),
    ("score_or_interval_transfer", "Scores/intervals D2H"),
    (
        "frequency_or_interval_construction",
        "Frequency/interval or CDF construction",
    ),
    ("range_coding", "Range coding/update"),
    ("arithmetic_decoding", "Frequency/CDF construction + range decoding"),
    ("decoded_token_transfer", "Decoded tokens D2H"),
    ("detokenization", "Detokenization"),
    ("framework_residual", "Framework/orchestration residual"),
)

ENCODER_LABELS = {
    "python_full_logits": "Historical Python full logits (MSAC-v1)",
    "python_full_logits_v2": "Python full logits",
    "python": "Python target interval",
    "cuda_v1": "CUDA v1 (device coder)",
    "cuda_v2": "CUDA v2 (+fused intervals)",
    "cuda_v3": "CUDA v3 (+no FP64 copy)",
    "cuda_v4": "CUDA exact fixed total",
}

DECODER_LABELS = {
    "host": "Python host",
    "device": "PyTorch device",
    "cuda": "Fused CUDA",
}


def _aggregate(samples: list[dict]) -> dict:
    valid = [sample for sample in samples if sample.get("stage_timing")]
    if not valid:
        required = {
            "seconds",
            "inference_seconds",
            "arithmetic_seconds",
            "detokenize_seconds",
        }
        valid = [sample for sample in samples if required <= sample.keys()]
        if not valid:
            raise ValueError(
                "input has neither stage_timing records nor coarse decoder timings"
            )
        totals = [sample["seconds"] for sample in valid]
        total = statistics.median(totals)
        keys = (
            "model_inference",
            "arithmetic_decoding",
            "detokenization",
            "framework_residual",
        )
        sample_seconds = []
        for sample in valid:
            assigned = (
                sample["inference_seconds"]
                + sample["arithmetic_seconds"]
                + sample["detokenize_seconds"]
            )
            sample_seconds.append(
                {
                    "model_inference": sample["inference_seconds"],
                    "arithmetic_decoding": sample["arithmetic_seconds"],
                    "detokenization": sample["detokenize_seconds"],
                    "framework_residual": max(0.0, sample["seconds"] - assigned),
                }
            )
        mean_shares = {
            key: statistics.fmean(
                stages[key] / sample["seconds"]
                for stages, sample in zip(sample_seconds, valid)
            )
            for key in keys
        }
        share_sum = sum(mean_shares.values())
        mean_shares = {key: value / share_sum for key, value in mean_shares.items()}
        return {
            "total_seconds": total,
            "shares": mean_shares,
            "seconds": {key: total * value for key, value in mean_shares.items()},
            "overlapped_seconds": {},
        }
    totals = [sample["stage_timing"]["total_seconds"] for sample in valid]
    total = statistics.median(totals)
    keys = list(valid[0]["stage_timing"]["exclusive_seconds"])
    mean_shares = {
        key: statistics.fmean(
            sample["stage_timing"]["exclusive_seconds"].get(key, 0.0)
            / sample["stage_timing"]["total_seconds"]
            for sample in valid
        )
        for key in keys
    }
    share_sum = sum(mean_shares.values())
    mean_shares = {key: value / share_sum for key, value in mean_shares.items()}
    overlapped = {
        key: statistics.median(
            sample["stage_timing"].get("overlapped_seconds", {}).get(key, 0.0)
            for sample in valid
        )
        for key in {key for sample in valid for key in sample["stage_timing"].get("overlapped_seconds", {})}
    }
    return {
        "total_seconds": total,
        "shares": mean_shares,
        "seconds": {key: total * value for key, value in mean_shares.items()},
        "overlapped_seconds": overlapped,
    }


def _stacked(ax, names, aggregates, keys, labels, *, normalized):
    y = np.arange(len(names))
    left = np.zeros(len(names))
    for key in keys:
        values = np.array([
            aggregates[name]["shares"].get(key, 0.0) * 100
            if normalized else aggregates[name]["seconds"].get(key, 0.0)
            for name in names
        ])
        ax.barh(
            y,
            values,
            height=0.52,
            left=left,
            label=LABELS[key],
            color=COLORS[key],
        )
        left += values
    ax.set_yticks(y, [labels.get(name, name) for name in names])
    ax.invert_yaxis()
    ax.set_xlabel("Share of end-to-end phase (%)" if normalized else "Time (s)")
    ax.grid(axis="x", alpha=0.2)
    if normalized:
        ax.set_xlim(0, 100)
    else:
        for index, name in enumerate(names):
            overlap = aggregates[name]["overlapped_seconds"].get("range_coding", 0.0)
            if overlap:
                ax.scatter(overlap, index, marker="D", facecolors="none",
                           edgecolors="black", zorder=5)


def _row_major_legend(fig, handles, labels, *, ncol):
    entries = list(zip(handles, labels))
    rows = [entries[index:index + ncol] for index in range(0, len(entries), ncol)]
    reordered = [
        row[column]
        for column in range(ncol)
        for row in rows
        if column < len(row)
    ]
    legend_handles, legend_labels = zip(*reordered)
    fig.legend(
        legend_handles,
        legend_labels,
        loc="outside lower center",
        ncol=ncol,
    )


def _set_panel_title(fig, grid_cell, title):
    bounds = grid_cell.get_position(fig)
    fig.text(
        0.5 * (bounds.x0 + bounds.x1),
        bounds.y1 + 0.018,
        title,
        ha="center",
        va="bottom",
        fontsize=plt.rcParams["axes.titlesize"],
    )


def _plot_absolute(fig, grid_cell, names, aggregates, keys, labels, *, title, highlight=None):
    totals = sorted(aggregates[name]["total_seconds"] for name in names)
    use_broken_axis = len(totals) > 1 and totals[-1] > 10 * totals[-2]
    if use_broken_axis:
        broken_grid = grid_cell.subgridspec(
            1, 2, width_ratios=(0.72, 0.28), wspace=0.08
        )
        absolute_low = fig.add_subplot(broken_grid[0, 0])
        absolute_high = fig.add_subplot(broken_grid[0, 1], sharey=absolute_low)
        _stacked(absolute_low, names, aggregates, keys, labels, normalized=False)
        _stacked(absolute_high, names, aggregates, keys, labels, normalized=False)
        # Both absolute axes share y; the two _stacked calls invert it twice.
        absolute_low.invert_yaxis()
        low_limit = 1.15 * totals[-2]
        absolute_low.set_xlim(0, low_limit)
        absolute_high.set_xlim(totals[-1] - low_limit, 1.015 * totals[-1])
        absolute_low.spines["right"].set_visible(False)
        absolute_high.spines["left"].set_visible(False)
        absolute_high.tick_params(axis="y", left=False, labelleft=False)
        absolute_high.set_ylabel("")
        absolute_low.set_xlabel("Absolute phase time (s)")
        absolute_high.set_xlabel("")
        diagonal = 0.012
        for axis, x in ((absolute_low, 1), (absolute_high, 0)):
            axis.plot(
                (x - diagonal, x + diagonal), (-diagonal, +diagonal),
                transform=axis.transAxes, color="black", clip_on=False,
            )
            axis.plot(
                (x - diagonal, x + diagonal), (1 - diagonal, 1 + diagonal),
                transform=axis.transAxes, color="black", clip_on=False,
            )
        annotation_axis = absolute_low
    else:
        annotation_axis = fig.add_subplot(grid_cell)
        _stacked(annotation_axis, names, aggregates, keys, labels, normalized=False)
    _set_panel_title(fig, grid_cell, title)
    if highlight is not None:
        if highlight not in names:
            raise ValueError(f"highlighted condition is not plotted: {highlight}")
        row = names.index(highlight)
        target = aggregates[highlight]["total_seconds"]
        x_min, x_max = annotation_axis.get_xlim()
        annotation_axis.annotate(
            "CUDA ★",
            xy=(target, row),
            xytext=(target + 0.18 * (x_max - x_min), row),
            ha="left",
            va="center",
            fontsize=11,
            color="#333333",
            fontweight="normal",
            arrowprops={
                "arrowstyle": "->",
                "color": "#333333",
                "linewidth": 1.2,
                "shrinkA": 10,
            },
        )


def _plot_share(fig, grid_cell, names, aggregates, keys, labels, *, title):
    axis = fig.add_subplot(grid_cell)
    _stacked(axis, names, aggregates, keys, labels, normalized=True)
    _set_panel_title(fig, grid_cell, title)


def _plot_combined(
    encoders,
    encoding,
    encoding_keys,
    decoders,
    decoding,
    decoding_keys,
    *,
    best_encoder=None,
    best_decoder=None,
):
    fig = plt.figure(figsize=(13.2, 6.6))
    panels = fig.add_gridspec(
        2,
        2,
        left=0.16,
        right=0.985,
        bottom=0.18,
        top=0.93,
        width_ratios=(1, 1),
        height_ratios=(1.35, 1),
        wspace=0.56,
        hspace=0.58,
    )
    _plot_absolute(
        fig,
        panels[0, 0],
        encoders,
        encoding,
        encoding_keys,
        ENCODER_LABELS,
        title="(a) Compression: absolute time",
        highlight=best_encoder,
    )
    _plot_share(
        fig,
        panels[0, 1],
        encoders,
        encoding,
        encoding_keys,
        ENCODER_LABELS,
        title="(b) Compression: time share",
    )
    _plot_absolute(
        fig,
        panels[1, 0],
        decoders,
        decoding,
        decoding_keys,
        DECODER_LABELS,
        title="(c) Decompression: absolute time",
        highlight=best_decoder,
    )
    _plot_share(
        fig,
        panels[1, 1],
        decoders,
        decoding,
        decoding_keys,
        DECODER_LABELS,
        title="(d) Decompression: time share",
    )
    present_keys = set(encoding_keys) | set(decoding_keys)
    handles = [
        Patch(facecolor=COLORS[key], label=label)
        for key, label in LEGEND_ENTRIES
        if key in present_keys
        or (key == "score_or_interval_transfer" and "score_transfer" in present_keys)
        or (
            key == "frequency_or_interval_construction"
            and "frequency_or_cdf_construction" in present_keys
        )
        or (key == "range_coding" and "inverse_cdf_and_range_update" in present_keys)
    ]
    _row_major_legend(
        fig,
        handles,
        [handle.get_label() for handle in handles],
        ncol=5,
    )
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument(
        "--encoding-supplement",
        action="append",
        default=[],
        type=Path,
        help="add encoding conditions missing from the primary benchmark JSON",
    )
    parser.add_argument(
        "--exclude-encoder",
        action="append",
        default=[],
        help="omit an encoding condition from the figure",
    )
    parser.add_argument(
        "--decoding-supplement",
        action="append",
        default=[],
        type=Path,
        help="add decoding conditions missing from the primary benchmark JSON",
    )
    parser.add_argument(
        "--decoding-override",
        type=Path,
        help="replace all decoder samples with a matched decoder benchmark JSON",
    )
    parser.add_argument(
        "--encoder-total-override",
        action="append",
        default=[],
        metavar="CONDITION=SECONDS",
        help=(
            "replace an encoder's absolute bar total while retaining its measured "
            "stage shares; useful when the total is a multi-run median and the "
            "stage profile comes from a separate diagnostic pass"
        ),
    )
    parser.add_argument(
        "--best-encoder",
        help="mark this compression condition with a left-pointing CUDA star arrow",
    )
    parser.add_argument(
        "--best-decoder",
        help="mark this decompression condition with a left-pointing CUDA star arrow",
    )
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument(
        "--figures-only",
        action="store_true",
        help="write only the combined PDF figure, omitting PNG and diagnostics",
    )
    args = parser.parse_args()
    report = json.loads(args.input.read_text())
    encoding_samples = dict(report["encoding"]["samples"])
    encoders = list(report["encoding"]["conditions"])
    for supplement_path in args.encoding_supplement:
        supplement = json.loads(supplement_path.read_text())
        for name in supplement["encoding"]["conditions"]:
            if name not in encoding_samples:
                encoding_samples[name] = supplement["encoding"]["samples"][name]
                encoders.append(name)
    encoders = [name for name in encoders if name not in args.exclude_encoder]
    encoder_order = {name: index for index, name in enumerate(ENCODER_LABELS)}
    encoders.sort(key=lambda name: encoder_order.get(name, len(encoder_order)))
    decoding_samples = dict(report["decoding"]["samples"])
    decoders = list(report["decoding"]["conditions"])
    if args.decoding_override is not None:
        decoder_report = json.loads(args.decoding_override.read_text())
        decoding_samples = dict(decoder_report["decoding"]["samples"])
        decoders = list(decoder_report["decoding"]["conditions"])
    for supplement_path in args.decoding_supplement:
        supplement = json.loads(supplement_path.read_text())
        for name in supplement["decoding"]["conditions"]:
            if name not in decoding_samples:
                decoding_samples[name] = supplement["decoding"]["samples"][name]
                decoders.append(name)
    decoder_order = {name: index for index, name in enumerate(DECODER_LABELS)}
    decoders.sort(key=lambda name: decoder_order.get(name, len(decoder_order)))
    encoding = {name: _aggregate(encoding_samples[name]) for name in encoders}
    decoding = {name: _aggregate(decoding_samples[name]) for name in decoders}
    for item in args.encoder_total_override:
        name, separator, raw_seconds = item.partition("=")
        if not separator or name not in encoding:
            raise ValueError(
                "encoder total overrides must be CONDITION=SECONDS for a plotted condition"
            )
        total_seconds = float(raw_seconds)
        if total_seconds <= 0:
            raise ValueError("encoder total overrides must be positive")
        encoding[name]["total_seconds"] = total_seconds
        encoding[name]["seconds"] = {
            key: total_seconds * share
            for key, share in encoding[name]["shares"].items()
        }
    encoding_keys = list(next(iter(encoding.values()))["shares"])
    decoding_keys = list(next(iter(decoding.values()))["shares"])

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig = _plot_combined(
        encoders,
        encoding,
        encoding_keys,
        decoders,
        decoding,
        decoding_keys,
        best_encoder=args.best_encoder,
        best_decoder=args.best_decoder,
    )
    fig.savefig(args.output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    if not args.figures_only:
        fig.savefig(args.output_prefix.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)

    if args.figures_only:
        return

    csv_path = args.output_prefix.with_name(args.output_prefix.name + "-values.csv")
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["phase", "implementation", "stage", "seconds", "share_percent"])
        for phase, names, values in (("encoding", encoders, encoding), ("decoding", decoders, decoding)):
            for name in names:
                for stage, seconds in values[name]["seconds"].items():
                    writer.writerow([phase, name, stage, seconds, 100 * values[name]["shares"][stage]])

    lines = ["# Pipeline stage timing conclusions", ""]
    for phase, names, values in (("Encoding", encoders, encoding), ("Decoding", decoders, decoding)):
        lines.extend((f"## {phase}", ""))
        for index, name in enumerate(names):
            bottleneck = max(values[name]["shares"], key=values[name]["shares"].get)
            sentence = (
                f"- **{name}**: {values[name]['total_seconds']:.4f} s; "
                f"largest stage is {LABELS[bottleneck]} "
                f"({100 * values[name]['shares'][bottleneck]:.1f}%)."
            )
            if index:
                previous = names[index - 1]
                speedup = values[previous]["total_seconds"] / values[name]["total_seconds"]
                reduction = 100 * (1 - values[name]["total_seconds"] / values[previous]["total_seconds"])
                if reduction >= 0:
                    sentence += f" Versus {previous}: {speedup:.2f}x speedup ({reduction:.1f}% less time)."
                else:
                    sentence += f" Versus {previous}: {-reduction:.1f}% slower."
            lines.append(sentence)
        lines.append("")
    lines.extend((
        "The stacked components are exclusive and sum to 100%. CUDA-v4 range-coder device work",
        "is overlapped with inference, so it is reported as an unfilled diamond in the absolute-time plot",
        "and is not added to the critical-path stack.",
    ))
    args.output_prefix.with_name(args.output_prefix.name + "-conclusions.md").write_text(
        "\n".join(lines) + "\n"
    )


if __name__ == "__main__":
    main()
