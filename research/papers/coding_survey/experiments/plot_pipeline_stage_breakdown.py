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
    "decoded_token_transfer": "#FF9DA6",
    "detokenization": "#9D755D",
}

ENCODER_LABELS = {
    "python_full_logits": "Python full logits",
    "python": "Python target interval",
    "cuda_v1": "CUDA v1",
    "cuda_v2": "CUDA v2",
    "cuda_v3": "CUDA v3",
    "cuda_v4": "CUDA v4",
}

DECODER_LABELS = {
    "host": "Python host",
    "device": "PyTorch device",
    "cuda": "Fused CUDA",
}


def _aggregate(samples: list[dict]) -> dict:
    valid = [sample for sample in samples if sample.get("stage_timing")]
    if not valid:
        raise ValueError("input has no stage_timing records; rerun the benchmark")
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
            if normalized else aggregates[name]["seconds"].get(key, 0.0) * 1000
            for name in names
        ])
        ax.barh(y, values, left=left, label=LABELS[key], color=COLORS[key])
        left += values
    ax.set_yticks(y, [labels.get(name, name) for name in names])
    ax.invert_yaxis()
    ax.set_xlabel("Share of end-to-end phase (%)" if normalized else "Time (ms)")
    ax.grid(axis="x", alpha=0.2)
    if normalized:
        ax.set_xlim(0, 100)
    else:
        for index, name in enumerate(names):
            overlap = aggregates[name]["overlapped_seconds"].get("range_coding", 0.0)
            if overlap:
                ax.scatter(overlap * 1000, index, marker="D", facecolors="none",
                           edgecolors="black", zorder=5)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.input.read_text())
    encoders = report["encoding"]["conditions"]
    decoders = report["decoding"]["conditions"]
    encoding = {name: _aggregate(report["encoding"]["samples"][name]) for name in encoders}
    decoding = {name: _aggregate(report["decoding"]["samples"][name]) for name in decoders}
    encoding_keys = list(next(iter(encoding.values()))["shares"])
    decoding_keys = list(next(iter(decoding.values()))["shares"])

    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.2), constrained_layout=True)
    _stacked(axes[0, 0], encoders, encoding, encoding_keys, ENCODER_LABELS, normalized=False)
    _stacked(axes[0, 1], encoders, encoding, encoding_keys, ENCODER_LABELS, normalized=True)
    _stacked(axes[1, 0], decoders, decoding, decoding_keys, DECODER_LABELS, normalized=False)
    _stacked(axes[1, 1], decoders, decoding, decoding_keys, DECODER_LABELS, normalized=True)
    axes[0, 0].set_title("Encoding: absolute time")
    axes[0, 1].set_title("Encoding: time share")
    axes[1, 0].set_title("Decoding: absolute time")
    axes[1, 1].set_title("Decoding: time share")
    handles, legend_labels = axes[0, 1].get_legend_handles_labels()
    handles2, legend_labels2 = axes[1, 1].get_legend_handles_labels()
    by_label = dict(zip(legend_labels + legend_labels2, handles + handles2))
    fig.legend(by_label.values(), by_label.keys(), loc="outside lower center", ncol=4)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(args.output_prefix.with_suffix(".png"), dpi=220, bbox_inches="tight")

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
