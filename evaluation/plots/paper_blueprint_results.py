#!/usr/bin/env python3
"""Build preliminary submission figures from all currently completed runs.

The full-vocabulary results use the new 10k-token evaluation. Until the
replacement long one-byte and beam sweeps finish, the script deliberately uses
the existing verified 1k-token one-byte run and 32-token beam diagnostic and
records that mixed provenance in the machine-readable summary.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


COLORS = {
    "blue": "#2563EB",
    "orange": "#EA580C",
    "green": "#16A34A",
    "purple": "#7C3AED",
    "muted": "#6B7280",
    "ink": "#111827",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path("artifacts/runs/paper-evaluation"),
    )
    parser.add_argument("--length", type=int, default=10_000)
    parser.add_argument("--search-length", type=int, default=512)
    parser.add_argument(
        "--pipeline-results",
        type=Path,
        default=Path("artifacts/runs/current/compression_results.json"),
    )
    parser.add_argument(
        "--one-byte-dir",
        type=Path,
        default=Path("artifacts/runs/adversarial/qwen_05b_ascii_bytes_n1000/full"),
    )
    parser.add_argument(
        "--beam-results",
        type=Path,
        default=Path(
            "artifacts/runs/compression-attacks/qwen_05b_ascii_n32_smoke/results.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/figures/neurips/final"),
    )
    return parser.parse_args()


def load(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing required experiment output: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def find_text8(pipeline: dict, *, reduce_tokens: bool, model_name: str) -> dict:
    matches = []
    for entry in pipeline.values():
        compression = entry.get("compression", {})
        args = compression.get("args", {})
        if (
            Path(str(args.get("input_path", ""))).name == "text8"
            and args.get("model_name") == model_name
            and args.get("first_n_tokens") == 100_000
            and args.get("context_length") == 1_000
            and args.get("retain_tokens") == 100
            and args.get("use_kv_cache") is True
            and args.get("batch_size") == 16
            and args.get("encoding") == "AC"
            and args.get("reduce_tokens") is reduce_tokens
            and args.get("lora_path") is None
        ):
            matches.append(compression)
    if len(matches) != 1:
        policy = "occurring" if reduce_tokens else "full"
        raise ValueError(f"Expected one matched text8 {policy} result, found {len(matches)}")
    return matches[0]


def mean_std(values) -> tuple[float, float]:
    array = np.asarray(list(values), dtype=float)
    return float(array.mean()), float(array.std(ddof=1)) if len(array) > 1 else 0.0


def clean_axes(axis) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(labelsize=7)
    axis.title.set_fontsize(8.5)
    axis.xaxis.label.set_fontsize(8)
    axis.yaxis.label.set_fontsize(8)


def main() -> None:
    args = parse_args()
    full_dir = args.run_root / f"full_vocab_n{args.length}"
    full = load(full_dir / "full" / "results.json")
    occurring = load(full_dir / "occurring" / "results.json")
    coding = load(full_dir / "arithmetic_payloads.json")
    ascii_payload = load(args.one_byte_dir / "results.json")
    ascii_check = load(args.one_byte_dir / "compression_verification.json")
    beam_payload = load(args.beam_results)
    pipeline = load(args.pipeline_results)

    for full_run, masked_run in zip(full["runs"], occurring["runs"]):
        if full_run["token_ids"] != masked_run["token_ids"]:
            raise ValueError("Occurring-token rescore changed adversarial token IDs")
    if any(len(run["token_ids"]) != args.length for run in full["runs"]):
        raise ValueError("Full-vocabulary generation is incomplete")

    model_name = full["metadata"]["model_name"]
    vocab_size = full["runs"][0]["summary"]["candidate_size"]
    fixed_bits = math.ceil(math.log2(vocab_size))
    text8 = {
        "full": find_text8(pipeline, reduce_tokens=False, model_name=model_name),
        "occurring": find_text8(pipeline, reduce_tokens=True, model_name=model_name),
    }

    def adversarial_rate(payload: dict, field: str) -> list[float]:
        return [
            100
            * (
                fixed_bits * len(run["start_token_ids"])
                - sum(value / math.log(2) for value in run[field])
            )
            / (8 * run["decoded_utf8_size_bytes"])
            for run in payload["runs"]
        ]

    adversarial_fixed = [
        100 * fixed_bits * len(run["token_ids"])
        / (8 * run["decoded_utf8_size_bytes"])
        for run in full["runs"]
    ]
    coding_full = [
        100 * row["payload_plus_seed_and_dictionary_ratio"]
        for row in coding["runs"] if row["policy"] == "full"
    ]
    if len(coding_full) != len(full["runs"]):
        raise ValueError("Full-vocabulary arithmetic payload scoring is incomplete")

    if not all(
        ascii_check[key]
        for key in (
            "token_round_trip", "byte_round_trip",
            "serialized_artifact_token_round_trip",
            "serialized_artifact_byte_round_trip",
        )
    ):
        raise ValueError("The one-byte serialized round trip is not verified")
    ascii_run = ascii_payload["runs"][0]
    ascii_compression = ascii_check["compression"]
    ascii_raw_bytes = ascii_compression["original_size_bytes"]
    ascii_tokens = ascii_run["summary"]["total_tokens"]
    ascii_fixed = [100 * fixed_bits * ascii_tokens / (8 * ascii_raw_bytes)]
    ascii_floor = [
        100 * (fixed_bits + ascii_compression["entropy"]) / (8 * ascii_raw_bytes)
    ]
    ascii_coded = [
        100 * ascii_compression["final_size_bytes"] / ascii_raw_bytes
    ]
    ascii_serialized = ascii_check["serialized_file_ratio_percent"]

    def text8_metrics(policy: str) -> dict[str, float]:
        result = text8[policy]
        n = result["args"]["first_n_tokens"]
        batches = result["args"]["batch_size"]
        raw_bits = 8 * result["original_size_bytes"]
        return {
            "fixed": 100 * n * fixed_bits / raw_bits,
            "floor": 100 * (batches * fixed_bits + result["entropy"]) / raw_bits,
            "total": 100 * result["final_size_bytes"] / result["original_size_bytes"],
        }

    text_metrics = {policy: text8_metrics(policy) for policy in ("full", "occurring")}
    full_fixed_mean, full_fixed_std = mean_std(adversarial_fixed)
    # The occurrence artifact is a continuous replay of the fixed token IDs.
    # Its raw probabilities therefore provide the checkpoint-independent full
    # model score, while its masked probabilities provide the post-hoc score.
    full_floor_mean, full_floor_std = mean_std(
        adversarial_rate(occurring, "model_log_probabilities")
    )
    full_coded_mean, full_coded_std = mean_std(coding_full)
    ascii_fixed_mean, ascii_fixed_std = mean_std(ascii_fixed)
    ascii_floor_mean, ascii_floor_std = mean_std(ascii_floor)
    ascii_coded_mean, ascii_coded_std = mean_std(ascii_coded)
    masked_floor_mean, masked_floor_std = mean_std(
        adversarial_rate(occurring, "masked_log_probabilities")
    )

    beam = {}
    for condition in ("beam-surprisal-per-byte", "beam-actual-ratio"):
        rows = [row for row in beam_payload["runs"] if row["condition"] == condition]
        beam[condition] = {
            "floor_percent": mean_std(100 * row["model_entropy_ratio"] for row in rows),
            "payload_percent": mean_std(100 * row["arithmetic_payload_ratio"] for row in rows),
        }

    summary = {
        "model_name": model_name,
        "long_sequence_tokens": args.length,
        "one_byte_sequence_tokens": ascii_tokens,
        "beam_sequence_tokens": beam_payload["metadata"]["total_length"],
        "runs_per_adversarial_condition": len(full["runs"]),
        "full_vocabulary": {
            "fixed_percent": [full_fixed_mean, full_fixed_std],
            "model_floor_percent": [full_floor_mean, full_floor_std],
            "arithmetic_total_percent": [full_coded_mean, full_coded_std],
        },
        "occurring_tokens": {
            "model_floor_percent": [masked_floor_mean, masked_floor_std],
            "reduction_percent": 100 * (1 - masked_floor_mean / full_floor_mean),
        },
        "one_byte_ascii": {
            "fixed_percent": [ascii_fixed_mean, ascii_fixed_std],
            "model_floor_percent": [ascii_floor_mean, ascii_floor_std],
            "arithmetic_payload_percent": [ascii_coded_mean, ascii_coded_std],
            "serialized_percent": ascii_serialized,
        },
        "text8": text_metrics,
        "beam": beam,
        "status": {
            "preliminary": True,
            "complete": [
                "fixed full-vocabulary token IDs (3 x 10,000 tokens)",
                "continuous full/occurring replay (3 x 10,000 tokens)",
                "full and occurring arithmetic replay",
            ],
            "fallback_artifacts": [
                "one-byte ASCII: verified 1,000-token run",
                "coder-aware beam: 32-token smoke test",
            ],
            "missing": [
                "replacement full-vocabulary generation after checkpoint-context fix",
                "printable-ASCII 10,000-token condition matrix",
                "one-byte ASCII 10,000-token condition matrix",
                "coder-aware beam search at 512 tokens",
            ],
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "paper_blueprint_results.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    figure, (left, right) = plt.subplots(
        1, 2, figsize=(7.05, 3.15), gridspec_kw={"width_ratios": [1.42, 1.0]}
    )
    conditions = ["Typical\ntext8", "Full-vocab.\nadversary", "One-byte\nadversary"]
    x = np.arange(3)
    width = 0.22
    plot_rows = [
        (
            [text_metrics["full"]["fixed"], full_fixed_mean, ascii_fixed_mean],
            [0, full_fixed_std, ascii_fixed_std],
            "Fixed-width token IDs", COLORS["muted"], "///",
        ),
        (
            [text_metrics["full"]["floor"], full_floor_mean, ascii_floor_mean],
            [0, full_floor_std, ascii_floor_std],
            "Model surprisal estimate", COLORS["blue"], "",
        ),
        (
            [text_metrics["full"]["total"], full_coded_mean, ascii_coded_mean],
            [0, full_coded_std, ascii_coded_std],
            "AC payload + dictionary", COLORS["orange"], "\\\\",
        ),
    ]
    for offset, (values, errors, label, color, hatch) in zip((-1, 0, 1), plot_rows):
        bars = left.bar(
            x + offset * width, values, width, yerr=errors, capsize=2,
            color=color, edgecolor="white", linewidth=0.7, hatch=hatch,
            label=label, zorder=3,
        )
        for bar, value in zip(bars, values):
            left.text(
                bar.get_x() + bar.get_width() / 2, value + 4,
                f"{value:.1f}", ha="center", va="bottom", fontsize=6.7,
            )
    left.axhline(100, color=COLORS["ink"], linestyle=(0, (3, 2)), linewidth=0.9)
    left.set(ylim=(0, max(170, ascii_coded_mean * 1.22)), ylabel="Encoded size / raw UTF-8 (%)")
    left.set_xticks(x, conditions)
    left.set_title("a   Two layers determine end-to-end size", loc="left")
    left.grid(axis="y", color="#E5E7EB", linewidth=0.6, zorder=0)
    left.legend(loc="upper left", frameon=False, fontsize=6.7)
    left.scatter(
        x[2] + width, ascii_serialized, marker="D", s=25,
        color=COLORS["purple"], edgecolor="white", linewidth=0.7, zorder=5,
        label="Serialized one-byte file",
    )
    left.text(x[2] + width, ascii_serialized + 4, f"{ascii_serialized:.1f}",
              ha="center", fontsize=6.7, color=COLORS["purple"])
    clean_axes(left)

    masking = [
        ("Typical text8", text_metrics["full"]["floor"], text_metrics["occurring"]["floor"], COLORS["green"], "o"),
        ("Adversarial worst case", full_floor_mean, masked_floor_mean, COLORS["orange"], "s"),
    ]
    for label, before, after, color, marker in masking:
        right.plot([0, 1], [before, after], color=color, lw=2, marker=marker,
                   markersize=5.5, markeredgecolor="white", markeredgewidth=0.7)
        right.text(0.03, before + 1.8, f"{before:.1f}%", color=color, fontsize=7, ha="center")
        right.text(0.97, after + 1.8, f"{after:.1f}%", color=color, fontsize=7, ha="center")
    reduction = summary["occurring_tokens"]["reduction_percent"]
    right.annotate(
        f"{reduction:.1f}% lower", xy=(1, masked_floor_mean),
        xytext=(0.5, max(full_floor_mean, masked_floor_mean) * 0.82),
        ha="center", fontsize=7.2, color=COLORS["orange"], fontweight="bold",
        arrowprops={"arrowstyle": "->", "color": COLORS["orange"], "lw": 0.8},
    )
    right.set(xlim=(-0.18, 1.18), ylim=(0, max(60, full_floor_mean * 1.18)),
              ylabel="Model surprisal / raw UTF-8 (%)")
    right.set_xticks([0, 1], ["Full vocabulary", "Occurring\ntokens"])
    right.set_title("b   Masking the same token IDs", loc="left")
    right.grid(axis="y", color="#E5E7EB", linewidth=0.6, zorder=0)
    right.legend(
        handles=[Line2D([0], [0], color=color, marker=marker, lw=2,
                        markersize=5, markeredgecolor="white", label=label)
                 for label, _, _, color, marker in masking],
        loc="upper right", frameon=False, fontsize=6.7,
    )
    clean_axes(right)
    figure.text(
        0.995, 0.015,
        "Preliminary: full-vocabulary n=10,000; one-byte n=1,000",
        ha="right", va="bottom", fontsize=6.3, color=COLORS["muted"],
    )
    figure.subplots_adjust(left=0.075, right=0.995, top=0.91, bottom=0.22, wspace=0.34)
    for suffix in ("pdf", "png"):
        figure.savefig(
            args.output_dir / f"layered_compression_robustness.{suffix}",
            dpi=300, bbox_inches="tight",
        )
    plt.close(figure)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
