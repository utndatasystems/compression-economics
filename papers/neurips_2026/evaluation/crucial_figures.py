"""Auditable helpers for the crucial NeurIPS compression figures.

The notebook in ``papers/neurips_2026/evaluation/crucial_figures.ipynb`` owns the
paper-specific analysis.  This module keeps binary serialization, round-trip
checks, and plotting logic in ordinary Python so they can be tested.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import struct
import subprocess
import tempfile
from typing import Iterable, Mapping, Sequence

import brotli
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TOKEN_MAGIC = b"QTOK"
TOKEN_VERSION = 1
TOKEN_HEADER = struct.Struct(">4sBBHIIQ")

PREDICTIVE_MAGIC = b"QAC1"
PREDICTIVE_VERSION = 1
PREDICTIVE_HEADER = struct.Struct(">4sBBBBIIIQ")

COLORS = {
    "qwen": "#0072B2",
    "anti": "#D55E00",
    "oracle": "#009E73",
    "token": "#6B7280",
    "fsst": "#009E73",
    "brotli": "#D55E00",
    "ink": "#202124",
    "grid": "#E5E7EB",
}


def find_repo_root(start: Path | None = None) -> Path:
    """Return the nearest parent containing ``pyproject.toml``."""
    start = (start or Path.cwd()).resolve()
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise FileNotFoundError("Could not locate repository root")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _pack_fixed_width(values: Sequence[int], width: int) -> bytes:
    if width < 1:
        raise ValueError("width must be positive")
    accumulator = 0
    buffered = 0
    output = bytearray()
    for value in values:
        if not 0 <= int(value) < (1 << width):
            raise ValueError(f"value {value} does not fit in {width} bits")
        accumulator = (accumulator << width) | int(value)
        buffered += width
        while buffered >= 8:
            buffered -= 8
            output.append((accumulator >> buffered) & 0xFF)
            accumulator &= (1 << buffered) - 1 if buffered else 0
    if buffered:
        output.append((accumulator << (8 - buffered)) & 0xFF)
    return bytes(output)


def _unpack_fixed_width(payload: bytes, count: int, width: int) -> list[int]:
    accumulator = 0
    buffered = 0
    output: list[int] = []
    for byte in payload:
        accumulator = (accumulator << 8) | byte
        buffered += 8
        while buffered >= width and len(output) < count:
            buffered -= width
            output.append((accumulator >> buffered) & ((1 << width) - 1))
            accumulator &= (1 << buffered) - 1 if buffered else 0
    if len(output) != count:
        raise ValueError(f"truncated token payload: decoded {len(output)} of {count}")
    return output


def serialize_token_ids(
    token_ids: Sequence[int], *, vocab_size: int, raw_size_bytes: int
) -> bytes:
    """Serialize fixed-width token IDs with a small self-describing header.

    The external Qwen token table is treated as shared decoder state, matching the
    paper's treatment of model weights.  Its bytes are therefore not embedded.
    """
    if not token_ids:
        raise ValueError("token_ids must not be empty")
    width = math.ceil(math.log2(vocab_size))
    header = TOKEN_HEADER.pack(
        TOKEN_MAGIC,
        TOKEN_VERSION,
        width,
        0,
        vocab_size,
        len(token_ids),
        raw_size_bytes,
    )
    return header + _pack_fixed_width(token_ids, width)


def deserialize_token_ids(blob: bytes) -> dict:
    if len(blob) < TOKEN_HEADER.size:
        raise ValueError("truncated token stream header")
    magic, version, width, reserved, vocab_size, count, raw_size = TOKEN_HEADER.unpack(
        blob[: TOKEN_HEADER.size]
    )
    if magic != TOKEN_MAGIC or version != TOKEN_VERSION or reserved != 0:
        raise ValueError("unsupported token stream")
    expected_width = math.ceil(math.log2(vocab_size))
    if width != expected_width:
        raise ValueError(f"stored width {width} != expected width {expected_width}")
    token_ids = _unpack_fixed_width(blob[TOKEN_HEADER.size :], count, width)
    return {
        "token_ids": token_ids,
        "vocab_size": vocab_size,
        "raw_size_bytes": raw_size,
        "width": width,
    }


def _pack_bits(bits: Iterable[int]) -> tuple[bytes, int]:
    values = [int(bit) for bit in bits]
    if any(bit not in (0, 1) for bit in values):
        raise ValueError("arithmetic payload must contain only bits")
    padding = (-len(values)) % 8
    return _pack_fixed_width(values + [0] * padding, 1), padding


def serialize_predictive_payload(
    bits: Sequence[int],
    *,
    seed_token: int,
    model_code: int,
    token_count: int,
    raw_size_bytes: int,
) -> bytes:
    """Serialize a model-coded block; model weights remain shared state."""
    payload, padding = _pack_bits(bits)
    header = PREDICTIVE_HEADER.pack(
        PREDICTIVE_MAGIC,
        PREDICTIVE_VERSION,
        model_code,
        padding,
        0,
        seed_token,
        token_count,
        raw_size_bytes,
        len(bits),
    )
    return header + payload


def parse_predictive_payload(blob: bytes) -> dict:
    if len(blob) < PREDICTIVE_HEADER.size:
        raise ValueError("truncated predictive stream header")
    fields = PREDICTIVE_HEADER.unpack(blob[: PREDICTIVE_HEADER.size])
    magic, version, model_code, padding, reserved, seed, count, raw_size, bit_count = fields
    if magic != PREDICTIVE_MAGIC or version != PREDICTIVE_VERSION or reserved != 0:
        raise ValueError("unsupported predictive stream")
    payload = blob[PREDICTIVE_HEADER.size :]
    if len(payload) != math.ceil(bit_count / 8) or padding != (-bit_count) % 8:
        raise ValueError("predictive payload length metadata is inconsistent")
    bits = _unpack_fixed_width(payload, bit_count, 1)
    return {
        "bits": bits,
        "seed_token": seed,
        "model_code": model_code,
        "token_count": count,
        "raw_size_bytes": raw_size,
    }


def benchmark_brotli(data: bytes, *, quality: int = 11) -> dict:
    if not data:
        raise ValueError("data must not be empty")
    encoded = brotli.compress(data, quality=quality)
    if brotli.decompress(encoded) != data:
        raise AssertionError("Brotli round trip failed")
    return {
        "codec": f"Brotli q{quality}",
        "raw_size_bytes": len(data),
        "serialized_size_bytes": len(encoded),
        "round_trip": True,
    }


def benchmark_tokenization(
    data: bytes, token_ids: Sequence[int], *, tokenizer, vocab_size: int
) -> dict:
    encoded = serialize_token_ids(
        token_ids, vocab_size=vocab_size, raw_size_bytes=len(data)
    )
    decoded = deserialize_token_ids(encoded)
    reconstructed = tokenizer.decode(
        decoded["token_ids"],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    ).encode("utf-8")
    if reconstructed != data:
        raise AssertionError("tokenization-only byte round trip failed")
    return {
        "codec": "Qwen token IDs",
        "raw_size_bytes": len(data),
        "serialized_size_bytes": len(encoded),
        "round_trip": True,
        "token_count": len(token_ids),
        "bits_per_token": math.ceil(math.log2(vocab_size)),
    }


def benchmark_fsst(data: bytes, *, executable: Path) -> dict:
    """Measure the official FSST CLI format, including its table and framing."""
    executable = executable.resolve()
    if not executable.is_file():
        raise FileNotFoundError(f"FSST executable not found: {executable}")
    with tempfile.TemporaryDirectory(prefix="compression-economics-fsst-") as tmp:
        tmpdir = Path(tmp)
        raw_path = tmpdir / "input.bin"
        encoded_path = tmpdir / "input.fsst"
        decoded_path = tmpdir / "decoded.bin"
        raw_path.write_bytes(data)
        subprocess.run(
            [str(executable), str(raw_path), str(encoded_path)],
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            [str(executable), "-d", str(encoded_path), str(decoded_path)],
            check=True,
            capture_output=True,
            text=True,
        )
        if decoded_path.read_bytes() != data:
            raise AssertionError("FSST byte round trip failed")
        size = encoded_path.stat().st_size
    return {
        "codec": "FSST",
        "raw_size_bytes": len(data),
        "serialized_size_bytes": size,
        "round_trip": True,
    }


def add_ratio(row: Mapping) -> dict:
    result = dict(row)
    raw = int(result["raw_size_bytes"])
    encoded = int(result["serialized_size_bytes"])
    if raw <= 0 or encoded < 0:
        raise ValueError("sizes must be non-negative and raw size must be positive")
    result["relative_size_percent"] = 100.0 * encoded / raw
    return result


def validate_bar_results(
    frame: pd.DataFrame,
    expected_datasets: Sequence[str],
    *,
    include_preliminary: bool = False,
) -> pd.DataFrame:
    required = {
        "dataset",
        "codec",
        "raw_size_bytes",
        "serialized_size_bytes",
        "round_trip",
        "status",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing result columns: {sorted(missing)}")
    accepted_statuses = (
        {"measured", "preliminary", "payload-only"} if include_preliminary else {"measured"}
    )
    measured = frame[frame["status"].isin(accepted_statuses)].copy()
    verified = measured[measured["status"] == "measured"]
    if not verified["round_trip"].fillna(False).all():
        raise AssertionError("every measured bar must pass a round trip")
    duplicates = measured.duplicated(["dataset", "codec"], keep=False)
    if duplicates.any():
        raise ValueError("duplicate measured dataset/codec rows")
    unknown = set(measured["dataset"]) - set(expected_datasets)
    if unknown:
        raise ValueError(f"unexpected datasets: {sorted(unknown)}")
    measured["relative_size_percent"] = (
        100.0 * measured["serialized_size_bytes"] / measured["raw_size_bytes"]
    )
    return measured


def plot_surprisal_scatter(frame: pd.DataFrame):
    required = {"block", "model", "surprisal_bits_per_token", "relative_size_percent"}
    if required - set(frame.columns):
        raise ValueError(f"scatter data missing {sorted(required - set(frame.columns))}")
    fig, ax = plt.subplots(figsize=(7.05, 3.15))
    model_specs = (
        ("Qwen2.5-0.5B", COLORS["qwen"], "o"),
        ("Anti-Qwen", COLORS["anti"], "s"),
    )
    means = frame.groupby("model", observed=True)[
        ["surprisal_bits_per_token", "relative_size_percent"]
    ].mean()
    model_order = [model for model, _, _ in model_specs if model in means.index]
    if len(model_order) > 1:
        ax.plot(
            means.loc[model_order, "surprisal_bits_per_token"],
            means.loc[model_order, "relative_size_percent"],
            color=COLORS["grid"],
            linewidth=1.2,
            zorder=1,
        )
    for model, color, marker in model_specs:
        subset = frame[frame["model"] == model]
        if subset.empty:
            continue
        ax.scatter(
            subset["surprisal_bits_per_token"],
            subset["relative_size_percent"],
            s=20,
            marker=marker,
            color=color,
            alpha=0.30,
            edgecolor="white",
            linewidth=0.6,
            label=f"{model} runs",
            zorder=2,
        )
        ax.scatter(
            means.loc[model, "surprisal_bits_per_token"],
            means.loc[model, "relative_size_percent"],
            s=60,
            marker=marker,
            color=color,
            edgecolor="white",
            linewidth=1.0,
            label=f"{model} mean",
            zorder=4,
        )
    ax.scatter(
        [0], [0], marker="*", s=80, facecolor="white", edgecolor=COLORS["oracle"],
        linewidth=1.4, label="Oracle entropy bound (perfect predictions)", zorder=4,
    )
    ax.axhline(
        100,
        color=COLORS["token"],
        linestyle=(0, (3, 2)),
        linewidth=0.9,
        zorder=1,
    )
    ax.text(
        x=ax.get_xlim()[0] + 0.05 * (ax.get_xlim()[1] - ax.get_xlim()[0]),
        y=110,
        s="raw UTF-8",
        color=COLORS["token"],
        fontsize=6.8,
    )
    ax.set(
        xlabel="Mean model surprisal (bits / predicted token)",
        ylabel=r"Realized size ratio $R_{\mathrm{real}}$ (%)",
    )
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.6, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(
        frameon=False,
        fontsize=7.2,
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
    )
    fig.subplots_adjust(left=0.105, right=0.99, top=0.82, bottom=0.20)
    return fig, ax


def plot_compressor_bars(
    frame: pd.DataFrame,
    dataset_order: Sequence[str],
    *,
    include_preliminary: bool = False,
):
    measured = validate_bar_results(
        frame, dataset_order, include_preliminary=include_preliminary
    )
    codec_styles = [
        ("Qwen + AC", "Qwen + AC", COLORS["qwen"], ""),
        ("Qwen token IDs", "Qwen token IDs", COLORS["token"], "///"),
        ("FSST", "FSST", COLORS["fsst"], ""),
        ("Brotli q11", "Brotli q11", COLORS["brotli"], "\\\\"),
    ]
    best_by_dataset = measured.groupby("dataset", observed=True)[
        "relative_size_percent"
    ].min()
    fig, ax = plt.subplots(figsize=(7.05, 3.15))
    baseline_mask = np.array([
        dataset in {"text8", "random text"} for dataset in dataset_order
    ])
    x = np.arange(len(dataset_order), dtype=float)
    adversary_count = int((~baseline_mask).sum())
    x[~baseline_mask] += 0.65 + 0.35 * np.arange(adversary_count)
    width = 0.19
    for index, (codec, display_name, color, hatch) in enumerate(codec_styles):
        values = []
        for dataset in dataset_order:
            hit = measured[(measured["dataset"] == dataset) & (measured["codec"] == codec)]
            values.append(float(hit.iloc[0]["relative_size_percent"]) if len(hit) else np.nan)
        bars = ax.bar(
            x + (index - 1.5) * width,
            values,
            width,
            label=display_name,
            color=color,
            edgecolor="white",
            linewidth=0.7,
            hatch=hatch,
            zorder=3,
        )
        for bar, value, dataset in zip(bars, values, dataset_order):
            if np.isfinite(value):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + 3,
                    f"{value + 1e-9:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=6.4,
                    fontweight=(
                        "bold"
                        if np.isclose(value, best_by_dataset.loc[dataset])
                        else "normal"
                    ),
                    rotation=90,
                )

    ax.axhline(
        100,
        color=COLORS["token"],
        linestyle=(0, (3, 2)),
        linewidth=0.9,
        zorder=1,
    )
    ax.text(
        x[0] - 0.38,
        110,
        "raw UTF-8 (100%)",
        color=COLORS["token"],
        fontsize=6.8,
    )
    display_labels = {
        "text8": "text8",
        "random text": "random\ntext",
        "token-level adversary": "MinProb\n($v^\\star_{\\mathrm{tok}}$)",
        "MinProb adversary": "MinProb\n($v^\\star_{\\mathrm{tok}}$)",
        "byte-aware adversary": "byte-aware\nadversary",
        "MaxSurprisal/Byte adversary": (
            "MaxSurprisal/Byte\n"
            "($v^\\star_{\\mathrm{byte}}$; N=1,024)"
        ),
        "one-byte adversary": "one-byte\nadversary",
        "all one-byte adversary": "one-byte UTF-8\nablation",
    }
    display_labels = [display_labels.get(dataset, dataset) for dataset in dataset_order]
    ax.set_xticks(x, display_labels, fontsize=7.2)
    ax.tick_params(axis="y", labelsize=7.5)
    group_specs = [
        ("BASELINES", x[baseline_mask]),
        ("INPUT ADVERSARIES", x[~baseline_mask]),
    ]
    for label, positions in group_specs:
        if len(positions):
            left = float(positions.min() - 0.42)
            right = float(positions.max() + 0.42)
            ax.plot(
                [left, right],
                [-0.185, -0.185],
                transform=ax.get_xaxis_transform(),
                color=COLORS["token"],
                linewidth=0.65,
                clip_on=False,
            )
            ax.text(
                float(positions.mean()),
                -0.200,
                label,
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=6.6,
                color=COLORS["token"],
                clip_on=False,
            )
    if baseline_mask.any() and (~baseline_mask).any():
        separator = (x[baseline_mask].max() + x[~baseline_mask].min()) / 2
        ax.axvline(separator, color=COLORS["grid"], linewidth=0.8, zorder=0)
    plot_top = ax.get_ylim()[1] * 1.15
    ax.set_ylim(top=plot_top)
    missing_datasets = [
        dataset for dataset in dataset_order
        if measured[measured["dataset"] == dataset].empty
    ]
    for dataset in missing_datasets:
        position = x[list(dataset_order).index(dataset)]
        ax.text(
            position,
            0.12 * plot_top,
            "pending\n10k run",
            ha="center",
            va="center",
            fontsize=6.4,
            fontstyle="italic",
            color=COLORS["token"],
            bbox={
                "boxstyle": "round,pad=0.3",
                "facecolor": "#F9FAFB",
                "edgecolor": "#9CA3AF",
                "linestyle": "--",
                "linewidth": 0.8,
            },
            zorder=4,
        )
    has_preliminary = (measured["status"] != "measured").any()
    ax.set_ylabel(
        "Encoded size / raw UTF-8 (%)"
        if has_preliminary
        else r"Realized size ratio $R_{\mathrm{real}}$ (%)",
        fontsize=8.5,
    )
    if has_preliminary:
        ax.text(
            0.995,
            0.985,
            "MaxSurprisal/Byte Qwen+AC: AC payload only",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=6.4,
            color=COLORS["qwen"],
        )
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.6, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(
        frameon=False,
        fontsize=7.2,
        ncol=4,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
    )
    fig.subplots_adjust(left=0.105, right=0.99, top=0.82, bottom=0.20)
    return fig, ax


def plot_paper_prediction_difficulty(
    frame: pd.DataFrame,
    *,
    text8_bits_per_token: float,
):
    """Plot paper-table prediction costs against the shared text8 reference."""
    required = {
        "condition",
        "prediction_bits_per_token",
        "std_bits_per_token",
        "single_run",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"prediction-difficulty data missing {sorted(missing)}")
    if frame.empty or text8_bits_per_token <= 0:
        raise ValueError("prediction-difficulty values must be non-empty and positive")
    values = frame["prediction_bits_per_token"].astype(float).to_numpy()
    error_values = pd.to_numeric(
        frame["std_bits_per_token"], errors="coerce"
    ).to_numpy(dtype=float)
    errors = np.nan_to_num(error_values, nan=0.0)
    if np.any(values <= 0) or np.any(errors < 0):
        raise ValueError("prediction costs and standard deviations must be non-negative")

    x = np.arange(len(frame), dtype=float)
    width = 0.32
    typical_positions = x - width / 2
    attack_positions = x + width / 2
    ratios = values / text8_bits_per_token

    fig, ax = plt.subplots(figsize=(7.05, 3.15))
    typical_bars = ax.bar(
        typical_positions,
        np.full(len(frame), text8_bits_per_token),
        width,
        color=COLORS["qwen"],
        edgecolor="white",
        linewidth=0.7,
        label="Natural text (text8)",
        zorder=3,
    )
    attack_bars = ax.bar(
        attack_positions,
        values,
        width,
        color=COLORS["anti"],
        edgecolor="white",
        linewidth=0.7,
        label="Adversarial condition",
        zorder=3,
    )
    error_mask = np.isfinite(error_values)
    ax.errorbar(
        attack_positions[error_mask],
        values[error_mask],
        yerr=error_values[error_mask],
        fmt="none",
        ecolor=COLORS["ink"],
        elinewidth=0.8,
        capsize=3,
        capthick=0.8,
        zorder=4,
    )
    value_offset = max(values) * 0.025
    for bar in typical_bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            text8_bits_per_token + value_offset,
            f"{text8_bits_per_token:.2f}",
            ha="center",
            va="bottom",
            fontsize=6.4,
            fontweight="bold",
        )
    for bar, value, error in zip(attack_bars, values, errors):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + error + value_offset,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=6.4,
            fontweight="bold",
        )
    for position, ratio in zip(x, ratios):
        ax.text(
            position,
            0.025,
            f"{ratio:.2f}×",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=6.6,
            color=COLORS["token"],
            fontweight="bold",
            bbox={
                "boxstyle": "round,pad=0.15",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.92,
            },
        )

    labels = frame["condition"].astype(str).tolist()
    labels = [
        label.replace("One-byte UTF-8 ablation", "One-byte UTF-8\nablation")
        for label in labels
    ]
    objective_labels = {
        "MinProb": "MinProb\n($v^\\star_{\\mathrm{tok}}$)",
        "MaxSurprisal/Byte": (
            "MaxSurprisal/Byte\n($v^\\star_{\\mathrm{byte}}$)"
        ),
    }
    labels = [objective_labels.get(label, label) for label in labels]
    labels = [
        f"{label}$^{{\\dagger}}$" if single_run else label
        for label, single_run in zip(labels, frame["single_run"].astype(bool))
    ]
    ax.set_xticks(x, labels)
    ax.tick_params(axis="x", labelsize=7.2)
    ax.tick_params(axis="y", labelsize=7.5)
    ax.set_ylabel("Prediction cost (bits/token)", fontsize=8.5)
    ax.set_ylim(0, max(values + errors) * 1.20)
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.6, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(
        frameon=False,
        fontsize=7.2,
        ncol=2,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
    )
    fig.text(
        0.99,
        0.025,
        r"$^\dagger$ $N=1{,}024$; one run",
        ha="right",
        va="bottom",
        fontsize=6.4,
        color=COLORS["token"],
    )
    fig.subplots_adjust(left=0.105, right=0.99, top=0.82, bottom=0.20)
    return fig, ax
