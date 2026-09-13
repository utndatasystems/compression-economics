#!/usr/bin/env python3
"""Plot CIDR model compression ratio against encode throughput."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, action="append", required=True)
    parser.add_argument(
        "--output", type=Path,
        default=Path("artifacts/papers/cidr-2027/model-survey/text8-model-tradeoff.png"),
    )
    parser.add_argument(
        "--qwen-results", type=Path,
        help="Optional main.py compression_results.json. It is included only when its source-byte count matches.",
    )
    return parser.parse_args()


def load_survey(paths: list[Path]) -> list[dict]:
    rows = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in payload["results"]:
            required = {
                "model", "source_bytes", "payload_compression_ratio",
                "encode_mib_per_second", "decode_mib_per_second",
                "roundtrip_valid", "accounting_mode",
            }
            missing = required - row.keys()
            if missing:
                raise ValueError(f"{path} row missing {sorted(missing)}")
            if not row["roundtrip_valid"]:
                raise ValueError(f"{path} contains a failed round trip")
            if row["accounting_mode"] != "shared_model_payload_only":
                raise ValueError(f"{path} uses unsupported accounting mode")
            rows.append(dict(row))
    if not rows:
        raise ValueError("no model rows found")
    sizes = {row["source_bytes"] for row in rows}
    if len(sizes) != 1:
        raise ValueError("all plotted survey rows must use the same source_bytes")
    return rows


def load_matching_qwen(path: Path, source_bytes: int) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    matches = []
    for run in payload.values():
        row = run.get("compression", {})
        if row.get("original_size_bytes") == source_bytes:
            matches.append(row)
    if not matches:
        raise ValueError(
            "No Qwen run has matching source_bytes. Re-run Qwen on the same "
            "byte-exact target and accounting mode before overlaying it."
        )
    best = max(matches, key=lambda row: row["throughput_kibibytes_per_sec"])
    return {
        "model": "Qwen2.5-0.5B",
        "source_bytes": source_bytes,
        "payload_compression_ratio": best["compression_factor"],
        "encode_mib_per_second": best["throughput_kibibytes_per_sec"] / 1024,
        "decode_mib_per_second": None,
        "roundtrip_valid": True,
        "accounting_mode": "main.py stream",
    }


def main() -> None:
    args = parse_args()
    rows = load_survey(args.results)
    if args.qwen_results:
        rows.append(load_matching_qwen(args.qwen_results, rows[0]["source_bytes"]))

    fig, axis = plt.subplots(figsize=(7.0, 4.2))
    colors = {"byte": "#0072B2", "token": "#D55E00"}
    for row in rows:
        marker = "*" if row["model"].startswith("Qwen") else "o"
        color = "#009E73" if marker == "*" else colors[row["symbol_kind"]]
        axis.scatter(
            row["encode_mib_per_second"], row["payload_compression_ratio"],
            color=color, marker=marker, s=82, edgecolor="white", linewidth=0.8,
            zorder=3,
        )
        axis.annotate(
            row["model"], (row["encode_mib_per_second"], row["payload_compression_ratio"]),
            xytext=(5, 4), textcoords="offset points", fontsize=7,
        )
    axis.set_xscale("log")
    axis.axhline(1.0, color="#6B7280", linestyle=(0, (3, 2)), linewidth=0.9)
    axis.set(
        xlabel="Encoding throughput (MiB/s; log scale)",
        ylabel="Compression ratio (raw / payload)",
        title=f"text8: {rows[0]['source_bytes']:,}-byte held-out continuation",
    )
    axis.grid(axis="both", color="#E5E7EB", linewidth=0.6, zorder=0)
    axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)

    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {args.output} and {csv_path}")


if __name__ == "__main__":
    main()
