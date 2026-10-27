#!/usr/bin/env python3
"""Evaluate every Table 4 codec on the same complete MinProb runs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import sys


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.papers.neurips_2026.evaluation.crucial_figures import (
    benchmark_brotli,
    benchmark_fsst,
    benchmark_tokenization,
)
from transformers import AutoTokenizer


CAMERA_ROOT = (
    REPO_ROOT
    / "artifacts/papers/neurips-2026/camera-ready/minprob-h100-three-seed"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, default=CAMERA_ROOT / "retained-source.json"
    )
    parser.add_argument(
        "--qwen-results",
        type=Path,
        default=CAMERA_ROOT / "finalized/results.json",
    )
    parser.add_argument("--fsst-executable", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=CAMERA_ROOT / "codec-results.json"
    )
    return parser.parse_args()


def write_atomic(path: Path, payload: dict) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    source = json.loads(args.source.read_text(encoding="utf-8"))
    qwen = json.loads(args.qwen_results.read_text(encoding="utf-8"))
    model_name = source["metadata"]["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=".cache")
    qwen_by_run = {row["run_index"]: row for row in qwen["runs"]}

    output_runs = []
    for row in sorted(source["runs"], key=lambda item: item["run_index"]):
        run_index = row["run_index"]
        token_ids = list(row["token_ids"])
        data = row["decoded_text"].encode("utf-8")
        qwen_row = qwen_by_run[run_index]
        if len(data) != qwen_row["raw_size_bytes"]:
            raise AssertionError("Source and Qwen raw sizes disagree")
        token_result = benchmark_tokenization(
            data,
            token_ids,
            tokenizer=tokenizer,
            vocab_size=tokenizer.vocab_size,
        )
        brotli_result = benchmark_brotli(data, quality=11)
        fsst_result = benchmark_fsst(data, executable=args.fsst_executable)
        raw_size = len(data)
        token_count = len(token_ids)
        prediction_bits = row["summary"]["model_surprisal_bits"]
        sizes = {
            "AC payload": qwen_row["payload_bits"] / 8,
            "Qwen + AC": qwen_row["serialized_size_bytes"],
            "Qwen token IDs": token_result["serialized_size_bytes"],
            "FSST": fsst_result["serialized_size_bytes"],
            "Brotli q11": brotli_result["serialized_size_bytes"],
        }
        output_runs.append(
            {
                "run_index": run_index,
                "start_token_ids": row["start_token_ids"],
                "token_count": token_count,
                "raw_size_bytes": raw_size,
                "prediction_bits_per_token": prediction_bits / (token_count - 1),
                "fertility_bytes_per_token": raw_size / token_count,
                "model_ideal_bits_per_byte": prediction_bits / raw_size,
                "ac_bits_per_byte": qwen_row["payload_bits"] / raw_size,
                "encoded_sizes_bytes": sizes,
                "relative_size_percent": {
                    codec: 100 * size / raw_size for codec, size in sizes.items()
                },
                "token_round_trip": token_result["round_trip"],
                "fsst_round_trip": fsst_result["round_trip"],
                "brotli_round_trip": brotli_result["round_trip"],
                "qwen_token_round_trip": qwen_row["token_round_trip"],
                "qwen_byte_round_trip": qwen_row["byte_round_trip"],
            }
        )

    scalar_metrics = (
        "prediction_bits_per_token",
        "fertility_bytes_per_token",
        "model_ideal_bits_per_byte",
        "ac_bits_per_byte",
    )
    aggregate = {
        metric: statistics.mean(row[metric] for row in output_runs)
        for metric in scalar_metrics
    }
    codecs = output_runs[0]["relative_size_percent"]
    aggregate["relative_size_percent"] = {
        codec: statistics.mean(
            row["relative_size_percent"][codec] for row in output_runs
        )
        for codec in codecs
    }
    aggregate["aggregation"] = "unweighted mean over three complete runs"
    retained_source = {
        "metadata": source["metadata"],
        "runs": [
            {
                "run_index": row["run_index"],
                "start_token_ids": row["start_token_ids"],
                "token_ids": row["token_ids"],
                "decoded_utf8_size_bytes": row["decoded_utf8_size_bytes"],
                "decoded_sha256": qwen_by_run[row["run_index"]][
                    "decoded_sha256"
                ],
                "decoded_text_round_trips": row[
                    "decoded_text_round_trips"
                ],
                "summary": row["summary"],
            }
            for row in sorted(
                source["runs"], key=lambda item: item["run_index"]
            )
        ],
    }
    write_atomic(args.output.parent / "retained-source.json", retained_source)
    write_atomic(
        args.output,
        {
            "metadata": {
                "model_name": model_name,
                "fsst_commit": "e638d4cf8c26129d73c242a4127b42b975de5b63",
                "brotli_quality": 11,
            },
            "runs": output_runs,
            "aggregate": aggregate,
        },
    )
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
