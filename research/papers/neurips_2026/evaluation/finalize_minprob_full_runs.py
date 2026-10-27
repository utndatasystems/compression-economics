#!/usr/bin/env python3
"""Create and verify complete Qwen+AC streams for full MinProb runs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.papers.neurips_2026.experiments.generate_adversarial import (
    make_predictor_args,
    token_ids_round_trip,
)
from src.adversarial.compression import (
    decode_arithmetic_payloads,
    encode_arithmetic_payloads,
)
from src.adversarial.qwen_stream import (
    parse_qwen_stream,
    serialize_qwen_stream,
    verify_decoded_bytes,
)
from src.prediction import TokenPredictor


DEFAULT_INPUT = (
    REPO_ROOT
    / "artifacts/papers/neurips-2026/camera-ready/"
    "minprob-h100-three-seed/retained-source.json"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "artifacts/papers/neurips-2026/camera-ready/"
    "minprob-h100-three-seed/finalized"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def decoded_bytes(tokenizer, token_ids: list[int]) -> bytes:
    return tokenizer.decode(
        token_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    ).encode("utf-8")


def main() -> None:
    args = parse_args()
    source = json.loads(args.input.read_text(encoding="utf-8"))
    metadata = source["metadata"]
    runs = sorted(source["runs"], key=lambda row: row["run_index"])
    if len(runs) != 3 or any(len(row["token_ids"]) != 10_000 for row in runs):
        raise ValueError("Expected exactly three complete 10,000-token runs")

    predictor = TokenPredictor(
        make_predictor_args(
            argparse.Namespace(model_name=metadata["model_name"])
        ),
        bitmap_data=None,
    )
    tokenizer = predictor.tokenizer
    sequences = [list(row["token_ids"]) for row in runs]
    seeds = [list(row["start_token_ids"]) for row in runs]
    raw = [decoded_bytes(tokenizer, sequence) for sequence in sequences]
    if any(not token_ids_round_trip(tokenizer, sequence) for sequence in sequences):
        raise AssertionError("A saved MinProb sequence is not canonical")
    if any(len(data) != row["decoded_utf8_size_bytes"] for data, row in zip(raw, runs)):
        raise AssertionError("Saved decoded-byte lengths do not match")

    stream_paths = [
        args.output_dir / f"run_{row['run_index']:02d}.qac" for row in runs
    ]
    if args.force or not all(path.is_file() for path in stream_paths):
        payloads = encode_arithmetic_payloads(
            predictor,
            sequences,
            start_length=len(seeds[0]),
            context_length=metadata["context_length"],
            retain_tokens=metadata["retain_tokens"],
            use_kv_cache=metadata["use_kv_cache"],
            progress_desc="Full MinProb: encoding",
        )
        for path, bits, seed, sequence, data in zip(
            stream_paths, payloads, seeds, sequences, raw
        ):
            write_atomic(
                path,
                serialize_qwen_stream(
                    bits,
                    model_name=metadata["model_name"],
                    seed_token_ids=seed,
                    token_count=len(sequence),
                    raw_size_bytes=len(data),
                    context_length=metadata["context_length"],
                    retain_tokens=metadata["retain_tokens"],
                    use_kv_cache=metadata["use_kv_cache"],
                    decoded_bytes=data,
                ),
            )

    parsed = [parse_qwen_stream(path.read_bytes()) for path in stream_paths]
    for item, seed, sequence, data in zip(parsed, seeds, sequences, raw):
        if item["seed_token_ids"] != seed or item["token_count"] != len(sequence):
            raise AssertionError("Serialized stream metadata does not match source")
        verify_decoded_bytes(item, data)

    recovered = decode_arithmetic_payloads(
        predictor,
        [item["bits"] for item in parsed],
        seed_token_ids=seeds,
        token_counts=[len(sequence) for sequence in sequences],
        context_length=metadata["context_length"],
        retain_tokens=metadata["retain_tokens"],
        use_kv_cache=metadata["use_kv_cache"],
        progress_desc="Full MinProb: decoding",
    )

    output_runs = []
    for row, path, item, sequence, result, data in zip(
        runs, stream_paths, parsed, sequences, recovered, raw
    ):
        if result != sequence or decoded_bytes(tokenizer, result) != data:
            raise AssertionError(f"Run {row['run_index']} failed round-trip")
        output_runs.append(
            {
                "run_index": row["run_index"],
                "start_token_ids": row["start_token_ids"],
                "token_count": len(sequence),
                "raw_size_bytes": len(data),
                "decoded_sha256": item["decoded_sha256"],
                "payload_bits": item["payload_bit_count"],
                "serialized_size_bytes": path.stat().st_size,
                "stream": str(path.relative_to(REPO_ROOT)),
                "token_round_trip": True,
                "byte_round_trip": True,
            }
        )

    result = {
        "metadata": {
            "model_name": metadata["model_name"],
            "context_length": metadata["context_length"],
            "retain_tokens": metadata["retain_tokens"],
            "frequency_total": parsed[0]["frequency_total"],
            "statesize": parsed[0]["statesize"],
            "selection": "complete-sequence",
            "aggregation": "three independent seed runs",
        },
        "runs": output_runs,
    }
    write_atomic(
        args.output_dir / "results.json",
        json.dumps(result, indent=2).encode("utf-8"),
    )


if __name__ == "__main__":
    main()
