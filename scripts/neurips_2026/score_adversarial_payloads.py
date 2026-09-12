#!/usr/bin/env python3
"""Measure arithmetic payloads for saved adversarial token sequences.

The scorer is deliberately separate from sequence generation.  It checkpoints
each run and dictionary policy atomically, so an interrupted paper evaluation can
resume without regenerating inputs or losing completed coding measurements.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys

from pyroaring import BitMap

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.generate_adversarial import make_predictor_args
from src.compression_attacks import score_arithmetic_payloads
from src.prediction import TokenPredictor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Adversarial run directory containing full/results.json.",
    )
    parser.add_argument(
        "--policy",
        choices=("full", "occurring"),
        action="append",
        help="Dictionary policy to score; default: both.",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _write_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _load_source(input_dir: Path) -> dict:
    path = input_dir / "full" / "results.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing generated sequences: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not payload.get("runs"):
        raise ValueError(f"No generated sequences in {path}")
    return payload


def main() -> None:
    args = parse_args()
    source = _load_source(args.input_dir)
    source_metadata = source["metadata"]
    output = args.output or args.input_dir / "arithmetic_payloads.json"
    policies = list(dict.fromkeys(args.policy or ["full", "occurring"]))
    metadata = {
        "model_name": source_metadata["model_name"],
        "source": str((args.input_dir / "full" / "results.json").resolve()),
        "context_length": source_metadata["context_length"],
        "retain_tokens": source_metadata["retain_tokens"],
        "use_kv_cache": source_metadata["use_kv_cache"],
        "policies": policies,
        "checkpointing": "atomic after each run and dictionary policy",
    }
    results = []
    if output.exists() and not args.force:
        checkpoint = json.loads(output.read_text(encoding="utf-8"))
        if checkpoint.get("metadata") != metadata:
            raise ValueError(
                f"Incompatible checkpoint at {output}; use --force to replace it"
            )
        results = list(checkpoint.get("runs", []))

    predictor_args = argparse.Namespace(
        model_name=source_metadata["model_name"]
    )
    predictor = TokenPredictor(
        make_predictor_args(predictor_args), bitmap_data=None
    )
    vocab_size = min(
        predictor.tokenizer.vocab_size, predictor.model.config.vocab_size
    )
    fixed_seed_bits = math.ceil(math.log2(vocab_size))

    for policy in policies:
        for run in source["runs"]:
            run_index = run["run_index"]
            if any(
                row["policy"] == policy and row["run_index"] == run_index
                for row in results
            ):
                print(f"Skipping completed {policy} run {run_index}.")
                continue
            tokens = run["token_ids"]
            start_length = len(run["start_token_ids"])
            candidates = sorted(set(tokens)) if policy == "occurring" else None
            payload_bits = score_arithmetic_payloads(
                predictor,
                [tokens],
                start_length=start_length,
                context_length=source_metadata["context_length"],
                retain_tokens=source_metadata["retain_tokens"],
                candidate_sets=[candidates] if candidates is not None else None,
                use_kv_cache=source_metadata["use_kv_cache"],
            )[0]
            raw_bits = 8 * run["decoded_utf8_size_bytes"]
            dictionary_bytes = (
                len(BitMap(candidates).serialize())
                if candidates is not None
                else 0
            )
            row = {
                "policy": policy,
                "run_index": run_index,
                "total_tokens": len(tokens),
                "decoded_utf8_size_bytes": run["decoded_utf8_size_bytes"],
                "candidate_size": (
                    len(candidates) if candidates is not None else vocab_size
                ),
                "seed_bits": fixed_seed_bits * start_length,
                "arithmetic_payload_bits": payload_bits,
                "dictionary_size_bytes": dictionary_bytes,
                "payload_ratio": payload_bits / raw_bits,
                "payload_plus_seed_and_dictionary_ratio": (
                    payload_bits
                    + fixed_seed_bits * start_length
                    + 8 * dictionary_bytes
                ) / raw_bits,
            }
            results = [
                old
                for old in results
                if not (
                    old["policy"] == policy and old["run_index"] == run_index
                )
            ]
            results.append(row)
            results.sort(
                key=lambda item: (
                    policies.index(item["policy"]), item["run_index"]
                )
            )
            _write_atomic(output, {"metadata": metadata, "runs": results})
            print(f"Saved {policy} run {run_index} to {output}.")


if __name__ == "__main__":
    main()
