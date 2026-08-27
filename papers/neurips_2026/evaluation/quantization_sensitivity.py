#!/usr/bin/env python3
"""Replay fixed adversarial sequences at several AC frequency precisions."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.generate_adversarial import make_predictor_args
from src.compression_attacks import _full_vocab_logits
from src.encoding import ArithmeticEncoder, BitOutputStream
from src.prediction import TokenPredictor

ARTIFACT_ROOT = REPO_ROOT / "artifacts/papers/neurips-2026"
DEFAULT_OUTPUT = ARTIFACT_ROOT / "ablations/arithmetic-quantization/results.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--frequency-total", type=int, action="append")
    parser.add_argument("--statesize", type=int, default=32)
    parser.add_argument("--context-length", type=int, default=1000)
    parser.add_argument("--retain-tokens", type=int, default=100)
    parser.add_argument("--limit", type=int, help="Diagnostic predicted-token limit")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_conditions() -> list[dict]:
    minprob_path = ARTIFACT_ROOT / "runs/attacks/minprob/full-vocabulary/n10000/full/results.json"
    max_path = ARTIFACT_ROOT / "runs/attacks/max-surprisal-per-byte/full-vocabulary/n1024/results.json"
    minprob = read_json(minprob_path)
    max_payload = read_json(max_path)
    rows = [{
        "condition": "MinProb",
        "run_index": row["run_index"],
        "token_ids": list(row["token_ids"]),
        "source": str(minprob_path.relative_to(REPO_ROOT)),
        "reported_model_surprisal_bits": row["summary"]["model_surprisal_bits"],
    } for row in minprob["runs"]]
    matches = [row for row in max_payload["runs"]
               if row["condition"] == "surprisal-per-byte" and row["run_index"] == 0]
    if len(matches) != 1:
        raise ValueError("Expected one final MaxSurprisal/Byte run")
    rows.append({
        "condition": "MaxSurprisal/Byte",
        "run_index": 0,
        "token_ids": list(matches[0]["token_ids"]),
        "source": str(max_path.relative_to(REPO_ROOT)),
        "reported_model_surprisal_bits": matches[0]["model_entropy_bits"],
    })
    return rows


def decode_size(tokenizer, token_ids: list[int]) -> int:
    return len(tokenizer.decode(token_ids, skip_special_tokens=False,
                                clean_up_tokenization_spaces=False).encode("utf-8"))


def quantize_frequencies(probabilities: np.ndarray, total: int,
                         descending_indices: np.ndarray) -> np.ndarray:
    """Match src.encoding_utils.build_cumul while reusing the probability sort."""
    alphabet_size = probabilities.size
    if total < alphabet_size:
        raise ValueError(f"Frequency total {total} is below alphabet size {alphabet_size}")
    frequencies = (probabilities * (total - alphabet_size)).astype(np.int64) + 1
    difference = int(total - frequencies.sum())
    if difference < 0 or difference >= alphabet_size:
        raise AssertionError(f"Unexpected rounding difference {difference}")
    frequencies[descending_indices[:difference]] += 1
    if int(frequencies.sum()) != total or np.any(frequencies < 1):
        raise AssertionError("Invalid quantized frequency table")
    return frequencies


def cumulative(frequencies: np.ndarray) -> np.ndarray:
    result = np.empty(frequencies.size + 1, dtype=np.int64)
    result[0] = 0
    result[1:] = np.cumsum(frequencies)
    return result


def replay(predictor, rows: list[dict], totals: list[int], args: argparse.Namespace) -> list[dict]:
    for row in rows:
        if args.limit is not None:
            row["token_ids"] = row["token_ids"][:args.limit + 1]
        row["raw_size_bytes"] = decode_size(predictor.tokenizer, row["token_ids"])
        row["raw_surprisal_bits"] = 0.0
        row["coders"] = {}
        for total in totals:
            bitout = BitOutputStream()
            row["coders"][total] = {
                "encoder": ArithmeticEncoder(args.statesize, bitout),
                "bitout": bitout,
                "quantized_surprisal_bits": 0.0,
                "clipped_tokens": 0,
                "sum_delta_bits": 0.0,
                "max_abs_delta_bits": 0.0,
            }

    contexts = [[row["token_ids"][0]] for row in rows]
    predictor.reset_kv_cache()
    max_length = max(len(row["token_ids"]) for row in rows)
    for position in tqdm(range(1, max_length), desc="Quantization sensitivity", unit="step"):
        if len(contexts[0]) >= args.context_length:
            contexts = [context[-args.retain_tokens:] for context in contexts]
        token_ids, logits, _, _ = predictor.run_batched_inference(contexts, enable_kv_cache=True)
        logits = _full_vocab_logits(token_ids, logits)
        for row_index, (row, logit_row) in enumerate(zip(rows, logits)):
            sequence = row["token_ids"]
            if position >= len(sequence):
                contexts[row_index].append(sequence[-1])
                continue
            target = sequence[position]
            probabilities = torch.softmax(logit_row.float(), dim=0).cpu().numpy()
            raw_bits = -math.log2(max(float(probabilities[target]), 1e-300))
            row["raw_surprisal_bits"] += raw_bits
            order = np.argsort(-probabilities)
            for total, state in row["coders"].items():
                frequencies = quantize_frequencies(probabilities, total, order)
                target_frequency = int(frequencies[target])
                quantized_bits = -math.log2(target_frequency / total)
                delta = quantized_bits - raw_bits
                state["quantized_surprisal_bits"] += quantized_bits
                state["clipped_tokens"] += target_frequency == 1
                state["sum_delta_bits"] += delta
                state["max_abs_delta_bits"] = max(state["max_abs_delta_bits"], abs(delta))
                state["encoder"].write(cumulative(frequencies), target)
            contexts[row_index].append(target)

    output = []
    for row in rows:
        predicted_tokens = len(row["token_ids"]) - 1
        for total, state in row.pop("coders").items():
            state["encoder"].finish()
            payload_bits = len(state["bitout"].get_bits())
            output.append({
                "condition": row["condition"],
                "run_index": row["run_index"],
                "source": row["source"],
                "frequency_total": total,
                "min_representable_probability": 1 / total,
                "token_count": len(row["token_ids"]),
                "predicted_tokens": predicted_tokens,
                "raw_size_bytes": row["raw_size_bytes"],
                "reported_model_ideal_bits_per_byte": (
                    None
                    if args.limit is not None
                    else row["reported_model_surprisal_bits"] / row["raw_size_bytes"]
                ),
                "coder_support_ideal_bits_per_byte": row["raw_surprisal_bits"] / row["raw_size_bytes"],
                "quantized_ideal_bits_per_byte": state["quantized_surprisal_bits"] / row["raw_size_bytes"],
                "realized_ac_bits_per_byte": payload_bits / row["raw_size_bytes"],
                "realized_ac_bits": payload_bits,
                "clipped_fraction": state["clipped_tokens"] / predicted_tokens,
                "mean_quantized_minus_raw_surprisal_bits": state["sum_delta_bits"] / predicted_tokens,
                "max_abs_quantized_minus_raw_surprisal_bits": state["max_abs_delta_bits"],
            })
    return output


def write_atomic(path: Path, payload: dict) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    totals = args.frequency_total or [2**18, 2**22, 2**26, 2**30]
    if totals != sorted(set(totals)):
        raise ValueError("Frequency totals must be unique and increasing")
    rows = load_conditions()
    predictor = TokenPredictor(make_predictor_args(SimpleNamespace(model_name=args.model_name)), bitmap_data=None)
    vocabulary_size = len(predictor.tokenizer)
    invalid = [total for total in totals if total < vocabulary_size]
    max_total = (1 << args.statesize) // 4 + 2
    if invalid:
        raise ValueError(f"Totals {invalid} are below vocabulary size {vocabulary_size}")
    if totals[-1] > max_total:
        raise ValueError(f"Total {totals[-1]} exceeds {args.statesize}-bit coder maximum {max_total}")
    results = replay(predictor, rows, totals, args)
    payload = {
        "metadata": {
            "model_name": args.model_name,
            "statesize": args.statesize,
            "frequency_totals": totals,
            "vocabulary_size": vocabulary_size,
            "context_length": args.context_length,
            "retain_tokens": args.retain_tokens,
            "clipped_definition": "selected token receives the mandatory minimum frequency count of one",
            "surprisal_delta_definition": "quantized surprisal minus unquantized surprisal after restricting to the coder support",
            "diagnostic_limit": args.limit,
        },
        "runs": results,
    }
    write_atomic(args.output, payload)
    print(f"Saved {args.output.resolve()}")


if __name__ == "__main__":
    main()
