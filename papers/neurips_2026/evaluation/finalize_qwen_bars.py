#!/usr/bin/env python3
"""Create complete, decoder-verified Qwen streams for Figure 2."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.generate_adversarial import make_predictor_args, token_ids_round_trip
from src.compression_attacks import (
    decode_arithmetic_payloads,
    encode_arithmetic_payloads,
)
from src.prediction import TokenPredictor
from src.qwen_stream import (
    parse_qwen_stream,
    serialize_qwen_stream,
    verify_decoded_bytes,
)


ARTIFACT_ROOT = REPO_ROOT / "artifacts/papers/neurips-2026"
DEFAULT_OUTPUT = ARTIFACT_ROOT / "finalized/qwen-bars/b10000"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--byte-budget", type=int, default=10_000)
    parser.add_argument("--context-length", type=int, default=1000)
    parser.add_argument("--retain-tokens", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dataset", action="append", default=[], help="Finalize only the named dataset; repeat to select multiple datasets")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _condition(path: Path, name: str, run_index: int = 0) -> dict:
    payload = _read_json(path)
    matches = [
        row for row in payload["runs"]
        if row["condition"] == name and row["run_index"] == run_index
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one {name} run {run_index} in {path}")
    return matches[0]


def _decode(tokenizer, token_ids: list[int]) -> bytes:
    return tokenizer.decode(
        token_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    ).encode("utf-8")


def _matched_prefix(tokenizer, token_ids: list[int], budget: int) -> tuple[list[int], bytes]:
    """Select the longest canonical token prefix no larger than the byte budget."""
    if budget <= 0 or len(token_ids) < 2:
        raise ValueError("A positive budget and at least two tokens are required")
    complete = _decode(tokenizer, token_ids)
    if len(complete) <= budget:
        if not token_ids_round_trip(tokenizer, token_ids):
            raise AssertionError("Saved token sequence is not canonical")
        return token_ids, complete

    low, high = 2, len(token_ids)
    while low < high:
        middle = (low + high + 1) // 2
        if len(_decode(tokenizer, token_ids[:middle])) <= budget:
            low = middle
        else:
            high = middle - 1
    for end in range(low, 1, -1):
        prefix = token_ids[:end]
        data = _decode(tokenizer, prefix)
        if len(data) <= budget and token_ids_round_trip(tokenizer, prefix):
            if len(data) < 0.9 * budget:
                raise ValueError(
                    f"Nearest canonical prefix is only {len(data)} of {budget} bytes"
                )
            return prefix, data
    raise ValueError("No canonical prefix fits the byte budget")


def _source_rows(root: Path) -> list[dict]:
    printable_path = root / "runs/auxiliary/printable-ascii/n10000/results.json"
    one_byte_path = root / "runs/ablations/one-byte-utf8/n10000/results.json"
    full_path = root / "runs/attacks/minprob/full-vocabulary/n10000/full/results.json"
    max_surprisal_path = root / "runs/attacks/max-surprisal-per-byte/full-vocabulary/n1024/results.json"
    printable = _read_json(printable_path)
    full = _read_json(full_path)
    full_runs = [row for row in full["runs"] if row["run_index"] == 0]
    if len(full_runs) != 1:
        raise ValueError("Expected full-vocabulary run 0")
    specs = [
        ("text8", _condition(printable_path, "ordinary-text8"), printable_path, None),
        ("random text", _condition(printable_path, "random-printable-utf8"), printable_path, None),
        ("MinProb adversary", full_runs[0], full_path, full_runs[0]["start_token_ids"]),
        ("MaxSurprisal/Byte adversary", _condition(max_surprisal_path, "surprisal-per-byte"), max_surprisal_path, None),
        ("printable one-byte adversary", _condition(printable_path, "surprisal-per-byte"), printable_path, None),
        ("all one-byte adversary", _condition(one_byte_path, "surprisal-per-byte"), one_byte_path, None),
    ]
    rows = []
    for dataset, row, source, explicit_seed in specs:
        tokens = list(row["token_ids"])
        generated = row.get("generated_tokens")
        seed_length = len(explicit_seed) if explicit_seed is not None else len(tokens) - int(generated)
        seed = tokens[:seed_length]
        if explicit_seed is not None and seed != list(explicit_seed):
            raise AssertionError(f"Seed mismatch for {dataset}")
        rows.append({
            "dataset": dataset,
            "source": str(source.relative_to(REPO_ROOT)),
            "token_ids": tokens,
            "seed_token_ids": seed,
        })
    return rows


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def _write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    if not 0 < args.retain_tokens <= args.context_length:
        raise ValueError("Require 0 < retain_tokens <= context_length")
    predictor_args = SimpleNamespace(model_name=args.model_name)
    predictor = TokenPredictor(make_predictor_args(predictor_args), bitmap_data=None)
    tokenizer = predictor.tokenizer

    rows = _source_rows(ARTIFACT_ROOT)
    if args.dataset:
        available = {row["dataset"] for row in rows}
        unknown = set(args.dataset) - available
        if unknown:
            raise ValueError(f"Unknown datasets: {sorted(unknown)}")
        rows = [row for row in rows if row["dataset"] in args.dataset]
    for row in rows:
        tokens, data = _matched_prefix(
            tokenizer, row.pop("token_ids"), args.byte_budget
        )
        row["token_ids"] = tokens
        row["decoded_bytes"] = data
        row["stream_path"] = args.output_dir / f"{_slug(row['dataset'])}.qac"

    can_resume = not args.force and all(row["stream_path"].is_file() for row in rows)
    if can_resume:
        print("Reusing serialized Qwen streams; running decoder verification.")
        parsed = [parse_qwen_stream(row["stream_path"].read_bytes()) for row in rows]
    else:
        payloads = encode_arithmetic_payloads(
            predictor,
            [row["token_ids"] for row in rows],
            start_length=1,
            context_length=args.context_length,
            retain_tokens=args.retain_tokens,
            use_kv_cache=True,
            progress_desc="Qwen bars: encoding",
        )
        parsed = []
        for row, bits in zip(rows, payloads):
            blob = serialize_qwen_stream(
                bits,
                model_name=args.model_name,
                seed_token_ids=row["seed_token_ids"],
                token_count=len(row["token_ids"]),
                raw_size_bytes=len(row["decoded_bytes"]),
                context_length=args.context_length,
                retain_tokens=args.retain_tokens,
                use_kv_cache=True,
                decoded_bytes=row["decoded_bytes"],
            )
            _write_atomic(row["stream_path"], blob)
            parsed.append(parse_qwen_stream(blob))
            print(f"Saved {row['stream_path'].relative_to(REPO_ROOT)}")

    for row, metadata in zip(rows, parsed):
        if metadata["model_name"] != args.model_name:
            raise ValueError(f"Model mismatch in {row['stream_path']}")
        if metadata["seed_token_ids"] != row["seed_token_ids"]:
            raise ValueError(f"Seed mismatch in {row['stream_path']}")
        if metadata["token_count"] != len(row["token_ids"]):
            raise ValueError(f"Token-count mismatch in {row['stream_path']}")
        verify_decoded_bytes(metadata, row["decoded_bytes"])

    decoded = decode_arithmetic_payloads(
        predictor,
        [metadata["bits"] for metadata in parsed],
        seed_token_ids=[metadata["seed_token_ids"] for metadata in parsed],
        token_counts=[metadata["token_count"] for metadata in parsed],
        context_length=args.context_length,
        retain_tokens=args.retain_tokens,
        candidate_sets=[metadata["candidate_token_ids"] for metadata in parsed],
        use_kv_cache=True,
        statesize=parsed[0]["statesize"],
        frequency_total=parsed[0]["frequency_total"],
        progress_desc="Qwen bars: decoding",
    )

    output_rows = []
    for row, metadata, decoded_tokens in zip(rows, parsed, decoded):
        if decoded_tokens != row["token_ids"]:
            raise AssertionError(f"Token round trip failed for {row['dataset']}")
        decoded_bytes = _decode(tokenizer, decoded_tokens)
        verify_decoded_bytes(metadata, decoded_bytes)
        if decoded_bytes != row["decoded_bytes"]:
            raise AssertionError(f"Byte round trip failed for {row['dataset']}")
        slug = _slug(row["dataset"])
        token_path = args.output_dir / f"{slug}.tokens.json"
        text_path = args.output_dir / f"{slug}.txt"
        _write_atomic(token_path, json.dumps(decoded_tokens).encode("utf-8"))
        _write_atomic(text_path, decoded_bytes)
        output_rows.append({
            "dataset": row["dataset"],
            "source": row["source"],
            "stream": str(row["stream_path"].relative_to(REPO_ROOT)),
            "tokens": str(token_path.relative_to(REPO_ROOT)),
            "decoded_text": str(text_path.relative_to(REPO_ROOT)),
            "seed_token_ids": row["seed_token_ids"],
            "token_count": len(decoded_tokens),
            "raw_size_bytes": len(decoded_bytes),
            "payload_bits": metadata["payload_bit_count"],
            "serialized_size_bytes": row["stream_path"].stat().st_size,
            "token_round_trip": True,
            "byte_round_trip": True,
            "status": "measured",
        })

    result = {
        "metadata": {
            "format": "QACS version 1",
            "model_name": args.model_name,
            "byte_budget": args.byte_budget,
            "context_length": args.context_length,
            "retain_tokens": args.retain_tokens,
            "frequency_total": parsed[0]["frequency_total"],
            "statesize": parsed[0]["statesize"],
            "model_weights_shared": True,
        },
        "runs": output_rows,
    }
    _write_atomic(
        args.output_dir / "results.json",
        json.dumps(result, indent=2).encode("utf-8"),
    )
    print(f"Finalized {len(output_rows)} Qwen bars in {args.output_dir}")


if __name__ == "__main__":
    main()
