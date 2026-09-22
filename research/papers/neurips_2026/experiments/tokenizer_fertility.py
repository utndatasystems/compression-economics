#!/usr/bin/env python3
"""Compare character, trained BPE, and pretrained Qwen tokenization on text8."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

DEFAULT_TRAIN_BYTES = 90_000_000
DEFAULT_EVAL_BYTES = 5_000_000
DEFAULT_QWEN_TOKENIZER = "Qwen/Qwen2.5-0.5B"


@dataclass(frozen=True)
class FertilityResult:
    tokenizer: str
    tokenizer_kind: str
    vocabulary_size: int
    token_count: int
    unique_tokens: int
    word_count: int
    character_count: int
    utf8_bytes: int
    fertility_tokens_per_word: float
    tokens_per_character: float
    tokens_per_utf8_byte: float
    characters_per_token: float
    utf8_bytes_per_token: float
    encoding_seconds: float
    round_trip_exact: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/text8"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/tokenizer-fertility"))
    parser.add_argument("--train-bytes", type=int, default=DEFAULT_TRAIN_BYTES)
    parser.add_argument("--eval-bytes", type=int, default=DEFAULT_EVAL_BYTES)
    parser.add_argument("--eval-offset", type=int, default=None,
                        help="Byte offset of evaluation data (default: final --eval-bytes bytes).")
    parser.add_argument("--bpe-vocab-size", type=int, default=32_000)
    parser.add_argument("--bpe-min-frequency", type=int, default=2)
    parser.add_argument("--training-chunk-bytes", type=int, default=1_000_000)
    parser.add_argument("--qwen-tokenizer", default=DEFAULT_QWEN_TOKENIZER)
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache"))
    return parser.parse_args()


def read_ascii_region(path: Path, offset: int, length: int) -> str:
    with path.open("rb") as source:
        source.seek(offset)
        data = source.read(length)
    if len(data) != length:
        raise ValueError(f"Requested {length:,} bytes at offset {offset:,}, got {len(data):,}")
    try:
        return data.decode("ascii")
    except UnicodeDecodeError as error:
        raise ValueError("The character baseline requires ASCII input") from error


def iter_ascii_chunks(path: Path, offset: int, length: int,
                      chunk_bytes: int) -> Iterable[str]:
    if chunk_bytes <= 0:
        raise ValueError("training chunk size must be positive")
    remaining = length
    with path.open("rb") as source:
        source.seek(offset)
        while remaining:
            data = source.read(min(chunk_bytes, remaining))
            if not data:
                raise ValueError(f"Input ended with {remaining:,} training bytes still requested")
            remaining -= len(data)
            try:
                yield data.decode("ascii")
            except UnicodeDecodeError as error:
                raise ValueError("The BPE training region must be ASCII") from error


def sha256_region(path: Path, offset: int, length: int,
                  chunk_bytes: int = 1_000_000) -> str:
    digest = hashlib.sha256()
    remaining = length
    with path.open("rb") as source:
        source.seek(offset)
        while remaining:
            data = source.read(min(chunk_bytes, remaining))
            if not data:
                raise ValueError(f"Input ended with {remaining:,} bytes still requested")
            digest.update(data)
            remaining -= len(data)
    return digest.hexdigest()


def train_byte_level_bpe(path: Path, *, train_bytes: int, vocab_size: int,
                         min_frequency: int, chunk_bytes: int):
    try:
        from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("Install project dependencies to train the BPE tokenizer") from error
    if vocab_size < 256:
        raise ValueError("BPE vocabulary size must be at least 256 for the byte alphabet")
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=["[UNK]"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    tokenizer.train_from_iterator(
        iter_ascii_chunks(path, 0, train_bytes, chunk_bytes), trainer=trainer
    )
    return tokenizer


def measure_token_ids(*, name: str, kind: str, vocabulary_size: int,
                      token_ids: Sequence[int], text: str,
                      encoding_seconds: float, decoded_text: str) -> FertilityResult:
    words = len(text.split())
    characters = len(text)
    byte_count = len(text.encode("utf-8"))
    tokens = len(token_ids)
    if not words or not characters or not byte_count or not tokens:
        raise ValueError("Evaluation text and token stream must be non-empty")
    return FertilityResult(
        tokenizer=name, tokenizer_kind=kind, vocabulary_size=vocabulary_size,
        token_count=tokens, unique_tokens=len(set(token_ids)), word_count=words,
        character_count=characters, utf8_bytes=byte_count,
        fertility_tokens_per_word=tokens / words,
        tokens_per_character=tokens / characters,
        tokens_per_utf8_byte=tokens / byte_count,
        characters_per_token=characters / tokens,
        utf8_bytes_per_token=byte_count / tokens,
        encoding_seconds=encoding_seconds, round_trip_exact=decoded_text == text,
    )


def measure_ascii(text: str) -> FertilityResult:
    start = time.perf_counter()
    try:
        token_ids = list(text.encode("ascii"))
    except UnicodeEncodeError as error:
        raise ValueError("The character baseline requires ASCII input") from error
    return measure_token_ids(
        name="ASCII (one byte per token)", kind="character", vocabulary_size=128,
        token_ids=token_ids, text=text, encoding_seconds=time.perf_counter() - start,
        decoded_text=bytes(token_ids).decode("ascii"),
    )


def measure_bpe(tokenizer, text: str) -> FertilityResult:
    start = time.perf_counter()
    encoding = tokenizer.encode(text)
    return measure_token_ids(
        name="BPE (trained on text8)", kind="byte-level BPE",
        vocabulary_size=tokenizer.get_vocab_size(), token_ids=encoding.ids, text=text,
        encoding_seconds=time.perf_counter() - start,
        decoded_text=tokenizer.decode(encoding.ids),
    )


def measure_qwen(tokenizer, text: str, model_name: str) -> FertilityResult:
    start = time.perf_counter()
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    elapsed = time.perf_counter() - start
    decoded = tokenizer.decode(token_ids, skip_special_tokens=False,
                               clean_up_tokenization_spaces=False)
    return measure_token_ids(
        name=model_name, kind="pretrained Qwen tokenizer",
        vocabulary_size=len(tokenizer), token_ids=token_ids, text=text,
        encoding_seconds=elapsed, decoded_text=decoded,
    )


def write_outputs(output_dir: Path, metadata: dict,
                  results: Sequence[FertilityResult]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [asdict(result) for result in results]
    with (output_dir / "results.json").open("w", encoding="utf-8") as target:
        json.dump({"metadata": metadata, "results": rows}, target, indent=2)
        target.write("\n")
    with (output_dir / "results.csv").open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def print_results(results: Sequence[FertilityResult]) -> None:
    print("\nTokenizer fertility on held-out text8")
    print(f"{'tokenizer':34} {'vocab':>9} {'tokens':>12} {'tokens/word':>12} {'chars/token':>12}")
    for result in results:
        print(f"{result.tokenizer[:34]:34} {result.vocabulary_size:9,d} "
              f"{result.token_count:12,d} {result.fertility_tokens_per_word:12.4f} "
              f"{result.characters_per_token:12.4f}")


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"text8 input not found: {input_path}")
    file_size = input_path.stat().st_size
    eval_offset = file_size - args.eval_bytes if args.eval_offset is None else args.eval_offset
    if args.train_bytes <= 0 or args.eval_bytes <= 0 or eval_offset < 0:
        raise ValueError("Training/evaluation sizes and evaluation offset must be positive")
    if args.train_bytes > eval_offset:
        raise ValueError("BPE training and evaluation regions overlap")
    if eval_offset + args.eval_bytes > file_size:
        raise ValueError("Evaluation region extends beyond the input file")

    print(f"Training byte-level BPE on {args.train_bytes:,} bytes...")
    bpe = train_byte_level_bpe(
        input_path, train_bytes=args.train_bytes, vocab_size=args.bpe_vocab_size,
        min_frequency=args.bpe_min_frequency, chunk_bytes=args.training_chunk_bytes,
    )
    evaluation_text = read_ascii_region(input_path, eval_offset, args.eval_bytes)
    print(f"Loading Qwen tokenizer {args.qwen_tokenizer}...")
    try:
        from transformers import AutoTokenizer
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("Install project dependencies to load the Qwen tokenizer") from error
    qwen = AutoTokenizer.from_pretrained(args.qwen_tokenizer,
                                         cache_dir=args.cache_dir, use_fast=True)
    results = [measure_ascii(evaluation_text), measure_bpe(bpe, evaluation_text),
               measure_qwen(qwen, evaluation_text, args.qwen_tokenizer)]
    if not all(result.round_trip_exact for result in results):
        failures = [result.tokenizer for result in results if not result.round_trip_exact]
        raise RuntimeError(f"Non-lossless tokenization for: {', '.join(failures)}")

    output_dir = args.output_dir.resolve()
    bpe_path = output_dir / "text8-bpe.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    bpe.save(str(bpe_path))
    metadata = {
        "input": str(input_path), "input_bytes": file_size,
        "training": {"offset": 0, "bytes": args.train_bytes,
                     "sha256": sha256_region(input_path, 0, args.train_bytes)},
        "evaluation": {"offset": eval_offset, "bytes": args.eval_bytes,
                       "sha256": hashlib.sha256(evaluation_text.encode("ascii")).hexdigest()},
        "bpe": {"requested_vocab_size": args.bpe_vocab_size,
                "actual_vocab_size": bpe.get_vocab_size(),
                "min_frequency": args.bpe_min_frequency, "artifact": str(bpe_path)},
        "qwen_tokenizer": args.qwen_tokenizer,
        "fertility_definition": "number of corpus tokens / whitespace-delimited words",
    }
    write_outputs(output_dir, metadata, results)
    print_results(results)
    print(f"\nWrote {output_dir / 'results.json'} and {output_dir / 'results.csv'}")


if __name__ == "__main__":
    main()
