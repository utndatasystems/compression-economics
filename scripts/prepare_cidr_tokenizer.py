#!/usr/bin/env python3
"""Explicitly download the pinned tokenizer used by CIDR token baselines."""

from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_NAME = "Qwen/Qwen2.5-0.5B"
DEFAULT_REVISION = "060db6499f32faf8b98477b0a26969ef7d8b9987"


def _arguments() -> argparse.Namespace:
    """Parse tokenizer identity and cache destination."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default=DEFAULT_NAME)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache"))
    return parser.parse_args()


def main() -> None:
    """Download tokenizer-only assets after an explicit user invocation."""
    from transformers import AutoTokenizer

    args = _arguments()
    tokenizer = AutoTokenizer.from_pretrained(
        args.name,
        revision=args.revision,
        cache_dir=args.cache_dir,
    )
    print(
        f"Prepared {args.name}@{args.revision} "
        f"({len(tokenizer):,} vocabulary entries) in {args.cache_dir}"
    )


if __name__ == "__main__":
    main()
