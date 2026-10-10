#!/usr/bin/env python3
"""Validate and combine FineZip, LLMZip, and Nacrith result JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("finezip", type=Path)
    parser.add_argument("llmzip", type=Path)
    parser.add_argument("nacrith", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    rows = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in (args.finezip, args.llmzip, args.nacrith)
    ]
    if {row["codec"] for row in rows} != {"finezip", "llmzip", "nacrith"}:
        parser.error(
            "inputs must contain one FineZip, one LLMZip, and one Nacrith result"
        )
    hashes = {row["input"]["sha256"] for row in rows}
    sizes = {row["input"]["prefix_bytes"] for row in rows}
    if len(hashes) != 1 or len(sizes) != 1:
        parser.error("baseline inputs do not match")
    if not all(row["validation"]["exact_byte_recovery"] for row in rows):
        parser.error("a baseline failed exact byte recovery")
    summary = {
        "schema_version": 1,
        "experiment": "coding-survey LLM baselines",
        "input": {"bytes": sizes.pop(), "sha256": hashes.pop()},
        "accounting": "complete per-file archive; shared model/tokenizer weights excluded",
        "codecs": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    for row in sorted(rows, key=lambda item: item["codec"]):
        print(
            f'{row["codec"]:8s}  {row["archive"]["bytes"]:9d} bytes  '
            f'{row["archive"]["bits_per_byte"]:.4f} bpb  '
            f'{row["timing"]["compression_mib_per_second"]:.5f} MiB/s encode  '
            f'{row["timing"]["decompression_mib_per_second"]:.5f} MiB/s decode'
        )


if __name__ == "__main__":
    main()
