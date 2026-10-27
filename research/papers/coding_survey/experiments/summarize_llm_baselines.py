#!/usr/bin/env python3
"""Validate and combine FineZip, LLMZip, and Nacrith result JSON."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.papers.coding_survey.experiments.baseline_adapter import (
    combine_baseline_results,
    write_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("finezip", type=Path)
    parser.add_argument("llmzip", type=Path)
    parser.add_argument("nacrith", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    try:
        summary = combine_baseline_results(
            (args.finezip, args.llmzip, args.nacrith)
        )
    except (KeyError, ValueError) as error:
        parser.error(str(error))
    write_json(args.output, summary)
    rows = summary["codecs"]
    for row in sorted(rows, key=lambda item: item["codec"]):
        print(
            f'{row["codec"]:8s}  {row["archive"]["bytes"]:9d} bytes  '
            f'{row["archive"]["bits_per_byte"]:.4f} bpb  '
            f'{row["timing"]["compression_mib_per_second"]:.5f} MiB/s encode  '
            f'{row["timing"]["decompression_mib_per_second"]:.5f} MiB/s decode'
        )


if __name__ == "__main__":
    main()
