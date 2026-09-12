#!/usr/bin/env python3
"""Run the CIDR row-versus-column benchmark from a TOML specification."""

from __future__ import annotations

import argparse
from pathlib import Path
import statistics

from src.benchmark.config import load_sweep_config
from src.benchmark.runner import run_sweep


DEFAULT_CONFIG = Path(
    "papers/cidr_2027/experiments/configs/row_column_smoke.toml"
)


def _arguments() -> argparse.Namespace:
    """Parse the sweep specification and optional output override."""
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark reversible row-major and column-major table layouts "
            "with independently framed CPU codec blocks."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"TOML sweep definition (default: {DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help="Override the artifact root, useful for tests or temporary runs.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the requested sweep and print one median summary per condition."""
    args = _arguments()
    config = load_sweep_config(args.config)
    aggregate_rows, block_rows = run_sweep(
        config, output_root=args.output_root
    )

    print(f"Completed {config.experiment_id}")
    print(f"Aggregate rows: {len(aggregate_rows)}; block rows: {len(block_rows)}")
    print()
    print(
        f"{'layout':<14} {'codec':<10} {'mode':<24} {'factor':>8} "
        f"{'comp MiB/s':>12} {'decomp MiB/s':>14} {'blocks':>7} {'reps':>5}"
    )
    groups = {}
    for row in aggregate_rows:
        key = (
            row["condition"]["layout"],
            row["condition"]["codec"],
            row["accounting_mode"],
        )
        groups.setdefault(key, []).append(row)
    for (layout, codec, accounting_mode), rows in groups.items():
        factor = statistics.median(
            row["accounting"]["compression_factor"] for row in rows
        )
        compression_speed = statistics.median(
            row["timings"]["compression_mib_per_second"] for row in rows
        )
        decompression_speed = statistics.median(
            row["timings"]["decompression_mib_per_second"] for row in rows
        )
        print(
            f"{layout:<14} {codec:<10} {accounting_mode:<24} "
            f"{factor:>8.3f} {compression_speed:>12.2f} "
            f"{decompression_speed:>14.2f} "
            f"{rows[0]['block_compression_factor_summary']['count']:>7} "
            f"{len(rows):>5}"
        )


if __name__ == "__main__":
    main()
