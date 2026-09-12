# CIDR 2027 artifacts

This is the stable root for generated CIDR 2027 inputs and outputs. Generated
files are ignored by Git; `README.md` and `manifest.json` are tracked.

Planned layout:

- `datasets/`: prepared/checksummed table snapshots and source manifests.
- `runs/`: resolved configurations, streams, aggregate JSONL, and block JSONL.
- `derived/`: regenerated summaries, figures, tables, and findings reports.

Every result used by an analysis or manuscript figure must be addressable from
`manifest.json`. Large datasets and model weights stay outside this tree and are
identified by pinned IDs and checksums.

## Current smoke run

Reproduce the CPU-only raw-byte slice from the repository root:

```bash
.venv/bin/python -m scripts.run_cidr_benchmark
```

The generated `runs/row-column-smoke/` directory contains the resolved config,
dataset manifest, aggregate JSONL, block JSONL, and all measured archives. The
8,192-row run produced 11 blocks per condition and validated every archive by
reading it back and reconstructing both the canonical bytes and typed table.

At zstd level 3, row-major storage reached a 10.593 compression factor and
column-major storage reached 15.149. These are deterministic size results for a
controlled synthetic table, not a general claim about relational workloads.
