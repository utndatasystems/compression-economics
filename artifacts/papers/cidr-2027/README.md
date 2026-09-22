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

Reproduce the CPU-only raw-byte and token-ID slice from the repository root:

```bash
.venv/bin/python -m scripts.run_cidr_benchmark
```

The generated `runs/row-column-smoke/` directory contains the resolved config,
dataset manifest, aggregate JSONL, block JSONL, and all measured archives. The
8,192-row run produced 11 blocks per condition and validated every archive by
reading it back and reconstructing both the canonical bytes and typed table.

For shared dependencies, raw zstd reached compression factors of 10.586 for
row-major and 15.135 for column-major. Plain three-byte token IDs reached 0.471
for both layouts; token IDs followed by zstd reached 9.460 and 13.256,
respectively. Charging the 7,038,873-byte tokenizer asset pair makes every token
condition expansionary on this small table. These deterministic size results
describe a controlled synthetic table, not relational workloads generally.

## IMDb smoke run

Prepare the checksum-pinned, official non-commercial dataset and reproduce the
real-world slice with:

```bash
.venv/bin/python -m scripts.prepare_cidr_imdb
.venv/bin/python -m scripts.run_cidr_benchmark \
  --config research/papers/cidr_2027/experiments/configs/imdb_smoke.toml
```

The 8,192-row evaluation sample produced 14 blocks per condition and validated
all archives. Under shared-model accounting, raw zstd reached factors of 3.647
for row-major and 3.282 for column-major. Plain three-byte token IDs reached
0.542 for both layouts; token IDs followed by zstd reached 3.058 and 2.898.
Charging the 7,038,873-byte tokenizer assets makes the token conditions
expansionary at this sample size.
