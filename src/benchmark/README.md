# Relational compression benchmark core

This package implements the reusable, CPU-only foundation for the CIDR sweep:

- `config.py`: strict TOML loading and explicit pipeline definitions.
- `table.py`: typed tables and the seeded synthetic generator.
- `serialization.py`: canonical row-major and column-major bytes.
- `archive.py`: independent checksummed blocks, zstd/identity codecs, persisted
  indexes, decoding, and exact byte accounting.
- `runner.py`: round-trip enforcement, timings, environment capture, block
  summaries, bootstrap intervals, and JSONL output.

There are two intentionally separate byte contracts:

1. **Canonical source bytes** contain the physical-layout tag, table shape,
   schema once, and every typed value.
2. **Stored archive bytes** contain an archive header and schema, block frames,
   codec payloads, and an index.

`source_bytes` measures the first contract. Every charged accounting component
sums exactly to the length of the second. A value is never split between blocks,
and every archive is read back from disk and decoded before its result is valid.

Run the smoke sweep from the repository root:

```bash
.venv/bin/python -m scripts.run_cidr_benchmark
```
