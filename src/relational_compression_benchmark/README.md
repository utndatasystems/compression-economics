# Relational compression benchmark core

This package implements the reusable, CPU-only foundation for the CIDR sweep:

- `config.py`: strict TOML loading and explicit pipeline definitions.
- `datasets.py`: checksum-pinned real datasets and deterministic split sampling.
- `table.py`: typed tables and the seeded synthetic generator.
- `serialization.py`: canonical row-major and column-major bytes.
- `tokenization.py`: pinned local tokenizers, a reversible byte bridge, and
  fixed-width token-ID packing.
- `block_storage_format.py`: independent checksummed blocks, raw/token representations,
  zstd/identity codecs, persisted indexes, and exact byte accounting.
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

Token archives map each canonical byte block bijectively through Latin-1 before
calling the tokenizer. Token IDs use the smallest whole-byte big-endian width
covering the vocabulary (three bytes for the pinned Qwen tokenizer). The archive
stores the tokenizer name, immutable revision, vocabulary size, mapping version,
and packing width. Tests and benchmark runs load tokenizer assets locally only.

Prepare the pinned tokenizer explicitly if it is not already cached:

```bash
.venv/bin/python main.py cidr prepare-tokenizer
```

Prepare the official IMDb `title.basics` snapshot explicitly:

```bash
.venv/bin/python main.py cidr prepare-imdb
```

The download remains under ignored `data/`. Its exact SHA-256 is pinned in the
IMDb sweep configurations. Stable hashing of `tconst` produces disjoint
`tuning` and `evaluation` splits, and a streaming reservoir samples each full split. IMDb permits these datasets only for personal
and non-commercial use; consult the URL printed in the download manifest.

Run the smoke sweep from the repository root:

```bash
.venv/bin/python main.py cidr benchmark
```

Select the real-world smoke sweep with:

```bash
.venv/bin/python main.py cidr benchmark \
  --config papers/cidr_2027/experiments/configs/imdb_smoke.toml
```
