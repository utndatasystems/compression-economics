# CIDR 2027 paper package

This package owns the version-controlled definitions and analyses for the
row-versus-column compression study. Reusable implementations belong in `src/`
and reusable command-line entry points belong in `scripts/`.

## Layout

- `experiments/configs/`: declarative smoke and full-pilot sweep definitions.
- `evaluation/`: loaders, analyses, plots, and focused tests added as the pilot
  is implemented.
- `manuscript/`: paper text plus generated-figure and generated-table inclusion
  points.
- `../../docs/cidr_2027_experiment_plan.md`: inventory, gaps, accounting
  contract, and staged roadmap.
- `../../artifacts/papers/cidr-2027/`: ignored raw/generated outputs with a
  tracked manifest and README.

## Run the CPU slice

From the repository root:

```bash
.venv/bin/python main.py cidr benchmark
```

The default smoke sweep uses explicit raw-byte and Qwen token-ID representations,
each stored directly and through zstd. It writes aggregate and per-block JSONL
records plus every validated archive under
`artifacts/papers/cidr-2027/runs/row-column-smoke/`. If the pinned tokenizer is
not cached, prepare it through an explicit network-enabled command:

```bash
.venv/bin/python main.py cidr prepare-tokenizer
```

The larger pilot uses the same runner with:

```bash
.venv/bin/python main.py cidr benchmark \
  --config papers/cidr_2027/experiments/configs/row_column_full.toml
```

## Run the IMDb slice

Download the official, non-commercial `title.basics` source and then run its
checksum-pinned smoke configuration:

```bash
.venv/bin/python main.py cidr prepare-imdb
.venv/bin/python main.py cidr benchmark \
  --config papers/cidr_2027/experiments/configs/imdb_smoke.toml
```

Use `imdb_full.toml` for the 100,000-row, two-block-size pilot. The source file
and generated run remain ignored; the tracked configuration records the exact
source checksum and deterministic split policy.

Predictive Qwen and arithmetic-coding pipelines remain deliberately absent.
They will be added after their probability, restart-state, and decoder-side
causality contracts are implemented.
