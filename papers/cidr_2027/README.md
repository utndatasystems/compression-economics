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
.venv/bin/python -m scripts.run_cidr_benchmark
```

The default smoke sweep uses explicit raw-byte identity and zstd pipelines. It
writes aggregate and per-block JSONL records plus every validated archive under
`artifacts/papers/cidr-2027/runs/row-column-smoke/`. The larger raw-byte pilot
uses the same runner with:

```bash
.venv/bin/python -m scripts.run_cidr_benchmark \
  --config papers/cidr_2027/experiments/configs/row_column_full.toml
```

Predictive/tokenized pipelines are deliberately not listed yet: they will be
added as explicit valid pipelines after their format and accounting adapters
exist.
