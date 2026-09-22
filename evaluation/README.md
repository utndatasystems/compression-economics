# Evaluation

This directory contains code and notebooks that consume experiment results.
Paper-specific runs live under `research/papers/<paper>/experiments/`;
generated tables and figures belong in the ignored `artifacts/` directory.

- `loaders.py`: shared result-loading and normalization helpers.
- `baselines/`: implementations of non-LLM comparison methods.
- `plots/`: reproducible plotting entry points and shared plotting code.
- `notebooks/`: exploratory analyses not owned by a paper package.
- `reference_data/`: small, version-controlled inputs used by evaluations.

Run Python entry points from the repository root. Notebook paths are organized by
paper and may need their data path set explicitly when opened interactively.

## Paper-specific evaluation

The current NeurIPS evaluation lives in `research/papers/neurips_2026/evaluation/`. Its
raw and finalized inputs are indexed by
`artifacts/papers/neurips-2026/manifest.json`.

The CIDR 2027 evaluation scaffold lives in `research/papers/cidr_2027/evaluation/` and
uses `artifacts/papers/cidr-2027/manifest.json` as the stable index for raw and
derived outputs.
