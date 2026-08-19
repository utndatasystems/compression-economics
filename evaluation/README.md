# Evaluation

This directory contains code and notebooks that consume experiment results.
Launching runs belongs in `experiments/`; generated tables and figures belong in
the ignored `artifacts/` directory.

- `loaders.py`: shared result-loading and normalization helpers.
- `baselines/`: implementations of non-LLM comparison methods.
- `plots/`: reproducible plotting entry points and shared plotting code.
- `notebooks/`: paper-specific and exploratory analyses.
- `reference_data/`: small, version-controlled inputs used by evaluations.

Run Python entry points from the repository root. Notebook paths are organized by
paper and may need their data path set explicitly when opened interactively.

## Matched adversarial comparison

The matched 1,000-token text8 comparison, reproduction commands, table, and
publication plot are integrated into
`notebooks/neurips/adversarial_worst_case.ipynb`.
