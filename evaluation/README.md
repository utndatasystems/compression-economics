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

The auditable NeurIPS figures and paper-specific analysis are integrated into
`notebooks/neurips/crucial_figures.ipynb`, with shared plotting and validation
helpers in `crucial_figures.py`.
