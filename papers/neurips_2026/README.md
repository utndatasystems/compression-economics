# NeurIPS 2026 paper package

This directory contains everything specific to the NeurIPS ML for Systems paper.
Reusable compression implementations remain in `src/` and general command-line
tools remain in `scripts/`.

## Layout

- `manuscript/`: LaTeX sources, included tables, and included figures.
- `experiments/`: reproducible entry points for every paper experiment.
- `evaluation/`: audited loaders, plotting code, notebook, and focused tests.
- `../../artifacts/papers/neurips-2026/`: raw runs, finalized streams, studies,
  and derived audit data. See its README and `manifest.json` for the mapping from
  paper conditions to files.

## Reproduce experiments

Run commands from the repository root:

```bash
bash papers/neurips_2026/experiments/run_all.sh all
```

Individual stages are `fertility`, `natural`, `attacks`, and `search`. Set
`DRY_RUN=1` to inspect commands without executing them.

## Rebuild evaluation outputs

```bash
.venv/bin/python papers/neurips_2026/evaluation/plot_prediction_difficulty.py
jupyter notebook papers/neurips_2026/evaluation/crucial_figures.ipynb
```

Included figures are written to `manuscript/plots/`. Rebuildable exploratory
figures and audit tables are written below the paper artifact root.

## Build the manuscript

```bash
cd papers/neurips_2026/manuscript
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```
