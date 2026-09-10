# NeurIPS 2026 paper package

This directory contains everything specific to the NeurIPS ML for Systems paper.
Reusable compression implementations remain in `src/` and general command-line
tools remain in `scripts/`.

## Layout

- `manuscript/`: LaTeX sources, included tables, and included figures.
- `supplementary/`: metadata and locked environment for the anonymous code artifact.
- `experiments/`: reproducible entry points for every paper experiment.
- `evaluation/`: audited loaders, plotting code, notebook, and focused tests.
- `../../artifacts/papers/neurips-2026/`: raw runs, finalized streams, studies,
  and derived audit data. See its README and `manifest.json` for the mapping from
  paper conditions to files.

## Reproduce experiments

The reported runs used Python 3.12.3 and the exact dependency resolution in
`uv.lock`. From the repository root, create that environment with:

```bash
uv sync --frozen
```

The model is `Qwen/Qwen2.5-0.5B` at Hugging Face revision
`060db6499f32faf8b98477b0a26969ef7d8b9987`. Download that revision into the
cache used by the implementation, pin the cache's `main` reference to it, and
run offline so subsequent runs cannot silently follow a changed model branch:

```bash
.venv/bin/hf download Qwen/Qwen2.5-0.5B \
  --revision 060db6499f32faf8b98477b0a26969ef7d8b9987 \
  --cache-dir .cache
mkdir -p .cache/models--Qwen--Qwen2.5-0.5B/refs
printf '%s' 060db6499f32faf8b98477b0a26969ef7d8b9987 \
  > .cache/models--Qwen--Qwen2.5-0.5B/refs/main
```

Download and verify the 100,000,000-byte text8 input:

```bash
wget -O /tmp/text8.zip http://mattmahoney.net/dc/text8.zip
unzip -o /tmp/text8.zip -d data
sha256sum data/text8
```

The expected SHA-256 digest is
`6e890197040d37d85beb962ae1f041ff1d9a9ca8d20c7d99c85027eebf51dca7`.

Run the experiment stages with the pinned local model:

```bash
HF_HUB_OFFLINE=1 bash papers/neurips_2026/experiments/run_all.sh fertility
HF_HUB_OFFLINE=1 bash papers/neurips_2026/experiments/run_all.sh natural
HF_HUB_OFFLINE=1 bash papers/neurips_2026/experiments/run_all.sh attacks
```

The `natural` stage is the auxiliary 100,000-token text8 evaluation. The main
10,000-token text8 and random-printable rows are generated as part of the
`attacks` stage so they use the same source-size budget as the adversarial rows.
The expensive full-vocabulary MaxSurprisal/Byte condition has its separate
paper budget of 1,024 tokens. Set `DRY_RUN=1` to inspect every expanded command
without executing it. The optional `search` stage is not used by the current
manuscript.

The FSST bars use the official implementation at commit
`e638d4cf8c26129d73c242a4127b42b975de5b63`. Build it before executing the
figure notebook:

```bash
git clone https://github.com/cwida/fsst.git /tmp/compression-economics-fsst
git -C /tmp/compression-economics-fsst checkout \
  e638d4cf8c26129d73c242a4127b42b975de5b63
cmake -S /tmp/compression-economics-fsst \
  -B /tmp/compression-economics-fsst/build
cmake --build /tmp/compression-economics-fsst/build --parallel
```

## Rebuild evaluation outputs

```bash
HF_HUB_OFFLINE=1 .venv/bin/python \
  papers/neurips_2026/evaluation/finalize_qwen_bars.py
HF_HUB_OFFLINE=1 .venv/bin/python \
  papers/neurips_2026/evaluation/quantization_sensitivity.py
.venv/bin/python papers/neurips_2026/evaluation/plot_prediction_difficulty.py
HF_HUB_OFFLINE=1 \
FSST_EXECUTABLE=/tmp/compression-economics-fsst/build/fsst \
  .venv/bin/jupyter nbconvert \
  --execute --to notebook --inplace \
  papers/neurips_2026/evaluation/crucial_figures.ipynb
```

Included figures are written to `manuscript/plots/`. Rebuildable exploratory
figures and audit tables are written below the paper artifact root.
`../../artifacts/papers/neurips-2026/manifest.json` maps every manuscript
condition to its raw input, finalized round-trip-verified stream, and derived
table or figure input.

## Primary execution platform

The primary experiments ran on Ubuntu 24.04.1 LTS on one Intel Xeon Gold 5318Y
CPU (24 physical cores, 48 hardware threads) with 125 GiB RAM. No GPU was
available on this machine. The installed key packages were PyTorch 2.10.0,
Transformers 5.3.0, tokenizers 0.22.2, NumPy 2.4.3, and Brotli 1.2.0; `uv.lock`
is authoritative for the complete environment.

## Build the manuscript

```bash
cd papers/neurips_2026/manuscript
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```
