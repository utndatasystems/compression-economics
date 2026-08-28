# NeurIPS 2026 artifacts

This directory mirrors the experiments and result groupings used by the paper.
Raw artifacts are ignored by Git; this README and `manifest.json` are tracked.

## Primary runs

- `runs/auxiliary/printable-ascii/n10000/results.json`: the 10,000-token
  natural-text and random-printable conditions used by the main comparison.
- `runs/natural-text/text8/n100000/`: auxiliary 100,000-token text8 runs.
- `runs/attacks/minprob/full-vocabulary/n10000/`: full-vocabulary MinProb runs.
- `runs/attacks/max-surprisal-per-byte/full-vocabulary/`: byte-aware attacks.
- `runs/ablations/one-byte-utf8/n10000/`: one-byte UTF-8 ablation.
- `runs/auxiliary/printable-ascii/n10000/`: printable-alphabet source bundle.

## Final and derived data

- `finalized/`: decoder-verified Qwen arithmetic streams used by paper bars.
- `studies/tokenizer-fertility/`: tokenizer fertility experiment.
- `studies/max-surprisal-length/`: N=32, 64, 128, and 1024 length study.
- `derived/audit-tables/`: rebuildable CSV inputs for paper figures.
- `derived/exploratory-figures/`: figures not directly included in the paper.

Every paper condition is mapped to its raw and finalized source in
`manifest.json`. Legacy artifact directories are retained as copy-first backups
for other local branches.
