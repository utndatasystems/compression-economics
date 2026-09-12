# Experiments

This directory contains version-controlled definitions that launch or configure
compression experiments. It must not contain generated results, model weights,
logs, or figures; those belong under `artifacts/`.

Run commands from the repository root so paths remain reproducible. For example:

```bash
bash experiments/sweeps/model_encoding_sweep.sh
```

Add reusable sweeps under `sweeps/`. Give each sweep a descriptive name and keep
machine-specific paths out of committed files.

Paper-specific sweep definitions live with their paper package. The CIDR 2027
row-versus-column configurations are under
`papers/cidr_2027/experiments/configs/`; reusable runners and benchmark
components should remain in the repository-level `scripts/` and `src/`
directories.
