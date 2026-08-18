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
