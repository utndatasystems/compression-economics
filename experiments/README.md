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

## Plain-text small-model sweep

The small-model runner accepts repeated context and inference-batch arguments
and writes canonical schema-v1 records. N-gram contexts remain fixed at one and
two symbols; every distinct neural architecture is expanded across the supplied
contexts. Training batch size is controlled separately.

```bash
.venv/bin/python scripts/plain_text_compression/evaluate_global_mask_models.py \
  --training-mode disjoint \
  --context-length 4 --context-length 16 --context-length 64 \
  --batch-size 1 --batch-size 8 --batch-size 32 --batch-size 128 \
  --training-batch-size 128 \
  --warmups 1 --repetitions 3 \
  --output-dir artifacts/papers/cidr-2027/model-survey/context-batch-sweep
```

Each measured repetition has its own stream artifact. Repetitions of the same
model/context/batch condition share a `condition_id` and differ by their
top-level `repetition` value.
