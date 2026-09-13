# CIDR 2027 experiments

Keep immutable, machine-independent sweep definitions in `configs/`. A runner
must write resolved configurations and raw records beneath
`artifacts/papers/cidr-2027/runs/`; it must never modify a committed config.

Configuration names distinguish the fast CPU validation slice from the full
matrix:

- `configs/row_column_smoke.toml`: one small deterministic table, both layouts,
  and inexpensive codec paths.
- `configs/row_column_full.toml`: the intended row-versus-column pilot axes.

Run the smoke definition from the repository root:

```bash
.venv/bin/python main.py cidr benchmark \
  --config papers/cidr_2027/experiments/configs/row_column_smoke.toml
```

The runner rejects unknown fields, expands the matrix deterministically, derives
content-addressed condition IDs, persists the measured archives, and records the
fully resolved configuration in every output row.
