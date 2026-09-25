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
  --config research/papers/cidr_2027/experiments/configs/row_column_smoke.toml
```

The runner rejects unknown fields, expands the matrix deterministically, derives
content-addressed condition IDs, persists the measured archives, and records the
fully resolved configuration in every output row.


## Parallel arithmetic coding

Install the optional compiled backend with `uv sync --extra parallel`. Text
compression selects it with `--encoding AC_MULTISTREAM --ac-backend
numba_parallel --ac-threads 4`; decoding uses the same portable MSAC archive
regardless of the encoder backend. The default remains `python`.

Run a matched probability-trace comparison with
`research/papers/cidr_2027/experiments/evaluate_coders.py --trace TRACE.npz
--coder AC_MULTISTREAM --ac-streams 4 --ac-backend numba_parallel
--ac-threads 4`. Its total encode time includes Python probability
quantization and archive framing. The reported `range_encode_seconds` excludes
model inference but includes interval buffering and the worker phase. Its
`tracemalloc` profiler can affect that timing.

For a clean worker-scaling measurement of the same compiled range kernel, run
`.venv/bin/python research/papers/cidr_2027/experiments/benchmark_parallel_ac.py
--workers 1 2 4`. This uses fixed integer intervals, warms the JIT before
timing, and checks output hashes across worker counts. It measures range
coding only; it does not predict end-to-end text compression speed.
