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


### Small GPT/Qwen model comparison

`compare_coders_model.py` loads a locally cached causal model in float32 and
scores a fixed text8 prefix once. It builds one shared token alphabet from that
prefix, then compares AC, MSAC, ANS, PMATIC, Huffman rank, and bit-packed rank
on the same saved probability trace. Run from the repository root:

```bash
.venv/bin/python research/papers/cidr_2027/experiments/compare_coders_model.py
```

The default is GPT-2, 512 target tokens, and three timing repetitions.
`--model Qwen/Qwen2.5-0.5B` uses the cached small Qwen model instead.
Results, the exact trace, and the serialized Roaring token mask are saved under
`artifacts/papers/cidr-2027/coder-comparison/`.

The mask uses token IDs observed in the evaluated text, so its size is reported
separately from coder archives. The saved probability trace provides decoder
distributions; these coder archives are not standalone compressed text files.
Coder times exclude model inference and memory tracing. PMATIC's separate
decoder-probability perturbation stress test is omitted from this comparison.
