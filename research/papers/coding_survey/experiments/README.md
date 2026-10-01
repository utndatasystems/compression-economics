# Coding survey experiments

See the [experiment tracker](EXPERIMENT_PLAN.md) for planned experiments,
the completed Qwen/text8 CPU pilot, its GPU repeat, and ablations.

Keep survey-specific benchmark definitions and analysis here. Reusable coder
implementations and trace measurements live in `src/`; raw traces, archives,
profiles, and result JSON belong under the ignored
`artifacts/papers/coding-survey/` directory.

## First slice: independent AC streams

The shared frozen-trace runner can compare `AC` and `AC_MULTISTREAM` without
rerunning the predictor. Both variants use the same target symbols,
probabilities, and `build_cumul` frequency rule. The multistream variant divides
the trace into contiguous, independently decodable streams. Its new MSAC v1
payload has big-endian fields, explicit symbol and bit counts, and a CRC32 for
each stream. It does not contain model contexts or seed tokens, so this is a
coder benchmark rather than an end-to-end text archive.

From the repository root:

```bash
.venv/bin/python research/papers/cidr_2027/experiments/evaluate_coders.py \
  --synthetic-symbols 10000 --alphabet-size 256 \
  --coder AC --coder AC_MULTISTREAM --ac-streams 4 \
  --output-dir artifacts/papers/coding-survey/ac-trace-smoke
```

The runner charges the MSAC directory, per-stream byte rounding, and padding.
Both implementations remain Python references; these timings do not establish
the throughput of a compiled multistream coder.

## Reproduce the coder bit-cost figure

The manuscript figure uses one deterministic synthetic trace, not a model
trace. From the repository root:

    .venv/bin/python research/papers/cidr_2027/experiments/evaluate_coders.py \
      --synthetic-symbols 4096 --alphabet-size 64 --seed 2027 \
      --perturbation-scale 1e-6 \
      --output-dir artifacts/papers/coding-survey/coder-bit-cost-trace
    .venv/bin/python research/papers/coding_survey/experiments/plot_coder_bit_cost.py \
      artifacts/papers/coding-survey/coder-bit-cost-trace/results.json \
      artifacts/papers/coding-survey/coder-bit-cost-trace/synthetic-trace.npz \
      research/papers/coding_survey/manuscript/figures/coder_bit_cost.pdf --synthetic-source

The bars report serialized archive bits per token. The probability-coder
reference is quantized model log-loss; the rank-coder reference is empirical
rank entropy. The expected Shannon lower bound is computed from the
trace probabilities because this synthetic generator sampled symbols
from those same distributions. PMATIC's helper bits are included in
its payload.

## Text compression

`main.py --encoding AC_MULTISTREAM` now assigns one independent arithmetic
stream to each token batch. The versioned GMMS file stores model settings, one
seed token per batch, the token bitmap, and the MSAC coder payload. The model
or n-gram checkpoint remains an external dependency. The JSON run result
records `saved_archive_size_bytes`, which includes the file metadata and seeds.
The older AC and ANS file formats remain readable. Standard decompression is
supported; speculative decompression is not enabled for this mode.

Next slices will add a matched compiled encoder and decoder, then engine-specific
handoff experiments. Each performance condition must preserve its resolved
environment, archive, exact round-trip result, and separate encode/decode wall
times.

## Device-side arithmetic decoding pilot

For MSAC v2 target-interval archives with the Transformer engine, pass
`--ac-decode-backend device` during decompression. This opt-in path keeps
probabilities and arithmetic state on the model's CPU/CUDA device; only token
IDs return to the host prompt buffers. The default `host` path remains the
reference. The current torch decoder performs host-visible synchronization in
its renormalization loop, so it is a correctness baseline for a later fused
kernel, not an established GPU speedup.

A frozen-trace comparison, including exact recovery, can be run from the repo
root with:

```bash
.venv/bin/python research/papers/coding_survey/experiments/benchmark_device_decode.py \
  --device cpu --steps 128 --streams 4 --alphabet 64 --repeats 5
```

Use `--device cuda` on a CUDA machine. This benchmark excludes model inference,
archive I/O, and prompt transfer; live archive timings must be reported
separately. CUDA synchronization brackets measured decode loops.

The live CPU baseline uses the existing E02 Qwen/text8 archive and excerpt,
loads one cached float32 model, rotates host/device condition order after
warmup, and writes every exact-recovery timing sample:

```bash
.venv/bin/python research/papers/coding_survey/experiments/benchmark_live_device_decode.py \
  --repeats 5
```

The default output is ignored by Git at
`artifacts/papers/coding-survey/in-engine-decoding/live-cpu.json`. The frozen
trace runner also accepts `--output` for its raw samples. Include these files
and the E02 archive in the anonymous submission artifact.

## GPU server handoff

Use the [GPU setup guide](GPU_SETUP.md) to install the locked environment,
validate CUDA, run the frozen and live decode benchmarks, and preserve the
machine and archive metadata. The [CUDA arithmetic coder plan](../../../../docs/cuda_arithmetic_coder_plan.md)
specifies the MSAC v2 byte contract, implementation stages, and performance
gates for a compiled kernel.


## Compiled CUDA interval encoder

`src/coding/cuda_ac.py` implements the first custom-kernel slice from the CUDA
plan. It accepts exact integer lower, upper, and actual-total tensors with
shape `[steps, streams]`; one CUDA thread owns each sequential arithmetic
stream. The host wrapper adds the unchanged MSAC v2 directory and CRCs. It
checks private-workspace overflow and retries rather than truncating output.
The extension builds lazily on first use and requires Ninja plus a matching CUDA toolkit, so exclude compilation from timings.

On a GPU allocation, run the byte-exact reference and overflow tests with:

```bash
MAX_JOBS=1 .venv/bin/python -m pytest -q tests/test_device_ac.py
```

The next implementation slice is the device decoder kernel, followed by
in-engine interval capture that removes the remaining host synchronization.
