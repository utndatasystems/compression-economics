# Coding survey experiments

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
