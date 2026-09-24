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

Next slices will add a matched compiled encoder and decoder, then engine-specific
handoff experiments. Each performance condition must preserve its resolved
environment, archive, exact round-trip result, and separate encode/decode wall
times.
