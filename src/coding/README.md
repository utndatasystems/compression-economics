# Coding subsystem

## Modules

- `encoding.py` implements arithmetic, ANS, Huffman, bit-packed rank, and
  PMATIC coders.
- `encoding_utils.py` contains shared probability quantization.
- `multistream_ac.py` and `paired_ac.py` implement the portable MSAC/MSAP
  archives and independent arithmetic streams.
- `parallel_ac.py` provides the optional Numba encoder.
- `target_interval.py`, `device_ac.py`, and `cuda_ac.py` implement compact
  target intervals and the device/CUDA backends.
- `trace_benchmark.py` compares coders on an immutable probability trace.

Import maintained implementations through `src.coding.*`.

## Target-interval arithmetic coding

`AC_TARGET_INTERVAL` transfers the known token's interval instead of a full
probability vector during compression:

```bash
python main.py --mode compress --engine transformer \
  --encoding AC_TARGET_INTERVAL --input_path data/text8 \
  --output_path artifacts/runs/o3.bin
python main.py --mode decompress --input_path artifacts/runs/o3.bin \
  --output_path artifacts/runs/o3.txt
```

The Python encoder is the default. Use `--ac-backend numba_parallel
--ac-threads 4` for compiled CPU encoding or `--ac-backend cuda` for the CUDA
path. Transformer decompression selects `--ac-decode-backend host|device|cuda`;
`host` remains the reference implementation.

The optional vLLM engine captures intervals in its worker. It supports standard
decoder-only inference without LoRA or speculative decompression. Archive
creation and decoding must use matching model, tokenizer, vocabulary, precision,
and context policies.

### Archive contracts

- MSAC v1 uses the original fixed-total distribution.
- MSAC v2 flag 1 uses `floor_float64_v1`: normalize in float64, assign
  `1 + floor(p * (nominal_total - alphabet_size))`, and store the resulting
  per-row total. MPS is unsupported for this policy.
- MSAC v3 flag 2 uses `fixed_total_stable_v2`, the deterministic fixed-total
  policy used by CUDA v4.

MSAC v2 CUDA variants v1--v3 must produce the same integer intervals and archive
bytes as the reference. CUDA v4 uses its separately versioned contract. Existing
v1 archives remain readable; old experimental dictionary payloads are not part
of this compatibility promise.

The outer GMMS archive stores seeds and model metadata but not model weights.
Standard and paired layouts include stream counts, declared bit lengths,
big-endian packed payloads, zero padding, and CRC32 checks. Malformed or
exhausted streams must be rejected.

## CUDA implementation and measurement

`cuda_ac.py` and `csrc/cuda_ac_*` provide byte-exact supplied-interval encoding,
fused target-interval construction, persistent CUDA-v4 encoding, and batched
inverse-CDF decoding. CUDA v4 overlaps arithmetic coding with subsequent model
inference on a dedicated stream; masking and softmax remain outside the custom
kernel.

Keep the host decoder as the validator and fallback. Correctness gates are exact
integer intervals, byte-identical archives where contracts match, and exact
token and UTF-8 recovery. Performance runs must exclude JIT compilation, warm
the workload, synchronize timing boundaries, retain raw repetitions, and report
complete compression and decompression separately from kernel-only timings.

## Matched coder comparison

The frozen-trace benchmark holds symbols and probabilities fixed and excludes
model inference. A smoke run is:

```bash
.venv/bin/python research/papers/cidr_2027/experiments/evaluate_coders.py \
  --synthetic-symbols 10000 --alphabet-size 256 \
  --ans-block-size 256 --ans-lanes 4 \
  --perturbation-scale 1e-8 --perturbation-scale 1e-6
```

A model trace NPZ must contain a probability matrix named `probabilities` and a
one-dimensional integer target array named `symbols`:

```bash
.venv/bin/python research/papers/cidr_2027/experiments/evaluate_coders.py \
  --trace artifacts/traces/qwen-text8.npz
```

Compare payload and complete archive sizes, framing or helper costs, exact
recovery, and encode/decode time. Rank transforms are not probability coders;
multistream results charge per-stream framing and padding but exclude model
context seeds. Frozen traces establish coder behavior, not end-to-end model
throughput or target-interval transfer savings.

Remaining validation and ablations belong in the
[coding-survey runbook](../../research/papers/coding_survey/README.md), not in a
parallel implementation plan.
