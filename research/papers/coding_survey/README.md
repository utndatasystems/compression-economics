# Coding survey

This package contains the experiments and manuscript for the lossless
LLM-compression systems study. Reusable implementations and archive contracts
live in [`src/coding/`](../../../src/coding/README.md); generated results belong
under ignored `artifacts/papers/coding-survey/` paths.

`manuscript/` is a separate Git submodule. `artifact/` builds the anonymous,
allowlisted reproduction repository.

## Environment

Use Python 3.12 and the locked project environment. On an NVIDIA server, the
setup script installs the required extras, checks CUDA, runs coder tests, and
writes `artifacts/papers/coding-survey/gpu-setup/environment.json`:

```bash
git submodule update --init research/papers/coding_survey/manuscript  # optional
bash research/papers/coding_survey/experiments/setup_gpu.sh
```

CUDA extension work requires a compatible driver, PyTorch wheel, toolkit with
`nvcc`, and Ninja. Model snapshots and datasets are intentionally not fetched by
the setup script. Run the byte-exact CUDA checks with:

```bash
research/papers/coding_survey/experiments/run_cuda_ac_tests.sh
```

## Primary workflows

The main 10 MiB text8 comparison covers full-logit references, compact
target-interval capture, CUDA v1--v4 encoding, and host/device/CUDA decoding:

```bash
.venv/bin/python \
  research/papers/coding_survey/experiments/benchmark_text8_cuda_roundtrip.py \
  --input-bytes 10485760 --batch-size 256 --context-length 256 \
  --retain-tokens 100 --repeats 3 \
  --output artifacts/papers/coding-survey/cuda-ac/text8-roundtrip.json
```

Other maintained entry points are:

| Purpose | Entry point |
| --- | --- |
| CUDA kernel and end-to-end ablations | `benchmark_cuda_encode.py`, `benchmark_cuda_compression.py`, `benchmark_cuda_interval_fusion.py` |
| Decoder isolation and live comparison | `benchmark_device_decode.py`, `benchmark_live_device_decode.py`, `benchmark_inference_coder_isolation.py` |
| Traditional codecs | `benchmark_traditional_codecs.py` |
| FineZip, LLMZip, and Nacrith | `benchmark_finezip.py`, `benchmark_llmzip.py`, `benchmark_nacrith.py` |
| Teacher-forced scoring | `benchmark_teacher_forcing.py` |
| Speculative feasibility | `benchmark_speculative_replay.py`, `benchmark_speculative_full.py`, `audit_speculative_numerics.py` |

All paths above are below `research/papers/coding_survey/experiments/`. Use each
program's `--help` for its complete, current interface. Pinned commands for
third-party LLM baselines are in
[`artifact/EXTERNAL_BASELINES.md`](artifact/EXTERNAL_BASELINES.md).

The teacher-forcing comparison must check integer target-interval equality, not
only aggregate loss or approximate probabilities:

```bash
.venv/bin/python \
  research/papers/coding_survey/experiments/benchmark_teacher_forcing.py \
  --input data/text8 --tokens 2048 --batch-size 16 \
  --context-length 256 --retain-tokens 128 --repeats 5 --device cuda \
  --model Qwen/Qwen2.5-0.5B \
  --output artifacts/papers/coding-survey/teacher-forcing/qwen-text8.json
```

Regenerate manuscript figures with the `plot_*.py` programs in `experiments/`.
Plotters validate source identity, recovery status, and model revisions before
combining results.

## Current state

- Complete 10 MiB H100 encoding and decoding runs cover the primary CUDA paths
  and recover the exact source.
- CUDA v4 implements deterministic fixed-total capture with pipelined arithmetic
  coding; the supplied-CDF CUDA decoder performs batched inverse lookup and
  arithmetic updates.
- Two operating points quantify the throughput/compression tradeoff; broader
  repeated stream, batch, and context sweeps remain.
- Six traditional codecs and matched FineZip, LLMZip, and Nacrith runs are
  complete on the same text8 prefix.
- Teacher-forced windows reduce forward calls, but measured multi-token BF16
  intervals disagree with incremental decoding and are not lossless-compatible.
- Speculative replay and full-corpus feasibility runners exist; integration with
  the target-interval archive remains future work.

### Remaining work

- Validate live vLLM/CUDA round trips and pin its attention and tensor-parallel
  configuration.
- Establish cross-precision, cross-device, and cross-engine numerical
  portability, including vocabulary-size edge cases.
- Resolve teacher-forced interval disagreement before completing the O1/O2/O3
  factorial.
- Add physical transfer, peak-memory, active-tail compaction, and fused-CDF
  measurements.
- Replicate beyond Qwen/text8 and complete the systematic coder, frequency,
  stream-count, and batch/context sweeps.

## Evidence contract

Every reported condition must retain:

- source, model, and tokenizer hashes or revisions plus the complete command;
- hardware, software, precision, engine, quantizer, context, and stream layout;
- warm-up policy, raw repetitions, failures, and separate encode/decode times;
- payload and complete archive sizes with seeds, masks, framing, padding, and
  external-model accounting stated explicitly;
- exact token and byte recovery before any performance result is accepted.

Use uncompressed source bytes divided by complete wall time for live throughput.
Keep decimal MB/s and binary MiB/s distinct, and label kernel or stage rates as
diagnostic. Frozen-trace results do not establish end-to-end model throughput.

## Anonymous artifact

Build the standalone reviewer repository with:

```bash
bash research/papers/coding_survey/artifact/build_anonymous_repository.sh \
  /tmp/sigmod-2027-artifact
```

The builder uses an explicit allowlist and prints an audit report. Review that
report before publishing the generated repository.
