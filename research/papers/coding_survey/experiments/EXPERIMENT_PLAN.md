# Coding survey experiment tracker

This is the working inventory of experiments needed for the coding survey.
Update statuses and evidence links after each run. A completed pilot does not
establish a general performance claim.

New raw outputs belong under `artifacts/papers/coding-survey/`. Keep the
existing CPU pilot in its original CIDR artifact directory. See the
[experiment README](README.md) for runner commands and the
[O3 design TODOs](../../../../docs/ac_target_interval_todos.txt) for related
implementation choices. This document tracks experiments; that file tracks
design decisions and possible implementation ablations.

## Status and priorities

- **Pilot complete:** evidence exists; broader replication remains.
- **Pending:** execution and analysis remain.
- **Needs implementation:** a necessary path or measurement is missing.
- **Exploratory:** pursue after core results, or if retained in the paper.

Prioritize E02–E04 next. The sweep sizes below are proposed starting points;
record changes here before expanding them.

## Core experiment inventory

| ID | Experiment / question | Conditions and measurements | Status |
| --- | --- | --- | --- |
| E01 | Matched frozen traces: compare AC, multistream AC, rANS (`ANS`), PMATIC, rank Huffman, and bitpacked ranks. | Same symbols and probabilities; synthetic and model traces; serialized bits/token, payload bits, encode/decode time, memory, exact recovery. Use each family's appropriate entropy reference. | Synthetic figure workflow exists; model-trace replication pending. |
| E02 | Live Qwen/text8 CPU: does target interval capture improve complete compression? | AC vs AC_MULTISTREAM vs AC_TARGET_INTERVAL with matched source/model/inference settings and coder backend; full archive sizes and exact recovery. | **Pilot complete**; longer runs and decoder timing repetitions pending. Details below. |
| E03 | Repeat E02 on GPU: does compact capture reduce transfer and wall time? | Matched CUDA Transformers run first; measure physical transfer, synchronization, conversion, coding, total time, and memory. | **Pending GPU access and instrumentation.** Protocol below. |
| E04 | Floor counts vs fixed-total count distribution. | Reference fixed-total vs exact vectorized fixed-total vs floor/variable-total; frequency-budget sweep; quantized log-loss, archive bits, conversion time, and cumulative-count parity where expected. | Existing quantizers available; matched study pending. Equivalent fixed-total device capture needs implementation. |
| E05 | Stream count, batch size, and boundary overhead. | Frozen-trace stream sweep, then live batch/context sweep; charge seeds, directories, padding, checksums, and changed boundary context; encode/decode speed and memory. | Paths exist; systematic sweep pending. |
| E06 | Python vs compiled coders and parallelism. | Same quantizer, streams, and targets; available Numba paths, thread sweep; separate encoder/decoder timing; exclude and report JIT warmup. | Compiled encoding exists; establish decoder coverage before symmetric comparisons. |
| E07 | rANS block/lane tradeoffs. | Blocks 64/256/1024, lanes 1/4/16 where supported; state flushes, headers, finite-precision excess bits, buffering memory, and encode/decode speed. | Runner settings exist; sweep pending. |
| E08 | Probability mismatch and robust decoding. | Controlled encoder/decoder perturbations, seeds/scales, PMATIC helper budget and bits; exact recovery/failure rates. Study actual engine/device disagreement separately. | Synthetic perturbation machinery exists; systematic study pending. |
| E09 | Full vocabulary vs observed-token mask. | Charge mask bytes; compare entropy, archive size, inference/conversion/coding cost. Separate post-logit masking from restricted output projection. | Mask path exists; sweep pending. Restricted projection needs implementation. |
| E10 | Inference engine performance and reproducibility. | Transformers CPU/CUDA and optional vLLM CUDA; pin model/tokenizer, versions, dtype, attention backend, tensor parallel layout; verify complete archives. | CPU pilot complete; live CUDA/vLLM validation pending. TensorRT-LLM would require a separate port. |
| E11 | O1/O2/O3 interactions. | Teacher-forced windows (O1), inference/coder overlap (O2), target interval capture (O3); 2×2×2 factorial with matched effective contexts and quantization. | **Needs implementation** for O1/O2. Compiled interval buffering alone is not pipelining. |
| E12 | KV cache and parallel sequence scoring. | Cache on/off; incremental vs teacher-forced/prefill scoring; identical context truncation/retention; verify distribution equivalence where claimed. | Cache path exists; matched study pending. Parallel scoring/window semantics need validation. |

### Suggested sweep sizes

- Source lengths: 1,024 tokens for pilots, then 8,192 and 65,536.
- Live batches: 1, 4, 16, 64, then 256 if memory permits. Record actual streams,
  seeds, and targets: changing batch boundaries can change model distributions.
- Contexts: 128, 512, 2,048; specify retained context explicitly.
- Synthetic alphabets: 64, 256, 4,096, and a large/model-sized vocabulary.
- Frequency budgets near 2^12, 2^16, 2^18, and 2^20. Skip invalid budgets
  that cannot assign positive counts to every symbol.
- Paper timing claims: start with at least five measured repetitions after
  warmup, rotating condition order; increase if variability requires it.

Run small pilots before full sweeps. Isolate coder effects on frozen traces,
then vary live inference conditions explicitly; do not multiply every axis
into one matrix.

## E02 — completed Qwen/text8 CPU pilot

Evidence: [report](../../../../artifacts/papers/cidr-2027/target-interval/Qwen2.5-0.5B-text8-1024-cpu/report.md),
[results](../../../../artifacts/papers/cidr-2027/target-interval/Qwen2.5-0.5B-text8-1024-cpu/results.json),
[runner](../../../../artifacts/papers/cidr-2027/target-interval/Qwen2.5-0.5B-text8-1024-cpu/run_benchmark.py).
These artifacts are ignored by Git; preserve them separately for sharing.

Configuration:

- `Qwen/Qwen2.5-0.5B`, revision
  `060db6499f32faf8b98477b0a26969ef7d8b9987`.
- First 1,024 text8 tokens: 5,645 source bytes, 1,008 encoded targets,
  16 seed tokens. Excerpt SHA-256:
  `c2b659e2bb154bedd2c2d1f8c33081052928a52e3c84323a52c660a922b05f12`.
- Transformers CPU float32; Intel Xeon Gold 5318Y; eight Torch threads;
  batch 16, context 128, retained context 64, KV cache enabled,
  observed vocabulary mask of 462 tokens; nominal total 262,144.
- Python coder backend for all three; one warmed reused model; three
  compression repetitions with rotated order. Model loading, archive disk
  I/O, and verification excluded from compression timing.

| Coder | Median compression (s) | Tokens/s | Median conversion + coding (s) | Full archive (bytes) |
| --- | ---: | ---: | ---: | ---: |
| AC | 6.166 | 166.1 | 0.130 | 2,135 |
| AC_MULTISTREAM | 6.185 | 165.6 | 0.145 | 2,503 |
| AC_TARGET_INTERVAL | 6.084 | 168.3 | 0.038 | 2,551 |

All persisted archives passed exact token/text recovery. Single verification
decode times were 6.167 / 6.166 / 6.134 seconds respectively; these are not
repeated decoder timing estimates. Conversion + coding is the median of each
repetition's stage sum. Conversion is included in inference timing, so do not
count it twice in total time.

O3 showed about a 1.4% throughput gain over AC in this small CPU pilot.
This does not establish a general speedup or GPU transfer benefit.
Archive differences include metadata/stream overhead and different
quantization: O3 uses floor counts with a variable actual total; AC and
multistream AC redistribute counts to a fixed total. E04 must isolate capture
from this distribution change before claiming a pure O3 comparison.

Remaining work:

- [ ] Longer excerpts and at least five compression repetitions.
- [ ] Repeated decoder timings with variability.
- [ ] Available compiled paths, excluding compilation.
- [ ] Equivalent-quantization capture comparison after implementation.

## E03 — matched GPU repeat

- [ ] Parameterize the CPU runner for CUDA; CUDA installation alone does not
  make its existing CPU configuration a GPU experiment.
- [ ] Reuse exact excerpt/hash, Qwen revision, tokenizer, mask, batch 16,
  context 128, retained context 64, KV cache, and Python backend. Begin with
  float32; label any required deviations as separate conditions.
- [ ] Save GPU, driver, CUDA/Torch/Transformers versions, attention backend,
  device count, CPU thread settings, and resolved configuration.
- [ ] Warm the model/device, rotate coder order, and measure at least five
  repetitions. Record cold load and JIT compilation separately.
- [ ] Synchronize total timing boundaries; use CUDA events/profiling for
  device stages. Host-only stage timers cannot establish asynchronous GPU
  costs. Avoid per-step synchronization that alters the execution under test.
- [ ] Measure physical device-to-host traffic and copies. Report logical
  32-byte interval records separately: the CPU pilot's 32,256 interval bytes
  are not a PCIe measurement.
- [ ] Record inference, interval conversion, host conversion, encoding,
  complete compression/decompression, peak host/device memory, full archive
  bytes, payload bits, and exact recovery. Explain overlapping stages instead
  of adding them as if they were serial.
- [ ] Verify each archive on the same GPU execution configuration.
  Cross-device decoding is a separate reproducibility study.
- [ ] Follow with full-vocabulary and compiled-backend conditions, changing
  one factor at a time; repeat through vLLM after live validation (E10).
- [ ] Save raw samples, variability, and comparison with E02 under
  `artifacts/papers/coding-survey/target-interval/`.

## Additional optimization ablations

| ID | Ablation | Dependencies / status |
| --- | --- | --- |
| A01 | Float64 vs float32 quantization: cost, threshold/count disagreement, archive size, exact recovery on CPU/CUDA. | Floor float64 exists; float32/MPS policies require implementation. No MPS O3 claim yet. |
| A02 | Prefix scan vs masked reduction vs fused mask/softmax/quantization/interval kernel. | Alternative kernels needed; measure launches, device memory traffic, and total time. |
| A03 | Batched records, pinned buffers, asynchronous copies, shared-memory allocation/reuse. | Profile existing capture first; alternatives need implementation. Charge synchronization/allocation. |
| A04 | Inactive final-batch rows vs active-row compaction. | Alternative needed; measure tail utilization and result parity. |
| A05 | In-engine arithmetic decoding and batched inverse CDF. | Opt-in torch device-state decoder implemented for MSAC v2 and paired layout; exact CPU and end-to-end tests pass. CUDA tests and live GPU profiling remain. Run `benchmark_device_decode.py` for a frozen-trace host/device comparison. |
| A06 | vLLM attention backends, tensor parallel layouts, vocabulary padding, worker vs dense probabilities. | Live GPU required; pin validated version and test count-level agreement before performance claims. |
| A07 | Speculative decompression/coding and early exit. | Exploratory; define lossless fallback, charge rejected work/helper data, compare multistream decoding. |
| A08 | More models, datasets, and adaptation. | After Qwen/text8, select additional model sizes/source types. Other architectures/LoRA only if supported and retained in scope; vLLM LoRA is not currently validated. |
| A09 | Classical compression baselines and compression/throughput Pareto curves. | Baseline selection pending; same source bytes and accounting convention; disclose external model dependency. |

## Evidence required for every reported condition

- [ ] Source hash, token IDs/counts, model/tokenizer revisions, seeds, mask,
  stream/chunk layout, context/retention, dtype, quantizer and total.
- [ ] Software/hardware, engine/backend/threads, command/configuration,
  warmup policy, raw timing samples, and failures.
- [ ] Separate frozen-trace coder results from live model results. Use
  identical distributions for coder implementation comparisons.
- [ ] Full archive bytes and payload bits separately; charge seeds, masks,
  codebooks, helpers, directories, padding, checksums. State external-model
  accounting.
- [ ] Quantized model log-loss reference for probability coders; rank entropy
  for rank coders. Synthetic source entropy only under valid sampling assumptions.
- [ ] Exact decode/token/text recovery as applicable and decoder configuration.
- [ ] Encode/decode separately, repetition variability, memory, cold/warm
  costs; logical bytes distinguished from physical traffic.
- [ ] Plots/tables linked to raw evidence, with interpretation limits.

## Run ledger template

Copy for each new run; retain both completed and failed runs.

```text
Experiment ID / condition:
Date / status:
Question / hypothesis:
Source and model revisions:
Hardware / engine / precision / coder / quantizer:
Command or configuration:
Warmup / repetitions / timing boundaries:
Artifact directory:
Exact recovery:
Results / variability:
Limitations / deviations:
Next action / design TODO:
```

## D1--D5 — decoder-side integration study

The manuscript section `sections/in_engine_decoding.tex` states the hypotheses,
controls, and current results. The implementation is opt-in with
`--ac-decode-backend device` for Transformer/MSAC-v2 decompression.

| ID | Comparison | Current evidence and next step |
| --- | --- | --- |
| D1 | Frozen MSAC-v2 host versus batched torch decoder; stream/alphabet sweep. | **CPU complete.** Four 128-step conditions, seed 2027, five timed repeats after warmup, exact recovery. Raw JSON: `artifacts/papers/coding-survey/in-engine-decoding/frozen-cpu-*.json`. Run GPU equivalents with probability rows preloaded on device. |
| D2 | Live host versus torch decode of identical Qwen/text8 archive. | **CPU complete.** Five rotated repeats after one warmup per path; 1,024 tokens, exact text recovery. Medians: host 6.187 s, torch 6.215 s. Raw JSON: `artifacts/papers/coding-survey/in-engine-decoding/live-cpu.json`. |
| D3 | Matched live CUDA decode, full archive and physical transfer. | **Pending GPU.** First verify the same archive; if GPU probabilities produce different counts, create a matched GPU archive and report cross-device failure separately. |
| D4 | Device decode stage ablations and fused kernel. | **Pending GPU.** Separate full-row host copy, device CDF with host state, current torch device state, and fused device state. Measure synchronization and memory. |
| D5 | Cross-device/engine exactness and mismatch. | **CPU tests complete; GPU/engine work pending.** Never time failed decodes as successful conditions. |

D1 reproduction (from the repository root):

```bash
.venv/bin/python research/papers/coding_survey/experiments/benchmark_device_decode.py \
  --device cpu --steps 128 --streams 4 --alphabet 64 --repeats 5 \
  --output artifacts/papers/coding-survey/in-engine-decoding/frozen-cpu-b4-a64.json
```

Repeat with streams/alphabet `(1,64)`, `(16,64)`, and `(4,512)` for the manuscript
table. D2 uses the existing ignored E02 archive and excerpt:

```bash
.venv/bin/python research/papers/coding_survey/experiments/benchmark_live_device_decode.py \
  --repeats 5
```

These JSON outputs are ignored by Git; include them, the archived source/model
identifiers, and the E02 archive in the anonymous artifact package.
