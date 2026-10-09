# Coding survey experiments

See the [experiment tracker](EXPERIMENT_PLAN.md) for planned experiments,
the completed Qwen/text8 CPU pilot, its GPU repeat, and ablations.

## Traditional-codec frontier

The rate-constrained frontier uses the exact first 10 MiB of `data/text8` and
six preregistered CPU baselines: FSST8, LZ4 default, gzip level 6, Zstandard
levels 3 and 19, and Brotli quality 11. The in-memory runner charges complete
framed output, times compression and decompression separately, and checks every
decompression byte-for-byte. FSST compression includes symbol-table training
and export, and its archive includes that table.

Build the small C ABI wrapper against a pinned checkout of the official FSST
implementation, then run the benchmark:

```bash
git clone https://github.com/cwida/fsst.git /tmp/compression-economics-fsst
g++ -std=c++17 -O3 -DNDEBUG -march=native -fPIC -shared \
  -I/tmp/compression-economics-fsst \
  research/papers/coding_survey/experiments/fsst8_wrapper.cpp \
  /tmp/compression-economics-fsst/libfsst.cpp \
  /tmp/compression-economics-fsst/fsst_avx512.cpp \
  -o /tmp/libcoding_survey_fsst8.so
FSST_REVISION=$(git -C /tmp/compression-economics-fsst rev-parse HEAD)
.venv/bin/python \
  research/papers/coding_survey/experiments/benchmark_traditional_codecs.py \
  --fsst-library /tmp/libcoding_survey_fsst8.so \
  --fsst-revision "$FSST_REVISION" \
  --cpu 0
```

The Python environment must provide `brotli`, `zstandard`, and `lz4`. Raw JSON
is written to `artifacts/papers/coding-survey/traditional-codecs/`.

## External LLM baselines

FineZip and Nacrith are executed from pinned upstream checkouts; their source is
not vendored here. Both runners use the exact first 10 MiB of `data/text8`,
exclude shared model/tokenizer weights from per-file size, time compression and
decompression separately after model loading, and reject anything short of
byte-exact recovery.

FineZip's practical paper configuration is dynamic-context token ranks followed
by bzip2; the arithmetic-coding ablation took roughly nine days for 10 MB and
was not pursued by the authors. `benchmark_finezip.py` calls the upstream
`ZipModel` rank generation and recovery unchanged, and packages its bzip2
payload and decode parameters in a small `FZB1` accounting container. This is
a repository-local wrapper, not an upstream FineZip interchange format. The
matched comparison uses `HuggingFaceTB/SmolLM2-135M` without online LoRA
memorization; it is therefore a FineZip-algorithm baseline, not a reproduction
of the paper's fine-tuned Llama-3-8B headline number.

Nacrith is run with its official NC05 archive and default ensemble settings.
Its llama.cpp environment is intentionally separate from the locked project
environment. Example H100 commands (pin and record exact upstream revisions):

```bash
git clone https://github.com/fazalmittu/FineZip.git /tmp/finezip
git clone https://github.com/robtacconelli/Nacrith-GPU.git /tmp/nacrith

.venv/bin/python research/papers/coding_survey/experiments/benchmark_finezip.py \
  --finezip-repo /tmp/finezip --dtype bfloat16 \
  --batch-size 1024 --context-size 512

/tmp/nacrith-venv/bin/python \
  research/papers/coding_survey/experiments/benchmark_nacrith.py \
  --nacrith-repo /tmp/nacrith \
  --gguf /path/to/SmolLM2-135M-BF16.gguf --workers 8

.venv/bin/python \
  research/papers/coding_survey/experiments/summarize_llm_baselines.py \
  artifacts/papers/coding-survey/llm-baselines/finezip-text8-first-10MiB.json \
  artifacts/papers/coding-survey/llm-baselines/nacrith-text8-first-10MiB.json \
  artifacts/papers/coding-survey/llm-baselines/text8-first-10MiB.json
```

The JSON records upstream revisions, source/model hashes, configuration,
complete archive size/hash, cold model-load time, warm operation times,
throughput, and exact-recovery status. FineZip and Nacrith use different
inference engines, so their BF16 conversions are identified and hashed rather
than assumed numerically identical.

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

On CUDA the frozen runner compares `host`, `torch_device`, and `cuda_cdf`.
The last condition retains canonical PyTorch CDF construction and uses the
custom supplied-CDF decoder kernel. Use `--backends` to isolate one condition
for profiling. Its throughput is reported in symbols/s only: the synthetic
Dirichlet symbols have no canonical source-byte representation, so an MB/s
conversion would be arbitrary. A matched live archive can be created and
benchmarked with:

```bash
.venv/bin/python research/papers/coding_survey/experiments/benchmark_live_device_decode.py \
  --device cuda --dtype auto \
  --pilot-dir artifacts/papers/coding-survey/in-engine-decoding/matched-cuda-qwen-1024 \
  --prepare-input data/text8 --prepare-tokens 1024 --repeats 5 \
  --output artifacts/papers/coding-survey/in-engine-decoding/live-cuda-v1.json
```

The live JSON records uncompressed-source throughput in both decimal MB/s
($10^6$ bytes/s) and binary MiB/s ($2^{20}$ bytes/s). End-to-end throughput
uses complete decompression wall time. The separately named AC-stage-equivalent
rate is diagnostic and must not be compared with complete codec throughput.
The CUDA compression benchmark records the same two units for its compression
phase and whole call; the compression-phase value excludes model construction
and is the relevant cross-codec rate.

Regenerate the paper's compression-ratio/throughput figure from the versioned
five-run H100 data with:

```bash
.venv/bin/python research/papers/coding_survey/experiments/plot_optimization_pareto.py
```

This writes a vector PDF and a 600-DPI PNG under the manuscript's `figures/`
directory. The charged compressed size includes the framed MSAC payload,
serialized vocabulary bitmap, and uint32 stream seeds. The shared external
model and tokenizer are excluded and disclosed in the figure.

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
The extension builds lazily on first use and requires Ninja plus a matching
CUDA toolkit, so exclude compilation from timings.

On a GPU allocation, run the byte-exact reference and overflow tests with:

```bash
MAX_JOBS=1 .venv/bin/python -m pytest -q tests/test_device_ac.py
```

The next implementation slice is the device decoder kernel, followed by
in-engine interval capture that removes the remaining host synchronization.


## Run the compiled CUDA encoder

On Helma, the test wrapper obtains one H100 through Slurm when needed, loads
CUDA 12.8, NCCL, and Ninja, and caps extension compilation to one job:

```bash
research/papers/coding_survey/experiments/run_cuda_ac_tests.sh
```

Live compression selects the buffered CUDA encoder with
`--encoding AC_TARGET_INTERVAL --engine transformer --ac-backend cuda`. The
CUDA path retains `[steps, streams]` interval tensors until finalization, uses
one controlled synchronization, and records kernel and final device-to-host
transfer metrics under `cuda_ac_metrics`.

Target-interval construction is a separate experimental factor. The default
`--target-interval-quantizer torch` is the original CUDA-encoder pipeline and
materializes normalized probabilities, integer frequencies, and full CDFs.
`--target-interval-quantizer cuda_fused` keeps the same float64 conversion and
row-sum reduction, then computes only the selected low/high interval, actual
total, and target probability in one kernel. This isolates fusion from both
model inference and arithmetic coding. Every comparison must require exact
integer intervals and byte-identical archives before reporting timing.

`--target-interval-quantizer cuda_v3` keeps the same floor-count contract and
the same CUDA arithmetic encoder while avoiding CUDA v2's vocabulary-sized
float64 probability tensor. It casts source probabilities inside the float64
row reduction and interval-extraction kernel. This is a separately selectable
ablation; CUDA v1, v2, and v3 must produce identical intervals and archives.

`--target-interval-quantizer cuda_v4` is the advanced encoding path. Its custom
kernel fuses deterministic exact fixed-total normalization, residual
calculation, stable residual-symbol selection, and target-only bound emission.
It writes MSAC v3 and incrementally updates persistent multistream arithmetic
state on a dedicated CUDA stream, allowing coding at step `t` to overlap the
next inference step rather than retaining the complete interval trace. Its
decoder reconstructs the same fixed-total CDF before the existing batched CUDA
inverse-CDF/state-update kernel. Because v4 changes the quantized distribution,
it is round-trip checked separately from the byte-identical MSAC-v2 v1--v3
ablation group. Model masking and softmax remain outside the custom kernel.

Kernel-only and matched live-Qwen benchmarks are:

```bash
.venv/bin/python research/papers/coding_survey/experiments/benchmark_cuda_encode.py \
  --output artifacts/papers/coding-survey/cuda-ac/kernel.json
.venv/bin/python research/papers/coding_survey/experiments/benchmark_cuda_compression.py \
  --input data/text8 --tokens 2048 --batch-size 16 --repeats 5 \
  --variants python cuda_v1 cuda_v2 cuda_v3 cuda_v4 \
  --output artifacts/papers/coding-survey/cuda-ac/unified-end-to-end.json
```

This is the manuscript comparison: it records one full-workload warm-up per
condition, excludes warm-ups from the five-run medians, rotates and reverses
timed condition order, and requires byte-identical archives after every run.
Byte identity applies within the MSAC-v2 v1--v3 group; CUDA v4 uses the
separately versioned fixed-total contract and must pass exact round-trip checks.
The legacy names `cuda_torch` and `cuda_fused` remain accepted as aliases for
`cuda_v1` and `cuda_v2`.

The 10 MiB text8 round-trip comparison additionally includes the preserved
pre-CUDA Python encoder and host decoder. The paper-matched quality setting is
batch 256, context 256, and retain 100:

```bash
.venv/bin/python \
  research/papers/coding_survey/experiments/benchmark_text8_cuda_roundtrip.py \
  --input-bytes 10485760 --batch-size 256 --context-length 256 \
  --retain-tokens 100 --repeats 3 \
  --output artifacts/papers/coding-survey/cuda-ac/\
text8-first-10MiB-roundtrip-b256-c256-r100.json
.venv/bin/python \
  research/papers/coding_survey/experiments/plot_text8_roundtrip_timing.py
```

The runner requires identical archives across all encoders and exact recovery
from all decoders. The plot uses non-overlapping timing components: interval
construction is subtracted from its enclosing inference timer before stacking.

The dedicated fusion microbenchmark holds the probability trace and CUDA
arithmetic encoder constant. It reports device-event time for interval
construction alone and for interval construction plus encoding:

```bash
.venv/bin/python \
  research/papers/coding_survey/experiments/benchmark_cuda_interval_fusion.py \
  --steps 4096 --streams 32 --alphabet 256 --repeats 50 \
  --output artifacts/papers/coding-survey/cuda-ac/fusion-kernel.json
```
