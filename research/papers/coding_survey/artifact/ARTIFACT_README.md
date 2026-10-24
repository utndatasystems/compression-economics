# Anonymous artifact: fast lossless text compression with LLMs

This repository contains the minimal code needed to reproduce the paper's
main experiments. It includes:

- the three reported end-to-end CUDA operating points on the first 10 MiB of
  text8 (batch/context/retention: 1024/512/256, 8192/128/64, and
  16384/128/64);
- the single-threaded conventional-codec comparison;
- adapters for the pinned FineZip, LLMZip, and Nacrith baselines; and
- the speculative-decoding and teacher-forcing studies used for the paper's
  limits-of-token-parallelism result.

Appendix ablations, parameter sweeps, notebooks, manuscript sources, and tests
are intentionally absent. Model weights, text8, and third-party repositories
are downloaded separately and are not redistributed.

## Hardware and expected runtime

The headline LLM measurements require an NVIDIA H100-class GPU, a CUDA 12.8
toolkit with `nvcc`, and enough device memory for the largest stream batch.
The paper used Python 3.12.13, PyTorch 2.10.0, CUDA 12.8, Transformers 5.3.0,
and NumPy 2.4.3. Exact throughput is hardware-dependent. The full 10 MiB LLM
runs are long; use the smoke command first.

The traditional codecs are CPU experiments. FSST8 is compiled from its pinned
upstream revision. The external LLM baselines each use their own pinned
upstream checkout; Nacrith also requires its documented llama.cpp environment
and a matching GGUF model.

## 1. Environment

Create a Python 3.12 environment and install the small runtime dependency set:

```bash
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Install the PyTorch wheel appropriate for the machine's CUDA stack if the
default package index does not provide it. Confirm that both CUDA and the CUDA
compiler are visible:

```bash
python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
nvcc --version
```

## 2. Inputs

Download text8 and verify the complete file and the evaluated 10 MiB prefix:

```bash
mkdir -p data
wget -O /tmp/text8.zip http://mattmahoney.net/dc/text8.zip
unzip -o /tmp/text8.zip -d data
sha256sum data/text8
head -c 10485760 data/text8 | sha256sum
```

Expected SHA-256 values:

- complete text8: `6e890197040d37d85beb962ae1f041ff1d9a9ca8d20c7d99c85027eebf51dca7`
- first 10 MiB: `d14c3bba3810d20335ded84f81248fa97c8ae2709744f7762c793bcc189c1dad`

Download the pinned model snapshot:

```bash
.venv/bin/hf download Qwen/Qwen2.5-0.5B \
  --revision 060db6499f32faf8b98477b0a26969ef7d8b9987 \
  --cache-dir .cache
mkdir -p .cache/models--Qwen--Qwen2.5-0.5B/refs
printf '%s' 060db6499f32faf8b98477b0a26969ef7d8b9987 \
  > .cache/models--Qwen--Qwen2.5-0.5B/refs/main
```

## 3. Smoke check

This compiles the CUDA extension and performs a short exact round trip:

```bash
bash reproduce.sh smoke
```

Generated JSON and figures are written below `results/`, which is ignored by
Git. Every end-to-end result records source/model hashes, environment details,
raw timing samples, complete archive accounting, and exact recovery status.

## 4. Main experiments

Run the paper's own CUDA operating points:

```bash
bash reproduce.sh cuda
```

Build FSST8 and run the matched conventional baselines:

```bash
bash reproduce.sh traditional
```

Run the limits-of-token-parallelism experiments:

```bash
bash reproduce.sh parallelism
```

`bash reproduce.sh all` runs those three groups. It does not run external LLM
baselines because they require separate upstream environments and model
conversion. Their pinned commands are documented in
`research/papers/coding_survey/experiments/EXTERNAL_BASELINES.md`.

After all baseline JSON files are present, regenerate the main operating-point
figure with the explicit input flags documented in that file. Plotting rejects
unmatched sources, incomplete runs, failed recovery, and inconsistent model
revisions.

## Scope and licensing

The code in this artifact is MIT licensed. Qwen2.5-0.5B is Apache-2.0
licensed. FSST, Brotli, LZ4, Zstandard, FineZip, LLMZip, Nacrith, and text8 are
third-party works governed by their own terms. No human-subject data is used.

Paths and machine names recorded in generated JSON describe the reviewer's
machine; they are not part of this repository.
