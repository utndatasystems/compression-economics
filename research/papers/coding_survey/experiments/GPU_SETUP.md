# GPU server setup for the coding survey

Run these commands from the repository root on a Linux server with an NVIDIA GPU. The repository uses Python 3.12 (`.python-version`) and `uv.lock`. Install a compatible NVIDIA driver, Python 3.12, Git, and [uv](https://docs.astral.sh/uv/getting-started/installation/) first. `nvidia-smi` should see the allocated GPU. A CUDA toolkit with `nvcc` is needed when compiling the proposed custom kernel; it is not needed for the existing PyTorch decoder. Check that the installed PyTorch wheel supports the driver's CUDA version. Avoid the root `setup.sh` here: it downloads unrelated large datasets.

## 1. Transfer the exact experiment revision

Clone the `coding-survey-benchmarks` branch, which contains the device decoder, GPU setup files, and this guide. The manuscript is a Git submodule; initialize it only if you need the LaTeX source. Its remote uses SSH, so configure a GitHub SSH key with access to both repositories on the new machine.

```bash
git clone --branch coding-survey-benchmarks git@github.com:utndatasystems/compression-economics.git
cd compression-economics
git submodule update --init research/papers/coding_survey/manuscript  # optional
bash research/papers/coding_survey/experiments/setup_gpu.sh
```

`setup_gpu.sh` installs the locked Python dependencies with the `parallel` extra, runs a CUDA quantization check, and runs the relevant coder tests. It writes the machine snapshot to `artifacts/papers/coding-survey/gpu-setup/environment.json`. The script intentionally does not fetch a model or dataset. If the driver or PyTorch wheel is incompatible, fix that before taking measurements; the preflight script exits with an error. Keep the resolved dependency versions and the snapshot with the submission artifact.

## 2. Run the frozen-trace GPU comparison

```bash
.venv/bin/python research/papers/coding_survey/experiments/benchmark_device_decode.py \
  --device cuda --steps 128 --streams 16 --alphabet 64 --repeats 5 \
  --output artifacts/papers/coding-survey/in-engine-decoding/frozen-cuda-b16-a64.json
```

The runner builds one deterministic synthetic MSAC v2 archive, verifies every decoded symbol, warms both backends, rotates the run order, and stores raw times. It includes the probability handoff but no model inference or archive I/O. It is a baseline for the current PyTorch reference, which still synchronizes during renormalization. Run other batch sizes and masked alphabet sizes with distinct output paths; keep exact recovery as a mandatory gate. Record `nvidia-smi` state and whether other jobs occupy the GPU.

## 3. Run a live model experiment

The local E02 CPU pilot under `artifacts/papers/cidr-2027/target-interval/Qwen2.5-0.5B-text8-1024-cpu/` is ignored by Git, as are the model cache and most datasets. Copy the archive and `excerpt.txt` separately if you need that pilot. The live benchmark requires both files in its `--pilot-dir`, and requires the model and tokenizer to be present in the repository `.cache` (`local_files_only=True`). A CPU-made archive may fail to decode with CUDA model probabilities. Treat this as an exactness failure, not a timing sample: archive creation and decompression must use matching model weights, tokenizer, precision, backend, and quantization.

To create a matched GPU archive, place the exact input text on the server, run `main.py --mode compress --encoding AC_TARGET_INTERVAL` on that GPU, and save its `.bin` output as `<pilot-dir>/ac_target_interval.bin`. Save an `excerpt.txt` containing exactly the source tokens selected by `--first_n_tokens`; the live runner checks token count and exact reconstructed text. The current E02 pilot uses Qwen/Qwen2.5-0.5B, 1024 tokens, batch size 16, context length 128, and retain length 64. Its `excerpt.txt` is suitable input for a matched rerun, provided it is available on the server:

```bash
mkdir -p artifacts/papers/coding-survey/in-engine-decoding/gpu-pilot
cp /path/to/excerpt.txt artifacts/papers/coding-survey/in-engine-decoding/gpu-pilot/excerpt.txt
.venv/bin/python main.py --mode compress \
  --input_path artifacts/papers/coding-survey/in-engine-decoding/gpu-pilot/excerpt.txt \
  --output_path artifacts/papers/coding-survey/in-engine-decoding/gpu-pilot/ac_target_interval.bin \
  --model_name Qwen/Qwen2.5-0.5B --encoding AC_TARGET_INTERVAL \
  --first_n_tokens 1024 --batch_size 16 --context_length 128 --retain_tokens 64 --force
.venv/bin/python research/papers/coding_survey/experiments/benchmark_live_device_decode.py \
  --pilot-dir artifacts/papers/coding-survey/in-engine-decoding/gpu-pilot \
  --device cuda --dtype auto --repeats 5 \
  --output artifacts/papers/coding-survey/in-engine-decoding/live-cuda.json
```

Use `--dtype float32` for a matched float32 archive. The live benchmark loads one model, warms both decoding backends, rotates order, checks exact token and text recovery on every run, and records raw samples plus archive/source hashes. `main.py` may also write its standard run records under `artifacts/runs/current/`; preserve these. Pin and record the resolved model revision for publication, since the model name alone is mutable. For CPU and GPU comparisons, report the precision and whether each archive was generated and decoded on the same backend. Never compare compression ratios across unmatched text or model configurations.

## 4. Preserve the artifact

Keep the Git commit and submodule commit, `environment.json`, benchmark JSON, command line, model revision, archive, excerpt, source and archive hashes, and any profiler outputs. Generated files under `artifacts/`, downloaded models under `.cache`, and data files are ignored by Git, so transfer them explicitly. The [CUDA kernel plan](../../../../docs/cuda_arithmetic_coder_plan.md) gives the compatibility contract, staged implementation, and profiler gates for the next step.
