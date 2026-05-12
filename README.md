# LLM Compression
LLM-guided text compression experiments with global token masks.

This repo explores how a language model can act as a probabilistic oracle to compress
text. A single bitmap (global mask) reduces the effective vocabulary, and the model's
next-token probabilities are encoded using arithmetic coding or rank-based schemes.

## How it works (high level)
- Tokenize input text with the chosen LLM tokenizer.
- Optionally reduce the vocabulary to only tokens seen in the input (global mask).
- Run batched, one-step LLM inference to predict next-token distributions.
- Encode each next token via:
  - `AC` (arithmetic coding), or
  - `bitpacked` / `huffman` (rank-based coding).

## Project layout
- `main.py`: CLI entry point for compression and decompression.
- `src/prediction.py`: tokenization, masking, and batched inference.
- `src/global_mask_compressor.py`: core compression/decompression loops.
- `src/encoding.py`: various encoding schemes (AC, bitpacked, Huffman).
- `src/utils.py`: binary IO helpers and experiment result storage.
- `data/text8`: sample dataset used in experiments.
- `adaptors`: location of trained LoRa / VeRa adaptors 

## Setup
1. Clone the repo:
```
git clone https://github.com/utndatasystems/summer-offsite.git
cd summer-offsite
```
2. Run the setup script to download the text8 dataset, create a virtual environment, install dependencies, and pre-populate the Hugging Face cache used by the benchmark models:
```
./setup.sh --model-cache-dir /path/to/shared/hf-cache
source .venv/bin/activate
```

The selected cache directory is persisted in `.model_cache_dir`, and the runtime code will reuse it on future runs. You can override it per job by setting `COMPRESSION_ECONOMICS_MODEL_CACHE=/path/to/shared/hf-cache`. If you need authenticated downloads during setup, export `HF_TOKEN` before running the script.

If you only want the environment and dataset setup, skip the model pre-download step:
```
./setup.sh --skip-model-downloads
```

To enable the CPU ONNX Runtime backend, install the optional extra after activating the virtual environment:
```
pip install -e .[onnxruntime]
```

To enable the direct in-process llama.cpp backend (`--engine llamacpp_direct`), install the optional extra after activating the virtual environment:
```
pip install -e .[llamacpp_direct]
```

To enable the Apple Silicon MLX backend on macOS arm64, install the optional extra in an Apple-hosted environment:
```
pip install -e .[mlx]
```

## Basic usage
### Compression
```
python main.py \
  --mode compress
```

Compression using an existing adaptor:
```
python main.py \
    --mode compress \
    --input_path ./data/text8 \
    --first_n_tokens 100000 \
    --batch_size 16 \
    --lora_path ./adapters/vera/text8/r4_lr0.0005_lsconstant_bs64_ep4_gas2/
```

### Decompression
```
python main.py \
  --mode decompress \
  --input_path compression_data.bin
```

Decompression reads all required settings from the binary header, so only the
compressed file path is required.

### Adaptor Training
```
python adapter_training.py \
    --adapter_type lora \
    --lr 0.0005 \
    --batch_size 64 \
    --r 4 \
    --epoch 4
```

## Key options
- `--input_path`: Text file to compress (compress mode) or `.bin` to decompress.
- `--output_path`: Override default output file path.
- `--model_name`: HuggingFace model name (tokenizer + LLM). Default: `Qwen/Qwen2.5-0.5B`.
- `--context_length`: Max context length for inference. Default: 1000.
- `--retain_tokens`: Context tail length when trimming. Default: 100.
- `--first_n_tokens`: Limit number of tokens processed. Default: 1000.
- `--batch_size`: Number of parallel sequences per step. Default: 1.
- `--use_kv_cache`: Enable KV cache for faster incremental inference. Default: enabled.
- `--reduce_tokens/--no_reduce_tokens`: Toggle global vocabulary reduction. Default: enabled.
- `--encoding`: `AC`, `bitpacked`, or `huffman`. Default: `AC`.
- `--gpu_memory_utilization`: vLLM-only. Override the fraction of GPU memory reserved for model weights and KV cache.
- `--tensorrt_engine_dir`: TensorRT-only. Path to a prebuilt TensorRT-LLM engine directory.
- `--sglang_mem_fraction_static`: SGlang-only. Override the fraction of GPU memory reserved by the SGlang engine.
- `--sglang_enable_deterministic_inference/--no_sglang_enable_deterministic_inference`: SGlang-only. Control deterministic engine execution.
- `--llamacpp_model_path`: llama.cpp-only. Path to a local GGUF model used by both `llamacpp` and `llamacpp_direct`.
- `--llamacpp_binary`: llama.cpp server-only. Path to the `llama-server` binary for `--engine llamacpp`.
- `--llamacpp_host`: llama.cpp server-only. Host for the managed local server used by `--engine llamacpp`.
- `--llamacpp_port`: llama.cpp server-only. Port for the managed local server used by `--engine llamacpp`.
- `--llamacpp_threads`: llama.cpp-only. CPU thread count used by either the managed server or the direct binding.
- `--llamacpp_direct_threads_batch`: direct llama.cpp-only. Batch-processing thread count for `--engine llamacpp_direct`.
- `--llamacpp_direct_n_batch`: direct llama.cpp-only. Prompt-processing batch size for `--engine llamacpp_direct`.
- `--llamacpp_direct_n_ubatch`: direct llama.cpp-only. Physical micro-batch size for `--engine llamacpp_direct`.
- `--llamacpp_direct_use_mmap/--no_llamacpp_direct_use_mmap`: direct llama.cpp-only. Toggle mmap-backed GGUF loading.
- `--llamacpp_direct_use_mlock/--no_llamacpp_direct_use_mlock`: direct llama.cpp-only. Toggle locking GGUF weights in RAM.
- `--llamacpp_n_gpu_layers`: llama.cpp-only. Number of layers to offload to GPU.
- `--mlx_model_source`: MLX-only. Local MLX model directory or MLX-compatible Hugging Face repo such as `mlx-community/...`.
- `--mlx_tokenizer_source`: MLX-only. Optional tokenizer source when it differs from the MLX model source.
- `--onnx_model_dir`: ONNX Runtime-only. Path to a local exported ONNX model directory.
- `--onnx_tokenizer_source`: ONNX Runtime-only. Optional tokenizer source when the ONNX export directory does not include tokenizer files.
- `--onnx_execution_provider`: ONNX Runtime-only. Execution provider for inference. The current implementation is scoped to `CPUExecutionProvider`.
- `--onnx_intra_op_threads`: ONNX Runtime-only. Intra-op CPU thread count.
- `--onnx_inter_op_threads`: ONNX Runtime-only. Inter-op CPU thread count.
- `--onnx_graph_optimization_level`: ONNX Runtime-only. Graph optimization level passed to ONNX Runtime.
- `--print_results`: Print detailed stats to stdout. Default: disabled.

When `--engine vllm` is used and `CUDA_VISIBLE_DEVICES` is not already set, the runtime will prefer the visible GPU with the most free memory and will clamp the requested memory reservation to fit current free memory.

Native vLLM arithmetic coding is verified against vLLM `0.17.1` in this repo. Rank-based encodings still use the bulk `prompt_logprobs` path, while arithmetic coding uses a separate internal full-logits compatibility path to retrieve dense next-token logits before sampler truncation.

The TensorRT backend expects a prebuilt TensorRT-LLM engine directory via `--tensorrt_engine_dir`. The current implementation is scoped to CUDA plus prebuilt engine artifacts and keeps HuggingFace tokenization outside the runtime for compression/decompression consistency. LoRA adapters are not yet supported on this backend in this repo.

The SGlang backend uses the embedded `sglang.Engine` API and reconstructs per-step next-token scores by probing the active token set with `token_ids_logprob`. This keeps the predictor contract identical to the transformer and vLLM backends. In reduced-token mode this is a practical AC path; in unreduced full-vocab mode it is functionally correct but heavier than vLLM's native dense-logits capture path.

The validated environment for this backend used `sglang 0.5.9`. Older SGlang releases such as `0.5.2` can fail during engine initialization against the current `transformers` build with `TypeError: AutoImageProcessor.register() got multiple values for argument 'exist_ok'`. If you see that error, upgrade SGlang in the active environment before retrying.

The ONNX Runtime backend expects a local exported ONNX model directory via `--onnx_model_dir` and currently targets `CPUExecutionProvider` only. It reuses the repository's predictor contract and Hugging Face tokenizer path, making it a CPU-oriented backend for Linux x86_64, ARM/Graviton, and Apple Silicon CPU execution. This first implementation is intentionally export-first: the repo does not yet generate ONNX artifacts for you, and LoRA adapters are not supported on this backend.

The `llamacpp` backend keeps the original managed `llama-server` subprocess path and expects a local GGUF model via `--llamacpp_model_path` plus a usable `llama-server` binary. Tokenization and detokenization are handled through llama.cpp runtime endpoints so compression and decompression stay aligned with the loaded GGUF tokenizer. This backend currently requires `--reduce_tokens`.

The `llamacpp_direct` backend is the new in-process `llama-cpp-python` path. It loads the GGUF directly, uses save/load state for cache reuse under the repo's exact-one-token prompt-extension rule, and can run either reduced-token or full-vocabulary inference. The hot path now skips vocabulary reindexing when you are already in full-vocabulary mode, returns raw logits directly for rank-based encodings, and exposes direct-runtime tuning knobs for `n_batch`, `n_ubatch`, `n_threads_batch`, `mmap`, and `mlock`.

The MLX backend targets macOS on Apple Silicon and loads models through `mlx-lm`. Use an MLX-compatible repo or a local converted MLX directory via `--mlx_model_source`; `mlx-community/...` repos are the usual starting point. This first implementation uses direct MLX forward passes and supports incremental prompt-cache reuse for single-sequence runs, while larger batched runs currently fall back to full-prompt evaluation for correctness. LoRA adapters are not yet supported on this backend in this repo.

If that compatibility path is unavailable for the installed environment, the runtime falls back to the transformer backend for arithmetic coding with a warning that names the missing capability. The verified native AC path is currently limited to `tensor_parallel_size=1`.

## Separate backend benchmark commands
If you want a concrete list of benchmark commands for the Linux-supported backends, generate them with:
```
python evaluation/generate_backend_commands.py \
  --output-file artifacts/backend_benchmarks/run_all.sh
```

This writes one compression command and one matching decompression command for every requested dataset, backend, and model combination. The generated shell script covers:
- Hugging Face-native backends: `transformer`, `vllm`, `sglang`
- Artifact-backed Linux backends: `onnxruntime`, `tensorrt`, `llamacpp`, `llamacpp_direct`
- The benchmark model list from `src/model_registry.py`

The generator omits `mlx` on Linux because that backend is macOS/Apple-Silicon only.

Artifact-backed engines need local export paths, so the generator uses placeholder templates by default. Override them when you have real artifacts:
```
python evaluation/generate_backend_commands.py \
  --engines onnxruntime tensorrt llamacpp llamacpp_direct \
  --onnx-model-dir-template /models/onnx/{model_slug} \
  --tensorrt-engine-dir-template /models/tensorrt/{model_slug} \
  --llamacpp-model-path-template /models/gguf/{model_slug}.gguf \
  --output-file artifacts/backend_benchmarks/run_artifact_backends.sh
```

If you only want a smaller subset, restrict `--engines`, `--models`, or `--datasets`:
```
python evaluation/generate_backend_commands.py \
  --engines vllm sglang \
  --models Qwen/Qwen2.5-0.5B distilbert/distilgpt2
```

Each generated command pair writes metrics into a backend-specific JSON file under `artifacts/backend_benchmarks/{engine}/results.json` by default, so you can run the commands separately and still plot them together later.

## Cost versus speed plot
After you have run the compression and decompression command pairs, plot total speed versus total cost directly from the saved results JSON files:
```
python evaluation/plot_cost_vs_speed.py \
  --results-json \
    artifacts/backend_benchmarks/transformer/results.json \
    artifacts/backend_benchmarks/vllm/results.json \
    artifacts/backend_benchmarks/sglang/results.json \
    artifacts/backend_benchmarks/onnxruntime/results.json \
    artifacts/backend_benchmarks/tensorrt/results.json \
    artifacts/backend_benchmarks/llamacpp/results.json \
    artifacts/backend_benchmarks/llamacpp_direct/results.json \
  --output figures/cost_vs_speed.png \
  --summary-output artifacts/backend_benchmarks/cost_vs_speed.tsv \
  --annotate
```

The plot script computes:
- `total_cost_usd = (total_compression_time + total_decompression_time) * hourly_cost / 3600`
- `total_speed_tok_s = input_tokens_count / (total_compression_time + total_decompression_time)`

Override hardware or hourly costs when needed:
```
python evaluation/plot_cost_vs_speed.py \
  --results-json artifacts/backend_benchmarks/vllm/results.json \
  --gpu-cost 1.20 \
  --cpu-cost 0.03 \
  --engine-cost tensorrt=1.45 \
  --engine-hardware transformer=gpu
```

[ToDo: update key options with new training arguments]
## Outputs
- `compression_results.json`: Aggregated metrics keyed by experiment settings.
- `compression_data.bin`: Binary artifact (header + bitstream + bitmap).
- `text_results.txt`: Reconstructed text from decompression (default output).

## AWS one-shot runners
The repo now includes a repo-local EC2 launcher that packages the current workspace,
uploads the bundle to S3, starts one EC2 instance for a benchmark job, and relies on
instance-initiated shutdown behavior set to `terminate` so the machine powers off after
the run finishes.

### What the launcher does
- Packages the current workspace, including uncommitted local changes.
- Uploads the workspace bundle, resolved spec, rendered user-data, and optional assets to S3.
- Boots an EC2 instance from a launch template or explicit AMI config.
- Restores the workspace, runs `setup.sh` with the requested backend profile, executes one benchmark command, syncs results and logs to S3, and shuts the instance down.

### Prerequisites
1. Install the repo dependencies with the backend you want to run locally or remotely:
```
./setup.sh --backend-profile transformer
```

Use `vllm`, `sglang`, `tensorrt`, or `llamacpp` instead of `transformer` when you want a backend-specific environment through `setup.sh`. The new `llamacpp_direct` backend is installed separately with the optional extra shown above.

2. Create backend-specific EC2 launch templates. Each launch template should provide:
- A suitable AMI for the backend.
- An instance profile with S3 write access, optional SSM `GetParameter` access for secrets, and permission to use IMDSv2.
- Instance-initiated shutdown behavior set to `terminate`.
- Any GPU instance type, subnet, or security-group defaults you want to reuse.

3. Edit one of the sample specs in `aws/specs/` and replace the placeholder bucket, launch-template, and secret parameter names.

### Dry run
The launcher is safe by default. Without `--launch` it only renders the plan locally under `tmp/aws_launcher/`:
```
python aws/launch_experiment.py \
  --spec aws/specs/transformer_smoke.json
```

This writes:
- A workspace tarball.
- A resolved JSON spec with the run ID and bundle hash.
- The rendered cloud-init user-data script.

### Launch an EC2 job
```
python aws/launch_experiment.py \
  --spec aws/specs/vllm_grid_smoke.json \
  --launch
```

The launcher prints the run ID and the S3 prefix where logs and outputs will land.

### Spec format
Each spec is JSON and contains these high-level sections:
- `name`, `region`, `s3_bucket`, `s3_prefix`
- `backend_profile`: `transformer`, `vllm`, `sglang`, `tensorrt`, or `llamacpp`
- `ec2`: launch template or AMI details, optional spot settings, and tags
- `execution`: entrypoint, CLI args, environment variables, optional SSM-backed secret environment variables, timeout, setup mode, and upload paths
- `assets`: optional local files or directories to upload to S3 and restore on the instance, useful for TensorRT engine directories, adapters, or GGUF models

Only these entrypoints are allowed remotely:
- `main.py`
- `grid_search.py`
- `evaluation/multi_model_runner.py`
- `adapter_training.py`

### Backend notes
- `vllm` and `sglang` should run on separate backend-specific images or launch templates. Their extras conflict in `pyproject.toml`.
- `tensorrt` support assumes you already have a prebuilt TensorRT-LLM engine directory and provide it through an uploaded asset plus `--tensorrt_engine_dir`.
- `llamacpp` support assumes you provide a GGUF model and a usable `llama-server` binary on the AMI or as an asset.
- `llamacpp_direct` is not a dedicated `backend_profile` in `setup.sh` today; install its Python extra manually if you want to use it remotely.
- `skip-model-downloads` only skips the pre-download phase in `setup.sh`. If the model is missing from cache, the benchmark can still pull it at runtime.
