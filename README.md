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
- `--print_results`: Print detailed stats to stdout. Default: disabled.

When `--engine vllm` is used and `CUDA_VISIBLE_DEVICES` is not already set, the runtime will prefer the visible GPU with the most free memory and will clamp the requested memory reservation to fit current free memory.

Native vLLM arithmetic coding is verified against vLLM `0.17.1` in this repo. Rank-based encodings still use the bulk `prompt_logprobs` path, while arithmetic coding uses a separate internal full-logits compatibility path to retrieve dense next-token logits before sampler truncation.

If that compatibility path is unavailable for the installed environment, the runtime falls back to the transformer backend for arithmetic coding with a warning that names the missing capability. The verified native AC path is currently limited to `tensor_parallel_size=1`.

[ToDo: update key options with new training arguments]
## Outputs
- `compression_results.json`: Aggregated metrics keyed by experiment settings.
- `compression_data.bin`: Binary artifact (header + bitstream + bitmap).
- `text_results.txt`: Reconstructed text from decompression (default output).
