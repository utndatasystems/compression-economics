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
2. Run the setup script to download the text8 dataset, create a virtual environment, and install dependencies:
```
./setup.sh
source .venv/bin/activate
```

## tensorrt_llm backend

Unfortunately this requires working with prebuilt wheels hence setuptools. You should do the following after setting up your uv envrionment as it probably won't sync correctly for your platform. It is reccomended to use the most recent version of CUDA possible for your system (13+ is best). Please note that GPU's prior to ampere are not supported.

Versions from 1.0.0 to 1.2.0 should work, but you may need to try a few versions to find one that works for your platform and CUDA/TensorRT stack. If you have a compatible version, the following command should install the package without errors:

```
source .venv/bin/activate
uv pip install --upgrade pip setuptools
uv pip install tensorrt_llm
```

to check if things will work.

```
from tensorrt_llm.runtime import ModelRunner
print("TensorRT-LLM runtime import OK")
```

### Engine requirements

If you want to build your own engine for the same model/tokenizer passed via `--model_name`. The
engine limits must cover the compression run:

- engine `max_batch_size` >= `--batch_size`
- engine `max_input_len` >= `--context_length`
- engine `max_output_len` >= 1
- generation logits must be enabled when building the engine

For example, if the compression run uses `--batch_size 1024` and
`--context_length 128`, build the TensorRT-LLM engine with at least those
capacity limits. Engine plans are generally tied to the GPU architecture,
TensorRT-LLM version, and CUDA/TensorRT stack used to build them, so rebuild the
engine if you move to a materially different runtime.

See NVIDIA's current TensorRT-LLM documentation for installation and engine
build commands:

- https://nvidia.github.io/TensorRT-LLM/latest/installation/index.html
- https://nvidia.github.io/TensorRT-LLM/latest/commands/trtllm-build.html

### Run compression with tensorrt_llm
Example using a prebuilt engine:

```
python main.py \
  --mode compress \
  --input_path ./data/text8 \
  --model_name qwen/qwen2.5-0.5b \
  --engine tensorrt \
  --tensorrt_engine_dir trt_engines/qwen_qwen2.5-0.5b/ctx256_batch1024 \
  --encoding ac \
  --first_n_tokens 100000 \
  --batch_size 1024 \
  --context_length 128 \
  --force
```

or with automatic building
```
python main.py \
  --mode compress \
  --input_path ./data/text8 \
  --model_name qwen/qwen2.5-0.5b \
  --engine tensorrt \
  --encoding ac \
  --first_n_tokens 100000 \
  --batch_size 1024 \
  --context_length 128 \
  --force \
```

The tensorrt path ignores `--use_kv_cache`; tensorrt_llm manages runtime caching
internally. For decompression, the engine path is stored in the compressed file
header, so the same engine directory must still exist at that path:

```
python main.py \
  --mode decompress \
  --input_path compression_data.bin
```

If tensorrt_llm is unavailable, the engine directory is missing, or the engine
was built with insufficient batch/context capacity, startup will fail before the
compression loop begins.

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
- `--engine`: Inference backend, either `transformer` or `tensorrt`. Default: `transformer`.
- `--tensorrt_engine_dir`: Prebuilt TensorRT-LLM engine directory when using `--engine tensorrt`.
- `--encoding`: `AC`, `PMATIC`, `bitpacked`, or `huffman`. Default: `AC`.
- `--print_results`: Print detailed stats to stdout. Default: disabled.

[ToDo: update key options with new training arguments]
## Outputs
- `compression_results.json`: Aggregated metrics keyed by experiment settings.
- `compression_data.bin`: Binary artifact (header + bitstream + bitmap).
- `text_results.txt`: Reconstructed text from decompression (default output).
