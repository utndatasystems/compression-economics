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
  - `AC` (arithmetic coding),
  - `AC_MULTISTREAM` (versioned multi-stream arithmetic coding for higher throughput),
  - `AC_TARGET_INTERVAL` (target-interval arithmetic coding to reduce dense score movement),
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

## Basic usage
### Compression
```
python main.py \
  --mode compress
```

Fast arithmetic coding (target-interval example):
```
python main.py \
  --mode compress \
  --encoding AC_TARGET_INTERVAL
```

`AC_MULTISTREAM` and `AC_TARGET_INTERVAL` default to `--encode_backend auto`, which uses Numba
when available and falls back to the Python backend otherwise. Use
`--encode_backend python` for debugging or bitstream comparisons, or try
`--encode_backend numba_threaded --encode_threads N` for large row-stream
counts. For the transformer engine, `--encode_backend numba_packed --encode_threads N --pipeline_encoding`
writes packed byte streams and pipelines multistream interval staging behind the next inference batch.

### Decompression
```
python main.py \
  --mode decompress \
  --input_path compression_data.bin
```

Decompression reads all required settings from the binary header, so only the
compressed file path is required.

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
- `--encoding`: `AC`, `AC_MULTISTREAM`, `AC_TARGET_INTERVAL`, `bitpacked`, or `huffman`. Default: `AC`.
- `--print_results`: Print detailed stats to stdout. Default: disabled.

## Outputs
- `compression_results.json`: Aggregated metrics keyed by experiment settings.
- `compression_data.bin`: Binary artifact (header + bitstream + bitmap).
- `text_results.txt`: Reconstructed text from decompression (default output).
