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

- `main.py`: compression and decompression CLI.
- `src/`: maintained compression, prediction, encoding, and training code.
- `scripts/`: standalone training, quantization, and data-generation CLIs.
- `experiments/`: version-controlled sweep definitions and run configurations.
- `evaluation/`: result loaders, baselines, plots, notebooks, and reference data.
- `tests/`: automated tests for maintained code.
- `data/`: local datasets (ignored).
- `artifacts/`: generated runs, figures, model weights, and logs (ignored).
- `archive/`: historical prototypes that are not part of the supported workflow.

See `experiments/README.md` and `evaluation/README.md` for the boundary between
running experiments and analyzing their output.

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

Compression using an existing adaptor:
```
python main.py \
    --mode compress \
    --input_path ./data/text8 \
    --first_n_tokens 100000 \
    --batch_size 16 \
    --lora_path ./artifacts/models/adapters/vera/text8/r4_lr0.0005_lsconstant_bs64_ep4_gas2/
```

### Decompression
```
python main.py \
  --mode decompress \
  --input_path artifacts/runs/current/compression_data.bin
```

Decompression reads all required settings from the binary header, so only the
compressed file path is required.

### Adaptor Training
```
python scripts/train_adapter.py \
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
- `--print_results`: Print detailed stats to stdout. Default: disabled.

[ToDo: update key options with new training arguments]

## Outputs
- `artifacts/runs/current/compression_results.json`: aggregated experiment metrics.
- `artifacts/runs/current/compression_data.bin`: binary bitstream artifact.
- `artifacts/runs/current/text_results.txt`: reconstructed text.

## Adversarial worst-case inputs

`scripts/generate_adversarial.py` creates several equal-token-length runs by repeatedly
selecting the allowed token with the lowest finite next-token logit. Since softmax
preserves ordering, this is exactly the lowest-probability token without numerical
underflow from materializing tiny probabilities.

The default variants are the full non-special vocabulary and an occurring-token
fixed point. The latter repeatedly generates, masks to the tokens that actually
occurred, and regenerates until the candidate set is stable. Top-k means the k most
frequent tokens in a supplied reference corpus. Reference processing is capped at
100,000 tokens by default; change it with --reference-max-tokens.

```bash
python scripts/generate_adversarial.py \
  --model-name Qwen/Qwen2.5-0.5B \
  --start-text "The" \
  --start-text "A" \
  --start-text "In" \
  --total-length 1000 \
  --candidate-mode full \
  --candidate-mode occurring
```

For a frequency-restricted dictionary:

```bash
python scripts/generate_adversarial.py \
  --start-text "The" \
  --start-text "A" \
  --total-length 1000 \
  --candidate-mode top-k \
  --top-k 1000 \
  --reference-path data/text8
```

Results are stored below `artifacts/runs/adversarial/<variant>/`. Each run has decoded
text and an authoritative `.tokens.json` file. The combined `results.json`
records fixed-point convergence, token IDs, text round-trip status, per-step log
probabilities, and full-vocabulary versus mask-renormalized surprisal. The fixed
total length includes the starting token or starting text.
