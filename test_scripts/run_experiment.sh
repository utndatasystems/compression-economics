#!/bin/bash

# Exit immediately if any command fails
set -e

# Model list
# models=(
#     "distilgpt2"
#     "Qwen/Qwen2.5-0.5B"
#     "Qwen/Qwen2.5-1.5B"
#     "Qwen/Qwen2.5-7B"
#     "Qwen/Qwen3-0.6B"
#     "Qwen/Qwen3-1.7B"
#     "Qwen/Qwen3-8B"
# )
models=(
    # "Qwen/Qwen2.5-0.5B"
    "Qwen/Qwen3-0.6B-FP8"
)

# Dataset list
datasets=(
    "./data/text8"
    "./data/combined_100mb.py"
)

FIRST_N_TOKENS=500000
CONTEXT_LENGTH=100
BATCH_SIZE=256
RETAIN_TOKENS=10  # ctx=100 → retain=10 (adjust if needed)

for model in "${models[@]}"; do
  for dataset in "${datasets[@]}"; do
    echo "=== Running model: $model on dataset: $dataset ==="

    # Compression
    python main.py \
      --mode compress \
      --input_path "$dataset" \
      --use_kv_cache \
      --first_n_tokens "$FIRST_N_TOKENS" \
      --batch_size "$BATCH_SIZE" \
      --context_length "$CONTEXT_LENGTH" \
      --retain_tokens "$RETAIN_TOKENS" \
      --model_name "$model"

    # Decompression (uncomment if needed)
    python main.py \
      --mode decompress
  done
done
