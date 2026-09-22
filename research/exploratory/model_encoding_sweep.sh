#!/usr/bin/env bash

set -e  # stop on first error

############################################
# Activate virtual environment
############################################
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
else
    echo "Virtual environment not found at .venv/bin/activate"
    exit 1
fi

############################################
# Configurations
############################################

INPUT_PATH="./data/text8"
FIRST_N_TOKENS=100000

ENCODINGS=("AC" "huffman" "bitpacked")
BATCH_SIZES=(16 32 64 128 256)

LORA_PATHS=(
    "./artifacts/models/adapters/lora/text8/r4_lr0.0005_lsconstant_bs64_ep2_gas2/"
    "./artifacts/models/adapters/vera/text8/r4_lr0.0005_lsconstant_bs64_ep2_gas2"
    "NONE"
)

#TODO: include quantization experiments as well

############################################
# Run experiments
############################################

for ENCODING in "${ENCODINGS[@]}"; do
  for BATCH_SIZE in "${BATCH_SIZES[@]}"; do
    for LORA_PATH in "${LORA_PATHS[@]}"; do

      echo "=============================================="
      echo "Encoding: $ENCODING"
      echo "Batch size: $BATCH_SIZE"
      echo "LoRA path: $LORA_PATH"
      echo "=============================================="

      if [ "$LORA_PATH" == "NONE" ]; then
        python main.py \
          --mode compress \
          --input_path "$INPUT_PATH" \
          --first_n_tokens "$FIRST_N_TOKENS" \
          --batch_size "$BATCH_SIZE" \
          --encoding "$ENCODING"
      else
        python main.py \
          --mode compress \
          --input_path "$INPUT_PATH" \
          --first_n_tokens "$FIRST_N_TOKENS" \
          --batch_size "$BATCH_SIZE" \
          --lora_path "$LORA_PATH" \
          --encoding "$ENCODING"
      fi

    done
  done
done

QUANT_PATHS=(
    "./quantization/text8/r4_lr0.0005_lsconstant_bs64_ep2_gas2/" # adjust this
    "NONE"
)

for ENCODING in "${ENCODINGS[@]}"; do
  for BATCH_SIZE in "${BATCH_SIZES[@]}"; do
    for QUANT_PAT in "${QUANT_PATHS[@]}"; do

      echo "=============================================="
      echo "Encoding: $ENCODING"
      echo "Batch size: $BATCH_SIZE"
      echo "Quantization pattern: $QUANT_PAT"
      echo "=============================================="

      if [ "$QUANT_PAT" == "NONE" ]; then
        python main.py \
          --mode compress \
          --input_path "$INPUT_PATH" \
          --first_n_tokens "$FIRST_N_TOKENS" \
          --batch_size "$BATCH_SIZE" \
          --encoding "$ENCODING"
      else
        python main.py \
          --mode compress \
          --input_path "$INPUT_PATH" \
          --first_n_tokens "$FIRST_N_TOKENS" \
          --batch_size "$BATCH_SIZE" \
          #TODO: include quantization args
          --encoding "$ENCODING"
      fi

    done
  done
done



echo "All runs completed."