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

ENCODINGS=("AC") # "huffman" "bitpacked")
BATCH_SIZES=(16 32 64 128 256)

declare -A MODEL_LORA_PATHS
MODEL_LORA_PATHS["Qwen/Qwen2.5-0.5B"]="NONE" #"./adapters/lora/text8/r4_lr0.0005_lsconstant_bs64_ep2_gas2/"
MODEL_LORA_PATHS["Qwen/Qwen2.5-3B"]="NONE"
MODEL_LORA_PATHS["Qwen/Qwen2.5-7B"]="NONE"
MODEL_LORA_PATHS["Qwen/Qwen3-0.6B"]="NONE" #"./adapters/vera/text8/r4_lr0.0005_lsconstant_bs64_ep2_gas2"
MODEL_LORA_PATHS["Qwen/Qwen3-2B"]="NONE"
MODEL_LORA_PATHS["Qwen/Qwen3-4B"]="NONE"
MODEL_LORA_PATHS["gpt2"]="NONE"

QUANT_PATHS="NONE"
#(    "./quantization/text8/r4_lr0.0005_lsconstant_bs64_ep2_gas2/"
 #   "NONE")

MODELS=(
    "Qwen/Qwen2.5-0.5B"
    "Qwen/Qwen2.5-3B"
    "Qwen/Qwen2.5-7B"
    "Qwen/Qwen3-0.6B"
    "Qwen/Qwen3-2B"
    "Qwen/Qwen3-4B"
    "gpt2",
    "state-spaces/mamba-130m-hf",

)

############################################
# Run LoRA experiments
############################################

for MODEL in "${MODELS[@]}"; do
  for ENCODING in "${ENCODINGS[@]}"; do
  #TODO : add condition for huffmann to only do Batch sizes 16

    for BATCH_SIZE in "${BATCH_SIZES[@]}"; do
        LORA_PATH="${MODEL_LORA_PATHS[$MODEL]}"

        if [ "$LORA_PATH" == "NONE" ]; then
          :
        else
          :
        fi

          for LORA_PATH in "${LORA_PATHS[@]}"; do

            echo "=============================================="
            echo "Model: $MODEL"
            echo "Encoding: $ENCODING"
            echo "Batch size: $BATCH_SIZE"
            echo "LoRA path: $LORA_PATH"
            echo "=============================================="

            if [ "$LORA_PATH" == "NONE" ]; then
              python main.py \
                --mode compress \
                --model "$MODEL" \
                --input_path "$INPUT_PATH" \
                --first_n_tokens "$FIRST_N_TOKENS" \
                --batch_size "$BATCH_SIZE" \
                --encoding "$ENCODING"
            else
              python main.py \
                --mode compress \
                --model "$MODEL" \
                --input_path "$INPUT_PATH" \
                --first_n_tokens "$FIRST_N_TOKENS" \
                --batch_size "$BATCH_SIZE" \
                --lora_path "$LORA_PATH" \
                --encoding "$ENCODING"
            fi

      done
    done
  done
done

############################################
# Run quantization experiments
############################################

for MODEL in "${MODELS[@]}"; do
  for ENCODING in "${ENCODINGS[@]}"; do
    for BATCH_SIZE in "${BATCH_SIZES[@]}"; do
      for QUANT_PAT in "${QUANT_PATHS[@]}"; do

        echo "=============================================="
        echo "Model: $MODEL"
        echo "Encoding: $ENCODING"
        echo "Batch size: $BATCH_SIZE"
        echo "Quantization pattern: $QUANT_PAT"
        echo "=============================================="

        if [ "$QUANT_PAT" == "NONE" ]; then
          python main.py \
            --mode compress \
            --model "$MODEL" \
            --input_path "$INPUT_PATH" \
            --first_n_tokens "$FIRST_N_TOKENS" \
            --batch_size "$BATCH_SIZE" \
            --encoding "$ENCODING"
        else
          python main.py \
            --mode compress \
            --model "$MODEL" \
            --input_path "$INPUT_PATH" \
            --first_n_tokens "$FIRST_N_TOKENS" \
            --batch_size "$BATCH_SIZE" \
            #TODO: include quantization args
            --encoding "$ENCODING"
        fi

      done
    done
  done
done

echo "All runs completed."