#!/bin/bash
set -e

# Configuration
MODELS_DIR="models"
TRT_CKPT_DIR="trt_checkpoints"
TRT_ENG_DIR="trt_engines"
GGUF_DIR="gguf_models"
DTYPE="bfloat16"

# Ensure directories exist
mkdir -p "$MODELS_DIR"
mkdir -p "$TRT_CKPT_DIR"
mkdir -p "$TRT_ENG_DIR"
mkdir -p "$GGUF_DIR"

# List of models to process (format: HF_Repo)
# Adjust these names if the official repositories for Qwen3 differ slightly
MODELS=(
    "Qwen/Qwen2.5-0.5B"
    "Qwen/Qwen2.5-1.5B"
    "Qwen/Qwen2.5-7B"
    "Qwen/Qwen3-0.6B"
    "Qwen/Qwen3-1.7B"
    "Qwen/Qwen3-8B"
)

echo "Starting Model Preparation Pipeline..."
echo "Models will be saved to: $MODELS_DIR"
echo "TRT Checkpoints will be saved to: $TRT_CKPT_DIR"
echo "TRT Engines will be saved to: $TRT_ENG_DIR"
echo "GGUF outputs will be saved to: $GGUF_DIR"

for MODEL_REPO in "${MODELS[@]}"; do
    # Extract just the model name, e.g., "Qwen2.5-0.5B"
    MODEL_NAME=$(basename "$MODEL_REPO")
    
    echo "============================================================"
    echo "Processing $MODEL_NAME"
    echo "============================================================"
    
    # Paths for this specific model
    LOCAL_MODEL_DIR="$MODELS_DIR/$MODEL_NAME"
    LOCAL_TRT_CKPT_DIR="$TRT_CKPT_DIR/$MODEL_NAME"
    LOCAL_TRT_ENG_DIR="$TRT_ENG_DIR/$MODEL_NAME"
    LOCAL_GGUF_FILE="$GGUF_DIR/${MODEL_NAME}.gguf"

    # Step 1: Download from Hugging Face
    echo "[1/4] Downloading $MODEL_REPO to $LOCAL_MODEL_DIR..."
    huggingface-cli download "$MODEL_REPO" --local-dir "$LOCAL_MODEL_DIR"

    # Step 2: Convert to TensorRT-LLM Checkpoint
    echo "[2/4] Creating TensorRT-LLM checkpoint in $LOCAL_TRT_CKPT_DIR..."
    python convert_checkpoint.py \
        --model_dir "$LOCAL_MODEL_DIR" \
        --output_dir "$LOCAL_TRT_CKPT_DIR" \
        --dtype "$DTYPE"

    # Step 3: Build TensorRT-LLM Engine
    echo "[3/4] Building TensorRT-LLM Engine in $LOCAL_TRT_ENG_DIR..."
    trtllm-build \
        --checkpoint_dir "$LOCAL_TRT_CKPT_DIR" \
        --output_dir "$LOCAL_TRT_ENG_DIR" \
        --gemm_plugin "$DTYPE" \
        --max_batch_size 512 \
        --max_input_len 1024 \
        --max_seq_len 1025 \
        --max_num_tokens 524288 \
        --gather_generation_logits

    # Step 4: Convert to GGUF format
    echo "[4/4] Exporting to GGUF format ($LOCAL_GGUF_FILE)..."
    if [ ! -f "llama.cpp/convert_hf_to_gguf.py" ]; then
        echo "Error: llama.cpp/convert_hf_to_gguf.py not found. Ensure llama.cpp is cloned in the root directory."
        exit 1
    fi
    
    # We use bf16 for outtype to match the bfloat16 request
    python llama.cpp/convert_hf_to_gguf.py \
        "$LOCAL_MODEL_DIR" \
        --outfile "$LOCAL_GGUF_FILE" \
        --outtype bf16

    echo "Finished processing $MODEL_NAME successfully!"
    echo "------------------------------------------------------------"
done

echo "Pipeline completed successfully for all models!"
