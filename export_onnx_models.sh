#!/usr/bin/env bash

set -euo pipefail

ONNX_TASK="${ONNX_TASK:-text-generation}"
FORCE_REEXPORT="${FORCE_REEXPORT:-0}"

MODELS=(
    "Qwen2.5-0.5B"
    "Qwen2.5-1.5B"
    "Qwen2.5-7B"
    "Qwen3-0.6B"
    "Qwen3-1.7B"
    "Qwen3-8B"
)

if [ "$#" -gt 0 ]; then
    MODELS=("$@")
fi

echo "Exporting Hugging Face models to ONNX using Optimum..."
source .venv/bin/activate || true

for model in "${MODELS[@]}"; do
    if [ "$FORCE_REEXPORT" = "1" ] || [ ! -f "models/onnx/$model/model.onnx" ]; then
        echo "Exporting $model..."
        rm -rf "models/onnx/$model"
        mkdir -p "models/onnx/$model"
        # Requires optimum[onnxruntime] installed in .venv
        optimum-cli export onnx --model "models/$model" --task "$ONNX_TASK" "models/onnx/$model/"
    else
        echo "$model ONNX already exists. Skipping."
    fi
done

echo "Done exporting all models to ONNX!"
