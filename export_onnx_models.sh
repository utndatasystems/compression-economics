#!/usr/bin/env bash

set -euo pipefail

MODELS=(
    "Qwen2.5-0.5B"
    "Qwen2.5-1.5B"
    "Qwen2.5-7B"
    "Qwen3-0.6B"
    "Qwen3-1.7B"
    "Qwen3-8B"
)

echo "Exporting Hugging Face models to ONNX using Optimum..."
source .venv/bin/activate || true

for model in "${MODELS[@]}"; do
    if [ ! -f "models/onnx/$model/model.onnx" ]; then
        echo "Exporting $model..."
        # Requires optimum[onnxruntime] installed in .venv
        optimum-cli export onnx --model "models/$model" --task text-generation "models/onnx/$model/"
    else
        echo "$model ONNX already exists. Skipping."
    fi
done

echo "Done exporting all models to ONNX!"
