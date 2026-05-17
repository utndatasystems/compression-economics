#!/usr/bin/env bash

set -euo pipefail

echo "Setting up uv virtual environments from pyproject.toml..."

MAPPINGS=(
  "vllm-env:vllm"
  "sglang-env:sglang"
  "transformers-env:transformer"
#   "tensorrt-llm-env:tensorrt"
  "onnx-env:onnx"
  "lamacpp-cpu-env:llamacpp"
  "lamacpp-gpu-env:llamacpp"
)

for MAP in "${MAPPINGS[@]}"; do
    ENV_DIR="${MAP%%:*}"
    EXTRA="${MAP##*:}"
    
    echo "==================================================="
    if [ ! -d "$ENV_DIR" ]; then
        echo "Creating Virtual Environment: $ENV_DIR"
        uv venv "$ENV_DIR"
    else
        echo "Environment $ENV_DIR already exists."
    fi

    echo "Syncing dependencies for $ENV_DIR with extra: [$EXTRA]"
    
    if [ "$ENV_DIR" == "lamacpp-gpu-env" ]; then
        echo "Building llama-cpp-python with CUDA support for $ENV_DIR..."
        CMAKE_ARGS="-DGGML_CUDA=on" FORCE_CMAKE=1 uv pip install -p "$ENV_DIR" llama-cpp-python --no-binary llama-cpp-python --force-reinstall
        uv pip install -p "$ENV_DIR" -e ".[$EXTRA]"
    elif [ "$ENV_DIR" == "tensorrt-llm-env" ]; then
        echo "Installing specific dependencies for $ENV_DIR..."
        uv pip install -p "$ENV_DIR" "tensorrt_llm<=1.0.0" tensorrt-cu12 nvidia-nccl-cu12 nvidia-cublas-cu12
        uv pip install -p "$ENV_DIR" -e ".[$EXTRA]"
    else
        uv pip install -p "$ENV_DIR" -e ".[$EXTRA]"
    fi
done

echo "==================================================="
echo "Done scaffolding environments. Dependencies are installed!"
