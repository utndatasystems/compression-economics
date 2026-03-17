#!/usr/bin/env bash
set -euo pipefail

MODEL_CACHE_DIR="${COMPRESSION_ECONOMICS_MODEL_CACHE:-$PWD/.cache}"
DOWNLOAD_MODELS=1
DATA_DIR="./data"
TEXT8_ZIP="$DATA_DIR/text8.zip"
TEXT8_FILE="$DATA_DIR/text8"
MODEL_CACHE_CONFIG_FILE=".model_cache_dir"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-cache-dir)
      if [[ $# -lt 2 ]]; then
        echo "[ERROR] --model-cache-dir requires a value" >&2
        exit 1
      fi
      MODEL_CACHE_DIR="$2"
      shift 2
      ;;
    --skip-model-downloads)
      DOWNLOAD_MODELS=0
      shift
      ;;
    *)
      echo "[ERROR] Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

mkdir -p "$MODEL_CACHE_DIR"
MODEL_CACHE_DIR="$(cd "$MODEL_CACHE_DIR" && pwd)"
printf '%s\n' "$MODEL_CACHE_DIR" > "$MODEL_CACHE_CONFIG_FILE"
export COMPRESSION_ECONOMICS_MODEL_CACHE="$MODEL_CACHE_DIR"
export HF_HOME="$MODEL_CACHE_DIR"
export HF_HUB_CACHE="$MODEL_CACHE_DIR/hub"
export TRANSFORMERS_CACHE="$MODEL_CACHE_DIR/transformers"

echo "[INFO] Model cache directory: $MODEL_CACHE_DIR"

# Install uv if not already installed
if ! command -v uv &> /dev/null; then
  echo "[INFO] Installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
else
  echo "[SKIP] uv already installed ($(uv --version))"
fi

# Python environment setup
if [ ! -d "./.venv" ]; then
  echo "[INFO] Creating Python virtual environment with uv"
  uv venv ./.venv
else
  echo "[SKIP] Python virtual environment already exists"
fi
source ./.venv/bin/activate
echo "[INFO] Installing Python dependencies with uv"
uv sync --group dev

# Download text8 dataset 

mkdir -p "$DATA_DIR"

if [ ! -f "$TEXT8_FILE" ]; then
  echo "[INFO] Downloading text8 dataset"

  if [ ! -f "$TEXT8_ZIP" ]; then
    wget -O "$TEXT8_ZIP" http://mattmahoney.net/dc/text8.zip
  else
    echo "[SKIP] text8.zip already exists"
  fi

  unzip -o "$TEXT8_ZIP" -d "$DATA_DIR"
  rm -f "$TEXT8_ZIP"
else
  echo "[SKIP] text8 dataset already exists"
fi

if [ "$DOWNLOAD_MODELS" -eq 1 ]; then
  echo "[INFO] Downloading benchmark models into cache"
  python - <<'PY'
import os

from huggingface_hub import snapshot_download

from src.model_registry import BENCHMARK_MODEL_IDS


cache_dir = os.environ["COMPRESSION_ECONOMICS_MODEL_CACHE"]
hf_token = os.environ.get("HF_TOKEN")

for model_id in BENCHMARK_MODEL_IDS:
    print(f"[INFO] Caching {model_id} into {cache_dir}")
    snapshot_download(
        repo_id=model_id,
        cache_dir=cache_dir,
        token=hf_token,
        resume_download=True,
    )
PY
else
  echo "[SKIP] Model pre-download skipped"
fi

echo "[DONE] All resources are ready."