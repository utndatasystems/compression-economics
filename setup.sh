#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="./data"
TEXT8_ZIP="$DATA_DIR/text8.zip"
TEXT8_FILE="$DATA_DIR/text8"

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
uv pip install -r requirements.txt

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

echo "[DONE] All resources are ready."