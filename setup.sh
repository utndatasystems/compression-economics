#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="./data"
TEXT8_ZIP="$DATA_DIR/text8.zip"
TEXT8_FILE="$DATA_DIR/text8"

# python environment setup
if [ ! -d "./.venv" ]; then
  echo "[INFO] Creating Python virtual environment"
  python3 -m venv ./.venv
else
  echo "[SKIP] Python virtual environment already exists"
fi
source ./.venv/bin/activate
echo "[INFO] Installing Python dependencies"
pip install --upgrade pip
pip install -r requirements.txt

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