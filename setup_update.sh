#!/usr/bin/env bash
set -euo pipefail

# --------------------------------------------------
# Resolve script directory (robust execution)
# --------------------------------------------------
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
cd "$SCRIPT_DIR"

DATA_DIR="./data"

TEXT8_ZIP="$DATA_DIR/text8.zip"
TEXT8_FILE="$DATA_DIR/text8"

DATASETS=("dbtext" "languages" "onpair" "bitext")

mkdir -p "$DATA_DIR"

# --------------------------------------------------
# Python environment setup
# --------------------------------------------------
if [ ! -d "./.venv" ]; then
  echo "[INFO] Creating Python virtual environment"
  python3 -m venv .venv
else
  echo "[SKIP] Python virtual environment already exists"
fi

source .venv/bin/activate

echo "[INFO] Installing Python dependencies"
pip install --upgrade pip
pip install -r requirements.txt

# --------------------------------------------------
# Download text8 dataset
# --------------------------------------------------
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

# --------------------------------------------------
# Download additional datasets
# --------------------------------------------------
for dataset in "${DATASETS[@]}"; do
  TARGET_DIR="$DATA_DIR/$dataset"

  if [ "$dataset" = "bitext" ]; then
    TARGET_DIR="$DATA_DIR/textcolumns"
  fi

  if [ ! -d "$TARGET_DIR" ]; then
    echo "[INFO] Downloading dataset: $dataset"

    ARCHIVE="$DATA_DIR/${dataset}.tar.gz"
    if [ ! -f "$ARCHIVE" ]; then
      wget \
        "https://db.in.tum.de/~schmidt/data/${dataset}.tar.gz" \
        -O "$ARCHIVE"
    else
      echo "[SKIP] Archive already exists: ${dataset}.tar.gz"
    fi

    tar -xzf "$ARCHIVE" -C "$DATA_DIR"
    if [ ! -d "$TARGET_DIR" ]; then
      echo "[ERROR] Failed to extract dataset: $dataset"
      exit 1
    else
      echo "[INFO] Successfully extracted dataset: $dataset"
      rm "$ARCHIVE"
    fi

  else
    echo "[SKIP] Dataset already exists: $dataset"
  fi
done

echo "[DONE] All resources are ready."