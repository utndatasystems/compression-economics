#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
MODEL="Qwen/Qwen2.5-0.5B"
MODEL_REVISION="060db6499f32faf8b98477b0a26969ef7d8b9987"
INPUT="$ROOT/data/text8"
RESULTS="$ROOT/results"
EXPERIMENTS="$ROOT/research/papers/coding_survey/experiments"

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "Required file is missing: $1" >&2
    exit 1
  fi
}

verify_input() {
  require_file "$INPUT"
  local actual
  actual="$(head -c 10485760 "$INPUT" | sha256sum | cut -d' ' -f1)"
  if [[ "$actual" != "d14c3bba3810d20335ded84f81248fa97c8ae2709744f7762c793bcc189c1dad" ]]; then
    echo "The first 10 MiB of data/text8 has the wrong SHA-256: $actual" >&2
    exit 1
  fi
}

run_cuda_point() {
  local batch="$1"
  local context="$2"
  local retain="$3"
  local bytes="$4"
  local repeats="$5"
  "$PYTHON" "$EXPERIMENTS/benchmark_text8_cuda_roundtrip.py" \
    --input "$INPUT" --input-bytes "$bytes" \
    --model "$MODEL" --batch-size "$batch" \
    --context-length "$context" --retain-tokens "$retain" \
    --repeats "$repeats" --encoders cuda_v3 --decoders cuda \
    --decoder-archive-encoder cuda_v3 \
    --output "$RESULTS/cuda/text8-b${batch}-c${context}-r${retain}.json"
}

run_smoke() {
  require_file "$PYTHON"
  mkdir -p "$RESULTS/cuda"
  run_cuda_point 16 128 64 32768 1
}

run_cuda() {
  require_file "$PYTHON"
  verify_input
  mkdir -p "$RESULTS/cuda"
  run_cuda_point 1024 512 256 10485760 3
  run_cuda_point 8192 128 64 10485760 3
  run_cuda_point 16384 128 64 10485760 3
}

run_traditional() {
  require_file "$PYTHON"
  verify_input
  local fsst_dir="${FSST_DIR:-$ROOT/third_party/fsst}"
  local revision="e638d4cf8c26129d73c242a4127b42b975de5b63"
  if [[ ! -d "$fsst_dir/.git" ]]; then
    echo "Clone FSST first: git clone https://github.com/cwida/fsst.git $fsst_dir" >&2
    exit 1
  fi
  local actual_revision
  actual_revision="$(git -C "$fsst_dir" rev-parse HEAD)"
  if [[ "$actual_revision" != "$revision" ]]; then
    echo "FSST must be checked out at $revision (found $actual_revision)" >&2
    exit 1
  fi
  mkdir -p "$RESULTS/traditional" "$ROOT/build"
  g++ -std=c++17 -O3 -DNDEBUG -march=native -fPIC -shared \
    -I"$fsst_dir" "$EXPERIMENTS/fsst8_wrapper.cpp" \
    "$fsst_dir/libfsst.cpp" "$fsst_dir/fsst_avx512.cpp" \
    -o "$ROOT/build/libartifact_fsst8.so"
  "$PYTHON" "$EXPERIMENTS/benchmark_traditional_codecs.py" \
    --input "$INPUT" --input-bytes 10485760 --repeats 7 \
    --fsst-library "$ROOT/build/libartifact_fsst8.so" \
    --fsst-revision "$revision" \
    --output "$RESULTS/traditional/text8-first-10MiB.json"
}

run_parallelism() {
  require_file "$PYTHON"
  verify_input
  mkdir -p "$RESULTS/parallelism"
  "$PYTHON" "$EXPERIMENTS/benchmark_speculative_full.py" \
    --dataset "$INPUT:10485760" --model "$MODEL" \
    --model-revision "$MODEL_REVISION" \
    --output "$RESULTS/parallelism/speculative-full.json"
  "$PYTHON" "$EXPERIMENTS/benchmark_teacher_forcing.py" \
    --input "$INPUT" --input-bytes 10485760 --tokens 0 \
    --model "$MODEL" --model-revision "$MODEL_REVISION" \
    --batch-size 256 --context-length 256 --retain-tokens 100 \
    --teacher-window-size 16 --repeats 5 --device cuda --dtype bfloat16 \
    --output "$RESULTS/parallelism/teacher-forcing-w16.json"
}

usage() {
  echo "Usage: bash reproduce.sh {smoke|cuda|traditional|parallelism|all}" >&2
}

case "${1:-}" in
  smoke) run_smoke ;;
  cuda) run_cuda ;;
  traditional) run_traditional ;;
  parallelism) run_parallelism ;;
  all)
    run_cuda
    run_traditional
    run_parallelism
    ;;
  *) usage; exit 2 ;;
esac
