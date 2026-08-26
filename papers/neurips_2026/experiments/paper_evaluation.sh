#!/usr/bin/env bash

set -euo pipefail

# Long greedy/replay experiments scale roughly linearly. Coder-aware beam search
# is much more expensive, so it has an independent, deliberately shorter budget.
PAPER_LENGTH="${PAPER_LENGTH:-10000}"
PAPER_SEARCH_LENGTH="${PAPER_SEARCH_LENGTH:-512}"
PAPER_BEAM_WIDTH="${PAPER_BEAM_WIDTH:-4}"
PAPER_BRANCH_FACTOR="${PAPER_BRANCH_FACTOR:-8}"
PAPER_MODEL="${PAPER_MODEL:-Qwen/Qwen2.5-0.5B}"
PAPER_ARTIFACT_ROOT="${PAPER_ARTIFACT_ROOT:-artifacts/papers/neurips-2026}"
PAPER_RUN_ROOT="${PAPER_RUN_ROOT:-$PAPER_ARTIFACT_ROOT/runs}"
PAPER_CONTEXT_LENGTH="${PAPER_CONTEXT_LENGTH:-1000}"
PAPER_RETAIN_TOKENS="${PAPER_RETAIN_TOKENS:-100}"
PAPER_TEXT8_PATH="${PAPER_TEXT8_PATH:-data/text8}"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
MODE="${1:-all}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python executable not found: $PYTHON_BIN" >&2
  exit 1
fi

case "$MODE" in
  all|core|search|payload) ;;
  *)
    echo "Usage: $0 [all|core|search|payload]" >&2
    exit 2
    ;;
esac

FULL_DIR="$PAPER_RUN_ROOT/attacks/minprob/full-vocabulary/n$PAPER_LENGTH"

run_full_generation() {
  if [[ -f "$FULL_DIR/full/results.json" && -f "$FULL_DIR/occurring/results.json" ]]; then
    echo "Skipping completed full-vocabulary generation at $FULL_DIR."
    return
  fi
  "$PYTHON_BIN" scripts/generate_adversarial.py \
    --model-name "$PAPER_MODEL" \
    --start-token-id 785 \
    --start-token-id 32 \
    --start-token-id 641 \
    --total-length "$PAPER_LENGTH" \
    --context-length "$PAPER_CONTEXT_LENGTH" \
    --retain-tokens "$PAPER_RETAIN_TOKENS" \
    --candidate-mode full \
    --candidate-mode occurring \
    --output-dir "$FULL_DIR"
}

score_full_payloads() {
  "$PYTHON_BIN" scripts/score_adversarial_payloads.py \
    --input-dir "$FULL_DIR"
}

run_long_byte_attacks() {
  local alphabet="$1"
  local output_dir
  case "$alphabet" in
    printable-ascii) output_dir="$PAPER_RUN_ROOT/auxiliary/printable-ascii/n$PAPER_LENGTH" ;;
    ascii-bytes) output_dir="$PAPER_RUN_ROOT/ablations/one-byte-utf8/n$PAPER_LENGTH" ;;
    *) echo "Unsupported paper alphabet: $alphabet" >&2; return 2 ;;
  esac
  "$PYTHON_BIN" scripts/run_compression_attacks.py \
    --model-name "$PAPER_MODEL" \
    --start-text A \
    --start-text B \
    --start-text C \
    --total-length "$PAPER_LENGTH" \
    --context-length "$PAPER_CONTEXT_LENGTH" \
    --retain-tokens "$PAPER_RETAIN_TOKENS" \
    --generation-alphabet "$alphabet" \
    --attack random-token \
    --attack min-probability \
    --attack surprisal-per-byte \
    --ordinary-text "$PAPER_TEXT8_PATH" \
    --random-utf8-bytes "$((2 * PAPER_LENGTH))" \
    --output-dir "$output_dir"
}

run_coder_aware_search() {
  "$PYTHON_BIN" scripts/run_compression_attacks.py \
    --model-name "$PAPER_MODEL" \
    --start-text A \
    --start-text B \
    --start-text C \
    --total-length "$PAPER_SEARCH_LENGTH" \
    --context-length "$PAPER_CONTEXT_LENGTH" \
    --retain-tokens "$PAPER_RETAIN_TOKENS" \
    --generation-alphabet ascii-bytes \
    --attack beam-surprisal-per-byte \
    --attack beam-actual-ratio \
    --beam-width "$PAPER_BEAM_WIDTH" \
    --branch-factor "$PAPER_BRANCH_FACTOR" \
    --output-dir "$PAPER_RUN_ROOT/ablations/one-byte-utf8/beam-n$PAPER_SEARCH_LENGTH"
}

if [[ "$MODE" == "all" || "$MODE" == "core" ]]; then
  run_full_generation
  score_full_payloads
  run_long_byte_attacks printable-ascii
  run_long_byte_attacks ascii-bytes
fi

if [[ "$MODE" == "all" || "$MODE" == "search" ]]; then
  run_coder_aware_search
fi

if [[ "$MODE" == "payload" ]]; then
  score_full_payloads
fi

echo "Paper evaluation stage '$MODE' complete under $PAPER_RUN_ROOT."
