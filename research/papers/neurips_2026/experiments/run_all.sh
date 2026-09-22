#!/usr/bin/env bash

# Run every experiment needed by Neurips_ML_for_Systems-3.pdf.
#
# The default `all` mode runs:
#   1. tokenizer fertility on the held-out 5 MB text8 region;
#   2. long full-vocabulary, printable-ASCII, and one-byte attacks; and
#   3. arithmetic payload scoring for full/occurring dictionaries.
#
# All experiment programs checkpoint or are skipped when their expected result
# already exists. Set FORCE=1 to rerun the non-checkpointed fertility experiment.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../../../.." && pwd)"
cd "$REPO_ROOT"

MODE="${1:-all}"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
MODEL="${PAPER_MODEL:-Qwen/Qwen2.5-0.5B}"
TEXT8_PATH="${TEXT8_PATH:-data/text8}"
ARTIFACT_ROOT="${PAPER_ARTIFACT_ROOT:-artifacts/papers/neurips-2026}"
RUN_ROOT="${PAPER_RUN_ROOT:-$ARTIFACT_ROOT/runs}"

# Long greedy attacks used by the main paper table.
PAPER_LENGTH="${PAPER_LENGTH:-10000}"
MAX_SURPRISAL_LENGTH="${MAX_SURPRISAL_LENGTH:-1024}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-1000}"
RETAIN_TOKENS="${RETAIN_TOKENS:-100}"

FERTILITY_OUTPUT="${FERTILITY_OUTPUT:-$ARTIFACT_ROOT/studies/tokenizer-fertility}"
FORCE="${FORCE:-0}"
DRY_RUN="${DRY_RUN:-0}"

case "$MODE" in
  all|fertility|attacks) ;;
  *)
    echo "Usage: $0 [all|fertility|attacks]" >&2
    exit 2
    ;;
esac

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python executable not found or not executable: $PYTHON_BIN" >&2
  exit 1
fi
if [[ ! -f "$TEXT8_PATH" ]]; then
  echo "text8 input not found: $TEXT8_PATH" >&2
  exit 1
fi
if [[ "$FORCE" != "0" && "$FORCE" != "1" ]]; then
  echo "FORCE must be 0 or 1, got: $FORCE" >&2
  exit 2
fi
if [[ "$DRY_RUN" != "0" && "$DRY_RUN" != "1" ]]; then
  echo "DRY_RUN must be 0 or 1, got: $DRY_RUN" >&2
  exit 2
fi

run() {
  printf '+ '
  printf '%q ' "$@"
  printf '\n'
  if [[ "$DRY_RUN" == "0" ]]; then
    "$@"
  fi
}

stage() {
  printf '\n===== %s =====\n' "$1"
}

run_fertility() {
  stage "Tokenizer fertility"
  if [[ -f "$FERTILITY_OUTPUT/results.json" && "$FORCE" == "0" ]]; then
    echo "Skipping existing $FERTILITY_OUTPUT/results.json (set FORCE=1 to rerun)."
    return
  fi
  run "$PYTHON_BIN" research/papers/neurips_2026/experiments/tokenizer_fertility.py \
    --input "$TEXT8_PATH" \
    --output-dir "$FERTILITY_OUTPUT" \
    --train-bytes 90000000 \
    --eval-bytes 5000000 \
    --bpe-vocab-size 32000 \
    --qwen-tokenizer "$MODEL"
}

run_long_attacks() {
  stage "Long adversarial and control runs"
  run env \
    PAPER_LENGTH="$PAPER_LENGTH" \
    PAPER_MODEL="$MODEL" \
    PAPER_RUN_ROOT="$RUN_ROOT" \
    PAPER_CONTEXT_LENGTH="$CONTEXT_LENGTH" \
    PAPER_RETAIN_TOKENS="$RETAIN_TOKENS" \
    PAPER_TEXT8_PATH="$TEXT8_PATH" \
    PYTHON_BIN="$PYTHON_BIN" \
    bash research/papers/neurips_2026/experiments/paper_evaluation.sh core

  stage "Full-vocabulary random canonical control"
  run "$PYTHON_BIN" research/papers/neurips_2026/experiments/run_compression_attacks.py \
    --model-name "$MODEL" \
    --start-token-id 785 \
    --start-token-id 32 \
    --start-token-id 641 \
    --total-length "$PAPER_LENGTH" \
    --generation-alphabet full \
    --attack random-token \
    --context-length "$CONTEXT_LENGTH" \
    --retain-tokens "$RETAIN_TOKENS" \
    --ordinary-text "$TEXT8_PATH" \
    --random-utf8-bytes "$((2 * PAPER_LENGTH))" \
    --output-dir "$RUN_ROOT/controls/random-canonical/full-vocabulary/n$PAPER_LENGTH"

  # This objective performs a context-sensitive decode for every vocabulary
  # item at every step, so the paper reports it at a separate 1,024-token budget.
  stage "Full-vocabulary MaxSurprisal/Byte attack"
  run "$PYTHON_BIN" research/papers/neurips_2026/experiments/run_compression_attacks.py \
    --model-name "$MODEL" \
    --start-token-id 32 \
    --total-length "$MAX_SURPRISAL_LENGTH" \
    --generation-alphabet full \
    --attack surprisal-per-byte \
    --context-length "$CONTEXT_LENGTH" \
    --retain-tokens "$RETAIN_TOKENS" \
    --output-dir "$RUN_ROOT/attacks/max-surprisal-per-byte/full-vocabulary/n$MAX_SURPRISAL_LENGTH"
}

if [[ "$MODE" == "all" || "$MODE" == "fertility" ]]; then
  run_fertility
fi
if [[ "$MODE" == "all" || "$MODE" == "attacks" ]]; then
  run_long_attacks
fi
echo
echo "NeurIPS evaluation stage '$MODE' complete."
