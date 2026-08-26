#!/usr/bin/env bash

# Run every experiment needed by Neurips_ML_for_Systems-3.pdf.
#
# The default `all` mode runs:
#   1. tokenizer fertility on the held-out 5 MB text8 region;
#   2. 100k-token text8 compression with full and occurring vocabularies;
#   3. long full-vocabulary, printable-ASCII, and one-byte attacks;
#   4. arithmetic payload scoring for full/occurring dictionaries;
#   5. bounded full-vocabulary and one-byte beam searches.
#
# All experiment programs checkpoint or are skipped when their expected result
# already exists. Set FORCE=1 to rerun the non-checkpointed fertility experiment.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

MODE="${1:-all}"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
MODEL="${PAPER_MODEL:-Qwen/Qwen2.5-0.5B}"
TEXT8_PATH="${TEXT8_PATH:-data/text8}"
ARTIFACT_ROOT="${PAPER_ARTIFACT_ROOT:-artifacts/papers/neurips-2026}"
RUN_ROOT="${PAPER_RUN_ROOT:-$ARTIFACT_ROOT/runs}"

# Long greedy attacks and natural-text settings used by the main paper table.
PAPER_LENGTH="${PAPER_LENGTH:-10000}"
TEXT8_TOKENS="${TEXT8_TOKENS:-100000}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-1000}"
RETAIN_TOKENS="${RETAIN_TOKENS:-100}"
TEXT8_BATCH_SIZE="${TEXT8_BATCH_SIZE:-16}"

# Beam search has a separate budget because it clones model and coder state.
PAPER_SEARCH_LENGTH="${PAPER_SEARCH_LENGTH:-512}"
PAPER_BEAM_WIDTH="${PAPER_BEAM_WIDTH:-4}"
PAPER_BRANCH_FACTOR="${PAPER_BRANCH_FACTOR:-8}"
PAPER_FIXED_OVERHEAD_BITS="${PAPER_FIXED_OVERHEAD_BITS:-0}"

FERTILITY_OUTPUT="${FERTILITY_OUTPUT:-$ARTIFACT_ROOT/studies/tokenizer-fertility}"
FORCE="${FORCE:-0}"
DRY_RUN="${DRY_RUN:-0}"

case "$MODE" in
  all|fertility|natural|attacks|search) ;;
  *)
    echo "Usage: $0 [all|fertility|natural|attacks|search]" >&2
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
  run "$PYTHON_BIN" experiments/tokenizer_fertility.py \
    --input "$TEXT8_PATH" \
    --output-dir "$FERTILITY_OUTPUT" \
    --train-bytes 90000000 \
    --eval-bytes 5000000 \
    --bpe-vocab-size 32000 \
    --qwen-tokenizer "$MODEL"
}

run_natural_text() {
  stage "Natural text8: full vocabulary"
  run "$PYTHON_BIN" main.py \
    --mode compress \
    --input_path "$TEXT8_PATH" \
    --output_path "$RUN_ROOT/natural-text/text8/n${TEXT8_TOKENS}/full/compression_data.bin" \
    --model_name "$MODEL" \
    --first_n_tokens "$TEXT8_TOKENS" \
    --context_length "$CONTEXT_LENGTH" \
    --retain_tokens "$RETAIN_TOKENS" \
    --batch_size "$TEXT8_BATCH_SIZE" \
    --encoding AC \
    --no_reduce_tokens

  stage "Natural text8: occurring-token vocabulary"
  run "$PYTHON_BIN" main.py \
    --mode compress \
    --input_path "$TEXT8_PATH" \
    --output_path "$RUN_ROOT/natural-text/text8/n${TEXT8_TOKENS}/occurring/compression_data.bin" \
    --model_name "$MODEL" \
    --first_n_tokens "$TEXT8_TOKENS" \
    --context_length "$CONTEXT_LENGTH" \
    --retain_tokens "$RETAIN_TOKENS" \
    --batch_size "$TEXT8_BATCH_SIZE" \
    --encoding AC \
    --reduce_tokens
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
    bash papers/neurips_2026/experiments/paper_evaluation.sh core

  # The existing core sweep constructs the full-vocabulary MinProb sequence,
  # then covers byte-restricted objectives. These are the missing full-vocabulary
  # random-canonical and byte-aware rows from the paper's decisive table.
  stage "Full-vocabulary random and surprisal-per-byte attacks"
  run "$PYTHON_BIN" scripts/run_compression_attacks.py \
    --model-name "$MODEL" \
    --start-token-id 785 \
    --start-token-id 32 \
    --start-token-id 641 \
    --total-length "$PAPER_LENGTH" \
    --generation-alphabet full \
    --attack random-token \
    --attack surprisal-per-byte \
    --context-length "$CONTEXT_LENGTH" \
    --retain-tokens "$RETAIN_TOKENS" \
    --ordinary-text "$TEXT8_PATH" \
    --random-utf8-bytes "$((2 * PAPER_LENGTH))" \
    --output-dir "$RUN_ROOT/attacks/max-surprisal-per-byte/full-vocabulary/n$PAPER_LENGTH"
}

run_searches() {
  stage "One-byte coder-aware beam search"
  run env \
    PAPER_SEARCH_LENGTH="$PAPER_SEARCH_LENGTH" \
    PAPER_BEAM_WIDTH="$PAPER_BEAM_WIDTH" \
    PAPER_BRANCH_FACTOR="$PAPER_BRANCH_FACTOR" \
    PAPER_MODEL="$MODEL" \
    PAPER_RUN_ROOT="$RUN_ROOT" \
    PAPER_CONTEXT_LENGTH="$CONTEXT_LENGTH" \
    PAPER_RETAIN_TOKENS="$RETAIN_TOKENS" \
    PAPER_TEXT8_PATH="$TEXT8_PATH" \
    PYTHON_BIN="$PYTHON_BIN" \
    bash papers/neurips_2026/experiments/paper_evaluation.sh search

  stage "Full-vocabulary ideal and realized-size beam search"
  run "$PYTHON_BIN" scripts/run_compression_attacks.py \
    --model-name "$MODEL" \
    --start-token-id 785 \
    --start-token-id 32 \
    --start-token-id 641 \
    --total-length "$PAPER_SEARCH_LENGTH" \
    --generation-alphabet full \
    --attack beam-surprisal-per-byte \
    --attack beam-actual-ratio \
    --context-length "$CONTEXT_LENGTH" \
    --retain-tokens "$RETAIN_TOKENS" \
    --beam-width "$PAPER_BEAM_WIDTH" \
    --branch-factor "$PAPER_BRANCH_FACTOR" \
    --fixed-overhead-bits "$PAPER_FIXED_OVERHEAD_BITS" \
    --output-dir "$RUN_ROOT/attacks/realized-size-beam/full-vocabulary/n$PAPER_SEARCH_LENGTH"
}

if [[ "$MODE" == "all" || "$MODE" == "fertility" ]]; then
  run_fertility
fi
if [[ "$MODE" == "all" || "$MODE" == "natural" ]]; then
  run_natural_text
fi
if [[ "$MODE" == "all" || "$MODE" == "attacks" ]]; then
  run_long_attacks
fi
if [[ "$MODE" == "all" || "$MODE" == "search" ]]; then
  run_searches
fi
echo
echo "NeurIPS evaluation stage '$MODE' complete."
