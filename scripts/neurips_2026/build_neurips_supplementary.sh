#!/usr/bin/env bash

# Build the anonymous, review-stage NeurIPS supplementary ZIP. The archive is
# intentionally assembled from an allowlist rather than from the Git tree.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
OUTPUT_PATH="${1:-$REPO_ROOT/papers/neurips_2026/supplementary/neurips-2026-supplementary.zip}"
ARCHIVE_ROOT_NAME="compression-economics-artifact"
MAX_BYTES=100000000

for command_name in grep zip; do
  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "Required command not found: $command_name" >&2
    exit 1
  fi
done

BUILD_DIR="$(mktemp -d)"
cleanup() {
  rm -r -- "$BUILD_DIR"
}
trap cleanup EXIT

ARCHIVE_ROOT="$BUILD_DIR/$ARCHIVE_ROOT_NAME"
mkdir -p -- "$ARCHIVE_ROOT"

copy_file() {
  local relative_path="$1"
  if [[ ! -f "$REPO_ROOT/$relative_path" ]]; then
    echo "Required artifact file is missing: $relative_path" >&2
    exit 1
  fi
  mkdir -p -- "$ARCHIVE_ROOT/$(dirname -- "$relative_path")"
  cp -- "$REPO_ROOT/$relative_path" "$ARCHIVE_ROOT/$relative_path"
}

for relative_path in \
  experiments/tokenizer_fertility.py \
  scripts/generate_adversarial.py \
  scripts/run_compression_attacks.py \
  scripts/score_adversarial_payloads.py \
  artifacts/papers/neurips-2026/manifest.json \
  papers/neurips_2026/experiments/paper_evaluation.sh \
  papers/neurips_2026/experiments/run_all.sh \
  src/__init__.py \
  src/adversarial.py \
  src/compression_attacks.py \
  src/encoding.py \
  src/encoding_utils.py \
  src/prediction.py; do
  copy_file "$relative_path"
done

cp -- "$REPO_ROOT/papers/neurips_2026/supplementary/ARTIFACT_CARD.md" "$ARCHIVE_ROOT/README.md"
cp -- "$REPO_ROOT/papers/neurips_2026/supplementary/LICENSE" "$ARCHIVE_ROOT/LICENSE"
cp -- "$REPO_ROOT/papers/neurips_2026/supplementary/pyproject.toml" "$ARCHIVE_ROOT/pyproject.toml"
cp -- "$REPO_ROOT/papers/neurips_2026/supplementary/uv.lock" "$ARCHIVE_ROOT/uv.lock"

# Source-tree caches are generated locally and are never part of the research
# artifact. Removal is confined to the freshly created temporary staging tree.
find "$ARCHIVE_ROOT" -type d \( \
  -name __pycache__ -o -name .pytest_cache -o -name .ipynb_checkpoints \
\) -prune -exec rm -r -- {} +
find "$ARCHIVE_ROOT" -type f \( -name '*.pyc' -o -name '*.pyo' \) -delete
find "$ARCHIVE_ROOT" -type f \( \
  -name '*.orig' -o -name '*.rej' -o -name '*~' -o -name '.DS_Store' \
\) -delete

find "$ARCHIVE_ROOT" -type d \( \
  -name .git -o -name .cache -o -name .venv -o -name __pycache__ \
\) -print -quit | grep -q . && {
  echo "Forbidden directory found in artifact" >&2
  exit 1
}

IDENTITY_PATTERN='(/home/|/Users/|@utn\.de|@in\.tum\.de|@campus\.lmu\.de|neubauer|zimmerer|heller|stoian|skander|ping-lin|tobias schmidt|v164be)'
if grep -EIRn "$IDENTITY_PATTERN" "$ARCHIVE_ROOT"; then
  echo "Potential author or institutional identity found in artifact" >&2
  exit 1
fi

SECRET_PATTERN='(-----BEGIN [A-Z ]*PRIVATE KEY-----|AKIA[0-9A-Z]{16}|gh[pousr]_[A-Za-z0-9_]{20,}|hf_[A-Za-z0-9]{20,})'
if grep -ERn "$SECRET_PATTERN" "$ARCHIVE_ROOT"; then
  echo "Potential credential found in artifact" >&2
  exit 1
fi

# Normalize timestamps and strip ZIP extra attributes to avoid leaking local
# filesystem metadata. ZIP archives do not include the source repository's Git
# history or remote configuration.
find "$ARCHIVE_ROOT" -exec touch -t 202601010000 {} +
mkdir -p -- "$(dirname -- "$OUTPUT_PATH")"
rm -f -- "$OUTPUT_PATH"
(
  cd "$BUILD_DIR"
  zip -X -q -r "$OUTPUT_PATH" "$ARCHIVE_ROOT_NAME"
)

ARCHIVE_BYTES="$(stat -c '%s' "$OUTPUT_PATH")"
if (( ARCHIVE_BYTES >= MAX_BYTES )); then
  echo "Archive is not below the 100MB NeurIPS limit: $ARCHIVE_BYTES bytes" >&2
  exit 1
fi

echo "Created $OUTPUT_PATH ($ARCHIVE_BYTES bytes)."
