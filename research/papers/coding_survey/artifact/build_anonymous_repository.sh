#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../../../.." && pwd)"
OUTPUT_PATH="${1:-/tmp/sigmod-2027-artifact}"

if [[ -e "$OUTPUT_PATH" ]]; then
  echo "Output path already exists; choose a new empty path: $OUTPUT_PATH" >&2
  exit 1
fi

STAGING="$(mktemp -d)"
cleanup() {
  rm -r -- "$STAGING"
}
trap cleanup EXIT
ARTIFACT="$STAGING/artifact"
mkdir -p "$ARTIFACT"

copy_file() {
  local source="$1"
  local destination="${2:-$1}"
  if [[ ! -f "$REPO_ROOT/$source" ]]; then
    echo "Required artifact file is missing: $source" >&2
    exit 1
  fi
  mkdir -p "$ARTIFACT/$(dirname -- "$destination")"
  cp -- "$REPO_ROOT/$source" "$ARTIFACT/$destination"
}

# Reviewer-facing files.
copy_file research/papers/coding_survey/artifact/ARTIFACT_README.md README.md
copy_file research/papers/coding_survey/artifact/requirements.txt requirements.txt
copy_file research/papers/coding_survey/artifact/reproduce.sh reproduce.sh
copy_file research/papers/coding_survey/artifact/EXTERNAL_BASELINES.md \
  research/papers/coding_survey/experiments/EXTERNAL_BASELINES.md
copy_file research/papers/coding_survey/artifact/LICENSE LICENSE
copy_file research/papers/coding_survey/artifact/gitignore .gitignore

# Main-paper experiment entry points only (no appendix ablations or tests).
for name in \
  benchmark_text8_cuda_roundtrip.py \
  benchmark_traditional_codecs.py \
  fsst8_wrapper.cpp \
  benchmark_finezip.py \
  benchmark_llmzip.py \
  benchmark_nacrith.py \
  summarize_llm_baselines.py \
  plot_optimization_operating_points.py \
  benchmark_speculative_full.py \
  audit_speculative_numerics.py \
  benchmark_teacher_forcing.py; do
  copy_file "research/papers/coding_survey/experiments/$name"
done

# Minimal in-repository runtime imported by the end-to-end CUDA runner.
for path in \
  src/__init__.py \
  src/global_mask_compressor.py \
  src/models.py \
  src/prediction.py \
  src/predictors.py \
  src/coding/__init__.py \
  src/coding/encoding.py \
  src/coding/encoding_utils.py \
  src/coding/multistream_ac.py \
  src/coding/parallel_ac.py \
  src/coding/cuda_ac.py \
  src/coding/device_ac.py \
  src/coding/paired_ac.py \
  src/coding/target_interval.py \
  src/coding/csrc/cuda_ac.cpp \
  src/coding/csrc/cuda_ac_kernel.cu \
  research/__init__.py \
  research/papers/__init__.py; do
  copy_file "$path"
done

# Keep package imports valid without exposing the manuscript-side directory.
mkdir -p "$ARTIFACT/research/papers/coding_survey"
printf '%s\n' '"""Anonymous artifact experiments."""' \
  > "$ARTIFACT/research/papers/coding_survey/__init__.py"
printf '%s\n' '"""Main-paper experiment entry points."""' \
  > "$ARTIFACT/research/papers/coding_survey/experiments/__init__.py"

chmod +x "$ARTIFACT/reproduce.sh"

# Generated/cached material and VCS metadata are forbidden even if the
# allowlist changes later.
if find "$ARTIFACT" -type d \( \
  -name .git -o -name .cache -o -name .venv -o -name __pycache__ \
\) -print -quit | grep -q .; then
  echo "Forbidden generated or VCS directory found in artifact" >&2
  exit 1
fi
if find "$ARTIFACT" -type f \( -name '*.pyc' -o -name '*.pyo' -o -name '*.so' \) \
  -print -quit | grep -q .; then
  echo "Forbidden generated binary found in artifact" >&2
  exit 1
fi

# Extend this expression with every author's name, username, email domain,
# affiliation, project name, and local cluster identifier before submission.
IDENTITY_PATTERN='(/home/|/Users/|@utn\.de|@in\.tum\.de|@campus\.lmu\.de|utndatasystems|UTN Data Systems|neubauer|andreas kipf|kipf|robin sternberg|sternberg|zimmerer|heller|stoian|skander|ping-lin|tobias schmidt|v164be)'
if grep -EIRn "$IDENTITY_PATTERN" "$ARTIFACT"; then
  echo "Potential author, institution, or local-path identity found" >&2
  exit 1
fi

SECRET_PATTERN='(-----BEGIN [A-Z ]*PRIVATE KEY-----|AKIA[0-9A-Z]{16}|gh[pousr]_[A-Za-z0-9_]{20,}|hf_[A-Za-z0-9]{20,})'
if grep -ERn "$SECRET_PATTERN" "$ARTIFACT"; then
  echo "Potential credential found" >&2
  exit 1
fi

# Syntax and shell checks do not execute model code or require dependencies.
python3 -m compileall -q "$ARTIFACT"
find "$ARTIFACT" -type d -name __pycache__ -prune -exec rm -r -- {} +
bash -n "$ARTIFACT/reproduce.sh"

mkdir -p -- "$(dirname -- "$OUTPUT_PATH")"
mv -- "$ARTIFACT" "$OUTPUT_PATH"
echo "Created standalone anonymous repository: $OUTPUT_PATH"
echo "Files: $(find "$OUTPUT_PATH" -type f | wc -l)"
echo "Bytes: $(du -sb "$OUTPUT_PATH" | cut -f1)"
echo "Review README.md and add all author-specific terms to IDENTITY_PATTERN before upload."
