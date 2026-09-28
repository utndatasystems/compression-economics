#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$repo_root"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required. Install it using https://docs.astral.sh/uv/getting-started/installation/" >&2
  exit 1
fi

# The broad root setup.sh downloads several datasets. This script prepares
# only the locked Python environment and validates the CUDA decoder path.
uv sync --locked --no-dev --extra parallel

output_dir="artifacts/papers/coding-survey/gpu-setup"
mkdir -p "$output_dir"
.venv/bin/python research/papers/coding_survey/experiments/gpu_preflight.py \
  --output "$output_dir/environment.json"
.venv/bin/python -m pytest -q tests/test_device_ac.py tests/test_target_interval.py

echo "GPU environment and decoder tests passed. Snapshot: $output_dir/environment.json"
