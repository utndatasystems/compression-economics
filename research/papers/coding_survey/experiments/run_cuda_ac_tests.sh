#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$repo_root"

partition="${CUDA_AC_PARTITION:-h100}"
gpu_type="${CUDA_AC_GPU_TYPE:-h100}"
time_limit="${CUDA_AC_TIME_LIMIT:-00:20:00}"
cpus="${CUDA_AC_CPUS:-4}"

if [[ "${1:-}" == "--allocated" ]]; then
  shift
elif [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  if ! command -v srun >/dev/null 2>&1; then
    echo "CUDA is not visible and Slurm is unavailable." >&2
    exit 1
  fi
  exec srun \
    --partition="$partition" \
    --gres="gpu:${gpu_type}:1" \
    --ntasks=1 \
    --cpus-per-task="$cpus" \
    --time="$time_limit" \
    bash "$0" --allocated "$@"
fi

module use /apps/modules/data-gpu/2025/development
module use /apps/modules/data-gpu/2025/tools
module load cuda/12.8.0
module load nccl/2.28.7-gcc11.5.0-cuda
module load ninja/1.12.1

export MAX_JOBS="${MAX_JOBS:-1}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${TMPDIR:-/tmp}/torch_extensions}"
mkdir -p "$TORCH_EXTENSIONS_DIR"

.venv/bin/python - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA is not visible inside the allocated job")
print(f"torch={torch.__version__} runtime={torch.version.cuda}")
print(f"device={torch.cuda.get_device_name()} capability={torch.cuda.get_device_capability()}")
PY

.venv/bin/python -m pytest -q \
  tests/test_device_ac.py tests/test_target_interval.py \
  tests/test_target_interval_cli.py \
  -k 'cuda or fused' -rs "$@"
