"""Check CUDA readiness and record a reproducible GPU environment snapshot.

Run from the repository root after syncing the project environment. This check
does not download models or datasets and requires a visible CUDA device.
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import transformers

from src.coding.target_interval import target_intervals_from_probs_tensor


def _version_command(command: list[str]) -> str | None:
    if shutil.which(command[0]) is None:
        return None
    result = subprocess.run(command, capture_output=True, text=True, timeout=15, check=False)
    return (result.stdout or result.stderr).strip() if result.returncode == 0 else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Write the environment snapshot as JSON")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("PyTorch cannot access CUDA; check the driver, GPU allocation, and Torch build")
    device = torch.device("cuda", torch.cuda.current_device())
    properties = torch.cuda.get_device_properties(device)
    probabilities = torch.tensor([[0.125, 0.375, 0.5]], dtype=torch.float32, device=device)
    low, high, total, _ = target_intervals_from_probs_tensor(
        probabilities, torch.tensor([1], device=device)
    )
    torch.cuda.synchronize(device)
    if (low.item(), high.item(), total.item()) != (32768, 131071, 262142):
        raise RuntimeError("CUDA target-interval quantization failed its known-value check")
    snapshot = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "transformers": transformers.__version__,
        "device_name": properties.name,
        "device_index": device.index,
        "compute_capability": f"{properties.major}.{properties.minor}",
        "device_memory_bytes": properties.total_memory,
        "device_count": torch.cuda.device_count(),
        "nvidia_smi": _version_command(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"]),
        "nvcc": _version_command(["nvcc", "--version"]),
        "quantizer_check": "passed",
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(snapshot, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(snapshot, indent=2))


if __name__ == "__main__":
    main()
