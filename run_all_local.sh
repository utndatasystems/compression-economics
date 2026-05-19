#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

HARDWARE="${1:-a100}"
NUM_GPUS="${2:-1}"
JOBS_DIR="jobs/${HARDWARE}"

if [ ! -d "$JOBS_DIR" ]; then
    echo "Error: $JOBS_DIR does not exist. Run python generate_jobs.py first."
    exit 1
fi

echo "Running all job scripts locally with ${NUM_GPUS} GPUs..."

# Array of available GPUs
declare -a available_gpus
for (( i=0; i<NUM_GPUS; i++ )); do
    available_gpus+=($i)
done

# Array of currently running jobs and their GPUs
declare -A running_pids
declare -A pid_to_gpu

for script in "$JOBS_DIR"/*.sh; do
    if [ ! -x "$script" ]; then
        continue
    fi

    # Wait if no GPUs are available
    while [ ${#available_gpus[@]} -eq 0 ]; do
        # Wait for any child process to finish
        wait -n
        # Find which PID finished and reclaim its GPU
        for pid in "${!running_pids[@]}"; do
            if ! kill -0 "$pid" 2>/dev/null; then
                gpu=${pid_to_gpu[$pid]}
                available_gpus+=($gpu)
                unset running_pids[$pid]
                unset pid_to_gpu[$pid]
            fi
        done
    done

    # Get the next available GPU
    gpu_idx=$(( ${#available_gpus[@]} - 1 ))
    gpu=${available_gpus[$gpu_idx]}
    unset available_gpus[$gpu_idx]

    echo "=========================================================="
    echo "Executing: $script on GPU $gpu"
    echo "=========================================================="
    
    # Run in background with specific GPU
    CUDA_VISIBLE_DEVICES=$gpu "$script" &
    pid=$!
    
    running_pids[$pid]=1
    pid_to_gpu[$pid]=$gpu
done

# Wait for all remaining jobs
wait

echo "All local jobs finished."
