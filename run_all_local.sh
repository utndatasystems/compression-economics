#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

HARDWARE="${1:-a100}"
JOBS_DIR="jobs/${HARDWARE}"

if [ ! -d "$JOBS_DIR" ]; then
    echo "Error: $JOBS_DIR does not exist. Run python generate_jobs.py first."
    exit 1
fi

echo "Running all job scripts locally..."

for script in "$JOBS_DIR"/*.sh; do
    if [ -x "$script" ]; then
        echo "=========================================================="
        echo "Executing: $script"
        echo "=========================================================="
        "$script"
    fi
done

echo "All local jobs finished."
