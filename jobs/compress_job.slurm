#!/bin/bash
#SBATCH --job-name=mamba-compress      # Job name
#SBATCH --output=logs/%x_%j.out        # Standard output (%x = job name, %j = job ID)
#SBATCH --error=logs/%x_%j.err         # Standard error
#SBATCH --gres=gpu:1                   # Request 1 GPU
#SBATCH --cpus-per-task=4              # Number of CPU cores
#SBATCH --time=02:00:00                # Max runtime (HH:MM:SS)
#SBATCH --mail-type=END,FAIL           # Email notification
#SBATCH --mail-user=luisa.neubauer@utn.de  # Your email

# Load necessary modules (CUDA, cuDNN, Python)
module load cuda/12.8.0
module load cudnn/8.9.7.29-12
module load python/3.12-conda          # Adjust to Python version you need

# Activate your uv environment (if using uv)
source /home/hpc/v164be/v164be10/src/compression-economics/.venv/bin/activate

# Optional: make logs directory
mkdir -p logs

# Run your Python script
python main.py \
  --mode compress \
  --first_n_tokens 100000 \
  --batch_size 16 \
  --model state-spaces/mamba-130m-hf