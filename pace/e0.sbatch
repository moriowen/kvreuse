#!/bin/bash
#SBATCH -J kv-e0
#SBATCH -p ice-gpu
#SBATCH -N1 -c 8
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=64G
#SBATCH -t 1:00:00
#SBATCH -o logs/%x-%j.out
# E0 gate on the real model: fp32 checks, then the bf16 noise floor.
# Exits nonzero on FAIL, which cancels the jobs that depend on it.
source pace/env.sh
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

python scripts/e0_real.py --model "$MODEL" --n 20 2>&1 | tee results/e0.txt
