#!/bin/bash
#SBATCH -J kv-profile
#SBATCH -p ice-gpu
#SBATCH -N1 -c 8
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=64G
#SBATCH -t 0:30:00
#SBATCH -o logs/%x-%j.out
# Where prefill time goes: full prefill vs. re-positioning on one real prompt.
source pace/env.sh
python scripts/profile_prefill.py --model "$MODEL" --question 0 2>&1 | tee results/profile.txt
