#!/bin/bash
#SBATCH -J kv-baselines
#SBATCH -p ice-gpu
#SBATCH -N1 -c 8
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=96G
#SBATCH -t 5:00:00
#SBATCH -o logs/%x-%j.out
# Week-7 milestone: baseline F1 over all 1,986 questions, bf16.
#   full          retrieved top-10 prompt (~2.4K tokens), the reference for every method
#   full_context  whole conversation (17K-31K tokens), no retrieval
source pace/env.sh
nvidia-smi --query-gpu=name,memory.total --format=csv

python scripts/run_eval.py --model "$MODEL" --method full
python scripts/run_eval.py --model "$MODEL" --method full_context
