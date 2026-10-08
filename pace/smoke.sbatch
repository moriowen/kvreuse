#!/bin/bash
#SBATCH -J kv-smoke
#SBATCH -p ice-gpu
#SBATCH -N1 -c 8
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=64G
#SBATCH -t 2:00:00
#SBATCH -o logs/%x-%j.out
# First real numbers: full prefill vs. re-positioning only, on the first N questions
# (default 100), bf16. Enough to see the pipeline answer questions; not a reportable result.
source pace/env.sh
N="${N:-100}"
nvidia-smi --query-gpu=name,memory.total --format=csv

python scripts/run_eval.py --model "$MODEL" --method full --limit "$N"
python scripts/run_eval.py --model "$MODEL" --method reposition --limit "$N" --kl
