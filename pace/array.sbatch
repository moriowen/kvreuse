#!/bin/bash
#SBATCH -J kv-sweep
#SBATCH -p ice-gpu
#SBATCH -N1 -c 8
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=64G
#SBATCH -t 3:00:00
#SBATCH -o logs/%x-%A_%a.out
# Runs lines of a job list on one GPU: task k runs lines k*PER_TASK+1 .. (k+1)*PER_TASK.
# Packing several runs per task keeps us far under ICE's 50-jobs-per-user cap (array tasks
# count, and the cap is shared with other coursework). Submit only after E0 passes:
#   sbatch --array=0-3 --export=ALL,PER_TASK=4 pace/array.sbatch pace/jobs/e1.txt
#   sbatch --array=0-1 --export=ALL,PER_TASK=8 pace/array.sbatch pace/jobs/e2.txt
source pace/env.sh
per=${PER_TASK:-1}
first=$(( SLURM_ARRAY_TASK_ID * per + 1 ))
nvidia-smi --query-gpu=name,memory.total --format=csv
echo "node $(hostname) task $SLURM_ARRAY_TASK_ID lines $first..$(( first + per - 1 )) start $(date)"
sed -n "${first},$(( first + per - 1 ))p" "$1" | while IFS= read -r cmd; do
    [ -z "$cmd" ] && continue
    echo "cmd: $cmd"
    eval "$cmd" < /dev/null
done
echo "done $(date)"
