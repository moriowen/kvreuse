#!/bin/bash
#SBATCH -J kv-serveprep
#SBATCH -p ice-cpu
#SBATCH -N1 -c 4
#SBATCH --mem=16G
#SBATCH -t 0:30:00
#SBATCH -o logs/%x-%j.out
# E3/E4 request workloads (C4); tokenizer only, so CPU. E4 also needs the chunk store for all
# ten conversations (~33 GB, GPU): pace/jobs/store_all.txt. E3 uses the conv-26 store from E5.
#   sbatch pace/serve_prep.sbatch
#   sbatch --array=0 pace/array.sbatch pace/jobs/store_all.txt
source pace/env.sh
mkdir -p work
for k in 2 4 6 8 10; do
    python scripts/make_workload.py locomo --model "$MODEL" --k "$k" --samples conv-26 --out "work/e3_k$k.jsonl"
done
for p in 0 0.25 0.5 0.75 0.9; do
    python scripts/make_workload.py zipf --model "$MODEL" --n 200 --k 6 --repeat-p "$p" --out "work/e4_p$p.jsonl"
done
