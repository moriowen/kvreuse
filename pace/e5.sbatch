#!/bin/bash
#SBATCH -J kv-e5
#SBATCH -p ice-gpu
#SBATCH -N1 -c 8
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=96G
#SBATCH -t 2:00:00
#SBATCH -o logs/%x-%j.out
# E5 on conversation conv-26's first N questions (default 100):
#   1. export the chunk store for conv-26 from the Transformers path
#   2. Transformers reference for the served mode (reposition_aligned)
#   3. vLLM + ChunkReuseConnector, and plain vLLM full prefill
#   4. compare
source pace/env.sh
N="${N:-100}"
STORE="${STORE:-$HOME/scratch/kvreuse-store-conv26}"
VLLM_VENV="${VLLM_VENV:-$HOME/scratch/kvreuse-vllm-venv312}"
nvidia-smi --query-gpu=name,memory.total --format=csv

[ -f "$STORE/meta.json" ] || python scripts/export_store.py --model "$MODEL" --samples conv-26 --out "$STORE"
python scripts/run_eval.py --model "$MODEL" --method reposition_aligned --limit "$N"
python scripts/run_eval.py --model "$MODEL" --method full --limit "$N" --name full_top10_bf16_n${N}_nocudnn

set +u; source "$VLLM_VENV/bin/activate"; set -u
python scripts/vllm_e5.py --model "$MODEL" --store "$STORE" --limit "$N" --dump-dir results/e5_dump
python scripts/vllm_e5.py --model "$MODEL" --store "$STORE" --limit "$N" --no-connector

python scripts/analyze.py results/reposition_aligned_top10_bf16_n${N}.jsonl \
    results/vllm_connector_top10_bf16_n${N}.jsonl results/full_top10_bf16_n${N}_nocudnn.jsonl \
    results/vllm_full_top10_bf16_n${N}.jsonl --out results/e5_table.md
python - <<EOF
import json
load = lambda p: {(r["sample_id"], r["qa_index"]): r for r in map(json.loads, open(p))}
for a, b in [("reposition_aligned_top10_bf16_n${N}", "vllm_connector_top10_bf16_n${N}"),
             ("full_top10_bf16_n${N}_nocudnn", "vllm_full_top10_bf16_n${N}")]:
    A, B = load(f"results/{a}.jsonl"), load(f"results/{b}.jsonl")
    keys = sorted(set(A) & set(B))
    same = sum(A[k]["prediction"] == B[k]["prediction"] for k in keys)
    print(f"{a} vs {b}: identical greedy answers {same}/{len(keys)}")
EOF
