#!/bin/bash
#SBATCH -J kv-prepare
#SBATCH -p ice-cpu
#SBATCH -N1 -c 4
#SBATCH --mem=16G
#SBATCH -t 1:00:00
#SBATCH -o logs/%x-%j.out
# CPU only: download the model weights once (so GPU jobs don't spend time downloading),
# then chunk LoCoMo and save BM25 retrievals.
source pace/env.sh

python - <<'EOF'
import os
from huggingface_hub import snapshot_download
# HF-format shards only; skip consolidated.safetensors (a second 14 GB copy for mistral-inference)
p = snapshot_download(os.environ["MODEL"],
                      allow_patterns=["*.json", "model-*.safetensors", "tokenizer*"])
print("model at", p)
EOF

python scripts/prepare_data.py --tokenizer "$MODEL" --k 10
