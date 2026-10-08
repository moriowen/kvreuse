#!/bin/bash
# One-time: a second venv for the vLLM path (vLLM 0.27.1 pins its own torch, so it must not
# share the Transformers venv). Python 3.12, because flashinfer (a vLLM dependency) annotates
# with array.array[int], which raises TypeError on 3.11. Run on the ICE login node from the
# project directory:
#   bash pace/setup_vllm.sh
set -eo pipefail
VLLM_VENV="${VLLM_VENV:-$HOME/scratch/kvreuse-vllm-venv312}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$HOME/scratch/pip-cache}"
module load python/3.12.5 >/dev/null 2>&1 || true
[ -f "$VLLM_VENV/bin/activate" ] || python3 -m venv "$VLLM_VENV"
source "$VLLM_VENV/bin/activate"
pip install -q --upgrade pip
pip install -q "vllm==0.27.1"
pip install -q rank_bm25 nltk pytest
pip install -q --no-deps -e .   # our package, without letting it change vLLM's torch
python -c "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__)"
python -c "import kvreuse.connector_logic; print('kvreuse importable')"
