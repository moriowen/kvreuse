# Sourced by setup.sh and every job. ICE home is small (~15 GB), so the venv, the HF
# cache (the model is ~15 GB), pip's cache and the project live under ~/scratch.
# Override any of these by exporting it before submitting.
set -eo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/scratch/kvreuse}"
VENV="${VENV:-$HOME/scratch/kvreuse-venv}"
export HF_HOME="${HF_HOME:-$HOME/scratch/hf}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$HOME/scratch/pip-cache}"
# Default: ungated mirror of Mistral-7B-Instruct-v0.3. Its three weight shards have the same
# SHA-256 as the official repo (checked Oct 7, 2026); only a pad_token and metadata differ.
# With HF_TOKEN and the license accepted, MODEL=mistralai/Mistral-7B-Instruct-v0.3 also works.
export MODEL="${MODEL:-unsloth/mistral-7b-instruct-v0.3}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

# Lmod and venv activation reference unset variables, so strict mode (-u) comes after them.
if command -v module >/dev/null 2>&1; then
    module load python/3.11.9 >/dev/null 2>&1 || true
fi
if [ -f "$VENV/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$VENV/bin/activate"
fi
set -u

cd "$PROJECT_DIR"
mkdir -p logs results data
