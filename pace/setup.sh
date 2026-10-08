#!/bin/bash
# One-time setup, run on the ICE login node from the project directory:
#
#   cd ~/scratch/kvreuse && bash pace/setup.sh
#
# The default MODEL is an ungated mirror (see env.sh), so no HF token is needed. For the
# official repo, export HF_TOKEN for an account that accepted the Mistral license.
source "$(dirname "$0")/env.sh"

python3 -c 'import sys; assert sys.version_info >= (3, 10), sys.version' \
    || { echo "need python >= 3.10 (module load python/3.11.9)"; exit 1; }

if [ ! -f "$VENV/bin/activate" ]; then
    python3 -m venv "$VENV"
fi
set +u; source "$VENV/bin/activate"; set -u
pip install -q --upgrade pip
pip install -q -e '.[dev]'

if [ ! -f data/locomo10.json ]; then
    curl -sSL -o data/locomo10.json \
        https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json
fi

# Fails fast if the token is missing or the gated license has not been accepted.
python - <<'EOF'
import os
from huggingface_hub import hf_hub_download
hf_hub_download(os.environ["MODEL"], "config.json")
print("Hugging Face access to", os.environ["MODEL"], "OK")
EOF

python -m pytest -q  # CPU-only toy-model tests, a few seconds
echo "setup done. next: bash pace/submit.sh"
