import sys

import torch


def pick_device(allow_cpu: bool) -> torch.device:
    """Real-model runs belong on a PACE GPU node. Refuse to fall back to CPU unless asked
    (``--allow-cpu`` is for smoke tests with a tiny local model)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if allow_cpu:
        return torch.device("cpu")
    sys.exit("no CUDA GPU found; run this on a PACE GPU node (or pass --allow-cpu with a tiny test model)")
