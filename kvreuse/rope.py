"""RoPE re-rotation of cached keys.

HF Mistral uses the NeoX pairing: dimension i rotates with i + D/2 (``rotate_half``),
not adjacent pairs as in the RoFormer paper. Rotations compose additively, so a key
cached at position p moves to p + delta with one more rotation by delta.

Always rotate from the stored canonical position (never chain p -> p1 -> p2): each cast
back to bf16 adds a rounding.
"""

import torch


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return x * cos + rotate_half(x) * sin


def shift_keys(
    k: torch.Tensor,
    delta: int,
    inv_freq: torch.Tensor,
    compute_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Move post-RoPE keys ``k[..., T, D]`` by ``delta`` positions (same delta for every token).

    The angle is computed and applied in ``compute_dtype``, then cast back once.
    Values are never rotated.
    """
    if delta == 0:
        return k  # cos=1, sin=0: bit-for-bit identity (E0 identity check)
    ang = delta * inv_freq.to(device=k.device, dtype=compute_dtype)
    emb = torch.cat((ang, ang), dim=-1)
    out = apply_rope(k.to(compute_dtype), emb.cos(), emb.sin())
    return out.to(k.dtype)
