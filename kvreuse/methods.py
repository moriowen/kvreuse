"""The correction methods, as plans for ``selective_prefill``.

They differ only in which tokens are recomputed (and, for AgentKVShift, how the rest
of each chunk is shifted). Every plan always recomputes the whole question. The system
prefix is exact and never recomputed.

Structural constraint: a token recomputed at layer l needs its fresh hidden state from
layer l-1, so the recompute set can only shrink with depth. Methods that select at
``check_layer`` therefore run every earlier layer in full and compute QKV for every token
at the check layer; ``PrefillResult.budget`` charges for that.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from .forward import Layout


def _cat_sorted(*parts: torch.Tensor) -> torch.Tensor:
    return torch.unique(torch.cat(parts))  # unique() returns sorted


def _all_recomputable(layout: Layout, device) -> torch.Tensor:
    idx = [torch.arange(a, b, device=device) for a, b in layout.chunk_spans]
    return _cat_sorted(*idx, layout.question_idx(device))


@dataclass
class FullRecompute:
    """Every chunk token at every layer. Must equal full prefill (E0 check c)."""

    def initial_set(self, layout, device):
        return _all_recomputable(layout, device)

    def on_layer(self, l, S, K, V, K0, V0, layout):
        return None


@dataclass
class Reposition:
    """Method 1: re-rotated chunk KV used as is; only the question is computed."""

    def initial_set(self, layout, device):
        return layout.question_idx(device)

    def on_layer(self, l, S, K, V, K0, V0, layout):
        return None


@dataclass
class RepositionAligned:
    """Method 1 as the vLLM connector serves it (E5 reference): reused KV only up to the last
    whole ``block`` of the covered prefix; the tail of the last chunk (< block tokens) and the
    question are computed fresh."""

    block: int = 16

    def initial_set(self, layout, device):
        end = min(layout.chunk_spans[-1][1] if layout.chunk_spans else layout.prefix_len, layout.n - 1)
        aligned = end // self.block * self.block
        return torch.arange(max(aligned, layout.prefix_len), layout.n, device=device)

    def on_layer(self, l, S, K, V, K0, V0, layout):
        return None


@dataclass
class EPIC:
    """Method 2 (LegoLink-k): recompute the first ``k`` tokens of every chunk at every
    layer, from layer 0. EPIC's code skips only the system prompt, so every retrieved
    chunk gets a boundary."""

    k: int = 16

    def initial_set(self, layout, device):
        parts = [torch.arange(a, min(a + self.k, b), device=device) for a, b in layout.chunk_spans]
        return _cat_sorted(*parts, layout.question_idx(device))

    def on_layer(self, l, S, K, V, K0, V0, layout):
        return None


def _deviation(X: torch.Tensor, X0: torch.Tensor, idx: torch.Tensor, squared: bool) -> torch.Tensor:
    """Per-token L2 distance between fresh and reused KV over heads and dims."""
    d = (X[0, :, idx].float() - X0[0, :, idx].float()).pow(2).sum(dim=(0, 2))
    return d if squared else d.sqrt()


@dataclass
class CacheBlend:
    """Method 3, following CacheBlend's released code rather than its paper: one
    selection at ``check_layer`` by squared L2 deviation of V only, top ``ratio`` of all
    chunk tokens (global, not per chunk). Departures from the code: an exact causal mask,
    and every question token is forced in (the code's ``last_len`` bug keeps only one)."""

    ratio: float = 0.15
    check_layer: int = 1
    use_k: bool = False  # ablation: rank by K+V deviation

    def initial_set(self, layout, device):
        return _all_recomputable(layout, device)

    def on_layer(self, l, S, K, V, K0, V0, layout):
        if l != self.check_layer:
            return None
        dev = S.device
        chunk_idx = layout.reusable_mask(dev).nonzero().squeeze(1)
        score = _deviation(V, V0, chunk_idx, squared=True)
        if self.use_k:
            score = score + _deviation(K, K0, chunk_idx, squared=True)
        n_sel = math.ceil(self.ratio * len(chunk_idx))
        sel = chunk_idx[torch.topk(score, n_sel).indices] if n_sel else chunk_idx[:0]
        return _cat_sorted(sel, layout.question_idx(dev))


@dataclass
class AgentKVShift:
    """Method 4 (AgentKVShift, Algorithm 1; reimplemented, no public code).

    At ``check_layer``: per-token divergence d = ||fresh - reused||_2, separately for K
    and V. Per chunk, the top b = ceil(ratio * n) tokens by d_K are K-probes and by d_V
    are V-probes; the union is recomputed at every later layer. At each later layer the
    chunk's mean residual over its probes is added to every non-recomputed token with
    weight w = min(d, 1).

    Our choices where the paper is silent: check_layer=1 (HF index; layer-0 divergence is
    zero after exact re-rotation), offsets averaged post-RoPE, the K/V probe union is what
    the budget counts.
    """

    ratio: float = 0.10
    check_layer: int = 1
    _state: dict = field(default_factory=dict, repr=False)

    def initial_set(self, layout, device):
        self._state = {}
        return _all_recomputable(layout, device)

    def on_layer(self, l, S, K, V, K0, V0, layout):
        dev = S.device
        if l == self.check_layer:
            chunks, keep = [], [layout.question_idx(dev)]
            for a, b in layout.chunk_spans:
                idx = torch.arange(a, b, device=dev)
                dK = _deviation(K, K0, idx, squared=False)
                dV = _deviation(V, V0, idx, squared=False)
                nb = math.ceil(self.ratio * len(idx))
                pK = idx[torch.topk(dK, nb).indices] if nb else idx[:0]
                pV = idx[torch.topk(dV, nb).indices] if nb else idx[:0]
                probes = _cat_sorted(pK, pV)
                rest = idx[~torch.isin(idx, probes)]
                chunks.append(dict(pK=pK, pV=pV, rest=rest,
                                   wK=dK.clamp(max=1.0)[rest - a], wV=dV.clamp(max=1.0)[rest - a]))
                keep.append(probes)
            self._state["chunks"] = chunks
            return _cat_sorted(*keep)
        if l > self.check_layer:
            for c in self._state["chunks"]:
                if len(c["rest"]) == 0:
                    continue
                for X, X0, p, w in ((K, K0, c["pK"], c["wK"]), (V, V0, c["pV"], c["wV"])):
                    if len(p) == 0:
                        continue
                    mu = (X[0, :, p].float() - X0[0, :, p].float()).mean(dim=1, keepdim=True)  # [H,1,D]
                    X[0, :, c["rest"]] = (X0[0, :, c["rest"]].float() + w[None, :, None] * mu).to(X.dtype)
        return None


def make_plan(name: str, budget: float | None = None, k: int | None = None):
    if name == "full":
        return FullRecompute()
    if name == "reposition":
        return Reposition()
    if name == "reposition_aligned":
        return RepositionAligned()
    if name == "epic":
        return EPIC(k=k if k is not None else 16)
    if name == "cacheblend":
        return CacheBlend(ratio=budget if budget is not None else 0.15)
    if name == "agentkvshift":
        return AgentKVShift(ratio=budget if budget is not None else 0.10)
    raise ValueError(f"unknown method {name!r}")
