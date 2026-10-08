"""Chunk KV store and prompt assembly.

A chunk is prefilled once after a fixed ``context`` (by default ``[BOS, [INST]]``, so the
chunk's first token is not the attention sink), and only the chunk's slice is kept,
at canonical positions ``len(context) ...``. Entries are keyed by a hash of
(context ids, chunk ids); a hit compares the ids too, so a hash collision cannot
return the wrong KV.

The system prefix ``[BOS][INST] system\\n\\n`` always sits at position 0, so its KV is exact
and is cached like an ordinary prefix cache.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import torch

from .forward import Layout, ModelView, full_prefill
from .rope import shift_keys


def ids_key(*seqs) -> str:
    h = hashlib.sha256()
    for s in seqs:
        h.update(np.asarray(s, dtype=np.int64).tobytes())
        h.update(b"|")
    return h.hexdigest()


@dataclass
class ChunkKV:
    ids: tuple[int, ...]
    k: torch.Tensor  # [L, n_kv, T, D], post-RoPE at canonical positions
    v: torch.Tensor
    start: int  # canonical position of the chunk's first token


class ChunkStore:
    def __init__(self, view: ModelView, context_ids: list[int], device=None):
        self.view = view
        self.context = tuple(context_ids)
        self.device = device or view.device
        self._chunks: dict[str, ChunkKV] = {}
        self._prefixes: dict[str, ChunkKV] = {}
        self.hits = 0
        self.misses = 0

    def _prefill_slice(self, context: tuple, ids: tuple) -> ChunkKV:
        full = torch.tensor(context + ids, device=self.view.device)
        res = full_prefill(self.view, full)
        a = len(context)
        k = torch.stack([K[0, :, a:] for K in res.K]).to(self.device)
        v = torch.stack([V[0, :, a:] for V in res.V]).to(self.device)
        return ChunkKV(ids, k, v, a)

    def chunk(self, ids) -> ChunkKV:
        ids = tuple(int(i) for i in ids)
        key = ids_key(self.context, ids)
        hit = self._chunks.get(key)
        if hit is not None:
            if hit.ids != ids:
                raise RuntimeError("chunk hash collision")
            self.hits += 1
            return hit
        self.misses += 1
        entry = self._prefill_slice(self.context, ids)
        self._chunks[key] = entry
        return entry

    def prefix(self, ids) -> ChunkKV:
        ids = tuple(int(i) for i in ids)
        key = ids_key(ids)
        hit = self._prefixes.get(key)
        if hit is None or hit.ids != ids:
            hit = self._prefill_slice((), ids)
            self._prefixes[key] = hit
        return hit

    def __len__(self):
        return len(self._chunks)


def assemble(
    view: ModelView,
    prefix: ChunkKV,
    chunks: list[ChunkKV],
    question_ids: list[int],
    compute_dtype: torch.dtype = torch.float32,
):
    """Build the full prompt ids, reused cache ``K0, V0`` ([L, n_kv, N, D]) and layout.

    Each chunk's keys are re-rotated from its canonical position to where it lands in
    this prompt. Question positions are left as zeros; they are always recomputed.
    """
    seg_ids = list(prefix.ids)
    spans = []
    for c in chunks:
        spans.append((len(seg_ids), len(seg_ids) + len(c.ids)))
        seg_ids.extend(c.ids)
    q_start = len(seg_ids)
    seg_ids.extend(question_ids)
    n = len(seg_ids)

    L, H, _, D = prefix.k.shape
    dev, dt = view.device, prefix.k.dtype
    K0 = torch.zeros(L, H, n, D, device=dev, dtype=dt)
    V0 = torch.zeros_like(K0)
    p = len(prefix.ids)
    K0[:, :, :p] = prefix.k.to(dev)
    V0[:, :, :p] = prefix.v.to(dev)
    for c, (a, b) in zip(chunks, spans):
        K0[:, :, a:b] = shift_keys(c.k.to(dev), a - c.start, view.inv_freq, compute_dtype)
        V0[:, :, a:b] = c.v.to(dev)

    layout = Layout(n=n, prefix_len=p, chunk_spans=spans, question_span=(q_start, n))
    ids = torch.tensor(seg_ids, device=dev)
    return ids, K0, V0, layout
