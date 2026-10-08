"""vLLM-independent pieces of the C3 connector, kept separate so they can be unit-tested
without vLLM installed.

The chunk store on disk (written by scripts/export_store.py from the Transformers path):

    <store>/meta.json     model, inv_freq, shapes, and one entry per segment:
                          {key: {"kind": "prefix"|"chunk", "start": canonical position,
                                 "ids": [...], "file": "<key>.safetensors"}}
    <store>/<key>.safetensors   "kv": [layers, T, n_kv_heads, 2*head_dim] bf16, vLLM's
                          packed layout (K post-RoPE at canonical positions, then V)

Prefix entries sit at position 0 and are exact. Chunk entries were prefilled after
``[BOS][INST]`` (start 2) and get re-rotated to wherever they land.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch

from .rope import rotate_half
from .store import ids_key

PROBE = 8  # tokens used to index chunk starts


@dataclass(frozen=True)
class Segment:
    key: str
    dst_start: int  # position in this prompt
    length: int
    src_start: int  # canonical position it was cached at


class StoreIndex:
    """Token-level index of the store: which segment starts with which tokens."""

    def __init__(self, meta: dict):
        self.meta = meta
        self.entries = meta["entries"]
        self.prefixes = [(k, e) for k, e in self.entries.items() if e["kind"] == "prefix"]
        self.by_probe: dict[tuple, list[str]] = {}
        for k, e in self.entries.items():
            if e["kind"] == "chunk":
                self.by_probe.setdefault(tuple(e["ids"][:PROBE]), []).append(k)

    @classmethod
    def load(cls, store_dir: str | Path) -> "StoreIndex":
        return cls(json.loads((Path(store_dir) / "meta.json").read_text()))

    def match(self, prompt: list[int]) -> list[Segment]:
        """Greedy left-to-right cover of the prompt by stored segments: one exact prefix at
        position 0, then chunks in any order. Stops at the first token no segment covers.
        A hit compares the full token ids, so a probe collision cannot return wrong KV."""
        segs: list[Segment] = []
        p = 0
        for k, e in self.prefixes:
            ids = e["ids"]
            if prompt[: len(ids)] == ids and len(ids) > p:
                segs, p = [Segment(k, 0, len(ids), e["start"])], len(ids)
        if not segs:
            return []
        while True:
            hit = None
            for k in self.by_probe.get(tuple(prompt[p : p + PROBE]), []):
                ids = self.entries[k]["ids"]
                if prompt[p : p + len(ids)] == ids and (hit is None or len(ids) > hit[1]):
                    hit = (k, len(ids))
            if hit is None:
                return segs
            k, n = hit
            segs.append(Segment(k, p, n, self.entries[k]["start"]))
            p += n


def matched_tokens(segs: list[Segment], n_prompt: int, n_computed: int, block_size: int) -> int:
    """Tokens the connector can supply beyond ``n_computed``: the covered prefix rounded down
    to whole blocks, and never the last prompt token (vLLM must compute at least one)."""
    if not segs:
        return 0
    end = segs[-1].dst_start + segs[-1].length
    end = min(end, n_prompt - 1)
    end = end // block_size * block_size
    return max(0, end - n_computed)


def slot_mapping(block_ids: list[int], start: int, n: int, block_size: int) -> torch.Tensor:
    pos = torch.arange(start, start + n)
    blocks = torch.tensor(block_ids, dtype=torch.long)[pos // block_size]
    return blocks * block_size + pos % block_size


def rotate_packed(kv: torch.Tensor, delta: int, inv_freq: torch.Tensor, head_dim: int) -> torch.Tensor:
    """Re-rotate the K half of packed ``[..., T, H, 2*D]`` KV by ``delta`` positions (fp32,
    cast once). Same math as kvreuse.rope.shift_keys, in vLLM's layout."""
    if delta == 0:
        return kv
    ang = delta * inv_freq.to(device=kv.device, dtype=torch.float32)
    emb = torch.cat((ang, ang))
    k = kv[..., :head_dim].float()
    k = k * emb.cos() + rotate_half(k) * emb.sin()
    return torch.cat((k.to(kv.dtype), kv[..., head_dim:]), dim=-1)


def assemble_packed(segs, start: int, n: int, get_kv, inv_freq, head_dim) -> torch.Tensor:
    """KV for prompt positions [start, start+n) as ``[layers, n, H, 2*D]``."""
    parts = []
    for s in segs:
        a, b = max(start, s.dst_start), min(start + n, s.dst_start + s.length)
        if a >= b:
            continue
        kv = get_kv(s.key)[:, a - s.dst_start : b - s.dst_start]
        parts.append(rotate_packed(kv, s.dst_start - s.src_start, inv_freq, head_dim))
    out = torch.cat(parts, dim=1)
    if out.shape[1] != n:
        raise RuntimeError(f"segments cover {out.shape[1]} of {n} requested tokens")
    return out


def to_packed(entry) -> torch.Tensor:
    """ChunkKV (k, v: [L, H, T, D]) -> vLLM packed [L, T, H, 2D], on CPU."""
    return torch.cat([entry.k.permute(0, 2, 1, 3), entry.v.permute(0, 2, 1, 3)], dim=-1).contiguous().cpu()


def segment_key(kind: str, context: list[int], ids: list[int]) -> str:
    return ids_key(context, ids) if kind == "chunk" else ids_key(ids)
