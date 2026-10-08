"""C3 logic without vLLM: matching, alignment, slots, and packed KV == the Transformers assembly."""

import random

import torch

from kvreuse.connector_logic import (
    StoreIndex, assemble_packed, matched_tokens, segment_key, slot_mapping, to_packed,
)
from kvreuse.forward import full_prefill, selective_prefill
from kvreuse.methods import RepositionAligned
from kvreuse.store import ChunkStore, assemble

CTX = [1, 3]


def _store(view, ids):
    prefix = ids(7)
    chunks = [ids(20 + 3 * i) for i in range(5)]
    store = ChunkStore(view, CTX)
    entries, tensors = {}, {}
    for kind, seg in [("prefix", prefix)] + [("chunk", c) for c in chunks]:
        e = store.prefix(seg) if kind == "prefix" else store.chunk(seg)
        key = segment_key(kind, CTX, seg)
        entries[key] = {"kind": kind, "start": e.start, "ids": list(seg), "file": f"{key}.safetensors"}
        tensors[key] = to_packed(e)
    meta = {"entries": entries, "inv_freq": view.inv_freq.tolist(), "head_dim": view.head_dim}
    return StoreIndex(meta), tensors, store, prefix, chunks


def test_match_any_order_and_stop_at_question(view, ids):
    index, _, _, prefix, chunks = _store(view, ids)
    order = random.Random(0).sample(range(5), 3)
    question = ids(9)
    prompt = prefix + [t for i in order for t in chunks[i]] + question
    segs = index.match(prompt)
    assert [s.length for s in segs] == [len(prefix)] + [len(chunks[i]) for i in order]
    assert segs[-1].dst_start + segs[-1].length == len(prompt) - len(question)
    assert index.match(ids(30)) == []  # no prefix, no hit


def test_alignment_and_slots():
    from kvreuse.connector_logic import Segment
    segs = [Segment("p", 0, 7, 0), Segment("c", 7, 30, 2)]  # covered prefix ends at 37
    assert matched_tokens(segs, n_prompt=50, n_computed=0, block_size=16) == 32
    assert matched_tokens(segs, n_prompt=50, n_computed=16, block_size=16) == 16
    assert matched_tokens(segs, n_prompt=33, n_computed=0, block_size=16) == 32  # never the last token
    assert matched_tokens(segs, n_prompt=32, n_computed=0, block_size=16) == 16
    assert slot_mapping([5, 2], 14, 4, 16).tolist() == [94, 95, 32, 33]


def test_seen_only_stops_at_first_unseen():
    from kvreuse.connector_logic import Segment, seen_only
    segs = [Segment("p", 0, 7, 0), Segment("a", 7, 30, 2), Segment("b", 37, 30, 2), Segment("c", 67, 30, 2)]
    assert [s.key for s in seen_only(segs, {"p", "a", "c"})] == ["p", "a"]
    assert seen_only(segs, {"a", "b"}) == []  # unseen prefix: no hit at all
    assert seen_only(segs, {"p", "a", "b", "c"}) == segs


def test_packed_kv_matches_transformers_assembly(view, ids):
    index, tensors, store, prefix, chunks = _store(view, ids)
    order = [3, 0, 4]
    question = ids(9)
    prompt = prefix + [t for i in order for t in chunks[i]] + question
    segs = index.match(prompt)
    n = matched_tokens(segs, len(prompt), 0, 16)
    kv = assemble_packed(segs, 0, n, tensors.__getitem__, torch.tensor(index.meta["inv_freq"]), view.head_dim)
    x, K0, V0, layout = assemble(view, store.prefix(prefix), [store.chunk(chunks[i]) for i in order], question)
    ref = torch.cat([K0[:, :, :n].permute(0, 2, 1, 3), V0[:, :, :n].permute(0, 2, 1, 3)], dim=-1)
    torch.testing.assert_close(kv, ref, atol=1e-6, rtol=0)
    # the HF reference for the served mode recomputes from the same block boundary
    S = RepositionAligned(16).initial_set(layout, "cpu")
    assert S[0].item() == n
    res = selective_prefill(view, x, K0, V0, layout, RepositionAligned(16))
    assert torch.isfinite(res.last_logits).all()
    assert (res.last_logits - full_prefill(view, x).last_logits).abs().max() > 0  # still approximate
