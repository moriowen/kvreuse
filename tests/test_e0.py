"""E0 correctness gate, fp32 on a tiny model. Every later number depends on these."""

import pytest
import torch

from kvreuse.forward import full_prefill, selective_prefill
from kvreuse.methods import EPIC, AgentKVShift, CacheBlend, FullRecompute, Reposition
from kvreuse.rope import shift_keys
from kvreuse.store import ChunkStore, assemble

ATOL = 1e-5


def _prompt(view, ids, n_chunks=3, chunk_len=20, prefix_len=6, q_len=5):
    prefix = ids(prefix_len)
    chunks = [ids(chunk_len + i) for i in range(n_chunks)]  # uneven lengths on purpose
    question = ids(q_len)
    return prefix, chunks, question


# (a) RoPE round trip
def test_rope_round_trip(view):
    k = torch.randn(4, 2, 30, 16)
    for d in (1, 7, 300, -45):
        back = shift_keys(shift_keys(k, d, view.inv_freq), -d, view.inv_freq)
        torch.testing.assert_close(back, k, atol=ATOL, rtol=0)


def test_zero_shift_is_bitwise_identity(view):
    k = torch.randn(4, 2, 30, 16).to(torch.bfloat16)
    assert torch.equal(shift_keys(k, 0, view.inv_freq), k)


# (b) layer-0 keys: shifted == computed directly at the target position
def test_layer0_shift_matches_direct(view, ids):
    chunk = torch.tensor(ids(25))
    h = view.embed(chunk[None])
    for src, dst in ((2, 40), (40, 2), (0, 500)):
        _, k_src, _ = view.qkv(0, h, torch.arange(src, src + 25))
        _, k_dst, _ = view.qkv(0, h, torch.arange(dst, dst + 25))
        moved = shift_keys(k_src, dst - src, view.inv_freq)
        # fp32 angle rounding grows with position (~p * 6e-8 rad), so scale the tolerance
        tol = ATOL + 6e-8 * max(src, dst) * k_src.abs().max().item() * 4
        torch.testing.assert_close(moved, k_dst, atol=tol, rtol=0)


# manual forward == HF forward (validates the code path every method shares)
def test_manual_forward_matches_hf(view, ids):
    x = torch.tensor(ids(50))
    ref = view.model(x[None]).logits[0, -1].float()
    ours = full_prefill(view, x).last_logits
    torch.testing.assert_close(ours, ref, atol=1e-4, rtol=1e-4)


# (c) 100% recompute == full prefill
def test_full_recompute_equals_full_prefill(view, ids):
    prefix, chunks, question = _prompt(view, ids)
    store = ChunkStore(view, context_ids=[1, 3])
    x, K0, V0, layout = assemble(view, store.prefix(prefix), [store.chunk(c) for c in chunks], question)
    full = full_prefill(view, x)
    res = selective_prefill(view, x, K0, V0, layout, FullRecompute())
    torch.testing.assert_close(res.last_logits, full.last_logits, atol=1e-4, rtol=1e-4)
    for l in range(view.n_layers):
        torch.testing.assert_close(res.K[l], full.K[l], atol=1e-4, rtol=1e-4)
    assert res.budget(layout.n_reusable) == pytest.approx(1.0)


# (d) identity: a chunk reused where and after what it was cached == full prefill
def test_identity_case(view, ids):
    prefix, chunks, question = _prompt(view, ids, n_chunks=1)
    store = ChunkStore(view, context_ids=prefix)  # cache the chunk after the real prefix
    x, K0, V0, layout = assemble(view, store.prefix(prefix), [store.chunk(chunks[0])], question)
    full = full_prefill(view, x)
    res = selective_prefill(view, x, K0, V0, layout, Reposition())
    torch.testing.assert_close(res.last_logits, full.last_logits, atol=1e-4, rtol=1e-4)
    assert res.budget(layout.n_reusable) == 0.0


@pytest.mark.parametrize(
    "plan",
    [EPIC(k=10_000), CacheBlend(ratio=1.0), AgentKVShift(ratio=1.0)],
    ids=["epic", "cacheblend", "agentkvshift"],
)
def test_every_method_at_full_budget_equals_full_prefill(view, ids, plan):
    prefix, chunks, question = _prompt(view, ids)
    store = ChunkStore(view, context_ids=[1, 3])
    x, K0, V0, layout = assemble(view, store.prefix(prefix), [store.chunk(c) for c in chunks], question)
    full = full_prefill(view, x)
    res = selective_prefill(view, x, K0, V0, layout, plan)
    torch.testing.assert_close(res.last_logits, full.last_logits, atol=1e-4, rtol=1e-4)


def test_reuse_error_is_real_and_shrinks_with_budget(view, ids):
    """Not part of the gate: a sanity check that the cross-chunk error exists (else E1 is
    vacuous) and that recomputing more never makes it worse on this toy model."""
    prefix, chunks, question = _prompt(view, ids)
    store = ChunkStore(view, context_ids=[1, 3])
    x, K0, V0, layout = assemble(view, store.prefix(prefix), [store.chunk(c) for c in chunks], question)
    full = full_prefill(view, x).last_logits
    err0 = (selective_prefill(view, x, K0, V0, layout, Reposition()).last_logits - full).abs().max()
    err5 = (selective_prefill(view, x, K0, V0, layout, CacheBlend(ratio=0.5)).last_logits - full).abs().max()
    assert err0 > 1e-3
    assert err5 < err0


def test_budgets_are_charged_for_selection_layers(view, ids):
    prefix, chunks, question = _prompt(view, ids)
    store = ChunkStore(view, context_ids=[1, 3])
    x, K0, V0, layout = assemble(view, store.prefix(prefix), [store.chunk(c) for c in chunks], question)
    L = view.n_layers
    res = selective_prefill(view, x, K0, V0, layout, CacheBlend(ratio=0.0))
    assert res.budget(layout.n_reusable) == pytest.approx(2 / L)  # layer 0 + layer-1 QKV, in full
    res = selective_prefill(view, x, K0, V0, layout, EPIC(k=4))
    assert res.budget(layout.n_reusable) == pytest.approx(4 * len(chunks) / layout.n_reusable)


def test_store_hits_and_ids(view, ids):
    store = ChunkStore(view, context_ids=[1, 3])
    c = ids(12)
    a = store.chunk(c)
    b = store.chunk(list(c))
    assert a is b and store.hits == 1 and store.misses == 1
    assert a.start == 2 and a.k.shape == (view.n_layers, view.n_kv, 12, view.head_dim)


def test_agentkvshift_offsets_move_nonprobe_kv(view, ids):
    prefix, chunks, question = _prompt(view, ids)
    store = ChunkStore(view, context_ids=[1, 3])
    x, K0, V0, layout = assemble(view, store.prefix(prefix), [store.chunk(c) for c in chunks], question)
    plan = AgentKVShift(ratio=0.2)
    res = selective_prefill(view, x, K0, V0, layout, plan)
    rest = plan._state["chunks"][0]["rest"]
    assert len(rest) > 0
    assert not torch.equal(res.K[2][0, :, rest], K0[2][:, rest])  # shifted after the check layer
    assert torch.isfinite(res.last_logits).all()
    assert 2 / view.n_layers < res.budget(layout.n_reusable) < 1.0
