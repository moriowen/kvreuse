"""C4: the synthetic stream controls reuse and never repeats a chunk inside one request."""

import pytest

from kvreuse.workload import reuse_rate, zipf_stream

POOL = [f"c{i}" for i in range(1300)]


def test_no_reuse_uses_fresh_chunks_only():
    reqs = zipf_stream(POOL, 50, 6, repeat_p=0.0)
    flat = [c for r in reqs for c in r["chunk_ids"]]
    assert len(flat) == len(set(flat)) == 300
    assert reuse_rate(reqs) == 0.0


@pytest.mark.parametrize("p", [0.25, 0.5, 0.9])
def test_reuse_rate_tracks_repeat_p(p):
    reqs = zipf_stream(POOL, 200, 6, repeat_p=p, seed=1)
    for r in reqs:
        assert len(set(r["chunk_ids"])) == 6
    # the first request has nothing to repeat, so the rate sits just under p
    assert abs(reuse_rate(reqs) - p) < 0.05
    seen = set()
    for r in reqs:  # 'repeats' counts exactly the chunks used by an earlier request
        assert r["repeats"] == sum(c in seen for c in r["chunk_ids"])
        seen.update(r["chunk_ids"])


def test_deterministic_and_pool_exhaustion():
    assert zipf_stream(POOL, 30, 6, 0.5, seed=3) == zipf_stream(POOL, 30, 6, 0.5, seed=3)
    with pytest.raises(ValueError):
        zipf_stream(POOL, 300, 6, repeat_p=0.0)
