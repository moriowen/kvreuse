"""C4: request workloads for the serving experiments, as token-id prompts.

E3 uses the real LoCoMo prompts (saved retrievals, first k chunks). E4 uses a synthetic stream
over the pool of all LoCoMo chunks where the share of repeated chunks is controlled:

    each of the k chunk slots in a request repeats an entry from an earlier request with
    probability ``repeat_p`` (picked by Zipf popularity over first-seen rank, exponent
    ``zipf_s``), and otherwise takes a chunk never used before.

Chunks within a request are shuffled, so a repeat rarely lands in the same prefix position
twice; that is the case prefix caching cannot serve. The measured reuse rate (repeated draws /
all draws) is the x-axis of E4, not ``repeat_p``. Every request carries the same system
prefix, as an agent's memory calls would. Requests are lists of ids, never re-tokenized text.
"""

from __future__ import annotations

import random

from . import locomo
from .prompt import PromptBuilder


def locomo_requests(pb: PromptBuilder, chunks: dict, retrieval: dict, k: int,
                    samples: list[str] | None = None, seed: int = 0) -> list[dict]:
    """One request per saved question, with the first k retrieved chunks in retrieval order."""
    convs = chunks["conversations"]
    by_id = {c["id"]: c for conv in convs.values() for c in conv["chunks"]}
    out = []
    for q in retrieval["questions"]:
        if samples and q["sample_id"] not in samples:
            continue
        if k > len(q["chunk_ids"]):
            raise ValueError(f"k={k} but only {len(q['chunk_ids'])} chunks were saved per question")
        question = locomo.Question(**{f: q[f] for f in ("sample_id", "qa_index", "category", "question", "answer",
                                                         "evidence")})
        conv = convs[q["sample_id"]]
        ids = pb.full(locomo.CONV_START_PROMPT.format(*conv["speakers"]),
                      [by_id[c]["token_ids"] for c in q["chunk_ids"][:k]], question.prompt(seed))
        out.append({"id": f"{q['sample_id']}/{q['qa_index']}", "prompt_ids": ids, "chunk_ids": q["chunk_ids"][:k]})
    return out


def zipf_stream(pool: list[str], n_requests: int, k: int, repeat_p: float, zipf_s: float = 1.0,
                seed: int = 0) -> list[dict]:
    """Chunk ids per request. ``pool`` is every chunk id that may be used; at most
    n_requests * k of them are needed (all of them when repeat_p = 0)."""
    rng = random.Random(seed)
    fresh = list(pool)
    rng.shuffle(fresh)
    seen: list[str] = []  # first-seen order = popularity rank
    out = []
    for i in range(n_requests):
        picked, repeats = [], 0
        for _ in range(k):
            # entries from earlier requests that this request has not used yet
            cands = [j for j, c in enumerate(seen) if c not in picked]
            if cands and rng.random() < repeat_p:
                w = [1.0 / (j + 1) ** zipf_s for j in cands]
                picked.append(seen[rng.choices(cands, weights=w)[0]])
                repeats += 1
            else:
                if not fresh:
                    raise ValueError(f"pool of {len(pool)} chunks exhausted at request {i}; "
                                     "use fewer requests, smaller k or a higher repeat_p")
                picked.append(fresh.pop())
        rng.shuffle(picked)
        seen.extend(c for c in picked if c not in seen)
        out.append({"chunk_ids": picked, "repeats": repeats})
    return out


def zipf_requests(pb: PromptBuilder, chunks: dict, retrieval: dict, n_requests: int, k: int,
                  repeat_p: float, zipf_s: float = 1.0, seed: int = 0) -> list[dict]:
    """E4 prompts: one fixed system prefix, k pool chunks from :func:`zipf_stream`, and a
    LoCoMo question (cycled in a seeded order). The content is not coherent; only the
    token counts and the reuse pattern matter for serving measurements."""
    convs = chunks["conversations"]
    by_id = {c["id"]: c for conv in convs.values() for c in conv["chunks"]}
    first = next(iter(convs.values()))
    system = locomo.CONV_START_PROMPT.format(*first["speakers"])
    qs = list(retrieval["questions"])
    random.Random(seed).shuffle(qs)
    out = []
    for i, r in enumerate(zipf_stream(sorted(by_id), n_requests, k, repeat_p, zipf_s, seed)):
        q = qs[i % len(qs)]
        question = locomo.Question(**{f: q[f] for f in ("sample_id", "qa_index", "category", "question", "answer",
                                                         "evidence")})
        ids = pb.full(system, [by_id[c]["token_ids"] for c in r["chunk_ids"]], question.prompt(seed))
        out.append({"id": f"z{i}", "prompt_ids": ids, **r})
    return out


def reuse_rate(requests: list[dict]) -> float:
    draws = sum(len(r["chunk_ids"]) for r in requests)
    return sum(r.get("repeats", 0) for r in requests) / draws if draws else 0.0
