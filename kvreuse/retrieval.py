"""BM25 top-k retrieval over one conversation's chunks.

Results are saved once and every configuration reads the same file, so no difference
between methods can come from retrieval. Order is retrieval rank (that varying order is
what breaks the prefix cache).
"""

from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

_WORD = re.compile(r"[a-z0-9]+")


def _tok(text: str) -> list[str]:
    return _WORD.findall(text.lower())


class BM25Retriever:
    def __init__(self, chunk_texts: list[str]):
        self.bm25 = BM25Okapi([_tok(t) for t in chunk_texts])

    def topk(self, query: str, k: int) -> list[int]:
        scores = self.bm25.get_scores(_tok(query))
        order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
        return order[:k]


class EmbeddingRetriever:
    """Dense retrieval with a sentence-transformers encoder (cosine similarity).

    Optional dependency: ``pip install sentence-transformers``. Small on purpose so it runs
    in the CPU prepare job; the encoder name is recorded in the retrieval file."""

    def __init__(self, chunk_texts: list[str], model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self.enc = SentenceTransformer(model, device="cpu")
        self.emb = self.enc.encode(chunk_texts, normalize_embeddings=True, convert_to_tensor=True)

    def topk(self, query: str, k: int) -> list[int]:
        q = self.enc.encode([query], normalize_embeddings=True, convert_to_tensor=True)
        scores = (self.emb @ q.T).squeeze(1).tolist()
        order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
        return order[:k]
