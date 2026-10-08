"""LoCoMo QA scoring, ported from snap-research/locomo task_eval/evaluation.py.

Kept behaviour: commas removed, punctuation stripped, a/an/the/and dropped, Porter
stemming, category 1 averaged over comma-split gold sub-answers, category 3 scored on
the first ``;``-separated gold answer, category 5 a substring check.
"""

from __future__ import annotations

import random
import re
import string
from collections import Counter

import numpy as np
from nltk.stem import PorterStemmer

_ps = PorterStemmer()
_PUNC = set(string.punctuation)


def normalize_answer(s: str) -> str:
    s = s.replace(",", "").lower()
    s = "".join(ch for ch in s if ch not in _PUNC)
    s = re.sub(r"\b(a|an|the|and)\b", " ", s)
    return " ".join(s.split())


def f1_score(prediction: str, ground_truth: str) -> float:
    p = [_ps.stem(w) for w in normalize_answer(prediction).split()]
    g = [_ps.stem(w) for w in normalize_answer(ground_truth).split()]
    same = sum((Counter(p) & Counter(g)).values())
    if same == 0:
        return 0.0
    prec, rec = same / len(p), same / len(g)
    return 2 * prec * rec / (prec + rec)


def f1_multi(prediction: str, ground_truth: str) -> float:
    preds = [x.strip() for x in prediction.split(",")]
    gts = [x.strip() for x in ground_truth.split(",")]
    return float(np.mean([max(f1_score(p, g) for p in preds) for g in gts]))


def score(prediction: str, answer: str, category: int) -> float:
    answer = str(answer)
    if category == 3:
        answer = answer.split(";")[0].strip()
    if category in (2, 3, 4):
        return f1_score(prediction, answer)
    if category == 1:
        return f1_multi(prediction, answer)
    if category == 5:
        low = prediction.lower()
        return float("no information available" in low or "not mentioned" in low)
    raise ValueError(f"unknown category {category}")


def clean_prediction(text: str) -> str:
    """Keep the first line; Mistral-Instruct tends to keep talking."""
    return text.strip().split("\n")[0].strip()


def paired_bootstrap(a, b, groups=None, n: int = 2000, seed: int = 0):
    """95% CI of mean(a - b). With ``groups`` (e.g. conversation ids), resamples whole
    groups (cluster bootstrap), which is the honest interval on LoCoMo's 10 conversations."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    rng = random.Random(seed)
    if groups is None:
        idx = [[i] for i in range(len(d))]
    else:
        by: dict = {}
        for i, g in enumerate(groups):
            by.setdefault(g, []).append(i)
        idx = list(by.values())
    stats = []
    for _ in range(n):
        pick = [i for _ in idx for i in idx[rng.randrange(len(idx))]]
        stats.append(d[pick].mean())
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi)
