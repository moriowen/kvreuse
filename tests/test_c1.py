import pytest

from kvreuse import locomo
from kvreuse.metrics import paired_bootstrap, score
from kvreuse.retrieval import BM25Retriever


def _sample():
    turns = lambda s, n: [{"speaker": "A" if i % 2 else "B", "dia_id": f"D{s}:{i}", "text": f"turn {i} " * 8}
                          for i in range(1, n + 1)]
    return {
        "sample_id": "conv-x",
        "conversation": {
            "speaker_a": "A", "speaker_b": "B",
            "session_1_date_time": "1:00 pm on 1 May, 2023", "session_1": turns(1, 9),
            "session_2_date_time": "2:00 pm on 9 May, 2023", "session_2": turns(2, 3),
        },
        "qa": [{"question": "q?", "answer": 2022, "category": 2, "evidence": ["D1:1"]},
               {"question": "adv?", "adversarial_answer": "x", "category": 5, "evidence": []}],
    }


def words(text):
    return len(text.split())


def test_chunks_pack_whole_turns_with_headers():
    s = _sample()
    chunks = locomo.chunk_conversation(s, words, target_tokens=60)
    all_turns = [t for c in chunks for t in c.turn_ids]
    assert all_turns == [t["dia_id"] for _, _, ts in locomo.sessions(s["conversation"]) for t in ts]
    for c in chunks:
        assert c.text.startswith("DATE: ")
        assert len(set(t.split(":")[0] for t in c.turn_ids)) == 1  # never crosses sessions
        assert words(c.text) <= 60 or len(c.turn_ids) == 1


def test_questions_and_cat5_prompt_is_seeded():
    qs = locomo.questions(_sample())
    assert qs[0].answer == "2022" and "approximate date" in qs[0].prompt()
    assert qs[1].answer == "x" and qs[1].prompt(seed=3) == qs[1].prompt(seed=3)
    assert locomo.NOT_MENTIONED in qs[1].prompt()


def test_scorer():
    assert score("7 May 2023", "7 May 2023", 2) == 1.0
    assert score("The painting", "painting", 4) == 1.0
    assert score("hiking, swimming", "swimming, hiking, running", 1) == pytest.approx(2 / 3)
    assert score("a; b", "a; b", 3) < 1.0  # gold truncated to "a" before scoring
    assert score("Not mentioned in the conversation", "x", 5) == 1.0


def test_bm25_and_bootstrap():
    r = BM25Retriever(["apples and pears", "car engines", "pear trees"])
    assert r.topk("pear", 2)[0] in (0, 2)
    m, lo, hi = paired_bootstrap([1, 1, 0, 1], [0, 1, 0, 0], groups=["a", "a", "b", "b"])
    assert lo <= m <= hi
