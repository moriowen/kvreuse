"""LoCoMo loading and chunking (C1).

A chunk is a run of whole turns from one session, packed greedily to ``target_tokens``,
and starts with the session's date header in LoCoMo's own format
(``DATE: ...\\nCONVERSATION:\\n``, from snap-research/locomo task_eval/gpt_utils.py), so
temporal questions stay answerable. A turn is never split; a turn that is too long on its
own becomes its own (oversized) chunk.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Callable

# from task_eval/gpt_utils.py
CONV_START_PROMPT = (
    "Below is a conversation between two people: {} and {}. The conversation takes place over "
    "multiple days and the date of each conversation is wriiten at the beginning of the conversation.\n\n"
)
QA_PROMPT = (
    "Based on the above context, write an answer in the form of a short phrase for the following "
    "question. Answer with exact words from the context whenever possible.\n\nQuestion: {} Short answer:"
)
QA_PROMPT_CAT_5 = "Based on the above context, answer the following question.\n\nQuestion: {} Short answer:"
TEMPORAL_SUFFIX = " Use DATE of CONVERSATION to answer with an approximate date."
NOT_MENTIONED = "Not mentioned in the conversation"


@dataclass
class Chunk:
    id: str  # "<sample_id>:<session>:<index>"
    sample_id: str
    session: int
    text: str
    turn_ids: list[str]
    token_ids: list[int] | None = None


def load(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def sessions(conv: dict) -> list[tuple[int, str, list[dict]]]:
    """(session number, date string, turns), in session order."""
    nums = sorted(int(m.group(1)) for k in conv if (m := re.fullmatch(r"session_(\d+)", k)))
    return [(i, conv[f"session_{i}_date_time"], conv[f"session_{i}"]) for i in nums if conv[f"session_{i}"]]


def header(date: str) -> str:
    return f"DATE: {date}\nCONVERSATION:\n"


def turn_text(turn: dict) -> str:
    t = f'{turn["speaker"]} said, "{turn["text"]}"'
    if turn.get("blip_caption"):
        t += f' and shared {turn["blip_caption"]}.'
    return t + "\n"


def chunk_conversation(
    sample: dict,
    count_tokens: Callable[[str], int],
    target_tokens: int = 256,
) -> list[Chunk]:
    sid = sample["sample_id"]
    out: list[Chunk] = []
    for num, date, turns in sessions(sample["conversation"]):
        head = header(date)
        body, ids = "", []
        for turn in turns:
            t = turn_text(turn)
            if body and count_tokens(head + body + t) > target_tokens:
                out.append(Chunk(f"{sid}:{num}:{len(out)}", sid, num, head + body, ids))
                body, ids = "", []
            body += t
            ids.append(turn["dia_id"])
        if body:
            out.append(Chunk(f"{sid}:{num}:{len(out)}", sid, num, head + body, ids))
    return out


@dataclass
class Question:
    sample_id: str
    qa_index: int
    category: int
    question: str
    answer: str
    evidence: list[str]

    def prompt(self, seed: int = 0) -> str:
        """LoCoMo's per-question prompt. Category 5 is multiple choice with a seeded order."""
        if self.category == 2:
            return QA_PROMPT.format(self.question + TEMPORAL_SUFFIX)
        if self.category == 5:
            import random

            opts = [NOT_MENTIONED, self.answer]
            if random.Random(f"{seed}:{self.sample_id}:{self.qa_index}").random() >= 0.5:
                opts.reverse()
            q = self.question + " Select the correct answer: (a) {} (b) {}. ".format(*opts)
            return QA_PROMPT_CAT_5.format(q)
        return QA_PROMPT.format(self.question)


def questions(sample: dict) -> list[Question]:
    out = []
    for i, qa in enumerate(sample["qa"]):
        ans = qa.get("answer", qa.get("adversarial_answer", ""))
        out.append(Question(sample["sample_id"], i, int(qa["category"]), qa["question"], str(ans),
                            list(qa.get("evidence", []))))
    return out


def chunks_to_json(chunks: list[Chunk]) -> list[dict]:
    return [asdict(c) for c in chunks]
