"""Prompt assembly by token-ID concatenation (Mistral-7B-Instruct-v0.3).

v0.3's chat template has no system turn: system text goes inside the last ``[INST]``
before the user content. Our layout is

    [BOS] [INST] system\\n\\n | chunk_1 | ... | chunk_k | \\n<question prompt> [/INST]
    `------ prefix ------'   `-- reused, any order --'   `----- question ------'

Every segment is tokenized once on its own (``add_special_tokens=False``) and prompts are
concatenations of those ids. Never decode and re-encode: SentencePiece can merge tokens
across a join, and the full-prefill reference must see exactly the same ids.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PromptBuilder:
    tokenizer: object

    def __post_init__(self):
        tok = self.tokenizer
        self.bos = tok.bos_token_id
        self.inst = tok.convert_tokens_to_ids("[INST]")
        self.inst_end = tok.convert_tokens_to_ids("[/INST]")
        if self.inst == tok.unk_token_id or self.inst_end == tok.unk_token_id:
            raise ValueError("tokenizer has no [INST]/[/INST] control tokens")

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def chunk_context(self) -> list[int]:
        """What a chunk is prefilled after when cached alone: BOS + [INST], so the chunk's
        own first token is not treated as the attention sink."""
        return [self.bos, self.inst]

    def prefix(self, system: str) -> list[int]:
        return [self.bos, self.inst] + self.encode(system.rstrip("\n") + "\n\n")

    def question(self, question_prompt: str) -> list[int]:
        return self.encode("\n" + question_prompt) + [self.inst_end]

    def full(self, system: str, chunk_ids: list[list[int]], question_prompt: str) -> list[int]:
        ids = self.prefix(system)
        for c in chunk_ids:
            ids += list(c)
        return ids + self.question(question_prompt)

    @property
    def stop_ids(self) -> set[int]:
        return {self.tokenizer.eos_token_id}
