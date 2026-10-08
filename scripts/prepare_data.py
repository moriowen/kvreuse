"""C1: chunk LoCoMo, tokenize each chunk once, retrieve top-k with BM25, save to disk.

    python scripts/prepare_data.py --k 10

Writes data/chunks.json and data/retrieval_k{K}.json. Every experiment reads these files;
never re-run retrieval inside an experiment.
"""

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

from kvreuse import locomo
from kvreuse.prompt import PromptBuilder
from kvreuse.retrieval import BM25Retriever, EmbeddingRetriever


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--locomo", default="data/locomo10.json")
    ap.add_argument("--tokenizer", default="mistralai/Mistral-7B-Instruct-v0.3")
    ap.add_argument("--target-tokens", type=int, default=256)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--out", default="data")
    ap.add_argument("--retriever", default="bm25", choices=["bm25", "embed"])
    ap.add_argument("--encoder", default="sentence-transformers/all-MiniLM-L6-v2")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    pb = PromptBuilder(tok)
    count = lambda text: len(pb.encode(text))
    samples = locomo.load(args.locomo)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    convs, retrieval = {}, []
    for s in samples:
        chunks = locomo.chunk_conversation(s, count, args.target_tokens)
        for c in chunks:
            c.token_ids = pb.encode(c.text)
        conv = s["conversation"]
        convs[s["sample_id"]] = {
            "speakers": [conv["speaker_a"], conv["speaker_b"]],
            "chunks": locomo.chunks_to_json(chunks),
        }
        texts = [c.text for c in chunks]
        retriever = BM25Retriever(texts) if args.retriever == "bm25" else EmbeddingRetriever(texts, args.encoder)
        for q in locomo.questions(s):
            top = retriever.topk(q.question, args.k)
            retrieval.append({**q.__dict__, "chunk_ids": [chunks[i].id for i in top]})
        lens = [len(c.token_ids) for c in chunks]
        print(f"{s['sample_id']}: {len(chunks)} chunks, {sum(lens)} tokens, "
              f"max chunk {max(lens)}, questions {len(s['qa'])}")

    meta = {"tokenizer": args.tokenizer, "target_tokens": args.target_tokens}
    ret_name = f"retrieval_k{args.k}.json" if args.retriever == "bm25" else f"retrieval_embed_k{args.k}.json"
    ret_meta = {"retriever": args.retriever, **({"encoder": args.encoder} if args.retriever == "embed" else {})}
    (out / "chunks.json").write_text(json.dumps({**meta, "conversations": convs}))
    (out / ret_name).write_text(json.dumps({**meta, **ret_meta, "k": args.k, "questions": retrieval}))
    print(f"wrote {out/'chunks.json'} and {out/ret_name} ({len(retrieval)} questions)")

    # One-time check of our hand-built layout against the official chat template.
    s0 = samples[0]
    system = locomo.CONV_START_PROMPT.format(*convs[s0["sample_id"]]["speakers"])
    first = convs[s0["sample_id"]]["chunks"][0]
    q = locomo.questions(s0)[0].prompt()
    ours = pb.full(system, [first["token_ids"]], q)
    official = tok.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": first["text"] + "\n" + q}],
        tokenize=True,
    )
    if hasattr(official, "input_ids"):
        official = official["input_ids"]
    print("\nlayout check vs apply_chat_template (expect small whitespace differences only):")
    print("  ours    :", repr(tok.decode(ours)[:160]), "...", repr(tok.decode(ours)[-60:]))
    print("  official:", repr(tok.decode(official)[:160]), "...", repr(tok.decode(official)[-60:]))


if __name__ == "__main__":
    main()
