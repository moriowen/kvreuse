"""C4: write a request workload (token-id prompts) for scripts/bench_serve.py.

    python scripts/make_workload.py locomo --k 6 --samples conv-26 --out work/e3_k6.jsonl
    python scripts/make_workload.py zipf --repeat-p 0.5 --n 200 --k 6 --out work/e4_p0.5.jsonl

Only needs the tokenizer, not the model.
"""

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

from kvreuse.prompt import PromptBuilder
from kvreuse.workload import locomo_requests, reuse_rate, zipf_requests


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["locomo", "zipf"])
    ap.add_argument("--model", default="mistralai/Mistral-7B-Instruct-v0.3")
    ap.add_argument("--chunks", default="data/chunks.json")
    ap.add_argument("--retrieval", default="data/retrieval_k10.json")
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--samples", nargs="*", help="locomo: conversations to include (default all)")
    ap.add_argument("--n", type=int, default=200, help="zipf: number of requests")
    ap.add_argument("--repeat-p", type=float, default=0.5)
    ap.add_argument("--zipf-s", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    pb = PromptBuilder(AutoTokenizer.from_pretrained(args.model))
    chunks = json.loads(Path(args.chunks).read_text())
    retrieval = json.loads(Path(args.retrieval).read_text())
    if args.kind == "locomo":
        reqs = locomo_requests(pb, chunks, retrieval, args.k, args.samples, args.seed)
    else:
        reqs = zipf_requests(pb, chunks, retrieval, args.n, args.k, args.repeat_p, args.zipf_s, args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for r in reqs:
            f.write(json.dumps(r) + "\n")
    meta = {**vars(args), "n_requests": len(reqs), "reuse_rate": reuse_rate(reqs),
            "mean_prompt_tokens": sum(len(r["prompt_ids"]) for r in reqs) / max(len(reqs), 1)}
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
