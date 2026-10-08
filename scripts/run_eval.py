"""Run one configuration over LoCoMo and score it (E1/E2, and the week-7 baselines).

    # baselines
    python scripts/run_eval.py --method full
    python scripts/run_eval.py --method full_context
    # reuse methods
    python scripts/run_eval.py --method reposition
    python scripts/run_eval.py --method epic --epic-k 16
    python scripts/run_eval.py --method cacheblend --budget 0.15 --kl
    python scripts/run_eval.py --method agentkvshift --budget 0.10 --kl

full          retrieved prompt, ordinary prefill (the reference for every reuse method)
full_context  the whole conversation in order, no retrieval (quality ceiling; the case
              prefix caching handles best)

Writes results/<name>.jsonl (one row per question) and results/<name>.summary.json.
"""

import argparse
import json
import platform
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from kvreuse import locomo
from kvreuse.device import pick_device
from kvreuse.forward import ModelView, full_prefill, greedy_decode, selective_prefill
from kvreuse.methods import make_plan
from kvreuse.metrics import clean_prediction, score
from kvreuse.prompt import PromptBuilder
from kvreuse.store import ChunkStore, assemble

DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32, "fp16": torch.float16}


class Timer:
    """CUDA-event timing on GPU (with sync), perf_counter on CPU. Milliseconds."""

    def __init__(self, device):
        self.cuda = device.type == "cuda"

    def __enter__(self):
        if self.cuda:
            self.a, self.b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            self.a.record()
        else:
            self.t = time.perf_counter()
        return self

    def __exit__(self, *exc):
        if self.cuda:
            self.b.record()
            torch.cuda.synchronize()
            self.ms = self.a.elapsed_time(self.b)
        else:
            self.ms = (time.perf_counter() - self.t) * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True,
                    choices=["full", "full_context", "reposition", "reposition_aligned", "epic", "cacheblend",
                             "agentkvshift"])
    ap.add_argument("--budget", type=float, help="recompute ratio for cacheblend / agentkvshift")
    ap.add_argument("--epic-k", type=int, help="tokens recomputed at the start of each chunk")
    ap.add_argument("--model", default="mistralai/Mistral-7B-Instruct-v0.3")
    ap.add_argument("--dtype", default="bf16", choices=DTYPES)
    ap.add_argument("--chunks", default="data/chunks.json")
    ap.add_argument("--retrieval", default="data/retrieval_k10.json")
    ap.add_argument("--categories", default="1,2,3,4,5")
    ap.add_argument("--limit", type=int, help="first N questions only (smoke tests)")
    ap.add_argument("--topk", type=int, help="use the first k of the saved retrievals (E2 k-sweep; must be <= saved k)")
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--kl", action="store_true", help="also run full prefill and log first-token KL")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--allow-cpu", action="store_true", help="smoke tests with a tiny model only")
    ap.add_argument("--name", help="output name (default derived from the config)")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    dev = pick_device(args.allow_cpu)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=DTYPES[args.dtype]).to(dev).eval()
    view = ModelView(model)
    pb = PromptBuilder(tok)

    chunks_data = json.loads(Path(args.chunks).read_text())
    ret = json.loads(Path(args.retrieval).read_text())
    assert chunks_data["tokenizer"] == ret["tokenizer"] == args.model, "tokenizer mismatch"
    convs = chunks_data["conversations"]
    topk = args.topk or ret["k"]
    if topk > ret["k"]:
        raise SystemExit(f"--topk {topk} > saved k {ret['k']}; rerun prepare_data.py with a larger --k")
    by_id = {c["id"]: c for conv in convs.values() for c in conv["chunks"]}

    cats = {int(c) for c in args.categories.split(",")}
    qs = [q for q in ret["questions"] if q["category"] in cats][: args.limit]
    reuse = args.method not in ("full", "full_context")
    plan = make_plan(args.method, args.budget, args.epic_k) if reuse else None
    store = ChunkStore(view, pb.chunk_context())

    name = args.name or "_".join(
        x for x in [args.method, args.budget and f"r{args.budget}", args.epic_k and f"k{args.epic_k}",
                    f"top{topk}" if args.method != "full_context" else None, args.dtype,
                    args.limit and f"n{args.limit}"] if x)
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    rows = []

    # warm-up (kernels, allocator) so the first question's timing is not an outlier
    full_prefill(view, torch.tensor(pb.prefix("warm up") * 4, device=dev))

    with open(out / f"{name}.jsonl", "w") as f:
        for i, q in enumerate(qs):
            conv = convs[q["sample_id"]]
            system = locomo.CONV_START_PROMPT.format(*conv["speakers"])
            question = locomo.Question(**{k: q[k] for k in
                                          ("sample_id", "qa_index", "category", "question", "answer", "evidence")})
            if args.method == "full_context":
                chunk_ids = [c["token_ids"] for c in conv["chunks"]]
            else:
                chunk_ids = [by_id[cid]["token_ids"] for cid in q["chunk_ids"][:topk]]
            q_ids = pb.question(question.prompt(args.seed))
            ids = torch.tensor(pb.full(system, chunk_ids, question.prompt(args.seed)), device=dev)

            row = {"i": i, **{k: q[k] for k in ("sample_id", "qa_index", "category", "question", "answer")},
                   "n_tokens": len(ids)}
            if reuse:
                misses0 = store.misses
                with Timer(dev) as t_load:
                    prefix = store.prefix(pb.prefix(system))
                    entries = [store.chunk(c) for c in chunk_ids]
                row["store_misses"] = store.misses - misses0
                row["store_ms"] = t_load.ms  # includes cold chunk prefills; reported separately
                with Timer(dev) as t_asm:
                    x, K0, V0, layout = assemble(view, prefix, entries, q_ids)
                assert torch.equal(x, ids), "assembled ids differ from the reference prompt"
                with Timer(dev) as t_pre:
                    res = selective_prefill(view, x, K0, V0, layout, plan)
                row.update(assemble_ms=t_asm.ms, prefill_ms=t_pre.ms, budget=res.budget(layout.n_reusable),
                           n_reusable=layout.n_reusable)
                if args.kl:
                    ref = full_prefill(view, ids).last_logits
                    row["kl_first"] = float(F.kl_div(F.log_softmax(res.last_logits, -1),
                                                     F.log_softmax(ref, -1), log_target=True, reduction="sum"))
                    row["top1_agree"] = bool(ref.argmax() == res.last_logits.argmax())
            else:
                with Timer(dev) as t_pre:
                    res = full_prefill(view, ids)
                row.update(prefill_ms=t_pre.ms, budget=1.0)

            out_ids = greedy_decode(view, res, len(ids), args.max_new_tokens, pb.stop_ids)
            raw = tok.decode(out_ids, skip_special_tokens=True)
            pred = clean_prediction(raw)
            row.update(raw=raw, prediction=pred, score=score(pred, q["answer"], q["category"]))
            f.write(json.dumps(row) + "\n")
            f.flush()
            rows.append(row)
            if i % 50 == 0:
                print(f"[{i}/{len(qs)}] {row['category']} {pred!r} | gold {q['answer']!r} | {row['score']:.2f}")

    by_cat = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r["score"])
    head = [r["score"] for r in rows if r["category"] != 5]
    mean = lambda xs: sum(xs) / len(xs) if xs else None
    summary = {
        "name": name,
        "config": {**vars(args), "torch": torch.__version__, "transformers": transformers.__version__,
                   "gpu": torch.cuda.get_device_name() if dev.type == "cuda" else platform.processor(),
                   "chunk_context": "bos_inst", "retrieval_k": ret["k"], "topk_used": topk},
        "n": len(rows),
        "f1_headline_cat1to4": mean(head),
        "per_category": {c: {"n": len(v), "mean": mean(v)} for c, v in sorted(by_cat.items())},
        "zero_f1_share": mean([float(r["score"] == 0) for r in rows if r["category"] != 5]),
        "mean_budget": mean([r["budget"] for r in rows]),
        "median_prefill_ms": sorted(r["prefill_ms"] for r in rows)[len(rows) // 2] if rows else None,
        "mean_prompt_tokens": mean([r["n_tokens"] for r in rows]),
        "mean_answer_words": mean([len(r["prediction"].split()) for r in rows]),
    }
    if args.kl and reuse:
        summary["mean_kl_first"] = mean([r["kl_first"] for r in rows])
        summary["top1_agree"] = mean([float(r["top1_agree"]) for r in rows])
    (out / f"{name}.summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in summary if k != "config"}, indent=2))


if __name__ == "__main__":
    main()
