"""E5: method 1 served by unmodified vLLM through ChunkReuseConnector, compared with the
Transformers reference for the same served mode (``run_eval.py --method reposition_aligned``).

    python scripts/vllm_e5.py --store store/ --limit 100            # with the connector
    python scripts/vllm_e5.py --store store/ --limit 100 --no-connector   # plain vLLM, full prefill

Prompts are sent as token ids (never text), so vLLM sees exactly the ids the Transformers path
and the chunk store were built from. Rows use run_eval.py's format; compare with
``analyze.py results/reposition_aligned_...jsonl results/vllm_...jsonl``. With --dump-dir the
connector also saves the first few assembled KV tensors it injected.
"""

import argparse
import json
import time
from pathlib import Path

from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig
from vllm.inputs import TokensPrompt

from kvreuse import locomo
from kvreuse.connector_logic import StoreIndex, matched_tokens
from kvreuse.metrics import clean_prediction, score
from kvreuse.prompt import PromptBuilder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mistralai/Mistral-7B-Instruct-v0.3")
    ap.add_argument("--store", required=True)
    ap.add_argument("--chunks", default="data/chunks.json")
    ap.add_argument("--retrieval", default="data/retrieval_k10.json")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--gpu-mem", type=float, default=0.6)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--no-connector", action="store_true")
    ap.add_argument("--dump-dir")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    convs = json.loads(Path(args.chunks).read_text())["conversations"]
    by_id = {c["id"]: c for conv in convs.values() for c in conv["chunks"]}
    qs = json.loads(Path(args.retrieval).read_text())["questions"][: args.limit]
    index = StoreIndex.load(args.store)

    kw = {}
    if not args.no_connector:
        extra = {"store_dir": str(Path(args.store).resolve())}
        if args.dump_dir:
            extra["dump_dir"] = str(Path(args.dump_dir).resolve())
        kw["kv_transfer_config"] = KVTransferConfig(
            kv_connector="ChunkReuseConnector", kv_connector_module_path="kvreuse.vllm_connector",
            kv_role="kv_both", kv_connector_extra_config=extra)
    llm = LLM(model=args.model, dtype="bfloat16", enable_prefix_caching=False, seed=args.seed,
              gpu_memory_utilization=args.gpu_mem, max_model_len=args.max_model_len, **kw)
    pb = PromptBuilder(llm.get_tokenizer())

    prompts, rows = [], []
    for i, q in enumerate(qs):
        conv = convs[q["sample_id"]]
        question = locomo.Question(**{k: q[k] for k in ("sample_id", "qa_index", "category", "question", "answer",
                                                         "evidence")})
        ids = pb.full(locomo.CONV_START_PROMPT.format(*conv["speakers"]),
                      [by_id[c]["token_ids"] for c in q["chunk_ids"]], question.prompt(args.seed))
        served = 0 if args.no_connector else matched_tokens(index.match(ids), len(ids), 0, 16)
        prompts.append(TokensPrompt(prompt_token_ids=ids))
        rows.append({"i": i, **{k: q[k] for k in ("sample_id", "qa_index", "category", "question", "answer")},
                     "n_tokens": len(ids), "served_tokens": served,
                     "budget": 1.0 - served / len(ids) if not args.no_connector else 1.0})

    sp = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens)
    t = time.perf_counter()
    outs = llm.generate(prompts, sp)  # batched; greedy outputs do not depend on batching order
    wall = time.perf_counter() - t
    for row, o in zip(rows, outs):
        raw = o.outputs[0].text
        pred = clean_prediction(raw)
        row.update(raw=raw, prediction=pred, score=score(pred, row["answer"], row["category"]),
                   out_ids=list(o.outputs[0].token_ids))

    name = f"vllm_{'full' if args.no_connector else 'connector'}_top10_bf16" + (f"_n{args.limit}" if args.limit else "")
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    with open(out / f"{name}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    head = [r["score"] for r in rows if r["category"] != 5]
    summary = {"name": name, "n": len(rows), "f1_headline_cat1to4": sum(head) / max(len(head), 1),
               "served_fraction": sum(r["served_tokens"] for r in rows) / sum(r["n_tokens"] for r in rows),
               "batch_wall_s": wall, "config": vars(args)}
    (out / f"{name}.summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
