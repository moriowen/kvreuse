"""Compare runs against a reference run (normally ``full``), question by question.

    python scripts/analyze.py results/full_top10_bf16.jsonl results/reposition_top10_bf16.jsonl ...
    python scripts/analyze.py --ref results/full_top10_bf16.jsonl results/*.jsonl --out results/table.md

Rows are matched on (sample_id, qa_index); questions missing from either run are dropped
from that comparison and the count says so. Headline F1 is categories 1-4. Delta F1 gets a
paired bootstrap CI over questions and a conversation-level (cluster) CI; with only 10
conversations the cluster interval is the honest one.
"""

import argparse
import json
import statistics
from pathlib import Path

from kvreuse.metrics import paired_bootstrap

CATS = (1, 2, 3, 4)


def load(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    return {(r["sample_id"], r["qa_index"]): r for r in rows}


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def fmt(x, nd=3):
    return "-" if x is None or x != x else f"{x:.{nd}f}"


def summarize(name, run, ref, n_boot):
    keys = sorted(set(run) & set(ref))
    head = [k for k in keys if run[k]["category"] in CATS]
    out = {"name": name, "n": len(keys), "n_run": len(run)}
    out["f1"] = mean(run[k]["score"] for k in head)
    out["per_cat"] = {c: mean(run[k]["score"] for k in head if run[k]["category"] == c) for c in CATS}
    cat5 = [k for k in keys if run[k]["category"] == 5]
    out["cat5"] = mean(run[k]["score"] for k in cat5)
    a = [run[k]["score"] for k in head]
    b = [ref[k]["score"] for k in head]
    if run is ref or not head:
        out["delta"] = out["ci"] = out["ci_conv"] = None
    else:
        out["delta"], lo, hi = paired_bootstrap(a, b, n=n_boot)
        groups = [k[0] for k in head]
        out["ci"] = (lo, hi)
        out["ci_conv"] = None  # a cluster bootstrap over one conversation is meaningless
        if len(set(groups)) >= 2:
            _, clo, chi = paired_bootstrap(a, b, groups=groups, n=n_boot)
            out["ci_conv"] = (clo, chi)
    out["n_conv"] = len({k[0] for k in head})
    answered = [k for k in head if ref[k]["score"] > 0]  # gap where the reference gets something right
    out["f1_on_answered"] = mean(run[k]["score"] for k in answered)
    out["zero_share"] = mean(float(run[k]["score"] == 0) for k in head)
    out["budget"] = mean(run[k].get("budget", 1.0) for k in keys)
    pre = [run[k]["prefill_ms"] for k in keys if "prefill_ms" in run[k]]
    ref_pre = [ref[k]["prefill_ms"] for k in keys if "prefill_ms" in ref[k]]
    out["prefill_ms"] = statistics.median(pre) if pre else None
    out["speedup"] = statistics.median(ref_pre) / out["prefill_ms"] if pre and ref_pre else None
    out["kl"] = mean(run[k]["kl_first"] for k in keys if "kl_first" in run[k]) if any(
        "kl_first" in run[k] for k in keys) else None
    out["top1"] = mean(float(run[k]["top1_agree"]) for k in keys if "top1_agree" in run[k]) if any(
        "top1_agree" in run[k] for k in keys) else None
    out["tokens"] = mean(run[k]["n_tokens"] for k in keys)
    return out


def table(rows):
    cols = ["run", "n", "convs", "F1 (1-4)", "ΔF1 vs ref", "95% CI (question)", "95% CI (conversation)",
            "multi-hop", "temporal", "open-dom", "single-hop", "cat5 acc", "F1 where ref>0",
            "zero-F1 share", "budget", "prefill ms (median)", "speedup", "KL first", "top-1 agree",
            "prompt tokens"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        ci = lambda c: "-" if c is None else f"[{c[0]:+.3f}, {c[1]:+.3f}]"
        pc = r["per_cat"]
        lines.append("| " + " | ".join([
            r["name"], f"{r['n']}" + ("" if r["n"] == r["n_run"] else f" of {r['n_run']}"),
            str(r["n_conv"]), fmt(r["f1"]), "-" if r["delta"] is None else f"{r['delta']:+.3f}", ci(r["ci"]), ci(r["ci_conv"]),
            fmt(pc[1]), fmt(pc[2]), fmt(pc[3]), fmt(pc[4]), fmt(r["cat5"]), fmt(r["f1_on_answered"]),
            fmt(r["zero_share"]), fmt(r["budget"]), fmt(r["prefill_ms"], 1),
            "-" if r["speedup"] is None else f"{r['speedup']:.2f}x", fmt(r["kl"], 4), fmt(r["top1"]),
            fmt(r["tokens"], 0),
        ]) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="results/*.jsonl")
    ap.add_argument("--ref", help="reference run (default: first argument)")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--out", help="also write the markdown table here")
    args = ap.parse_args()

    ref_path = args.ref or args.runs[0]
    ref = load(ref_path)
    rows = []
    for p in dict.fromkeys([ref_path] + args.runs):  # reference first, no duplicates
        run = ref if p == ref_path else load(p)
        rows.append(summarize(Path(p).stem, run, ref, args.boot))
    md = (f"Reference: `{Path(ref_path).stem}`. Category names follow LoCoMo's code comments "
          "(1 multi-hop, 2 temporal); check 3 and 4 against the paper before publishing.\n\n" + table(rows))
    print(md)
    if args.out:
        Path(args.out).write_text(md + "\n")


if __name__ == "__main__":
    main()
