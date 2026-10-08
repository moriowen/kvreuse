"""E2 figure: median prefill time and F1 (categories 1-4) against k, one line per method.

    python scripts/plot_e2.py results/e2 --out results/e2/e2_curve.png

Reads ``<method>_top<k>_bf16_n300.jsonl`` for full, reposition and agentkvshift_r0.1 (the
k-sweep in pace/jobs/e2.txt). Prefill times are from the cuDNN-free code (SDPA_BACKENDS).
"""

import argparse
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CATS = (1, 2, 3, 4)
METHODS = [("full", "full prefill", "#2a78d6", "o"), ("reposition", "re-position only", "#eb6834", "s"),
           ("agentkvshift_r0.1", "AgentKVShift (r=0.1)", "#1baf7a", "^")]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def stats(path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    head = [r["score"] for r in rows if r["category"] in CATS]
    return statistics.median(r["prefill_ms"] for r in rows), sum(head) / len(head)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    d = Path(args.dir)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), dpi=200)
    for name, label, color, marker in METHODS:
        pts = []
        for k in (2, 4, 6, 8, 10):
            p = d / f"{name}_top{k}_bf16_n300.jsonl"
            if p.exists():
                pts.append((k, *stats(p)))
        if not pts:
            continue
        ks, ms, f1 = zip(*pts)
        for ax, ys in zip(axes, (ms, f1)):
            ax.plot(ks, ys, color=color, lw=2, marker=marker, ms=6, label=label,
                    markeredgecolor="white", markeredgewidth=1.5)
    for ax, ylabel in zip(axes, ("median prefill time (ms)", "LoCoMo F1, categories 1-4")):
        ax.set_xlabel("chunks retrieved (k)", color=MUTED)
        ax.set_ylabel(ylabel, color=MUTED)
        ax.set_ylim(bottom=0)
        ax.set_xticks((2, 4, 6, 8, 10))
        ax.grid(color=GRID, lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=8)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("E2: prefill time and quality vs k (Mistral-7B v0.3, bf16, H100, first 300 questions)",
                 fontsize=9, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
