"""E1 figure: LoCoMo F1 (categories 1-4) against the charged recompute budget, one line per method.

    python scripts/plot_e1.py results/e1 --out results/e1/e1_curve.png

Reads every ``*_top10_bf16.jsonl`` in the directory. The x value is the measured mean budget
(the one formula for every method), not the nominal ratio or EPIC's k. ``full`` and
``reposition`` are the two ends of every curve; ``full_context`` is drawn as a ceiling line.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CATS = (1, 2, 3, 4)
METHODS = [("epic", "EPIC (boundary)", "#2a78d6", "o"),
           ("cacheblend", "CacheBlend", "#eb6834", "s"),
           ("agentkvshift", "AgentKVShift", "#1baf7a", "^")]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def point(path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    head = [r for r in rows if r["category"] in CATS]
    return (sum(r.get("budget", 1.0) for r in rows) / len(rows),
            sum(r["score"] for r in head) / len(head))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    d = Path(args.dir)

    full = point(d / "full_top10_bf16.jsonl")
    repo = point(d / "reposition_top10_bf16.jsonl")
    ctx = d / "full_context_bf16.jsonl"

    fig, ax = plt.subplots(figsize=(7, 4.4), dpi=200)
    for name, label, color, marker in METHODS:
        pts = sorted(point(p) for p in d.glob(f"{name}_*_top10_bf16.jsonl"))
        xs, ys = zip(*([repo] + pts))
        ax.plot(xs, ys, color=color, lw=2, marker=marker, ms=6, label=label,
                markeredgecolor="white", markeredgewidth=1.5, zorder=3)
        ax.annotate(label, (xs[-1], ys[-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=8, color=INK)

    ax.axhline(full[1], color=INK, lw=1, ls="--", zorder=1)
    ax.text(0.01, full[1], f"full prefill {full[1]:.3f}", va="bottom", fontsize=8, color=INK)
    ax.axhspan(full[1] - 0.02, full[1], color=GRID, alpha=0.6, lw=0, zorder=0)
    ax.text(0.01, full[1] - 0.02, "within 0.02 F1 (target)", va="bottom", fontsize=7, color=MUTED)
    if ctx.exists():
        c = point(ctx)[1]
        ax.axhline(c, color=MUTED, lw=1, ls=":", zorder=1)
        ax.text(0.99, c, f"full context {c:.3f}", va="bottom", ha="right", fontsize=8, color=MUTED)
    ax.plot(*repo, marker="D", ms=6, color=MUTED, zorder=4)
    ax.annotate(f"re-position only {repo[1]:.3f}", repo, xytext=(6, -10), textcoords="offset points",
                fontsize=8, color=MUTED)

    ax.set_xlim(-0.02, 1.0)
    ax.set_xlabel("recompute budget (charged fraction of reusable token-layers)", color=MUTED)
    ax.set_ylabel("LoCoMo F1, categories 1-4", color=MUTED)
    ax.set_title("E1: answer quality vs. recompute budget (Mistral-7B v0.3, top-10, bf16, n=1,986)",
                 fontsize=9, color=INK, loc="left")
    ax.grid(color=GRID, lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
