"""E3/E4 tables and figures from scripts/bench_serve.py summaries.

    python scripts/analyze_serve.py results/serve --out results/serve/serve

Each (mode, workload, config) cell is the median over repetitions of the per-run median TTFT
(and of throughput). E3's x-axis is k; E4's is the measured reuse rate of the workload.
Writes <out>.md and <out>_e3.png / <out>_e4.png.
"""

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CONFIGS = [("full", "full prefill", "#2a78d6", "o"), ("prefix", "prefix caching", "#eb6834", "s"),
           ("connector", "chunk connector (method 1)", "#1baf7a", "^")]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def load(d):
    cells = defaultdict(list)
    for p in sorted(Path(d).glob("*.summary.json")):
        s = json.loads(p.read_text())
        t, wm = s["tags"], s.get("workload_meta") or {}
        x = wm.get("k") if t["mode"] == "e3" else wm.get("reuse_rate")
        cells[(t["mode"], x, t["config"])].append(s)
    return cells


def med(runs, key):
    xs = [r[key] for r in runs if r.get(key) is not None]
    return statistics.median(xs) if xs else None


def fmt(x, nd=1):
    return "-" if x is None else f"{x:.{nd}f}"


def table(cells, mode):
    xs = sorted({x for (m, x, _) in cells if m == mode})
    xname = "k" if mode == "e3" else "reuse rate"
    lines = [f"| {xname} | config | reps | n | prompt tok | cached frac | TTFT med ms | TTFT p90 ms | "
             "latency med ms | req/s | vs full (TTFT) |", "|" + "---|" * 11]
    for x in xs:
        base = med(cells.get((mode, x, "full"), []), "ttft_ms_median")
        for c, *_ in CONFIGS:
            runs = cells.get((mode, x, c))
            if not runs:
                continue
            ttft = med(runs, "ttft_ms_median")
            lines.append("| " + " | ".join([
                fmt(x, 0 if mode == "e3" else 2), c, str(len(runs)), str(runs[0]["n"]),
                fmt(med(runs, "mean_prompt_tokens"), 0), fmt(med(runs, "cached_fraction"), 3), fmt(ttft),
                fmt(med(runs, "ttft_ms_p90")), fmt(med(runs, "latency_ms_median")), fmt(med(runs, "req_per_s"), 2),
                "-" if not (base and ttft) else f"{base / ttft:.2f}x"]) + " |")
    return "\n".join(lines)


def plot(cells, mode, out):
    panels = [("ttft_ms_median", "median TTFT (ms)")]
    if mode == "e4":
        panels.append(("req_per_s", "throughput (requests/s)"))
    fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 3.8), dpi=200, squeeze=False)
    for ax, (key, ylabel) in zip(axes[0], panels):
        for c, label, color, marker in CONFIGS:
            pts = sorted((x, med(r, key)) for (m, x, cc), r in cells.items() if m == mode and cc == c)
            pts = [(x, y) for x, y in pts if y is not None]
            if pts:
                ax.plot(*zip(*pts), color=color, lw=2, marker=marker, ms=6, label=label,
                        markeredgecolor="white", markeredgewidth=1.5)
        ax.set_xlabel("chunks retrieved (k)" if mode == "e3" else "measured chunk reuse rate", color=MUTED)
        ax.set_ylabel(ylabel, color=MUTED)
        ax.set_ylim(bottom=0)
        ax.grid(color=GRID, lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=8)
    axes[0][0].legend(frameon=False, fontsize=8)
    title = ("E3: TTFT vs k (LoCoMo conv-26, one request at a time)" if mode == "e3"
             else "E4: serving under load vs chunk reuse (synthetic, k=6, 16 concurrent)")
    fig.suptitle(title, fontsize=9, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--out", required=True, help="output prefix")
    args = ap.parse_args()
    cells = load(args.dir)
    md = []
    for mode in ("e3", "e4"):
        if any(m == mode for m, _, _ in cells):
            md.append(f"## {mode.upper()}\n\n" + table(cells, mode))
            plot(cells, mode, f"{args.out}_{mode}.png")
    text = "\n\n".join(md)
    Path(f"{args.out}.md").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
