# Prefill once, reuse anywhere

Non-prefix KV reuse for agent memory in vLLM. CS 6220 Big Data Systems, Group 7 (Fall 2026):
Atharva Mohite, Teppei Kawashima, Prakhar Langer, Vishal Puneyani.

**[View the slides](https://moriowen.github.io/kvreuse/walkthrough/slides.html)** ·
[Walkthrough site](https://moriowen.github.io/kvreuse/walkthrough/) ·
[Slides on Claude](https://claude.ai/artifact/1ix6SkJucKoX7bLywJgEMM) ·
[Proposal (PDF)](Group7-proposal-nonprefix-kv-reuse-agent-memory.pdf)

The slides and walkthrough are plain HTML in [`walkthrough/`](walkthrough/). GitHub shows
HTML files as source, so use the Pages links above, or clone the repo and open
`walkthrough/slides.html` in a browser. The slide page has a toggle for speaker notes.

| Page | What it covers |
|---|---|
| [slides.html](walkthrough/slides.html) | The 21-slide project check-in deck |
| [index.html](walkthrough/index.html) | Overview, research question, current status |
| [pipeline.html](walkthrough/pipeline.html) | One LoCoMo question followed from raw conversation to F1 |
| [methods.html](walkthrough/methods.html) | The four correction methods, with a token-by-layer grid |
| [experiments.html](walkthrough/experiments.html) | E0 to E5, what has run, numbers so far |
| [codebase.html](walkthrough/codebase.html) | Every file, its dependencies, and how jobs run on PACE |

## The problem

A memory agent retrieves a few chunks of old conversation for each question and puts them in
front of the question. Two questions about the same conversation often pull the same chunks in
a different order. vLLM's prefix cache only matches identical leading tokens, so after the
system prompt it misses, and chunks that were computed seconds earlier get prefilled again.

We cache each chunk's KV once under a hash of its tokens. When the chunk shows up again, we
re-rotate its keys (RoPE) to the new position, which is exact. The chunk still never attended to
the chunks now in front of it, so we recompute some tokens to patch that. The four methods differ
only in which tokens they recompute:

| # | Method | Recomputes |
|---|---|---|
| 1 | Re-positioning only (baseline) | nothing |
| 2 | Boundary (EPIC) | first 16 to 32 tokens of each chunk |
| 3 | Dynamic selection (CacheBlend) | the 10 to 15% of tokens that deviate most at an early layer |
| 4 | Offset correction (AgentKVShift) | a few probe tokens, then a mean shift added to the rest |

Research question: how much prefill can non-prefix KV reuse remove from a memory-augmented
agent, and how do these corrections trade answer quality against prefill cost? Served through
unmodified vLLM, does reuse lower TTFT and raise throughput, and how often must memories repeat
before it beats prefix caching?

The correction algorithms come from prior work. Our contribution is engineering and evaluation:
all four methods on one conversational-memory workload under one budget formula, a
reimplementation of AgentKVShift (which has no public code), and reuse served through vLLM's
public KV connector API without editing vLLM.

## Setup

- Model: Mistral-7B-Instruct-v0.3, the model AgentKVShift reports on LoCoMo
- Data: LoCoMo, 10 conversations and 1,986 questions, chunked into whole turns of about 256 tokens
- Retrieval: BM25 top-10, saved to disk once and read by every run
- Compute: H100 on Georgia Tech's PACE ICE cluster
- Serving: vLLM 0.27.1, pinned

## Status

| Piece | Status |
|---|---|
| C1 memory pipeline | built, ran on PACE |
| C2 KV store, re-rotation, selective recompute, 4 methods | built |
| E0 correctness gate | passed Oct 7 (worst logit error 4.6e-5 against a 1e-3 tolerance) |
| E1 quality sweep, baselines | running on PACE |
| E2 prefill time vs. k | job list ready |
| C3 vLLM connector | built; E5 match check not run yet |
| C4 Zipf workload, C5 serving benchmark | not built |

## Repo layout

```
kvreuse/        package: chunking, retrieval, KV store, RoPE, recompute loop, methods, vLLM connector
scripts/        prepare_data, e0_real, run_eval, analyze, export_store, vllm_e5, profile_prefill
tests/          E0 logic and connector logic on a tiny random Mistral (CPU, under a second)
pace/           Slurm job scripts for PACE ICE
walkthrough/    HTML slides and walkthrough pages
reports/        research review
```

## Running it

```bash
python3.11 -m venv .venv && .venv/bin/pip install -e '.[dev]'
.venv/bin/python -m pytest
curl -sSL -o data/locomo10.json https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json
python scripts/prepare_data.py --k 10
python scripts/e0_real.py --n 20          # needs a CUDA GPU
python scripts/run_eval.py --method full  # also reposition, epic, cacheblend, agentkvshift
```

The real model runs only on a GPU node. [AGENTS.md](AGENTS.md) has the PACE workflow, the fixed
design decisions, and the experiment rules.
