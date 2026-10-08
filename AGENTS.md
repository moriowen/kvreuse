# AGENTS.md

Working agreement for coding agents in `bds/project/`, the CS 6220 Big Data Systems semester
project. **This file is the source of truth.** `CLAUDE.md` is a stub that imports it. The
workspace-wide rules in `~/dev/acads/AGENTS.md` still apply; this file adds to them.

## What this is

**Prefill once, reuse anywhere: non-prefix KV reuse for agent memory in vLLM.** Team of 4,
45% of the course grade. The topic is final. Earlier candidates (FairCache, RAG compaction,
GH Archive memory, snapshot rebuild) were dropped, and their files were deleted. Don't bring
them back.

[`proposal-nonprefix-kv-reuse-agent-memory.md`](proposal-nonprefix-kv-reuse-agent-memory.md)
is the spec. The `.pdf` next to it is rendered from the `.md` and is the copy for submission. If
a decision changes, amend the `.md` and regenerate the PDF. Leave the old rationale out.
The `.md` was amended on Oct 7, 2026 (novelty wording, related work, chunk definition, E0
check d, baselines, statistics); the PDF has **not** been regenerated since.

[`reports/Non prefix KV reuse project evaluation.md`](reports/Non%20prefix%20KV%20reuse%20project%20evaluation.md)
is the Oct 7 research review: prior art, implementation pitfalls, vLLM 0.27.1 connector
internals, interview framing, learning path. Its sources are in `research_notes/`.

[`concept-guide.md`](concept-guide.md) explains the problem space (why reused KV is wrong,
how CacheBlend and AgentKVShift correct it) with section-level paper citations. Read it
before touching the correction methods.

Deadlines live in `~/dev/acads/calendar.md`, not here.

**Status:** C1 and C2 built. **E0 passed on the real model** (Oct 7, 2026, PACE job 6092311,
H100 80GB, 20 LoCoMo prompts, fp32): worst max-abs logit error 4.6e-5 vs HF, 7.3e-5 for 100%
recompute, 4.2e-5 for the identity case (gate 1e-3). bf16 noise floor: mean first-token KL
6.4e-4, max 4.0e-3, top-1 agreement 1.000. vLLM work (C3) is now unblocked.

**E1 done (Oct 8, 2026; jobs 6092366 + 6092367, H100, bf16, top-10, all 1,986 questions).**
F1 cat 1-4: full 0.319, full_context 0.326, re-position only 0.152. Best at each charged budget:
AgentKVShift 0.210 @ 0.12, 0.225 @ 0.18, 0.240 @ 0.23, 0.272 @ 0.40, 0.302 @ 0.70; CacheBlend
and EPIC are at or below it everywhere (EPIC tops out at 0.236 @ 0.28). **The 0.02-F1 @ 10-15%
target is not met**; AgentKVShift gets within 0.02 only at ~70%. The failure is mostly the
model answering "not mentioned" (43% of re-position answers vs 13% for full), which also
inflates category-5 accuracy (0.46 vs 0.29). E5 confirms the served path matches Transformers
(0.120 vs 0.119 on n=100). Table and plot are in `results/e1/` (`scripts/analyze.py`,
`scripts/plot_e1.py`). Nominal ratio != charged budget: layers 0-1 are charged in full.

**Timing finding (Oct 7, profile jobs 6092365, 6092397, 6092445, H100).** PyTorch picked cuDNN's
SDPA kernel, which rebuilds its plan for every new sequence length; every LoCoMo prompt has a
new length, so each call paid ~40-45 ms. Median prefill over 12 distinct prompt lengths (~2.3K
tokens): full 115.7 -> 78.9 ms and re-positioning 75.1 -> 29.9 ms once cuDNN is excluded
(`SDPA_BACKENDS` in `kvreuse/forward.py`; the default now). A fresh 67-token forward still
takes ~19 ms against a ~4 ms weight-read floor: the eager per-layer loop is launch-bound
(~1,500 kernel launches per call), so CUDA graphs or `torch.compile` are the next speedup.
**E1 sweep 6092366 and baselines 6092367 were submitted with the old code (cuDNN allowed):
use them for quality only, never their `prefill_ms`.** E2 timings must come from the new code.

## The research question

> How much prefill can non-prefix KV reuse remove from a memory-augmented agent, and how do
> different ways of correcting reused KV trade answer quality against prefill cost?

Secondary question (C3-C5, E3-E5): served through unmodified vLLM, does reuse lower TTFT and
raise throughput, and how often must memories repeat before it beats prefix caching?

Retrieved memory chunks come back in different orders, which breaks vLLM's prefix cache.
We cache each chunk's KV under a hash of its tokens, re-rotate it (RoPE) to its new position,
and then correct for the cross-chunk attention it missed. There are four correction methods,
and they differ **only in which tokens get recomputed**:

| # | Method | Recomputes |
|---|---|---|
| 1 | Re-positioning only (baseline) | nothing |
| 2 | Boundary (EPIC) | first 16–32 tokens of each chunk |
| 3 | Dynamic selection (CacheBlend) | the 10–15% of tokens that deviate most at an early layer |
| 4 | Offset correction (AgentKVShift) | a few probe tokens, then a mean shift added to the rest |

## Fixed decisions

Don't change these without the team. If one changes, update the proposal too.

- **Model:** Mistral-7B-Instruct-v0.3. It's the model AgentKVShift reports on LoCoMo, so our
  numbers can be compared with theirs.
- **Data:** LoCoMo for quality (10 conversations, 1,986 questions). A synthetic pool with
  Zipf-distributed popularity for systems measurements.
- **Compute:** PACE ICE, on an A100-80GB or H100.
- **No changes to upstream source.** vLLM, Transformers, the model and LoCoMo are used
  unmodified. vLLM can only serve method 1, through an external KV connector plugin. Methods
  2–4 stay in Transformers because they would need scheduler changes. Editing vLLM source is
  out of scope.
- **Chunks** are whole turns from one session, packed greedily to ~256 tokens, never split
  mid-turn, each starting with LoCoMo's `DATE: ...\nCONVERSATION:\n` header. Each chunk is
  tokenized once; prompts are token-ID concatenations, never re-tokenized text. With the
  Mistral tokenizer the 10 conversations are 17K-31K tokens.
- **Chunk cache convention:** a chunk is prefilled after `[BOS][INST]` and only its slice is
  kept. The system prefix sits at position 0 and is cached exactly.
- **Store** holds post-RoPE keys; re-rotation by a position delta in fp32, cast once.
- **Baselines:** `full` (full prefill of the same retrieved prompt) is the reference for every
  method; `full_context` (whole conversation, no retrieval) is the quality ceiling and the
  case prefix caching wins.
- **Headline F1 is categories 1-4.** Category 5 (adversarial, 446 of 1,986 questions) is
  reported separately. Per-category F1 is a headline result.
- **Pin one vLLM version: 0.27.1.** The connector API changes between releases. LMCache's connector
  is the reference implementation, but its blend path is known to be broken on recent vLLM
  (LMCache issues #4476 and #5101). Don't rely on it.

## Components (planned)

1. Memory pipeline: chunk conversations into entries of a few hundred tokens, retrieve
   top-k with BM25 or embeddings, and **save the retrieval results to disk**.
2. Transformers implementation: chunk KV store, RoPE re-rotation, and a custom per-layer
   forward pass that recomputes a chosen set of tokens against the assembled cache. Each of
   the 4 methods is just a token-selection function plugged into this one loop.
3. vLLM connector plugin: supplies the assembled, re-rotated KV as the prompt prefix, so vLLM
   only prefills the question.
4. Synthetic Zipf workload generator.
5. Benchmark harness: vLLM's benchmark tool with a custom dataset (TTFT, latency, throughput).

**Build order is part of the scope.** Components 1–2 and experiments E0–E2 are the core and
answer the research question by themselves. Components 3–5 and E3–E5 come after the core
works. The multi-agent workload is a stretch goal. Don't start vLLM work until E0 passes.

## Experiment rules

- **E0 is the correctness gate.** Four checks: (a) RoPE round trip returns the original keys;
  (b) first-layer shifted keys equal keys computed directly at the target position, with a
  tolerance that grows with position; (c) 100% recompute equals full-prefill logits; (d) a
  chunk reused where and after what it was cached equals full prefill. Run in fp32 with TF32
  off; separately measure the bf16 noise floor (bf16 vs fp32 full prefill). Re-rotation alone can't match full prefill once the
  preceding context changes; that gap is E1, not an E0 failure. If E0 fails,
  every later number is meaningless.
- Every configuration reads the **same saved retrieval results**. Never re-run retrieval
  inside an experiment.
- Greedy decoding, one warm-up run, report the median of 3 repetitions for timings. F1 runs
  are deterministic; compare methods with paired bootstrap CIs, and with a conversation-level
  bootstrap (only 10 conversations).
- **Budget accounting is one formula for every method:** sum over layers of reusable tokens
  with fresh QKV, divided by (layers x reusable tokens). Layers run in full before a selection
  layer, and the selection layer itself, are charged in full.
- Quality metric: LoCoMo F1, overall and per question category, compared against full prefill.
- E5 checks method 1 through the vLLM connector against Transformers: identical tokenization
  and matching sampled KV tensors first, then near-identical greedy outputs and F1. Same F1
  alone proves nothing.

Success criteria (from the proposal): AgentKVShift lands within 0.02 absolute F1 of full
prefill (or within its own reported relative gap) at a 10–15% recompute budget; CacheBlend and EPIC get a full quality-cost curve (0–60%)
with no fixed threshold; prefill time drops at least 2x at k ≥ 6; the connector
beats both full prefill and prefix caching on TTFT when entries repeat in different orders.

## Reporting results

- **Report what we measure.** If reuse loses more quality than the papers claim, that is a
  finding. Don't tune until the numbers hit the target.
- Never write a number into the proposal or a report unless a run produced it. Keep the run
  config (vLLM version, GPU, k, budget, seed) next to each result.
- The contribution is engineering and evaluation, not a new algorithm. Don't overclaim
  novelty in writeups.

## Prior art to cite

CacheBlend (arXiv 2405.16444), EPIC (2410.15332), AgentKVShift (2607.21604, the main
comparison point), LMCache, KVCOMM (2510.12872), KVCMAS (2609.34060), LoCoMo, ProphetKV (2602.02579), FusionRAG (2601.12904), RelaxKV
(2609.33503), the Cestola et al. reuse study (2603.20218), MiniPIC (2606.13126). The full list
is in the proposal's References section.

## Code

Python package `kvreuse/` (C1 and C2), scripts in `scripts/`, tests in `tests/`. The research
review's implementation section explains each choice. Things that bite:

- `kvreuse/forward.py` runs the decoder layers by hand on plain tensors with an explicit
  `kv_pos <= q_pos` mask. Don't swap in HF's cache or `model(...)` for selective recompute:
  HF's SDPA path with no mask assumes contiguous queries and crops K/V. Don't use
  FlashAttention-2 for selective recompute; it can't express the mask.
- The recompute set can only shrink with depth. Methods that select at layer 1 (CacheBlend,
  AgentKVShift) run layer 0 and layer-1 QKV for every token, and the budget charges it.
- CacheBlend follows its released code (one selection, V only, global ratio), not its paper.
  AgentKVShift is a reimplementation from Algorithm 1 (no public code); our choices where the
  paper is silent are in the class docstring.
- `scripts/run_eval.py` asserts the assembled ids equal the reference prompt. Keep that.

Setup and commands (local CPU for tests; GPU node for real runs):

```bash
python3.11 -m venv .venv && .venv/bin/pip install -e '.[dev]'
.venv/bin/python -m pytest                       # E0 logic on a tiny random Mistral, CPU, <1 s
curl -sSL -o data/locomo10.json https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json
python scripts/prepare_data.py --k 10            # data/chunks.json, data/retrieval_k10.json
python scripts/e0_real.py --n 20                 # E0 gate on the real model + bf16 noise floor
python scripts/run_eval.py --method full         # also: full_context, reposition, epic --epic-k 16,
                                                 #   cacheblend --budget 0.15 --kl, agentkvshift --budget 0.1 --kl
```

**Model source.** The official `mistralai/Mistral-7B-Instruct-v0.3` is gated. PACE jobs default
to the ungated mirror `unsloth/mistral-7b-instruct-v0.3` (`MODEL` in `pace/env.sh`). Its three
`model-0000*-of-00003.safetensors` shards have the same SHA-256 as the official ones (checked
Oct 7, 2026). Its config matches v0.3 (theta 1e6, no sliding window); it only adds a
`pad_token` and unsloth metadata. Every summary records the model id. Results land in
`results/<name>.jsonl` plus `.summary.json` with the full config.

Analysis and sweeps:

```bash
python scripts/analyze.py results/full_top10_bf16.jsonl results/*.jsonl --out results/table.md
python scripts/run_eval.py --method full --topk 6       # first k of the saved top-10 (E2)
python scripts/prepare_data.py --retriever embed        # dense retrieval; needs pip install -e '.[embed]'
```

**The real model runs only on PACE.** `run_eval.py` and `e0_real.py` exit when no CUDA GPU is
present; `--allow-cpu` exists only for smoke tests with a tiny local model.

### Running on PACE ICE

`pace/` holds the job scripts. Everything lives under `~/scratch` (home is ~15 GB).

```bash
# from the laptop (GT VPN on), copy the code
rsync -av --exclude .venv --exclude data --exclude results --exclude '*.egg-info' \
    ./ <gtuser>@login-ice.pace.gatech.edu:~/scratch/kvreuse/
# on the ICE login node
cd ~/scratch/kvreuse
export HF_TOKEN=hf_...                    # account that accepted the Mistral license
bash pace/setup.sh                        # venv, LoCoMo download, HF access check, tests
bash pace/submit.sh                       # prepare -> e0 -> smoke (100 questions)
bash pace/submit.sh --with-baselines      # adds full + full_context over all questions
squeue -u $USER                           # logs/ and results/ fill in as jobs finish
```

Jobs chain with `afterok`, so a failed E0 stops everything after it. ICE facts, checked
with `sinfo`/`sacctmgr` on Oct 7, 2026:

- GPU jobs go to partition `ice-gpu` (16 h max), CPU jobs to `ice-cpu`. Account `coc`, QOS
  `coc-ice`; no `-A` needed.
- GPU type names are lowercase. `ice-gpu` has h100 and h200 (6 nodes x 8 each), plus a few
  a100, a40, l40s, v100, rtx_6000. Default is `gpu:h100:1`; `GPU=h200 bash pace/submit.sh`
  overrides it.
- **ICE caps a user at 50 submitted jobs, array tasks included**, and that cap is shared
  with other coursework (HW3 runs as `tqa-*`). Pack runs: `pace/array.sbatch` runs several
  lines of a job list per GPU task.
- **ICE also caps GPU x requested-minutes across a user's running jobs**
  (`MaxGRESRunMinsPerUser`), shared with HW3. Request realistic `-t` (measured: an E1 config
  over 1,986 questions takes ~20 min on an H100; four per array task ~1.5 h). A pending or
  running job's limit can be lowered with `scontrol update JobId=<id> TimeLimit=HH:MM:SS`,
  never raised.
- Login with `ssh amohite8@login-ice.pace.gatech.edu` (GlobalProtect VPN on). Never compute on
  the login node.

E1 and E2 sweeps (only after E0 passes):

```bash
sbatch --array=0-3 --export=ALL,PER_TASK=4 pace/array.sbatch pace/jobs/e1.txt   # 16 configs
sbatch --array=0-1 --export=ALL,PER_TASK=8 pace/array.sbatch pace/jobs/e2.txt   # k-sweep, 300 q
```

### C3: vLLM connector (method 1)

- `kvreuse/vllm_connector.py` is `ChunkReuseConnector` (KVConnectorBase_V1, vLLM 0.27.1).
  `kvreuse/connector_logic.py` holds the vLLM-free parts (segment matching, block alignment,
  slot mapping, packed re-rotation) and is unit-tested locally.
- Facts checked in the 0.27.1 source: logical KV cache shape per layer is
  `(num_blocks, num_kv_heads, block_size, 2*head_dim)` with K in the first half; keys are
  cached post-RoPE (Mistral applies `rotary_emb` before `attn`); the scheduler offers the
  connector a block-aligned local hit and takes its count as the external hit.
- The store is built offline by `scripts/export_store.py` from the Transformers path (same
  bytes as method 1 uses), in vLLM's packed layout: about 128 KiB per token, so ~2.8 GB for
  conv-26 and ~35 GB for all ten conversations.
- vLLM gets its own venv (`pace/setup_vllm.sh`, `~/scratch/kvreuse-vllm-venv`) because it pins
  its own torch. Our package is installed there with `--no-deps`.
- E5: `pace/e5.sbatch` exports conv-26, runs the Transformers reference for the served mode
  (`--method reposition_aligned`: reused KV up to the last whole 16-token block, the rest
  fresh), vLLM with and without the connector, and compares greedy answers and F1.
- Always run the connector with prefix caching off.
- **Two code copies on PACE.** `~/scratch/kvreuse` is frozen while E1 (6092366) and baselines
  (6092367) run, because each config starts a new Python process that rereads the code.
  New work runs from `~/scratch/kvreuse-dev` with
  `--export=ALL,PROJECT_DIR=$HOME/scratch/kvreuse-dev,PYTHONPATH=$HOME/scratch/kvreuse-dev`.
  Sync dev into main only once those jobs finish.

Not yet built: C4 Zipf generator, C5 benchmark harness (served TTFT/throughput, E3/E4).
