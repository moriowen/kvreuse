# Non-prefix KV reuse: concept guide

Companion to [`proposal-nonprefix-kv-reuse-agent-memory.md`](proposal-nonprefix-kv-reuse-agent-memory.md).
The proposal says what we will build and measure. This guide explains why the problem exists
and how CacheBlend and AgentKVShift each repair the error that KV reuse introduces.

Written Oct 1, 2026. Paper claims below are taken from the papers themselves, with section
numbers so you can check them. Anything that is our own reasoning rather than a paper's claim
is marked as such.

## Contents

1. The short version
2. Background: prefill, the KV cache, positions
3. The problem: why reused KV is wrong, and by how much
4. CacheBlend: recompute the tokens that deviate most
5. AgentKVShift: correct every token with a shared offset
6. EPIC and other related work
7. The four methods side by side
8. How this maps onto our project
9. Open questions and pitfalls
10. Glossary
11. References

---

## 1. The short version

An agent retrieves memories A, B, C for one question and C, A, D for the next. The same text
comes back in a different order, so vLLM's prefix cache misses and every memory is prefilled
again.

We want to cache each memory's KV once and paste it into any prompt. Two things are wrong
with a pasted chunk:

- **Position.** Its keys were rotated for the old position. This is fixable exactly with a
  second RoPE rotation.
- **Context.** Its KV was computed without seeing the chunks now placed before it. This
  cannot be fixed without recomputing something.

Every method in this project is an answer to the second problem with a small compute budget:

| Method | Idea in one line |
|---|---|
| Re-positioning only | Ignore the context error. |
| EPIC | The error sits mostly at the start of each chunk, so recompute those first tokens. |
| CacheBlend | The error sits in a few tokens anywhere in the chunk, so find and recompute those. |
| AgentKVShift | The error is mostly one shared shift per chunk, so estimate it from a few tokens and add it to all of them. |

## 2. Background

### Prefill and the KV cache

A decoder-only transformer processes a prompt in one forward pass called prefill. At every
layer, each token produces a query, a key and a value. Keys and values are saved (the KV
cache) so that later tokens, including generated ones, can attend to them without
recomputing the prompt.

Prefill cost grows with prompt length (linear in the MLPs, quadratic in attention). For long
prompts it dominates **time to first token (TTFT)**. Decoding each later token is cheap by
comparison, because it reuses the cache.

### What a token's KV depends on

At layer 0, a token's key and value are a linear projection of its own embedding. They depend
only on the token itself and its position.

From layer 1 upward, the input to each layer is the output of the layer below, which mixed
in information from every earlier token through attention. So a token's KV at layer l > 0
depends on:

1. the token,
2. its position,
3. **every token before it**.

That third dependency is the whole problem. Our E0 test relies on the layer-0 fact: after
re-rotation, layer-0 KV must match full prefill exactly at any position, because layer 0
never saw the context.

### RoPE and why position is easy

Mistral uses rotary position embeddings (RoPE). Keys and queries are rotated by an angle
proportional to their position, and the attention score between a query at position m and a
key at position n depends only on the difference m − n.

So moving a cached key from position p to position p' is one more rotation by (p' − p).
CacheBlend states this directly: the correction is "done simply by multiplying the K vector
by a rotation matrix," with negligible overhead (CacheBlend §4.3, proof in Appendix A).
Values carry no position, so they need no change.

### Prefix caching in vLLM

vLLM stores KV in fixed-size blocks and keys each block by a hash of all tokens up to and
including it. Two prompts share cached blocks only up to the first token where they differ.
That is exact, because a shared prefix really does have identical KV. It is also brittle:
one changed token early in the prompt invalidates everything after it.

## 3. The problem

### The agent memory workload

Our prompt is:

```
[ system prompt ][ memory 1 ][ memory 2 ] ... [ memory k ][ question ]
```

Across a LoCoMo conversation (105 to 260 questions each), the retriever returns overlapping
sets of memories in different orders. Prefix caching covers the system prompt and nothing
after the first memory that differs. Every memory after that point is prefilled again, even
if it was prefilled one question earlier.

AgentKVShift frames the agent case the same way: when a memory "is retrieved under a
different query or dialogue state than when originally encoded, its contextual
representations diverge" (AgentKVShift §3.1). Its §2 also notes a difference from plain RAG:
agent memories are curated by the agent and may carry LLM-generated summaries, keywords and
tags, not just raw passages.

### Full reuse: what goes wrong

Suppose we cache each memory's KV by prefilling it alone (or after the system prompt only),
re-rotate it to its new position, and concatenate. Inside each chunk, tokens saw each other
correctly. Across chunks, nothing was ever computed: memory 2's KV never attended to
memory 1.

CacheBlend calls this missing piece **cross-attention** between chunks and shows that full
reuse makes the model "ramble and not produce the right answer" when the answer needs facts
from two chunks together (CacheBlend §3.3, the Messi vs. Ronaldo example).

The question is always computed fresh, so it attends to every memory correctly. The error
is only in the memories' own KV, which the question then reads.

### Measuring the error

AgentKVShift writes the error as a residual per layer l, for a reused chunk:

```
R_K[l] = K_fresh[l] − K_reuse[l]
R_V[l] = V_fresh[l] − V_reuse[l]
```

where "fresh" is what full prefill of the actual prompt would produce (AgentKVShift §3.1).
CacheBlend calls the per-token size of this residual **KV deviation**, and the resulting
change in what the question's tokens attend to **attention deviation** (CacheBlend §4.1).

Why the error is usually small (our summary of both papers' argument): most attention mass
stays inside a token's own chunk and on a few sink tokens, so cross-chunk context changes a
token's hidden state only a little. Both papers show it is small on average and concentrated
in structure that a cheap method can exploit. They disagree on what that structure is, and
that disagreement is the core of our comparison.

## 4. CacheBlend

Yao et al., *CacheBlend: Fast Large Language Model Serving for RAG with Cached Knowledge
Fusion*, EuroSys 2025. [arXiv 2405.16444](https://arxiv.org/abs/2405.16444).

### Claim: the error is sparse across tokens

CacheBlend observes that "about 10-15% of tokens have much higher KV deviation than others"
(§4.3). It attributes this to attention sparsity: high attention "typically only occurs
between a small number of tokens and their preceding tokens." Most tokens barely care what
came before their chunk, and a few care a lot.

So the fix is to find those few **high-KV-deviation (HKVD) tokens** and recompute them,
leaving the rest as reused. The paper's key insight: "On layer i, recomputing the KV of token
j who has a higher KV deviation reduces the attention deviation by a greater amount" (§4.3).

### How it finds HKVD tokens without full prefill

Knowing the true deviation would require the fresh KV, which is what we are trying to avoid
computing. CacheBlend's way out is that deviation is correlated across layers: "Tokens with
the highest KV deviations on one layer are likely to have the highest KV deviations on the
next layer," confirmed with high Spearman rank correlation (§4.3).

The procedure, which the paper calls **gradual filtering** (§4.3, Figure 9):

1. At an early layer, recompute KV for all tokens. This is cheap because it is one layer.
   Compare against the reused KV and keep the r1% of tokens with the highest deviation.
2. At the next layer, recompute only those tokens, measure their deviation, and keep a
   slightly smaller r2%.
3. Continue until the target ratio is reached; from then on, recompute only the selected set.
4. At each layer, the recomputed tokens' KV overwrites the reused KV. Every other token keeps
   its reused KV.

Recomputing a token means running it through the layer with attention over the full
assembled cache, so it now sees the preceding chunks. That restores the cross-attention for
exactly the tokens that needed it.

Our own note on "early layer": by the reasoning in section 2, deviation is exactly zero at
layer 0, so the first layer where selection can see anything is layer 1 (0-indexed). Check
which index the reference code uses before copying a constant.

### Recompute ratio

"Choosing 10-20% tokens as HKVD tokens ... suffices to greatly reduce the attention
deviation and preserve generation quality" (§4.3). The systems section uses r* = 15% as the
minimum quality-preserving ratio (§5.1), and the evaluation shows quality held across a 5 to
18% range (§7, Figure 16).

### Hiding the recompute behind loading

CacheBlend assumes cached KV may live in CPU memory or on disk, so it must be loaded. It
pipelines layer i's selective recompute with layer i+1's KV load, and picks the ratio so that
recompute delay roughly equals load delay, taking the larger of that and 15% (§5.1). In that
setting the recompute is close to free. In our Transformers setup KV stays on GPU, so we will
pay the recompute cost directly. That is fine for E1 and E2, but it means our latency numbers
are not comparable to CacheBlend's pipelined numbers.

### Reported results

Mistral-7B, Yi-34B and Llama-70B on 2WikiMQA, MuSiQue, SAMSum and MultiNews (§7):

- TTFT 2.2 to 3.3x lower than full recompute.
- Throughput 2.8 to 5x higher than full recompute.
- Quality within 0.02 F1 or Rouge-L of full recompute.
- 0.1 to 0.2 higher absolute F1 than full reuse on QA tasks.

Note these are RAG passages, not agent memories, and none of them is LoCoMo.

### What CacheBlend leaves alone

Tokens not selected keep their stale KV. CacheBlend's bet is that their error is too small to
matter. AgentKVShift attacks exactly that bet.

## 5. AgentKVShift

Pandey, Kong, Hu, Zhao, Zhao, Gungor, Zhang, Rosing, *AgentKVShift: Efficient KV Cache Reuse
for Agentic Memory Systems*, arXiv 2026. [arXiv 2607.21604](https://arxiv.org/abs/2607.21604).
This is our main comparison point, because it evaluates Mistral-7B-Instruct-v0.3 on LoCoMo.

### Claim: the error is mostly one shared shift per chunk

AgentKVShift's hypothesis (§3.2) is that each token's residual splits into a part shared by
the whole chunk and a small token-specific part:

```
r_i^K = μ_K + ξ_i^K
r_i^V = μ_V + ξ_i^V
```

μ is the chunk-level **offset**, and ξ is the token-wise **fluctuation**.

The evidence is a spectral analysis of the residual matrices (§3.2, Figure 2). The leading
singular value dominates at every layer ("sharp spectral concentration"), and subtracting the
chunk mean removes most of that dominant mode, especially for keys. In plain terms: the
residuals of all tokens in a chunk mostly point in the same direction. A large fraction of the
reuse error is "captured by a single shared chunk-level offset."

Intuition (ours, not the paper's): the preceding chunks change the chunk's overall "topic
context" the same way for every token in it, and a shared bias in hidden states shows up as a
shared shift in K and V.

### The method (Algorithm 1, §3.3)

1. **Pick probes.** At an early check layer, compute fresh K for the chunk's tokens and score
   each token by divergence d_j = ||K_fresh[j] − K_reuse[j]||. Take the top b tokens as the
   probe set S. Give each token a weight w_j = min(d_j, 1).
2. **Recompute probes at every layer.** Probe tokens get fresh K and V, exactly as
   CacheBlend's selected tokens do.
3. **Estimate the offset per layer.** Average the probes' residuals:
   `μ̂[l] = mean over j in S of (K_fresh[j][l] − K_reuse[j][l])`, and the same for V.
4. **Correct everyone else.** Every non-probe token gets
   `K_corr[j][l] = K_reuse[j][l] + w_j · μ̂[l]`, and the same for V.

Both keys and values are corrected. Offsets are recomputed at every layer, but the probe set
is chosen once and fixed across layers.

### How it differs from CacheBlend

Step 2 is CacheBlend. Steps 3 and 4 are the addition. The paper puts it this way:
token-selection methods "decide which tokens to recompute and leave the rest of the cache
stale," while AgentKVShift "also corrects the keys and values of the tokens it does not
recompute, turning the refresh budget into useful signal across the entire chunk" (§3.3,
§4.3).

The extra cost is one vector mean and one add per chunk per layer, which is negligible next to
recomputing tokens.

An oddity worth noticing: the probe selection rule (highest divergence) picks the tokens
least like the average, and then uses their mean as the average. The weights partly
compensate, since high-divergence tokens get a full correction and quiet ones a smaller one.
Whether this is optimal is an open question, and an easy ablation for us (random probes vs.
top-divergence probes).

### Reported results

Models: Qwen2.5-3B-Instruct, Qwen3-4B-Instruct and Mistral-7B-Instruct-v0.3 on LoCoMo;
Qwen3-32B on AMA-Bench (§4.1). Baselines: CacheBlend, ProphetKV, full recompute. The paper
also runs on top of memory systems (A-Mem and LiCoMemory) rather than plain chunked
conversations.

LoCoMo at a 10% refresh budget (Table 1):

- With LiCoMemory, the relative F1 gap to full recompute is 1.5% (Qwen2.5-3B), 2.5%
  (Qwen3-4B) and 3.5% (Mistral-7B).
- With A-Mem on Qwen2.5-3B: AgentKVShift 0.319 F1, full recompute 0.339, CacheBlend 0.178,
  ProphetKV 0.125.

Other results:

- Baselines need about 45 to 55% recompute to match full-recompute quality; AgentKVShift
  matches it at 10% (Figure 3).
- AMA-Bench-Recall at 30%: 0.284 F1 vs. 0.296 for full recompute.
- Prefill speedup 2 to 3.5x over no reuse on one A100 at 10% (§4.4, Table 3).
- Under 2-bit KIVI quantization, about 2x the F1 of the baselines (Table 4).

Stated limitations (Appendix E): K and V residuals behave differently, so the offset model
fits keys better than values. Questions that need reasoning across several retrieved chunks
show larger gaps to full recompute.

The paper does not discuss RoPE re-rotation in its main text. We handle it the same way as
CacheBlend.

## 6. EPIC and other related work

**EPIC** (Hu et al., *EPIC: Efficient Position-Independent Caching for Serving Large Language
Models*, [arXiv 2410.15332](https://arxiv.org/abs/2410.15332)). EPIC names the setting
position-independent caching (PIC). Its diagnosis is different from both papers above: a
chunk prefilled alone treats its own first tokens as an attention sink, and that sink is wrong
once the chunk sits in the middle of a prompt. Its LegoLink algorithm recomputes a small,
fixed number of tokens at the start of each chunk (our proposal uses 16 to 32). The selection
is static, so it costs nothing to compute. EPIC reports up to 8x better TTFT and 7x higher
throughput than existing systems with little or no accuracy loss.

**LMCache** ([GitHub](https://github.com/LMCache/LMCache)) is the KV caching layer that
CacheBlend's authors built for vLLM. Its connector is our reference for writing a vLLM
connector, but its blending path is reported broken on recent vLLM (issues #4476 and #5101).

**ProphetKV** is a token-selection baseline in AgentKVShift's tables. It sits in the same
family as CacheBlend.

**KVCOMM** (NeurIPS 2025, [arXiv 2510.12872](https://arxiv.org/abs/2510.12872)) and
**KVCMAS** ([arXiv 2609.34060](https://arxiv.org/abs/2609.34060)) reuse KV across agents. They
matter only for our multi-agent stretch goal.

## 7. The four methods side by side

All four share the same skeleton: load cached chunk KV, re-rotate keys, then run a per-layer
loop that recomputes some tokens against the assembled cache. They differ in which tokens
are recomputed and what happens to the others.

| | Re-positioning only | EPIC | CacheBlend | AgentKVShift |
|---|---|---|---|---|
| Error model | Context error is negligible | Error sits at each chunk's start (attention sink) | Error is sparse: a few tokens anywhere | Error is mostly a shared per-chunk shift |
| Tokens recomputed | None | First 16 to 32 of each chunk | Top 10 to 20% by KV deviation | Top b by divergence (probes) |
| Selection cost | None | None (static) | One full early layer, then filtering | One full early layer |
| Other tokens | Stale | Stale | Stale | Shifted by w_j · μ̂ per layer |
| Paper's evidence | (baseline) | Attention-sink analysis | Deviation sparsity, cross-layer rank correlation | Spectral concentration of residuals |
| Reported on LoCoMo | No | No | As a baseline in AgentKVShift | Yes, Mistral-7B among others |

How to read these as hypotheses we can test (our framing):

- If re-positioning alone is close to full prefill, the context error is small on LoCoMo and
  the corrections don't matter much.
- If EPIC does as well as CacheBlend, the error really is positional and dynamic selection
  isn't worth its cost.
- If CacheBlend beats EPIC, the error is spread through the chunk.
- If AgentKVShift beats CacheBlend at equal budget, stale tokens carry a meaningful shared
  error, as the spectral analysis claims.

E1's plot of F1 against recompute budget, one line per method, answers all four at once.

## 8. How this maps onto our project

| Concept | Where it lives in our plan |
|---|---|
| Chunk = one memory entry, cached under a hash of its tokens | Component 1 (memory pipeline) and the chunk KV store in component 2 |
| Same retrieved set for every method | Retrieval runs once; results saved to disk (component 1) |
| RoPE re-rotation | Component 2; verified by E0 |
| Layer-0 KV is context-free | E0's second check |
| Per-layer recompute loop over the assembled cache | Component 2's custom forward pass |
| Each method = a token-selection rule (+ offset for AgentKVShift) | Plugged into that one loop |
| Quality vs. budget | E1 (LoCoMo F1, 0 to 30% budget) |
| Prefill cost vs. number of chunks | E2 |
| Re-positioning only, served by vLLM | Component 3 (connector); E3 to E5 |
| How often memories must repeat for reuse to pay off | Component 4 (Zipf workload); E4 |

Why only re-positioning goes through vLLM: vLLM's connector can hand the engine KV for a
prompt prefix, and vLLM then computes the rest. Methods 2 to 4 need to recompute tokens
inside the prefix at every layer, which the connector interface doesn't allow without changing
vLLM's scheduler. That's out of scope, so those methods stay in Transformers.

E5 ties the two worlds together: re-positioning only must give the same F1 through vLLM as
through Transformers. If it doesn't, the connector is wrong, not the method.

## 9. Open questions and pitfalls

- **Our success criterion for CacheBlend may be optimistic.** The proposal expects
  CacheBlend within 2 F1 of full prefill at 10 to 15%. AgentKVShift's own LoCoMo table shows
  CacheBlend at 0.178 vs. 0.339 for full recompute at 10% (A-Mem, Qwen2.5-3B), and says
  baselines need 45 to 55% to catch up. CacheBlend's own 0.02 claim is on RAG datasets. If
  we see a large CacheBlend gap, that agrees with prior work rather than signaling a bug.
- **Absolute vs. relative F1.** AgentKVShift reports relative gaps (3.5% for Mistral-7B);
  our criterion uses absolute F1 points. Report both.
- **Memory format.** AgentKVShift runs on A-Mem and LiCoMemory, whose memories include
  generated metadata. Our memories are plain chunks of conversation. Numbers will not line up
  exactly; say so in the writeup.
- **How was each chunk prefilled when cached?** Alone, or after the system prompt? This
  changes the residual and, for EPIC, the attention sink. Pick one, document it, use it for
  every method.
- **Layer indexing.** Deviation is zero at layer 0. Confirm the check layer each paper uses
  is 0-indexed or 1-indexed before porting.
- **Budget accounting.** Count recomputed tokens the same way for all methods. AgentKVShift's
  selection layer recomputes every token once; CacheBlend's gradual filtering does too. Decide
  whether that layer counts toward the budget.
- **K vs. V.** AgentKVShift says the offset model fits keys better than values. An ablation
  correcting only K is cheap and tells us where the gain comes from.
- **Probe choice.** Top-divergence probes vs. random probes, as noted in section 5.
- **Cross-chunk questions.** Both papers suggest multi-hop questions suffer most. LoCoMo's
  question categories let us check this directly in E1.
- **Latency comparisons.** CacheBlend's latency numbers assume pipelined loading from slower
  storage. Ours keep KV on GPU. Don't compare those numbers directly.

## 10. Glossary

| Term | Meaning |
|---|---|
| Prefill | The forward pass over the prompt that builds the KV cache |
| TTFT | Time to first token; dominated by prefill for long prompts |
| KV cache | Per-layer keys and values for every prompt token |
| Prefix caching | Reusing KV for an identical leading run of tokens (exact) |
| Non-prefix reuse / PIC | Reusing a chunk's KV at any position (approximate) |
| RoPE | Rotary position embedding; position is a rotation of Q and K |
| Re-rotation | Rotating cached keys by the position difference to move them |
| Cross-attention (CacheBlend's sense) | Attention between tokens of different chunks, missing from reused KV |
| KV deviation | Per-token size of the difference between reused and fresh KV |
| HKVD tokens | High-KV-deviation tokens; the ones CacheBlend recomputes |
| Residual | AgentKVShift's term: fresh minus reused KV |
| Offset (μ) | Shared per-chunk component of the residual |
| Fluctuation (ξ) | Token-specific remainder of the residual |
| Probe tokens | Tokens AgentKVShift recomputes to estimate the offset |
| Recompute / refresh budget | Fraction of chunk tokens recomputed per layer |
| Attention sink | Tokens at a sequence start that absorb disproportionate attention |

## 11. References

- CacheBlend: https://arxiv.org/abs/2405.16444 (HTML: https://arxiv.org/html/2405.16444)
- AgentKVShift: https://arxiv.org/abs/2607.21604 (HTML: https://arxiv.org/html/2607.21604v1)
- EPIC: https://arxiv.org/abs/2410.15332
- LMCache: https://github.com/LMCache/LMCache (issues #4476, #5101)
- KVCOMM: https://arxiv.org/abs/2510.12872
- KVCMAS: https://arxiv.org/abs/2609.34060
- LoCoMo: https://github.com/snap-research/locomo
- vLLM: https://github.com/vllm-project/vllm
