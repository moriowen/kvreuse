# Non-prefix / position-independent KV reuse: landscape (to Oct 2026) and novelty of the Group 7 proposal

Researched 2026-10-07. Verification labels used throughout:
- **[V-primary]** read this session from the arXiv abs/HTML page or the GitHub issue itself (via a fetch tool that summarises the page; numbers were quoted back, but I did not see the PDF tables rendered).
- **[V-snippet]** seen only in a search-engine snippet of the primary page; not opened.
- **[Prior]** from my own background knowledge of the paper (training data through mid-2026); not re-checked this session. Treat numbers marked [Prior] as "check before citing".

## Q1. Which methods exist, and what does each one do?

### Takeaway
The field moved fast between 2024 and 2026. There are at least 30 methods, in four families: (a) training-free selective recompute (CacheBlend, EPIC, CacheClip, ProphetKV, QCFuse, RelaxKV, KVShare, Cache-Craft, CacheTune, KVBoost, RedKnot); (b) training-free correction without full recompute (AgentKVShift, KVCOMM, KVCMAS, APE); (c) fine-tuning approaches (Block-Attention, TurboRAG, KVLink, SemPIC, the Tachibana et al. hybrid); (d) serving and infrastructure work (LMCache, MiniPIC, Irminsul, Leyline, CacheSlide, RAGCache, FusionRAG). Most of these were evaluated on RAG passages (2WikiMQA, MuSiQue, LongBench, RULER). Only AgentKVShift (and, more loosely, Leyline, CacheSlide, Irminsul, the LMCache OpenClaw blog, KVCOMM and KVCMAS) target agents.

### Cited Findings

**Training-free, selective recompute**
- **CacheBlend** (Yao et al., EuroSys 2025): recomputes "High-KV-Deviation" tokens, found by fully recomputing an early layer, then recomputes a fixed fraction (~10-15%) of tokens. — [arXiv 2405.16444](https://arxiv.org/html/2405.16444v1); summary from a 2026 paper [V-snippet]: "CacheBlend locates high-KV-deviation tokens through full first-layer recomputation… the first-layer recomputation overhead is irreducible" — [CacheTune 2605.24022](https://arxiv.org/pdf/2605.24022). [Prior]: 2.2-3.3x TTFT and 2.8-5x throughput vs full prefill; models Mistral-7B, Yi-34B, Llama-70B; datasets 2WikiMQA, MuSiQue, SAMSum, MultiNews; code ships in LMCache.
- **EPIC** (Hu et al., ICML 2025): "recomputes only the first k attention sink positions of each chunk… but offers insufficient coverage of globally semantic tokens and fails to guarantee model accuracy in many scenarios" (critique from a later paper) [V-snippet] — [EPIC 2410.15332v2](https://arxiv.org/html/2410.15332v2); critique as summarised in search results from [CacheTune](https://arxiv.org/pdf/2605.24022). [Prior]: algorithm "LegoLink", k ≤ 32 tokens per chunk; reports up to 8x TTFT and 7x throughput vs existing systems, with accuracy loss of 0-7%; implemented in a vLLM fork.
- **CacheClip** (Oct 2025): small auxiliary model guides token selection, shared prefix removes redundant attention sinks, sliding-window grouping, auxiliary model runs on CPU; "up to 3.33x in prefill time" [V-snippet] — [arXiv 2510.10129](https://arxiv.org/abs/2510.10129).
- **ProphetKV** (Feb 2026): user-query-driven selection, dual-stage recompute that fuses layer-wise attention metrics; "retains 96%-101% of full-prefill accuracy with only a 20% recomputation ratio" [V-snippet] — [arXiv 2602.02579](https://arxiv.org/abs/2602.02579). It is a baseline in AgentKVShift and RelaxKV (below).
- **QCFuse** (Jun 2026): chunk-anchor query probing plus critical-layer profiling, so it avoids inspecting every layer; "average prefill-time speedup of 1.7x over full prefill and 1.5x over ProphetKV" [V-snippet] — [arXiv 2606.05875](https://arxiv.org/abs/2606.05875).
- **RelaxKV** (27 Sep 2026): treats repair as a joint choice of which tokens to recompute *and* which cached entries they attend to (sparse context during recompute). Baselines: CacheBlend, EPIC, KVShare, ProphetKV, full recompute, full reuse. Models: Llama-3.1-8B, Qwen3-14B, Phi-4-14B, Llama-3.2-3B. Data: LongBench, RULER-MV, LV-Eval. Example: Qwen3-14B at 32K RULER-MV, 95.00 vs ProphetKV 91.17. "Code will be released upon acceptance." [V-primary] — [arXiv 2609.33503v1](https://arxiv.org/html/2609.33503v1).
- **CacheTune / "Adaptive KV Cache Reuse for Fast Long-Context LLM Serving"** (Li et al., v1 20 May 2026, v2 4 Oct 2026): uses frequency-domain analysis to pick critical KV pairs to recompute, plus sparse KV transfer, multi-stream overlap, deferred positional encoding and tiered storage. Reports 3.72-4.86x TTFT, 3.93-6.21x throughput, and 2.34-2.36x TTFT with offloaded storage. The abstract names no model, dataset, code or vLLM integration. [V-primary] — [arXiv 2605.24022](https://arxiv.org/abs/2605.24022).
- **RedKnot** (Jun 2026): head-aware KV reuse plus "SegPagedAttention"; positions itself against "existing PIC systems such as CacheBlend, EPIC, and ProphetKV" [V-snippet] — [arXiv 2606.06256](https://arxiv.org/pdf/2606.06256). Numbers not verified.
- **KVShare** (Mar 2025): multi-tenant, semantic-similarity KV sharing. Built in vLLM (KV Retriever, Cache-Aware Scheduler, DHD Selector, KV Writer). "Reduces TTFT by up to 9.39x and increases 1.2x of the throughput" vs full recompute [V-snippet] — [arXiv 2503.16525](https://arxiv.org/html/2503.16525v2).
- **Cache-Craft** (Adobe, SIGMOD 2025): manages a chunk-cache store, decides which chunks are reusable, recomputes a subset of tokens; implemented on vLLM [V-snippet for existence] — [arXiv 2502.15734](https://arxiv.org/pdf/2502.15734). [Prior]: roughly 51% less redundant compute than prefix caching and 75% less than full recompute.
- **KVBoost** (Aug 2026): fixed 128-token chunks with a dual hash key, reused "regardless of their position", plus deviation-guided recompute [V-snippet] — [arXiv 2608.21362](https://arxiv.org/pdf/2608.21362).
- **FusionRAG** (Jan 2026): offline step folds related-chunk context into each chunk's cache, then recomputes selectively online; "by recomputing fewer than 15% of the tokens… up to 70% higher normalized F1" than prior approaches, and 2.66-9.39x lower TTFT than full attention [V-primary] — [arXiv 2601.12904](https://arxiv.org/abs/2601.12904).

**Training-free, correction rather than recompute**
- **AgentKVShift** (UCSD; Pandey, Kong, Hu, Zhao, Zhao, Gungor, Zhang, Rosing): spectral decomposition of the residual into one shared memory-level offset plus token-wise fluctuation. Picks high-divergence probes at layer 1 by L2 norm, estimates a per-layer chunk mean, and adds `w·μ̂` to non-probe tokens. K and V get separate probe sets, with weights capped at min(divergence, 1). [V-primary] — [arXiv 2607.21604](https://arxiv.org/html/2607.21604). Details under Q3.
- **APE** (ICLR 2025) [Prior]: parallel encoding with a shared prefix, attention temperature and scaling. Reports ~98% / 93% of sequential-encoding quality on RAG / ICL, and up to 4.5x speedup at 128K — [arXiv 2502.05431](https://arxiv.org/abs/2502.05431).
- **KVCOMM** (NeurIPS 2025) [Prior]: multi-agent cross-context KV sharing. It estimates the KV offset for a shared segment from "anchor" examples of earlier offsets; reports >70% reuse and up to ~7.8x speedup; code at FastMAS/KVCOMM — [arXiv 2510.12872](https://arxiv.org/abs/2510.12872).
- **KVCMAS** (Jeon et al., v1 28 Sep 2026, v2 6 Oct 2026): multi-agent shared context. Represents cross-agent cache deviations as low-rank states and chains corrections along the workflow, with no separate reference prefill. Reports 2.0x TTFT vs non-shared inference and up to 3.7x lower peak GPU memory than earlier correction methods. No code link found. [V-primary] — [arXiv 2609.34060](https://arxiv.org/abs/2609.34060).

**Fine-tuning approaches**
- **Block-Attention** [Prior]: fine-tunes with a block-diagonal mask and re-encodes positions, with the last block attending to everything; reports ~98.7% TTFT reduction — [arXiv 2409.15355](https://arxiv.org/abs/2409.15355).
- **TurboRAG** [Prior]: precomputes chunk KV under an independent-attention mask, resets position IDs, fine-tunes; reports up to 9.4x TTFT — [arXiv 2410.07590](https://arxiv.org/abs/2410.07590).
- **KVLink** [Prior]: KV position re-encoding plus trainable "link tokens"; fine-tuned; reports ~96% TTFT reduction — [arXiv 2502.16002](https://arxiv.org/abs/2502.16002).
- **SemPIC** (2026): trains a LoRA "Writer" to compile per-layer document KV by behavioural distillation, and keeps the pretrained decoder unchanged as the "Reader" [V-snippet] — [arXiv 2607.28069](https://arxiv.org/html/2607.28069).
- **"Fine-Tuning a KV Cache Concatenation-Aware Model or Recomputing KV Caches? Why Not Both?"** (Tachibana, Miyashita, Deguchi; 9 Sep 2026): fine-tunes for concatenation and also recomputes a subset of tokens selectively. On RULER at 124K tokens it reports +9.7 points over baseline and 80% lower TTFT vs full attention. No code link. [V-primary] — [arXiv 2609.09768](https://arxiv.org/abs/2609.09768).
- **LinearKV** (Aug 2026): "One Cached State Suffices for Position-Independent Caching in Hybrid LLMs", i.e. PIC for hybrid (linear-attention) models [V-snippet title only] — [arXiv 2608.11231](https://arxiv.org/pdf/2608.11231).

**Serving, systems and agent-specific work**
- **RAGCache** [Prior]: a knowledge tree of document KV with a PGDSF eviction policy. It is **prefix-based** (order-sensitive), not position-independent. Reports ~4x TTFT and 2.1x throughput vs vLLM+Faiss — [arXiv 2404.12457](https://arxiv.org/abs/2404.12457).
- **MiniPIC** (IBM Research, Ordonez and Parnell, 11 Jun 2026): positional-encoding-free KV cache plus user-controlled reuse primitives (block-aligned padding, span separator, prompt depend), in "<100 lines of core engine modifications plus a custom attention backend". It implements **Block-Attention, EPIC and Prompt Cache in the same running vLLM instance** with CPU offload. On 2WikiMultihopQA: +49% prefill throughput, TTFT down "up to two orders of magnitude" for cached spans, 5.7% worst-case overhead [V-primary]. The search-result title says "Code available at https://github.com/IBM/vllm, from commit 6631ff3 onwards" [V-snippet] — [arXiv 2606.13126](https://arxiv.org/abs/2606.13126), [HTML](https://arxiv.org/html/2606.13126).
- **Irminsul** (May 2026): extends SGLang's radix cache with content-hash keys over content-defined chunks, plus a δ-rotation rule for MLA's decoupled RoPE key, targeting agentic serving [V-snippet] — [arXiv 2605.05696](https://arxiv.org/html/2605.05696).
- **Leyline** (Ma, Eitzinger, Koestler; 31 May 2026): declarative "KV cache directives" for agents that edit context (splice, remove, retry), with RoPE-rotation correction kernels. Reports +11.2 pp cache hit, up to 241 ms lower latency, +14.3 pp solve rate on debug-gym. No code or engine integration stated. [V-primary] — [arXiv 2606.01065](https://arxiv.org/abs/2606.01065).
- **CacheSlide** (FAST '26): "Relative-Position-Dependent Caching", for agent workloads where reused segments keep their relative order while their absolute positions shift [V-snippet] — [USENIX FAST'26 PDF](https://www.usenix.org/system/files/fast26-liu-yang.pdf).
- **LMCache** (paper Oct 2025): the reference vLLM KV-connector library, which includes CacheBlend — [arXiv 2510.09665](https://arxiv.org/pdf/2510.09665), [GitHub](https://github.com/lmcache/lmcache).
- Also found, not read: LazyAttention, deferred positional encoding for RAG ([2606.04302](https://arxiv.org/pdf/2606.04302)); PCR, prefetch-enhanced reuse ([2603.23049](https://arxiv.org/pdf/2603.23049)); RcLLM, beyond-prefix caching for recommendation ([2605.07443](https://arxiv.org/html/2605.07443)); CachePrune ([2605.23640](https://arxiv.org/pdf/2605.23640)); RelayCaching ([2603.13289](https://arxiv.org/html/2603.13289v1)); "Can I Buy Your KV Cache?" ([2606.13361](https://arxiv.org/pdf/2606.13361)).

**Benchmark / survey**
- **"An experimental study of KV cache reuse strategies in chunk-level caching systems"** (Cestola, Xia, Zheng, Zheng, Didona; 3 Mar 2026). Evaluates **11 CLC systems**. Recompute-based: CacheBlend, EPIC, Link0, CacheClip, DroidSpeak. Attention-reshaping: APE, SEL, TurboRAG, Block-Attention, KVLink. Models: **Llama3.1-8B, Qwen3-8B, Mistral-7B**. Data: 2WikiMQA, MuSiQue, RULER. Findings: recompute methods score 7-18% below full prefill; only 8-20% of the selected tokens stay the same across layers; base models get F1 = 0 on 42-58% of queries. It proposes the hybrid **PSR** (Link0 sink-prefix + CacheBlend ΔK recompute + APE scaling), "up to 5% higher accuracy". The authors reimplemented closed-source systems. No prefix-caching or reuse-rate analysis. [V-primary] — [arXiv 2603.20218](https://arxiv.org/html/2603.20218).

### Inferences
- The proposal's related-work section (EPIC, CacheBlend, AgentKVShift, Prompt Cache, CacheGen, LMCache, KVCOMM, KVCMAS) leaves out the closest prior work: the 2603.20218 experimental study and MiniPIC. It also leaves out the newer training-free selectors (ProphetKV, QCFuse, RelaxKV, CacheTune) and the fine-tuned family (Block-Attention, TurboRAG, KVLink, APE, SemPIC). A grader who knows the area would notice those gaps.
- ProphetKV, not EPIC, is now the standard "strong baseline" (used by AgentKVShift, RelaxKV and QCFuse). A four-way comparison without ProphetKV may look dated.

### Gaps
- Not opened this session: RedKnot, QCFuse, SemPIC, CacheClip, KVBoost, LinearKV, Irminsul full text. Their numbers above come from snippets only.
- Code status could not be confirmed for most 2026 papers. Confirmed: RelaxKV ("upon acceptance"), AgentKVShift (no link found, see Q3). MiniPIC's IBM/vllm link comes from a snippet only.
- EPIC/CacheBlend/APE/Block-Attention/TurboRAG/KVLink/Cache-Craft/RAGCache/KVCOMM numbers are [Prior] and should be checked against the PDFs before they go into the report.

## Q2. Has anyone run a like-for-like comparison on conversational agent memory, or published a break-even analysis vs prefix caching?

### Takeaway
Nobody I found compares **EPIC + CacheBlend + AgentKVShift (+ naive re-positioning)** on one model and one conversational-memory benchmark. AgentKVShift already compares **CacheBlend and ProphetKV** on LoCoMo with Mistral-7B, though, and 2603.20218 already runs a broad like-for-like (11 methods, incl. EPIC and CacheBlend, incl. Mistral-7B) on multi-hop RAG. No published throughput/TTFT-vs-repetition-rate sweep in vLLM turned up, but several papers make adjacent hit-rate or break-even arguments.

### Cited Findings
- AgentKVShift baselines are CacheBlend, ProphetKV and full recompute. EPIC and APE do not appear; LongMemEval is not used; datasets are LoCoMo and AMA-Bench [V-primary] — [arXiv 2607.21604](https://arxiv.org/html/2607.21604).
- 2603.20218 compares CacheBlend, EPIC, Link0, CacheClip, DroidSpeak, APE, SEL, TurboRAG, Block-Attention, KVLink on Llama3.1-8B, Qwen3-8B, Mistral-7B, but on 2WikiMQA/MuSiQue/RULER (RAG), not agent memory. It "does not compare against prefix caching or analyze reuse rates" [V-primary] — [arXiv 2603.20218](https://arxiv.org/html/2603.20218).
- RelaxKV compares CacheBlend, EPIC, KVShare, ProphetKV on LongBench/RULER/LV-Eval (RAG, not memory) [V-primary] — [arXiv 2609.33503](https://arxiv.org/html/2609.33503v1).
- MiniPIC runs Block-Attention, EPIC and Prompt Cache in one vLLM instance on 2WikiMultihopQA [V-primary] — [arXiv 2606.13126](https://arxiv.org/abs/2606.13126).
- LMCache blog, "Accelerating OpenClaw Agents with CacheBlend" (1 Apr 2026): an agent workload (two-turn MTRAG Cloud RAG with documents that change position). Prefix caching got 48% hit rate and 5.553→4.055 s TTFT; CacheBlend got 98% hit rate and 2.325 s TTFT ("42% latency reduction"). No model or version stated, and the setup deliberately breaks prefix alignment (worst case for prefix caching) [V-primary] — [blog.lmcache.ai](https://blog.lmcache.ai/en/2026/04/01/accelerating-openclaw-agents-with-cacheblend/).
- Break-even claims, snippet only [V-snippet]: one source says reusing a resident KV "is 9-50x cheaper in compute than prefilling on Qwen3-4B… the break-even is essentially immediate, so reuse pays off by the second read", and another says that with D=20 positions, C=100 cache entries and moderately skewed Zipf popularity, "position-agnostic caching provides a 2.86x hit-ratio advantage". The search result does not say which claim comes from which paper. Candidates: [Can I Buy Your KV Cache? 2606.13361](https://arxiv.org/pdf/2606.13361), [LazyAttention 2606.04302](https://arxiv.org/pdf/2606.04302), [PCR 2603.23049](https://arxiv.org/pdf/2603.23049).
- [Prior] Cache-Craft reports savings relative to prefix caching on production-like RAG traces, which is a comparison against prefix caching but not a sweep over repetition rate — [arXiv 2502.15734](https://arxiv.org/pdf/2502.15734).

### Inferences
- "Like-for-like on conversational memory with EPIC and AgentKVShift included" is still open, but narrowly. The defensible framing is "first independent reproduction of AgentKVShift, plus the first EPIC and re-positioning-only numbers on LoCoMo, all under one budget accounting".
- The break-even analysis is the least pre-empted contribution, but it needs to cite the Zipf hit-ratio and "pays off by the second read" results and go beyond them. Hit ratio is not latency: the proposal measures end-to-end TTFT/throughput in vLLM including load, re-rotation and correction cost, and that is the part that adds something.

### Gaps
- Did not open 2606.13361, 2606.04302 or 2603.23049 to confirm which one contains the break-even/Zipf analysis or its exact setup. Do this before claiming novelty for E4.
- Did not search LongMemEval/AMA-Bench-specific KV-reuse papers beyond AgentKVShift; none surfaced.

## Q3. AgentKVShift exact details

### Takeaway
The proposal's quoted numbers check out against the arXiv HTML: Mistral-7B, LiCoMemory, r = 0.1, full recompute 0.509, CacheBlend 0.290, AgentKVShift 0.491. EPIC is **not** a baseline. No code release was found.

### Cited Findings (all [V-primary] from [arXiv 2607.21604 HTML](https://arxiv.org/html/2607.21604))
- Authors: Nilesh Prasad Pandey, Jason Kong, Lanxiang Hu, Quanling Zhao, Yujie Zhao, Onat Gungor, Hao Zhang, Tajana Rosing (UC San Diego). v1. The fetched page gave a submission date of **15 May 2026**, which does not fit a 2607 (July 2026) arXiv ID. Treat the date as unverified.
- LoCoMo F1, recompute ratio r = 0.1 (Full / CacheBlend / ProphetKV / AgentKVShift):

| Model | Memory system | Full | CacheBlend | ProphetKV | AgentKVShift |
|---|---|---|---|---|---|
| Qwen2.5-3B-Instruct | A-Mem | 0.339 | 0.178 | 0.125 | 0.319 |
| Qwen2.5-3B-Instruct | LiCoMemory | 0.390 | 0.286 | 0.329 | 0.384 |
| Qwen3-4B-Instruct | A-Mem | 0.444 | 0.360 | 0.315 | 0.429 |
| Qwen3-4B-Instruct | LiCoMemory | 0.446 | 0.360 | 0.385 | 0.435 |
| Mistral-7B-Instruct-v0.3 | A-Mem | 0.305 | 0.208 | 0.170 | 0.282 |
| Mistral-7B-Instruct-v0.3 | LiCoMemory | 0.509 | 0.290 | 0.357 | 0.491 |

- AMA-Bench-Recall, Qwen3-32B, r = 0.3, weighted average F1: CacheBlend 0.270 (91.2% retention), ProphetKV 0.277 (93.6%), AgentKVShift 0.284 (96.0%). Accuracy: 0.240 / 0.236 / 0.279 (83.6% / 82.2% / 97.2%).
- Claims: "within 1.5-6% relative F1 on LoCoMo" at r = 0.1. Baselines need "45-55% refresh" to match AgentKVShift at "10-30% recompute". "2-3.5x over no-KV-reuse on a single A100 GPU" at r = 0.1. Hardware: A100 80GB (small models), H200 (32B).
- Under 2-bit KIVI quantisation at r = 0.1: AgentKVShift ≈ 0.203 F1, CacheBlend 0.076, ProphetKV 0.012.
- Limitations (App. E): K/V asymmetry, since the sub-Gaussian fluctuation assumption "fits keys more cleanly than values, especially in shallow transformer layers"; and cross-chunk reasoning, where "all KV reuse methods, including AgentKVShift, leave a wider gap to full recompute on cross-chunk reasoning capabilities such as Causal Inference and State Updating" (AMA-Bench).
- No GitHub link and no vLLM/SGLang integration in the HTML text (checked both halves of the page).
- Note that the Mistral **A-Mem** numbers are much lower (full 0.305), so the proposal's choice of the LiCoMemory row as its reference is the favourable one.

### Inferences
- The proposal's plain-chunk memories match neither A-Mem nor LiCoMemory. The full-prefill F1 on Mistral could land anywhere from ~0.3 to ~0.5, so "relative gap" is the right comparison, as the proposal already says.
- Success criterion (1), "within 0.02 absolute F1 at 10-15%", is tighter than what AgentKVShift itself reports for Mistral/LiCoMemory (0.018 gap) and Mistral/A-Mem (0.023 gap). Expect to miss it in some configurations.
- Without released code, method 4 is a from-paper reimplementation, and the report should call it that.

### Gaps
- The fetch did not return per-category LoCoMo results (single-hop / multi-hop / temporal / open-domain / adversarial). The page appears to have none for LoCoMo.
- Did not check whether a v2 or a code link appeared on GitHub/HF after v1.

## Q4. Is there a working open-source path for non-prefix reuse in current vLLM (0.26-0.28) or SGLang?

### Takeaway
No stable, documented, unmodified-vLLM path was found. LMCache's CacheBlend has three open issues: a broken V1 non-prefix lookup, an out-of-date example, and no MP-mode client. The vLLM generalized-reuse RFC is closed. Working code exists only as forks or engine patches (MiniPIC on IBM/vllm, Irminsul on SGLang, KVShare/Cache-Craft/EPIC forks). The proposal's "plugin-only" framing still has room, but its engineering risk is higher than it states.

### Cited Findings
- **LMCache #3238** (opened 9 May 2026, open): CacheBlend non-prefix reuse fails under the vLLM V1 connector because the scheduler's rolling hash chain only detects prefix matches. Only the ~24-token system prompt was found cached instead of ~90%. Also reports a zero-division on empty chunks, a BOS-token assumption, and a double-unpin bug. Environment: LMCache dev f35456e1, vLLM 0.18.0+rocm700, MI300X, Qwen2.5-1.5B [V-primary] — [GitHub #3238](https://github.com/LMCache/LMCache/issues/3238).
- **LMCache #4476** "cacheblend out of date" (opened 10 Aug 2026, open, no maintainer reply visible): the `blend_kv_v1` README example cannot be run on the newest vLLM (LMCache v0.5.3 path) [V-primary] — [GitHub #4476](https://github.com/LMCache/LMCache/issues/4476).
- **LMCache #5101** (opened 14 Sep 2026, open, no maintainer reply visible): in MP mode with `--engine-type blend`, `lmcache_mp_connector.py` never sends the `CB_REGISTER_ROPE_V3` handshake ("grep -c -i blend" returns 0). Reordered chunks take the same latency as cold prefill (0.709 s vs 0.728 s). Image `lmcache/vllm-openai:v0.5.4` = **vLLM 0.27.1 + LMCache 0.5.4** [V-primary] — [GitHub #5101](https://github.com/LMCache/LMCache/issues/5101). This confirms the proposal's version pairing.
- **vLLM #25672** "[Feature]: Generalized KV cache reuse" (iddo10, 25 Sep 2025, **closed**): proposed attention masks over arbitrary key subsets, per-token success bitmaps from the KV connector instead of a prefix length, and scheduler masking arrays, opt-in and FlashInfer first; mentions "compositional context" for RAG [V-primary] — [GitHub #25672](https://github.com/vllm-project/vllm/issues/25672). Whether it closed as completed, superseded or stale was not visible in the fetch.
- **MiniPIC**: needs <100 LOC of vLLM core changes plus a custom attention backend, so it is not a pure plugin. Code reportedly at IBM/vllm commit 6631ff3+ [V-primary for the LOC claim; V-snippet for the repo] — [arXiv 2606.13126](https://arxiv.org/abs/2606.13126).
- **Irminsul** is an SGLang radix-cache extension (content-hash keys plus δ-rotation for MLA) [V-snippet] — [arXiv 2605.05696](https://arxiv.org/html/2605.05696).
- The LMCache OpenClaw blog references a demo repo, but gives no versions [V-primary] — [blog](https://blog.lmcache.ai/en/2026/04/01/accelerating-openclaw-agents-with-cacheblend/).

### Inferences
- The proposal's C3 relies on the connector supplying KV for "the first N tokens". That is exactly what vLLM's connector API supports (prefix-length semantics, as #25672 describes), so serving re-rotated, concatenated chunks *as the prefix* is consistent with the API. Two consequences follow. (a) vLLM's own prefix-cache block hashing must be bypassed or disabled for those tokens, the same hash-chain problem #3238 hits. (b) Keys stored in vLLM's paged cache are post-RoPE, so re-rotation has to happen in the connector before the blocks are written.
- Because methods 2-4 need per-layer recompute inside the prefix, which is the scheduler-level change #25672 asked for, keeping them in Transformers is the right scope choice.

### Gaps
- Did not check vLLM 0.26-0.28 release notes for any merged non-prefix or "partial hit" connector feature after #25672 closed. Worth a quick search before the report.
- Did not confirm whether SGLang mainline has merged any PIC feature.

## Q5. What do these papers say about full-context baselines and when reuse fails?

### Takeaway
Reuse fails most on cross-chunk reasoning: multi-hop, causal inference, state updating. Independent re-evaluation finds bigger losses than the original papers report. Base models are also weak on multi-hop data, which inflates how good reuse looks.

### Cited Findings
- AgentKVShift: every reuse method, including its own, leaves "a wider gap… on cross-chunk reasoning capabilities such as Causal Inference and State Updating" [V-primary] — [arXiv 2607.21604](https://arxiv.org/html/2607.21604).
- 2603.20218: recompute methods score 7-18% below full prefill, and attention-reshaping methods lose a further 2-9% relative to recompute. Base models score F1 = 0 on 42-58% of queries, which "masks true technique effectiveness". Selected tokens are unstable across layers (8-20% overlap) [V-primary] — [arXiv 2603.20218](https://arxiv.org/html/2603.20218).
- EPIC critique: start-of-chunk recompute "fails to guarantee model accuracy in many scenarios" [V-snippet; critique made in a 2026 paper] — [arXiv 2605.24022](https://arxiv.org/pdf/2605.24022).
- CacheBlend's own motivation is that full reuse makes the model fail when the answer needs facts from two chunks together (CacheBlend §3.3), per the project's concept guide; [Prior] consistent — [arXiv 2405.16444](https://arxiv.org/html/2405.16444v1).
- In the OpenClaw blog, LMCache's baseline is prefix caching rather than full recompute, and it reports no quality numbers [V-primary] — [blog](https://blog.lmcache.ai/en/2026/04/01/accelerating-openclaw-agents-with-cacheblend/).

### Inferences
- LoCoMo has multi-hop and temporal categories, and the proposal already plans per-category F1. Given the evidence above, per-category F1 should be a core plot, not an ablation.
- Following 2603.20218, report the share of questions where full prefill already gets F1 = 0, and possibly compute the gap only on questions full prefill answers.

### Gaps
- No LoCoMo per-category reuse numbers were found in any paper.

## Q6. Novelty assessment of each §3.3 claim

### Takeaway
None of the four claims is fully pre-empted, but two are overstated as written. The "new setting for CacheBlend" claim is **false**: AgentKVShift already ran CacheBlend on LoCoMo with Mistral-7B. The "like-for-like comparison" claim needs to cite 2603.20218 and AgentKVShift and narrow its scope. The plugin-serving and break-even claims hold up best, but both need citations to adjacent work (MiniPIC, LMCache, the Zipf/break-even papers).

### Cited Findings and assessment
1. **"Like-for-like comparison… all four methods with one model, one retrieval output and one budget accounting."** *Partially novel.* 2603.20218 already compares 11 methods incl. EPIC and CacheBlend on Mistral-7B under one harness, on RAG ([link](https://arxiv.org/html/2603.20218)). AgentKVShift already compares CacheBlend vs ProphetKV vs itself on LoCoMo/Mistral-7B ([link](https://arxiv.org/html/2607.21604)). RelaxKV compares CacheBlend/EPIC/KVShare/ProphetKV ([link](https://arxiv.org/html/2609.33503v1)). What is new: EPIC and re-positioning-only on conversational memory alongside AgentKVShift, plus an independent reimplementation of AgentKVShift, which has no public code. **Suggested rewording:** "the first comparison of boundary, dynamic-selection and offset correction on one conversational-memory workload, and an independent reproduction of AgentKVShift." Consider adding ProphetKV as a fifth method, since it is the de facto strong baseline.
2. **"A new setting for EPIC and CacheBlend, which were evaluated on RAG passages rather than conversational memory."** *True for EPIC, false for CacheBlend.* CacheBlend on LoCoMo is in AgentKVShift's main table (0.290 vs 0.509, Mistral/LiCoMemory) ([link](https://arxiv.org/html/2607.21604)). LMCache also ran CacheBlend on an agent workload (OpenClaw/MTRAG) ([link](https://blog.lmcache.ai/en/2026/04/01/accelerating-openclaw-agents-with-cacheblend/)). The proposal's own §4.3 cites the AgentKVShift CacheBlend number, which contradicts §3.3. **Fix:** restrict the claim to EPIC, and to plain-chunk memories (no generated metadata), which neither A-Mem nor LiCoMemory represent.
3. **"Non-prefix reuse served through vLLM's public plugin interface instead of engine changes."** *Plausibly novel as engineering, with caveats.* MiniPIC needs core-engine changes ([link](https://arxiv.org/abs/2606.13126)). LMCache's blend path is the only plugin-style route and is broken or incomplete (#3238, #4476, #5101) ([#3238](https://github.com/LMCache/LMCache/issues/3238), [#4476](https://github.com/LMCache/LMCache/issues/4476), [#5101](https://github.com/LMCache/LMCache/issues/5101)). EPIC, KVShare and Cache-Craft are forks. Caveat: only method 1 is served, and method 1 is uncorrected full reuse, the configuration every paper shows losing the most quality. The claim must be paired with E1's method-1 F1, or it serves a known-lossy mode quickly. Also note that LMCache's non-MP connector may already do RoPE re-rotation for blend (not verified).
4. **"Break-even analysis of how often memories must repeat before reuse beats prefix caching, which the papers above do not report."** *Mostly novel as an end-to-end vLLM measurement, but adjacent analyses exist.* There are Zipf hit-ratio results (2.86x hit-ratio advantage for position-agnostic caching) and a compute break-even ("pays off by the second read"), both [V-snippet] ([2606.13361](https://arxiv.org/pdf/2606.13361), [2606.04302](https://arxiv.org/pdf/2606.04302)). The OpenClaw blog reports prefix-caching vs CacheBlend hit rates (48% vs 98%) at one point rather than a sweep ([link](https://blog.lmcache.ai/en/2026/04/01/accelerating-openclaw-agents-with-cacheblend/)). "The papers above do not report" is literally true of the papers the proposal cites, but a reviewer would expect these to be cited. A TTFT/throughput curve against Zipf exponent in vLLM, including load and re-rotation overhead, was not found anywhere.

### Inferences
- Overall: still a reasonable course project, but its novelty rests on the specific combination (EPIC + AgentKVShift reproduction + plain conversational memory + plugin serving + repetition sweep), not on any one piece. The writeup should say so.
- Concrete edits for the proposal/report: (a) fix the CacheBlend "new setting" claim; (b) cite 2603.20218, MiniPIC, ProphetKV, RelaxKV, FusionRAG and the fine-tuned family in related work; (c) consider adding ProphetKV as a method; (d) make per-category (multi-hop/temporal) F1 a headline result; (e) loosen or justify success criterion (1).

### Gaps
- Primary-source checks are still needed for: the source of the "2.86x" and "second read" break-even claims; MiniPIC's repo and vLLM version; whether vLLM merged anything replacing #25672; AgentKVShift's true v1 date and any later code release.
- Search was limited to roughly 20 tool calls, so recent (Sep-Oct 2026) papers on agent-memory KV reuse could have been missed. The ones found (RelaxKV, KVCMAS, the Tachibana et al. hybrid) are RAG or multi-agent, not LoCoMo-style memory.
