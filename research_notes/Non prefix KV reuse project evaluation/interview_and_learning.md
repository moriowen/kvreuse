# Turning the non-prefix KV reuse project into an inference-engineering interview story: measurements, extensions, learning path

Scope: Atharva's own parts of the project (chunk KV store, RoPE re-rotation, vLLM KV-connector plugin, C3/E3/E4/E5). Model is Mistral-7B-Instruct-v0.3 in bf16 on one A100-80GB or H100 on PACE ICE. Schedule is weeks 7-15 (Oct 5 - Dec 1, 2026), and these notes were written 2026-10-07 (week 7). Facts carry a source. Anything I computed or recommend is labelled as an inference. Project facts come from `proposal-nonprefix-kv-reuse-agent-memory.md` and `AGENTS.md` in this folder.

---

## Q1. What do inference-engineering interviews and job descriptions emphasize, and how does this project map onto them?

### Takeaway
Postings and prep guides from 2025-2026 keep coming back to the same short list: KV-cache management (paging, prefix caching, offload), why prefill is compute-bound and decode is memory-bound, continuous batching, speculative decoding, quantization, parallelism, and hands-on vLLM/SGLang/TensorRT-LLM experience. This project gives first-hand, measured answers on KV-cache management, prefix caching, and prefill/decode economics, plus real vLLM plugin code. It says nothing about speculative decoding, quantization, or multi-GPU parallelism, so those need separate prep.

### Cited Findings
- An inference-infra interview prep guide (2026) lists the core topics as transformer forward-pass mechanics, KV-cache mechanics and memory cost, and "Prefill vs decode phase: very different compute profiles". It lists the major optimizations as continuous batching, paged attention, speculative decoding, quantization (FP8/INT8/INT4), and tensor/pipeline parallelism. — [techinterview.org, AI Inference Infrastructure Interview Prep 2026](https://www.techinterview.org/post/3233475331/ai-inference-infrastructure-interview-prep-2026-vllm-triton/)
- The same guide quotes these sample questions: "Walk me through how a transformer forward pass actually executes on a GPU"; "Why is decode memory-bound and prefill compute-bound?"; "How would you serve a 70B model on 4 A100 80GB?"; "Explain paged attention and why it works"; "Walk me through speculative decoding and its limits". Its system-design prompts include "Design an inference platform that serves 100K req/s of an 8B model", long-context (200K) serving, and multi-tenant serving with priority queues and rate limits. — [techinterview.org](https://www.techinterview.org/post/3233475331/ai-inference-infrastructure-interview-prep-2026-vllm-triton/)
- Companies that guide names as hiring: OpenAI, Anthropic, Google DeepMind, Meta, Together AI, Fireworks AI, Anyscale, Modal, AWS Bedrock, Azure OpenAI, GCP Vertex. Frameworks to know: vLLM, SGLang, TensorRT-LLM, TGI, llama.cpp, MLX. — [techinterview.org](https://www.techinterview.org/post/3233475331/ai-inference-infrastructure-interview-prep-2026-vllm-triton/)
- A common KV-cache question is "How do you handle the large memory requirements of KV cache in LLM inference?" The expected points are PagedAttention (fixed-size blocks, less fragmentation, larger batches), GQA/MQA (fewer KV heads), KV quantization, and offloading cold KV to CPU RAM. — [dataford.io, managing KV cache for LLMs](https://dataford.io/questions/managing-kv-cache-for-llms) (via search summary; page not fetched in full)
- Job postings collected through search:
  - Inferact, the company built around vLLM, has a "Member of Technical Staff, Inference" role ($200K-$400K). It asks for experience with vLLM/TensorRT-LLM/SGLang and prefers "deep understanding of KV-cache memory management, prefix caching, and hybrid model serving." — [search summary of Inferact posting](https://zerogtalent.com/ai-jobs/inferact/member-of-technical-staff-inference-9470565b-c62d-4de9-8b87-26d525ecec49) (direct fetch of startup.jobs mirror returned HTTP 403, so this is the search-engine summary)
  - An NVIDIA "Senior Software Engineer - AI Inference" role involves contributing upstream to vLLM and SGLang and building "KV-cache efficiency through paging and sharding". — [search summary](https://seedtable.com/jobs/14880)
  - Lightbits has a "Senior Inference Systems Engineer – KV Cache Optimization" role covering KV offloading, streaming, compression and scheduling, and optimizing vLLM/SGLang paged attention and prefix caching. — [Lightbits posting](https://www.lightbitslabs.com/position/senior-inference-systems-engineer-kv-cache-optimization/)
  - A "System Software Engineer, LLM Inference" role asks for "deep understanding of KV-cache management algorithms including PagedAttention, prefix caching, and speculative decoding." — [search summary](https://jobright.ai/jobs/info/6a55ba482ce8bf79a13a00da)
- vLLM's documented KV-connector family includes ExampleConnector/SharedStorage, LMCacheConnectorV1, NixlConnector, MooncakeConnector, OffloadingConnector, MultiConnector and FlexKVConnectorV1, all configured through `--kv-transfer-config '{"kv_connector":..., "kv_role":"kv_both"}'`. Industry KV offload and disaggregation work therefore runs through the same interface this project uses. — [vLLM docs, disaggregated prefill](https://docs.vllm.ai/en/latest/features/disagg_prefill.html)

### Inferences
- **Mapping of common questions to project evidence** (my synthesis):

| Interview question | What the project lets the user say from first-hand data |
|---|---|
| Why is prefill compute-bound and decode memory-bound? | The roofline numbers in Q2 for this exact model and GPU, plus a measured prefill-time-vs-tokens curve from E2. |
| How big is the KV cache, and how do you manage it? | 128 KiB/token for Mistral-7B with GQA, about 440K tokens fit on one 80 GB GPU. The chunk store and its hashing, hash collisions, and host vs GPU tiers. |
| Explain PagedAttention / prefix caching, and when it fails | Prefix caching is exact but only hits on identical leading tokens, and reordered retrievals miss. That is the motivating failure here, measured in E4. |
| Design an LLM serving system | The scheduler/worker split of the vLLM connector (Q3), an external KV tier, cache-hit-aware admission, and TTFT vs throughput tradeoffs. |
| When is loading KV better than recomputing it? | The break-even arithmetic in Q2, checked against E3 numbers. Few candidates can answer this with their own data. |
| How do you benchmark/profile an inference system? | The methodology in Q3, ideally with an nsys trace of the connector load path. |
| Speculative decoding, quantization, TP/PP | Not covered. Prepare separately. Don't stretch the project to fit. |

- The strongest interview signal is probably the tradeoff (approximate reuse loses some quality, and you can quantify how much) together with the break-even (how often memories must repeat). Interviewers at serving companies weigh decisions under cost/quality constraints, and the proposal's E1 and E4 are built for exactly that.
- Proposal method 1 (re-positioning only) is the only one served through vLLM, and it is the least accurate. The story has to own that honestly: "the systems path is upper-bound speed; the quality cost is measured in Transformers."

### Gaps
- I found no first-hand interview reports from Glassdoor, Blind or Reddit specific to inference teams at Anyscale, Fireworks, Baseten, Modal, OpenAI or Anthropic. Those sites are login-walled or didn't come up in search. The question list above comes from a prep guide and job postings, not verified candidate reports.
- I could not retrieve Together AI's individual posting text: the Greenhouse board page lists titles only.

---

## Q2. Concrete numbers: roofline, KV memory, and the load-vs-recompute break-even (with arithmetic)

### Takeaway
For Mistral-7B, prefill costs about 14 GFLOP per token, and its KV costs 128 KiB per token. That is roughly 106,000 FLOPs of recompute avoided per byte loaded. Loading cached KV over PCIe therefore beats recomputing it by about 10-20x per token on both A100 and H100, and any link faster than about 1-2 GB/s (A100) or 4-6 GB/s (H100) wins in steady state. A copy from HBM is effectively free. Once the KV is loaded, TTFT is bounded below by one forward pass over the question tokens, which is memory-bound at about 7 ms (A100) or 4 ms (H100).

### Cited Findings
- **Mistral-7B architecture:** dim 4096, 32 layers, head_dim 128, 32 query heads, 8 KV heads (GQA), FFN hidden 14336. Vocabulary is 32000 in v0.1 and 32768 in v0.3. — [Mistral 7B paper, arXiv:2310.06825](https://arxiv.org/abs/2310.06825) (table values from the paper; v0.3 vocab from the model card, not fetched this session)
- **kipply's per-token KV size:** `2·2·n_layers·n_heads·d_head` bytes for K and V in 16-bit precision. A decode step costs about `2·P` FLOPs because "we matmul through all the parameters". On A100, "312 TFLOPS ÷ 1.5 TB/s = 208", so below about 208 tokens per batch you are memory-bandwidth-bound and above it you are FLOPs-bound. Non-matmul ops (softmax, layernorm, elementwise) were about 43% of time in smaller models. — [kipply, Transformer Inference Arithmetic](https://kipp.ly/transformer-inference-arithmetic/)
- **A100 80GB:** 312 TFLOPS dense BF16 (624 with sparsity). HBM2e bandwidth is 1,935 GB/s (PCIe) or 2,039 GB/s (SXM). PCIe Gen4 is listed as 64 GB/s and NVLink as 600 GB/s. — [NVIDIA A100 page](https://www.nvidia.com/en-us/data-center/a100/)
- **H100 SXM:** BF16 is listed as "1,979 teraFLOPS" *with sparsity*, so dense is about 989 TFLOPS. HBM3 is 3.35 TB/s, NVLink 900 GB/s, and PCIe Gen5 is listed as 128 GB/s. The H100 NVL variant has 3.9 TB/s and 94 GB. — [NVIDIA H100 page](https://www.nvidia.com/en-us/data-center/h100/)
- **Real PCIe Gen4 host-to-device bandwidth:** the NVIDIA dev-forum consensus is that about 25 GB/s unidirectional H2D is expected for an A100 on PCIe4. Badly configured systems can measure far lower (one report saw 8.4 GB/s with pinned memory) depending on NUMA, slot and BIOS. — [NVIDIA forums, A100 simpleMultiCopy](https://forums.developer.nvidia.com/t/a100-simplemulticopy/303902), [PCIe H2D slow thread](https://forums.developer.nvidia.com/t/pcie-bandwidth-issue-h2d-very-slow-gen1-but-d2h-reaches-gen4/345210)
- **LMCache paper's crossover:** at 32 Gbps network bandwidth, LMCache's KV loading beats prefill only above about 256K context tokens. At 64-128 Gbps it beats prefill at all lengths, so loading should be adaptive. — [LMCache paper, arXiv:2510.09665](https://arxiv.org/pdf/2510.09665) (via search summary; the paper's model and GPU for this figure were not verified)
- **Break-even as a reuse window:** when caching cost is set equal to recompute cost, token count cancels, so the break-even is a reuse-time window rather than a token count. — [HackerNoon, "Prefill is the tax you keep paying twice"](https://hackernoon.com/prefill-is-the-tax-you-keep-paying-twice?source=rss) (secondary blog; treat as an opinion/framing)
- The proposal's own figures are 128 KB/token KV, about 15 GB of bf16 weights, and about 20 GB of KV for all ten LoCoMo conversations. — `proposal-nonprefix-kv-reuse-agent-memory.md` §4.1

### Inferences (my arithmetic; inputs cited above)

**Parameter and FLOP count (Mistral-7B v0.3):**
- Attention weights per layer: Wq + Wo = 2·4096·4096 = 33.6M, and Wk + Wv = 2·4096·(8·128) = 8.4M, for 41.9M total.
- MLP per layer (SwiGLU, 3 matrices): 3·4096·14336 = 176.2M.
- That gives 218.1M per layer, × 32 layers = 6.98B non-embedding parameters. Adding the input embedding and LM head (2·32768·4096 = 0.27B) gives about 7.25B total.
- **Linear-layer prefill FLOPs ≈ 2 · 6.98e9 ≈ 14.0 GFLOP per token.** The LM head is only needed for the last position in prefill.
- **Attention-score FLOPs**, causal, QKᵀ plus AV, with GQA not reducing the query-side math: about 2·L·d·N² = 2·32·4096·N².

| Prompt N | Linear TFLOP | Attention TFLOP | Attention share |
|---|---|---|---|
| 512 | 7.15 | 0.07 | 1% |
| 2,048 | 28.6 | 1.10 | 4% |
| 4,096 | 57.2 | 4.40 | 7% |
| 16,384 | 228.7 | 70.4 | 24% |

  At LoCoMo-style prompts of k=6 chunks × about 300 tokens plus the system prompt (about 2K tokens), prefill is essentially `2·P·N`. Attention only matters at 16K and above.

**KV memory:**
- Per token: 32 layers × 8 KV heads × 128 dims × 2 (K and V) × 2 bytes = **131,072 B = 128 KiB**.
- For comparison, a 7B multi-head-attention model with 32 KV heads (Llama-2-7B shape) would be 4× larger at 512 KiB. GQA is what makes loading cheap here.
- A 300-token chunk is about 39 MB of KV. k=6 chunks is about 236 MB. LoCoMo's 10 conversations × 16K tokens is about 21 GB, which matches the proposal.
- GPU KV capacity: 80 GB × 0.9 (vLLM's default `gpu_memory_utilization`, from my knowledge and not fetched) leaves about 72 GB. Subtracting about 14.5 GB of weights leaves about 57.5 GB, or **about 440K tokens**. Activations and CUDA-graph memory reduce that a bit.

**Roofline:**
- Ridge point (peak FLOPs ÷ HBM bandwidth): A100 SXM is 312e12 / 2.039e12 ≈ **153 FLOP/byte**, and H100 SXM is 989e12 / 3.35e12 ≈ **295 FLOP/byte**. kipply's 208 uses the older 1.5 TB/s A100-40GB figure.
- The arithmetic intensity of one forward pass over N tokens is about N FLOP/byte, because weights are read once (2 bytes per parameter) and each parameter does 2N FLOPs. So **a forward pass over fewer than about 150 tokens (A100) or 300 tokens (H100) is memory-bound**.
- Its floor is the weight-read time: 13.96 GB / 2.04 TB/s ≈ **6.8 ms on A100**, and 13.96 GB / 3.35 TB/s ≈ **4.2 ms on H100**.
- Consequences:
  - Decode at batch 1 is about 7 ms per token on A100, or at most about 145 tok/s for a single stream.
  - Decode stays memory-bound until the batch reaches roughly 150-300 concurrent sequences, and KV reads add to the bytes moved.
  - **With reuse, TTFT cannot go below about one weight read**, because vLLM still prefills the question (about 20-100 tokens), and that pass is memory-bound.

**Full-prefill TTFT estimate for a 2,000-token prompt at 50% MFU:**
- A100: (14.0e9·2000 + 2·32·4096·2000²) / (0.5·312e12) ≈ **186 ms**.
- H100: about **59 ms** at 50% of 989 TFLOPS.
- These are estimates. Real MFU for 7B prefill at 2K tokens is unmeasured here. E2 and E3 should measure it, and achieved MFU = measured FLOPs/s ÷ peak is a good number to report.

**Load vs recompute, per token:**

| GPU, prefill MFU | Recompute time/token (14.0 GFLOP ÷ achieved FLOP/s) | H2D load time/token (128 KiB ÷ link) | HBM→HBM copy (read+write 256 KiB) | Recompute ÷ load | Break-even link bandwidth |
|---|---|---|---|---|---|
| A100, 40% | 112 µs | 5.2 µs @ 25 GB/s | 0.13 µs | 21× | 1.2 GB/s |
| A100, 60% | 75 µs | 5.2 µs | 0.13 µs | 14× | 1.8 GB/s |
| H100, 40% | 35 µs | 2.6 µs @ ~50 GB/s (assumed practical Gen5) | 0.08 µs | 13× | 3.7 GB/s |
| H100, 60% | 24 µs | 2.6 µs | 0.08 µs | 9× | 5.6 GB/s |

- **General formula:** break-even bandwidth = `KV_bytes_per_token × achieved_FLOP/s ÷ FLOPs_per_token` = achieved FLOP/s ÷ 106,496 for this model. It does not depend on N (if you ignore attention's N² term, which only makes recompute worse).
- **Check against LMCache's figure:** LMCache's crossover at 32 Gbps (4 GB/s) sits between my A100 (1.2-1.8 GB/s) and H100 (3.7-5.6 GB/s) break-evens. If their setup was an H100-class GPU, the numbers are consistent. If it was an A100, their result is more pessimistic than this ideal model, which would point to per-transfer overheads, serialization and the inability to overlap at short lengths. **This is unresolved.** The model and GPU behind LMCache's figure need checking before citing it as agreement.
- **Implications for E3:** at k=6 (about 1,800 reused tokens) on an A100, the components are:

| Component | Time |
|---|---|
| Recompute the reused tokens | about 135-200 ms |
| Load their KV from host pinned memory at 25 GB/s | about 9.4 ms |
| Load if already in GPU HBM | about 0.24 ms |
| Prefill of the question | about 7 ms floor |

  An idealized TTFT is therefore about 10-20 ms against about 190 ms, an upper bound of roughly 10-15×. The measured gap will be smaller because of the system prompt, Python/connector overhead per layer, scheduler steps, and tokenization/HTTP. **Predict first, then measure, then explain the residual. That residual analysis is the interview story.**
- **RoPE re-rotation cost:** each cached key gets one more rotation, a few FLOPs per element over 32·8·128 = 32,768 key elements per token, and it is memory-bound (read and write K = 64 KiB/token). It is negligible next to recompute. **Rotation is never the bottleneck. Kernel-launch count is**, if the user implements it as many small PyTorch ops per chunk per layer (see Q4).

### Gaps
- I did not measure realistic prefill MFU for Mistral-7B on A100/H100 in vLLM. The 40-60% band is an assumption, and E2/E3 should replace it with measured values.
- The practical H2D bandwidth on PCIe Gen5 H100 hosts (I assumed about 50 GB/s) is unverified. Measure it on PACE with `nvidia-smi topo -m` plus a pinned-memory `torch` copy benchmark or CUDA's `bandwidthTest`.
- Which GPUs the course allocation can actually get is unverified. A search found ICE listing RTX6000/V100/A40/A100/MI210, with H100/H200 in the separate AI Makerspace, per [CoC GPU expansion proposal](https://support.cc.gatech.edu/sites/default/files/tech_fee_props/CoC%20-%20GPU%20Computing%20Expansion%20fore%20PACE-ICE.pdf) and the [GT AI Makerspace page](https://www.coe.gatech.edu/academics/ai-for-engineering/ai-makerspace). Whether "H100 on ICE" is real for this course should be confirmed. If only an A100 is available, use the A100 rows.

---

## Q3. Which measurements and analyses make the project credible: TTFT vs throughput, prefill savings to serving throughput, profiling, and honest reporting

### Takeaway
Credible serving claims need four things:
1. TTFT/TPOT percentiles under an open-loop arrival rate, not just single-request latency.
2. An explanation of how prefill saved turns into throughput through chunked prefill and prefill/decode interference.
3. A profiler trace showing where the connector's time goes.
4. Statistics that reveal variance, meaning warmups, enough repetitions, medians with an interval, and reporting of cold-cache costs.

### Cited Findings
- **Sarathi-Serve:** prefill saturates compute while decode underuses it, and batching them together creates a throughput-latency conflict ("generation stalls"). Chunked prefills split prefills into balanced chunks, and stall-free scheduling adds new requests without pausing ongoing decodes. It reports 2.6× higher serving capacity for Mistral-7B on a single A100, and up to 3.7× for Yi-34B on 2 A100s, compared to vLLM. — [Sarathi-Serve, arXiv:2403.02310](https://arxiv.org/abs/2403.02310)
- **Interface the user's connector implements**, from the vLLM `KVConnectorBase_V1` docstrings:
  - Scheduler side, which "binds metadata": `get_num_new_matched_tokens(request, num_computed_tokens)` ("number of new tokens that can be loaded from the external KV cache beyond the num_computed_tokens"; it may be called several times and should be side-effect free), `update_state_after_alloc(request, blocks, num_external_tokens)`, and `build_connector_meta(scheduler_output)` ("should NOT modify fields in the scheduler_output").
  - Worker side: `start_load_kv(forward_context)` ("Start loading the KV cache from the connector to vLLM's paged KV buffer… asynchronous loads may start after it is submitted"), `wait_for_layer_load(layer_name)` ("called from within attention layer to ensure async copying… is complete"), `save_kv_layer(...)`, and `wait_for_save()`.
  - Source: [vLLM API docs, kv_connector/v1/base](https://docs.vllm.ai/api/vllm/distributed/kv_transfer/kv_connector/v1/base.html). The docs are versioned, so check them against the pinned 0.27.1 tree.
- **Horace He's three regimes** are compute, memory bandwidth and overhead (Python, dispatch, kernel launch). To diagnose which one you are in, measure achieved FLOPs as a percentage of peak. Operator fusion removes memory round trips, and a fused `x.cos().cos()` takes about the same time as a single `x.cos()`. — [Horace He, Making Deep Learning Go Brrrr](https://horace.io/brrr_intro.html)
- **Stas Bekman's definitions:** TTFT is the time from submit to the first token. TPOT ("Time Per Output Token") is a per-user metric. His benchmarking advice is to run benchmarks "multiple times to get realistic numbers, as the variance between single runs can be quite large", and he notes that client implementation changes results (aiohttp scaled better than the OpenAI client). — [Stas Bekman, ML Engineering: inference](https://github.com/stas00/ml-engineering/tree/master/inference)
- The proposal's current method is "one warm-up run and the median of three repetitions", greedy decoding, and the config stored with every result. — proposal §5

### Inferences (recommendations; mark as mine)

**How prefill savings translate into throughput:**
- Think of throughput as how many token-slots the scheduler has per step. In vLLM's V1 scheduler with chunked prefill (default on in recent versions, from my knowledge and not verified for 0.27.1), each engine step has a token budget (`max_num_batched_tokens`). Decode tokens are scheduled first, and prefill chunks fill the rest.
- A connector hit lowers each request's prefill tokens from about P to about q, the question length. That frees budget for more decodes or new admissions.
- Simple model: GPU-seconds per request ≈ `P_computed / prefill_rate + output_len × (step_time_at_batch_B / B)`. Reuse shrinks only the first term.
- For LoCoMo-style QA (prompt about 2K tokens, answer about 10-50 tokens), prefill dominates, so request throughput should rise nearly in proportion to prefill removed. For long-generation workloads the gain shrinks.
- **Pick the E4 output length on purpose and report it.** Run one short-output and one long-output setting to show the Amdahl effect.
- Interference story to tell: without chunked prefill, a big prefill stalls everyone's decode (TPOT spikes), so removing prefill also improves **TPOT p99 for other users**. Measuring TPOT/ITL p99 alongside TTFT in E4 makes that visible. It is a strong interview point because it shows the user understands multi-tenant effects, not just single-request speed.

**Measurement plan additions, in priority order:**
1. **Open-loop serving runs.** Use Poisson arrivals at several request rates, from my knowledge of `vllm bench serve --request-rate`. Plot **TTFT p50/p99 vs achieved throughput**, a latency-throughput curve, for full prefill, prefix caching, and the connector. Closed-loop, one-at-a-time runs hide queueing.
2. **Cold vs warm store.** Report the first-retrieval cost of prefilling a chunk and writing it to the store separately from the hit cost. E4's Zipf exponent sets the hit rate, so plot gain against the measured hit rate, not just against the exponent.
3. **Predicted vs measured.** Put the Q2 roofline estimate next to each measured TTFT. Explaining the gap with profiler evidence is the most "inference-engineer" artifact the project can produce.
4. **Profiling:**
   - Use `torch.profiler` with CUDA activities for op-level breakdown in the Transformers path (E2).
   - Use **Nsight Systems** for the vLLM path: `nsys profile -t cuda,nvtx,osrt --cuda-graph-trace=node python ...`. Add NVTX ranges around `start_load_kv`, the re-rotation, and `wait_for_layer_load` to see whether the H2D copies (`cudaMemcpyAsync` HtoD) overlap with compute or serialize. The flag names come from my knowledge of nsys and should be checked against `nsys profile --help` on PACE.
   - Count kernels per request. Launch overhead (Horace's third regime) is the likely surprise.
5. **Timing hygiene:**
   - Use CUDA events or `torch.cuda.synchronize()` around GPU timing.
   - Discard warmup iterations, which absorb cuBLAS autotune, CUDA-graph capture, `torch.compile`, and allocator growth.
   - Fix GPU clocks if allowed, and record the GPU SKU, driver, and `nvidia-smi` clocks.
   - Pin vLLM, CUDA and PyTorch versions. The proposal already pins vLLM 0.27.1.
6. **Statistics:**
   - "Median of 3" is fine for expensive E1 F1 runs, which are deterministic under greedy decoding.
   - For latency microbenchmarks, which are cheap, use 20 or more repetitions and report the median plus IQR or a bootstrap 95% CI.
   - For serving runs, report percentiles (p50/p90/p99) over hundreds of requests and repeat each configuration 3 or more times with different seeds.
   - For F1 differences between methods, report a paired bootstrap CI over questions (1,986 LoCoMo questions), because 0.02 F1 differences may not be significant.

### Gaps
- I didn't verify the exact `vllm bench serve` flags or chunked-prefill defaults for vLLM 0.27.1. Check them in the pinned tree.
- I found no published measurement of per-layer connector overhead in vLLM V1. The user will have to measure it, which is itself a contribution.

---

## Q4. Which extensions add the most interview value per hour? (after implementing the proposal as written)

### Takeaway
**Recommendation, not sourced fact.** The best value per hour comes from analysis and profiling that build on what must exist anyway: the analytical break-even model checked against E3/E4, an nsys trace, and layer-wise async loading. System-prompt-conditioned precompute is cheap and testable with the E1 harness. A fused Triton kernel is the most "impressive-sounding" option but the most expensive for a beginner, and the arithmetic says the copy itself isn't the bottleneck. Justify the kernel only if profiling shows launch overhead dominates.

### Cited Findings
- The connector API is designed for per-layer async loading. `start_load_kv` may be asynchronous, and `wait_for_layer_load` blocks inside each attention layer. — [vLLM KVConnectorBase_V1 docs](https://docs.vllm.ai/api/vllm/distributed/kv_transfer/kv_connector/v1/base.html)
- Fusion pays off when an op is memory- or overhead-bound, not compute-bound. — [Horace He](https://horace.io/brrr_intro.html)
- LMCache argues that loading should adapt to bandwidth and context length, which favors an explicit cost model. — [LMCache paper, arXiv:2510.09665](https://arxiv.org/pdf/2510.09665)
- EPIC attributes much of the reuse error to chunk-start tokens acting as attention sinks when a chunk is cached alone. — proposal §2 and concept-guide, citing [EPIC, arXiv:2410.15332](https://arxiv.org/abs/2410.15332) and [StreamingLLM, arXiv:2309.17453](https://arxiv.org/abs/2309.17453)
- Methods 2-4 would need scheduler changes to run inside vLLM and are out of scope. Upstream source edits are forbidden. — `AGENTS.md`, proposal §3.2

### Inferences (ranked; hour estimates are my guesses for a beginner)

| Rank | Extension | Est. hours | Interview value | Why |
|---|---|---|---|---|
| 1 | **Analytical cost model checked against data.** TTFT ≈ max(load, compute overlap) + question prefill + overhead, and break-even Zipf hit rate vs prefix caching. | 4-8 | Very high | It directly answers "load vs recompute" and "when does caching pay". It reuses the Q2 math and E3/E4 data, and it is part of the proposal's novelty claim, so it is not scope creep. |
| 2 | **nsys timeline of the connector path, with NVTX ranges.** | 4-6 | High | It is concrete proof of profiling skill, and it produces one figure for the report and one for interviews ("here's where the 12 ms went"). |
| 3 | **Layer-wise async load.** Start H2D for layer i+1 while layer i computes, then measure overlap. | 6-12 | High | It is classic pipelining and the exact pattern LMCache/NIXL use. Do it only if the prototype is synchronous and the trace shows exposed load time. With KV kept in GPU HBM, load is about 0.24 ms and the gain is small. It matters for the host-RAM tier. |
| 4 | **System-prompt-conditioned chunk precompute.** Prefill `[system prompt; chunk]` once and cache only the chunk's KV, so the chunk was computed with a realistic sink and preceding context instead of standing alone. | 3-6 | Medium-high | It is a cheap store-population change that is servable through the connector with no scheduler change. It is measured with the existing E1 harness as a fifth configuration, and it may close part of method 1's quality gap. It is a hypothesis: no paper in the proposal reports this exact variant on LoCoMo, and the proposal forbids overclaiming novelty. |
| 5 | **Static offset correction in the connector.** Apply a precomputed per-chunk or per-layer mean offset (an AgentKVShift-style estimate made at store time, not per-request probes) during load. | 10-20 | Medium | It would make a correction method servable in vLLM, but the offset depends on the preceding context, so a static estimate is an approximation that needs its own quality study. It is risky in the remaining weeks. |
| 6 | **Fused Triton kernel for gather + re-rotate + scatter into paged KV blocks.** | 15-30 (beginner) | High for kernel/NVIDIA/Fireworks-style roles, but only if it is justified | Q2 shows the bytes moved are trivial, about 0.13 µs/token, so a kernel can only win by removing launch overhead, roughly k chunks × 32 layers × several ops per request. Measure the kernel count and time first. If overhead is under about 10% of TTFT, a kernel is resume decoration. If it is large, the kernel is well-motivated and makes an excellent story ("profiled, found launch-bound, fused, X→Y ms"). |

- Order of operations: finish C3/E5 correctness, then #2 (profile), then #1 (model), then pick #3, #4 or #6 depending on what the trace shows. The deciding question is "what does the trace say is slow?". Interviewers reward that sequence (measure, hypothesize, change, re-measure) more than the extension itself.
- Don't start anything past #2 before E3/E4 data exist (week 13). The proposal cuts E4 first if the core slips.

### Gaps
- I didn't check whether vLLM 0.27.1's paged KV layout (block size, K/V tensor layout per attention backend) makes the scatter non-trivial. That determines how hard #6 actually is.
- There are no published numbers for #4 or #5 on LoCoMo that I could find. They are open questions.

---

## Q5. Learning path in dependency order, with resources, mapped to weeks 7-15

### Takeaway
Learn in the order the project needs things:
1. Attention, the KV cache, and inference arithmetic (week 7).
2. RoPE and GQA, plus Hugging Face cache internals (week 8, when the KV store and re-rotation are built).
3. Profiling basics (week 9).
4. PagedAttention, prefix caching and the connector API (weeks 10-11).
5. Continuous batching and chunked prefill (weeks 11-12).
6. Disaggregation, FlashAttention and Triton last, and only as needed.

### Cited Findings (resources)
- **Inference arithmetic:** [kipply, Transformer Inference Arithmetic](https://kipp.ly/transformer-inference-arithmetic/) covers KV size, 2·P FLOPs and the 208 ridge. Add [Horace He, Making Deep Learning Go Brrrr](https://horace.io/brrr_intro.html) for the compute/bandwidth/overhead regimes and fusion.
- **Practitioner handbooks:** [Stas Bekman, ML Engineering: inference chapter](https://github.com/stas00/ml-engineering/tree/master/inference) covers prefill/decode, KV memory, batching, TTFT/TPOT and benchmarking. The [LLM Inference Handbook (BentoML, now Modular)](https://bentoml.com/llm) covers prefill/decode, PagedAttention, speculative decoding and PD disaggregation, with interactive visualizers; it has moved to [handbook.modular.com](https://handbook.modular.com/).
- **GPU MODE lectures:**
  - Lecture 8, CUDA Performance Checklist
  - Lecture 12, Flash Attention
  - Lecture 14, A Practitioner's Guide to Triton
  - Lecture 16, Hands-on profiling
  - Lecture 22, Speculative decoding in vLLM
  - Lecture 35, SGLang performance optimization
  - Lecture 40, FlashInfer
  - Source: [GPU MODE lectures repo](https://github.com/gpu-mode/lectures)
- **Chunked prefill / interference:** [Sarathi-Serve, arXiv:2403.02310](https://arxiv.org/abs/2403.02310).
- **Connector API:** [vLLM KVConnectorBase_V1 docs](https://docs.vllm.ai/api/vllm/distributed/kv_transfer/kv_connector/v1/base.html) and [vLLM disaggregated prefill docs](https://docs.vllm.ai/en/latest/features/disagg_prefill.html).
- **KV offload / load layer:** [LMCache paper, arXiv:2510.09665](https://arxiv.org/pdf/2510.09665).
- Papers already in the proposal bibliography with arXiv IDs: [PagedAttention 2309.06180](https://arxiv.org/abs/2309.06180), [RoFormer 2104.09864](https://arxiv.org/abs/2104.09864), [SGLang/RadixAttention 2312.07104](https://arxiv.org/abs/2312.07104), [StreamingLLM 2309.17453](https://arxiv.org/abs/2309.17453), [Mistral 7B 2310.06825](https://arxiv.org/abs/2310.06825).

### Inferences: week-by-week plan (my recommendation; resources marked † were not fetched this session, but the URLs are well known)

| Week | Project work (Atharva) | Concepts, in dependency order | 1-2 resources |
|---|---|---|---|
| 7 (Oct 5-9) | PACE setup, baseline F1 harness | Self-attention, the KV cache, prefill vs decode, FLOPs/bytes per token, roofline | kipply's Inference Arithmetic, then Horace He's brrr. Re-derive the Q2 tables by hand. Optional: Jay Alammar, *The Illustrated GPT-2*† (jalammar.github.io/illustrated-gpt2) if attention itself is shaky. |
| 8 (Oct 12-16) | Chunk KV store, RoPE re-rotation | RoPE (the rotation composes, so re-rotating by Δ is exact for keys). GQA (8 KV heads shared by 32 query heads, which is why KV is 128 KiB). The HF `past_key_values`/`DynamicCache` layout, and whether HF caches keys post-RoPE. | RoFormer §3 plus EleutherAI's *Rotary Embeddings* blog† (blog.eleuther.ai/rotary-embeddings). GQA paper† (arXiv 2305.13245). |
| 9 (Oct 19-23) | E0 tests, methods 1-2 | Attention sinks (why chunk-start tokens matter, for EPIC). Basic GPU timing and `torch.profiler`. | StreamingLLM paper, GPU MODE Lecture 16 (profiling). |
| 10 (Oct 26-30) | Study connector API | PagedAttention (blocks, block tables, copy-on-write). Prefix caching (vLLM hashes full blocks; SGLang uses a radix tree). Scheduler/worker split. | PagedAttention paper. vLLM *Automatic Prefix Caching* design doc† (docs.vllm.ai, design/prefix_caching). Read the `KVConnectorBase_V1` docstrings and the ExampleConnector source in the pinned 0.27.1 tree. |
| 11 (Nov 2-6) | Connector prototype | Continuous (iteration-level) batching. How a "matched token" count changes what the scheduler computes. Nsight Systems and NVTX. | Anyscale's *Continuous batching* blog† (anyscale.com/blog/continuous-batching-llm-inference). Stas Bekman's inference chapter (batching, TTFT/TPOT). |
| 12 (Nov 9-13) | Finish C3, E5 | Chunked prefill and prefill/decode interference, token budget per step. Benchmark methodology (open-loop arrivals, percentiles). | Sarathi-Serve, Stas Bekman's benchmarking section. |
| 13 (Nov 16-20) | E3, E4 | KV tiers and disaggregation (why connectors exist): prefill/decode disaggregation and KV-centric storage. | LMCache paper. DistServe† (arXiv 2401.09670) or Mooncake† (arXiv 2407.00079), skimming intro and design only. |
| 14 (Nov 23-27) | Report, figures; optional extension | FlashAttention (tiling and why the softmax is online, enough to discuss it). Triton basics, only if pursuing Q4 #6. | GPU MODE Lecture 12 (Flash Attention) and Lecture 14 (Triton). Official Triton tutorials† (triton-lang.org, "vector add" and "fused softmax"). |
| 15 (Nov 30-Dec 1) | Final edits, README | Interview prep: speculative decoding, quantization and TP basics (not covered by the project) | GPU MODE Lecture 22 (speculative decoding in vLLM), the LLM Inference Handbook. |

- For deeper roofline/parallelism study after the semester: the JAX team's *How to Scale Your Model*† (jax-ml.github.io/scaling-book) and Lilian Weng's *Large Transformer Model Inference Optimization*† (lilianweng.github.io/posts/2023-01-10-inference-optimization).
- Highest-leverage single habit: before each experiment, write the predicted number using Q2's arithmetic in the results log next to the measured one.

### Gaps
- URLs marked † were not fetched or verified in this session. Check that they resolve before citing them in a report.
- I didn't check whether vLLM's prefix-caching design doc is at the same path for 0.27.1.

---

## Q6. What should the resume bullet and 2-minute pitch look like, and with which metrics?

### Takeaway
The bullet should name the system (a vLLM KV-connector plugin with no engine changes), the mechanism (position-independent chunk KV reuse with exact RoPE re-rotation), and three measured numbers: TTFT reduction against both full prefill and prefix caching at a stated k and hardware, the quality cost (F1 delta), and the break-even reuse rate. **No number goes in until a run produces it.** That rule is in `AGENTS.md`, and it applies doubly to a resume.

### Cited Findings
- Project rules: "Never write a number into the proposal or a report unless a run produced it", and "The contribution is engineering and evaluation, not a new algorithm. Don't overclaim novelty". — `bds/project/AGENTS.md`
- Postings ask specifically for KV-cache memory management, prefix caching, and vLLM/SGLang experience. — see Q1 citations ([Lightbits](https://www.lightbitslabs.com/position/senior-inference-systems-engineer-kv-cache-optimization/), [Inferact via search](https://zerogtalent.com/ai-jobs/inferact/member-of-technical-staff-inference-9470565b-c62d-4de9-8b87-26d525ecec49))

### Inferences (templates with placeholders; fill only from measured runs)

**Resume bullets:**
- "Built a vLLM KV-connector plugin that serves position-independent cached KV for retrieved agent memories (exact RoPE re-rotation of cached keys, hash-addressed chunk store), cutting TTFT by **[X]×** versus full prefill and **[Y]×** versus vLLM prefix caching at k=[6] on one [A100], at a measured **[ΔF1]** quality cost."
- "Derived and validated a roofline/bandwidth model of load-vs-recompute for Mistral-7B (128 KiB KV/token vs 14 GFLOP/token); predicted TTFT within **[Z]%** of measurement and found reuse beats prefix caching above **[H]%** chunk-hit rate."
- Optional, only if done: "Profiled the connector with Nsight Systems; [fused gather+re-rotate+scatter into a Triton kernel / pipelined per-layer H2D loads], reducing connector overhead from **[a] ms to [b] ms**."
- Credit the team. The project has four members, and the Transformers correction methods (2-4) belong mostly to teammates.

**2-minute pitch, as a structure. Each part is a beat of about 20 seconds.**
1. **Problem.** Agents re-retrieve the same memories in different orders. vLLM's prefix cache needs identical leading tokens, so it misses, and every memory is re-prefilled. Prefill dominates TTFT for these about 2K-token prompts.
2. **Insight.** A cached chunk's KV is wrong in two ways. The position is fixable exactly, because RoPE rotations compose. The missing cross-chunk attention is only approximately fixable. Explain the four correction strategies in one sentence.
3. **What I built.** The chunk store, re-rotation (checked by exact-match tests: round trip, first layer, and 100% recompute equals full prefill), and a connector that plugs into unmodified vLLM through its scheduler/worker connector API.
4. **Numbers.** First the back-of-envelope: 14 GFLOP vs 128 KiB per token means loading is about 10-20× cheaper than recomputing over PCIe and essentially free from HBM. Then the measured result: TTFT [X] ms vs [Y] ms, and a throughput curve vs hit rate with the break-even at [H]%.
5. **Tradeoff and honesty.** The vLLM path serves the cheapest correction, whose F1 drops by [ΔF1]. The better corrections need scheduler changes. Here's what I'd build next (for example, store-time conditioning or layer-wise async loading).
6. **What I learned.** The bottleneck after reuse was [launch overhead / question prefill floor of about 7 ms / scheduler step], from the profile. That leads naturally into follow-up questions on batching and disaggregation.

- Anticipated follow-ups to rehearse:
  - Why not just use SGLang's RadixAttention? (It is still prefix-only.)
  - What happens under memory pressure? (Eviction policy for the chunk store.)
  - How would this work with tensor parallelism? (KV is sharded by heads.)
  - How does it interact with chunked prefill? (Matched tokens reduce the scheduled tokens.)
  - Is the quality loss acceptable? (Cite the E1 curve.)

### Gaps
- No real metrics exist yet: the project is at week 7 with no code. Every bracketed value above has to come from E1-E4 runs.
