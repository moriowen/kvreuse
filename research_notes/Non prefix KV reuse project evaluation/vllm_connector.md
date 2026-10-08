# vLLM KV connector (KVConnectorBase_V1) design guide for C3, as of vLLM v0.27.1

Method: read the source at tag `v0.27.1` (shallow clone of `vllm-project/vllm`, the version the project pins). Tags v0.26.0 through v0.29.1rc0 exist ([tags](https://github.com/vllm-project/vllm/tags)). Line numbers refer to v0.27.1. I did not diff v0.26 or v0.28 against it, so anything described as "current" means v0.27.1 only. Issues and RFCs were read through the GitHub API on 2026-10-07.

Source shorthand used below (all at tag v0.27.1):
- BASE = [vllm/distributed/kv_transfer/kv_connector/v1/base.py](https://github.com/vllm-project/vllm/blob/v0.27.1/vllm/distributed/kv_transfer/kv_connector/v1/base.py)
- EXAMPLE = [v1/example_connector.py](https://github.com/vllm-project/vllm/blob/v0.27.1/vllm/distributed/kv_transfer/kv_connector/v1/example_connector.py)
- FACTORY = [kv_connector/factory.py](https://github.com/vllm-project/vllm/blob/v0.27.1/vllm/distributed/kv_transfer/kv_connector/factory.py)
- SCHED = [vllm/v1/core/sched/scheduler.py](https://github.com/vllm-project/vllm/blob/v0.27.1/vllm/v1/core/sched/scheduler.py)
- KVCM = [vllm/v1/core/kv_cache_manager.py](https://github.com/vllm-project/vllm/blob/v0.27.1/vllm/v1/core/kv_cache_manager.py)
- FA = [vllm/v1/attention/backends/flash_attn.py](https://github.com/vllm-project/vllm/blob/v0.27.1/vllm/v1/attention/backends/flash_attn.py)
- FI = [vllm/v1/attention/backends/flashinfer.py](https://github.com/vllm-project/vllm/blob/v0.27.1/vllm/v1/attention/backends/flashinfer.py)
- KTU = [model_executor/layers/attention/kv_transfer_utils.py](https://github.com/vllm-project/vllm/blob/v0.27.1/vllm/model_executor/layers/attention/kv_transfer_utils.py)
- MIXIN = [v1/worker/kv_connector_model_runner_mixin.py](https://github.com/vllm-project/vllm/blob/v0.27.1/vllm/v1/worker/kv_connector_model_runner_mixin.py)
- KTC = [vllm/config/kv_transfer.py](https://github.com/vllm-project/vllm/blob/v0.27.1/vllm/config/kv_transfer.py)

## Q1. API surface of KVConnectorBase_V1, registration, and an example to copy

### Takeaway
In v0.27.1 a connector must implement five abstract methods: on the scheduler side `get_num_new_matched_tokens`, `update_state_after_alloc` and `build_connector_meta`, and on the worker side `start_load_kv`, `wait_for_layer_load`, `save_kv_layer` and `wait_for_save` (`request_finished` has a default). It is loaded out of tree with `--kv-transfer-config '{"kv_connector":"<ClassName>","kv_connector_module_path":"<pkg.module>","kv_role":"kv_both",...}'`. `ExampleConnector` (the old SharedStorageConnector) is the template to copy. It already does almost what C3 needs: it hashes token IDs, injects KV through `slot_mapping` in `start_load_kv`, and makes all decisions on the scheduler side.

### Cited Findings
**Class and roles**
- `KVConnectorBase_V1.__init__(self, vllm_config, role: KVConnectorRole, kv_cache_config: KVCacheConfig)`. `KVConnectorRole` is `SCHEDULER=0` ("Connector running in the scheduler process") or `WORKER=1`. The constructor raises if `vllm_config.kv_transfer_config` is None and logs "This API is experimental and subject to change". — [BASE](https://github.com/vllm-project/vllm/blob/v0.27.1/vllm/distributed/kv_transfer/kv_connector/v1/base.py)
- The same class is instantiated twice, once per role. The scheduler builds its copy with `KVConnectorFactory.create_connector(...)` (SCHED ~l.140) and then calls `connector.bind_gpu_block_pool(self.kv_cache_manager.block_pool)` (SCHED l.294). The only channel between the two copies is a `KVConnectorMetadata` subclass that you define. It is attached to `scheduler_output.kv_connector_metadata` (SCHED l.1236) and handed to the worker through `bind_connector_metadata()` before each forward pass. — [SCHED](https://github.com/vllm-project/vllm/blob/v0.27.1/vllm/v1/core/sched/scheduler.py); [MIXIN](https://github.com/vllm-project/vllm/blob/v0.27.1/vllm/v1/worker/kv_connector_model_runner_mixin.py)

**Scheduler-side methods (abstract ones marked \*)**
- \*`get_num_new_matched_tokens(request, num_computed_tokens) -> tuple[int | None, bool]`. It returns the number of tokens loadable from the external cache *beyond* `num_computed_tokens`, plus a flag saying the load is async. `None` means "ask again later", and the request is skipped this step. The async flag "Must be 'False' if the first element is 0." The docstring says the method "might be called multiple times for a given request and should be side-effect free", and that "the connector should only consider the largest prefix of prompt-tokens for which KV cache is actually available". — [BASE](https://github.com/vllm-project/vllm/blob/v0.27.1/vllm/distributed/kv_transfer/kv_connector/v1/base.py)
- \*`update_state_after_alloc(request, blocks: KVCacheBlocks, num_external_tokens)` runs after blocks are allocated. For async loads it can run twice. The docstring says to "decide whether to load based on `num_external_tokens`, not on whether `blocks` is empty". — [BASE](https://github.com/vllm-project/vllm/blob/v0.27.1/vllm/distributed/kv_transfer/kv_connector/v1/base.py)
- \*`build_connector_meta(scheduler_output) -> KVConnectorMetadata` runs once per step. It "should NOT modify fields in the scheduler_output" and "will reset the state of the connector". — [BASE](https://github.com/vllm-project/vllm/blob/v0.27.1/vllm/distributed/kv_transfer/kv_connector/v1/base.py)
- `request_finished(request, block_ids) -> (bool, dict | None)` is called exactly once, before the request's blocks are freed. Returning True means the connector will free them asynchronously. The default is `(False, None)`. — [BASE](https://github.com/vllm-project/vllm/blob/v0.27.1/vllm/distributed/kv_transfer/kv_connector/v1/base.py)
- Optional scheduler hooks that are newer than the original V1 API: `on_new_request(request)`, `update_connector_output(KVConnectorOutput)`, `take_events()`, `bind_gpu_block_pool(block_pool)`, `has_pending_push_work()`, `reset_cache()`, `get_finished_count()`, `build_kv_connector_stats` / `build_prom_metrics`, the class method `get_required_kvcache_layout(vllm_config)` (returns "NHD"/"HND"/None), and the class method `requires_piecewise_for_cudagraph(extra_config)`. Two properties also exist: `prefer_cross_layer_blocks` (default False) and `requires_kv_delivery` (default `is_kv_producer`). A mixin `SupportsHMA` with `request_finished_all_groups` covers the hybrid memory allocator. — [BASE](https://github.com/vllm-project/vllm/blob/v0.27.1/vllm/distributed/kv_transfer/kv_connector/v1/base.py)

**Worker-side methods**
- \*`start_load_kv(forward_context, **kwargs)` is called from the model runner inside the forward context, *before* the forward pass. The code is `kv_connector.bind_connector_metadata(scheduler_output.kv_connector_metadata)` and then `kv_connector.start_load_kv(get_forward_context())`. After the forward pass the runner calls `wait_for_save()`, `get_finished(finished_req_ids)` and `get_block_ids_with_load_errors()`, and later `clear_connector_metadata()`. — [MIXIN](https://github.com/vllm-project/vllm/blob/v0.27.1/vllm/v1/worker/kv_connector_model_runner_mixin.py)
- \*`wait_for_layer_load(layer_name)` and \*`save_kv_layer(layer_name, kv_layer, attn_metadata, **kw)` are called by the `maybe_transfer_kv_layer` decorator around each attention op. The decorator does nothing unless a V1 transfer group exists and `connector.has_connector_metadata()` returns True. — [KTU](https://github.com/vllm-project/vllm/blob/v0.27.1/vllm/model_executor/layers/attention/kv_transfer_utils.py)
- \*`wait_for_save()` blocks until saves finish, at forward-context exit. — [BASE](https://github.com/vllm-project/vllm/blob/v0.27.1/vllm/distributed/kv_transfer/kv_connector/v1/base.py)
- Optional worker hooks: `register_kv_caches(kv_caches: dict[str, Tensor])` (layer name to paged tensor), `register_cross_layers_kv_cache`, `handle_preemptions(meta)`, `get_finished(finished_req_ids) -> (sending_ids, recving_ids)`, `get_block_ids_with_load_errors() -> set[int]`, `build_connector_worker_meta()` (worker-to-scheduler metadata with an `aggregate()` method), `shutdown()`, `get_kv_connector_stats()`, `set_host_xfer_buffer_ops`, and the handshake-metadata methods (P/D only). — [BASE](https://github.com/vllm-project/vllm/blob/v0.27.1/vllm/distributed/kv_transfer/kv_connector/v1/base.py)

**Registration**
- `KVTransferConfig` fields: `kv_connector`, `kv_role` (`kv_producer`/`kv_consumer`/`kv_both`, required whenever `kv_connector` is set), `kv_connector_extra_config: dict`, `kv_connector_module_path` ("The Python module path to dynamically load the KV connector from. Only supported in V1."), `kv_load_failure_policy: "recompute" | "fail"` (default "fail"), `enable_permute_local_kv`, `kv_buffer_device`, and others. — [KTC](https://github.com/vllm-project/vllm/blob/v0.27.1/vllm/config/kv_transfer.py)
- `KVConnectorFactory.get_connector_class` runs `importlib.import_module(kv_connector_module_path)` and fetches the class named `kv_connector` from it. An empty-string module path raises an error. Built-in classes are registered lazily with `KVConnectorFactory.register_connector(name, module_path, class_name)`, e.g. `"ExampleConnector"`. — [FACTORY](https://github.com/vllm-project/vllm/blob/v0.27.1/vllm/distributed/kv_transfer/kv_connector/factory.py)
- Connector-specific settings are read through `self._kv_transfer_config.get_from_extra_config(key, default)`. ExampleConnector reads `shared_storage_path` this way. — [EXAMPLE](https://github.com/vllm-project/vllm/blob/v0.27.1/vllm/distributed/kv_transfer/kv_connector/v1/example_connector.py)

**Example connectors in v0.27.1** (`kv_connector/v1/`): `example_connector.py`, `example_hidden_states_connector.py`, `decode_bench_connector.py`, `simple_cpu_offload_connector.py`, `offloading_connector.py`, `lmcache_connector.py`, `lmcache_mp_connector.py`, `multi_connector.py`, `nixl/`, `mooncake/`, `flexkv_connector.py`, `hf3fs/`, `moriio/`. There is also `examples/disaggregated/kv_load_failure_recovery_offline/load_recovery_example_connector.py`. — [v1 dir](https://github.com/vllm-project/vllm/tree/v0.27.1/vllm/distributed/kv_transfer/kv_connector/v1)
- What ExampleConnector does: `get_num_new_matched_tokens` checks whether a folder named by the hash of `prompt_token_ids[:align(len-1)]` exists and returns `num_tokens_to_check - num_computed_tokens, False`. `update_state_after_alloc` records requests with `num_external_tokens > 0`. `build_connector_meta` walks `scheduler_output.scheduled_new_reqs`, takes `new_req.block_ids[0]` (KV cache group 0), and builds `slot_mapping = block_id * block_size + offset`, truncated to the aligned token count. It also handles requests resumed after preemption via `scheduled_cached_reqs.resumed_req_ids`. `start_load_kv` loops over `forward_context.no_compile_layers`, reads each layer's `layer.kv_cache`, and writes `dst[block_idxs, :, offsets] = src`. The class comment says it "does extra work which will overwrite the existing prefix-cache in GPU". — [EXAMPLE](https://github.com/vllm-project/vllm/blob/v0.27.1/vllm/distributed/kv_transfer/kv_connector/v1/example_connector.py)

### Inferences
- Skeleton for C3 (`ChunkReuseConnector`):
  - Scheduler side: parse `request.prompt_token_ids` into [header][chunk_1]...[chunk_k][question], look up each chunk by the hash of its token IDs, and return the longest contiguous covered prefix, block-aligned (see Q2). Cache the parse per `request_id` so repeated calls stay side-effect free.
  - `update_state_after_alloc`: remember `(request_id, num_external_tokens)`.
  - `build_connector_meta`: emit per request a list of `(chunk_id, src_pos_start, dst_pos_start, length)` plus the slot mapping built from `new_req.block_ids[0]`, the same way ExampleConnector does.
  - Worker side: in `start_load_kv`, load the K/V of each chunk and layer, re-rotate K by `delta = dst_pos - src_pos`, and scatter into `layer.kv_cache`.
  - `save_kv_layer` and `wait_for_save`: no-ops, because the chunk store is built offline by C2.
  - Register with `kv_role="kv_both"` or `"kv_consumer"`. Before relying on `kv_consumer`, check its side effects in v0.27.1, e.g. `requires_kv_delivery` keys off the producer role.
- The ExampleConnector's own `align_to_block_size` returns `(n-1)//bs*bs`, which drops a full block when n is an exact multiple. Do not copy it blindly.

### Gaps
- I did not diff base.py between v0.26, v0.27.1 and v0.28. In v0.28/v0.29 a method may have been added or changed (the `SupportsHMA`, `build_connector_worker_meta` and `has_pending_push_work` hooks look recent).
- I did not verify when SharedStorageConnector was renamed to ExampleConnector. Only the v0.27.1 name is confirmed.
- I did not verify whether `kv_role="kv_consumer"` alone changes scheduling behaviour in 0.27.1, beyond `requires_kv_delivery`.

## Q2. Block granularity, paged KV layout, slot_mapping, and whether RoPE is applied before caching

### Takeaway
Return block-aligned counts (multiples of 16). The scheduler presents a block-aligned local hit to the connector. Only full blocks are ever hashed into the prefix cache. ExampleConnector documents that alignment is expected. The tail of the last chunk that does not fill a block is simply prefilled by vLLM together with the question.

In v0.27.1 FlashAttention and FlashInfer both use a *packed* logical layout `(num_blocks, num_kv_heads, block_size, 2*head_size)`, with K in `[..., :128]` and V in `[..., 128:]`. The physical stride order defaults to NHD. This differs from the older `(2, num_blocks, block_size, H, D)` layout, so it is version-specific.

Keys in the paged cache are post-RoPE: Llama/Mistral apply `rotary_emb` before calling `self.attn`. The connector must therefore write fully rotated keys for the target positions.

### Cited Findings
- The scheduler queries the local prefix cache first and then calls the connector with `block_aligned_local = num_new_local_computed_tokens - (num_new_local_computed_tokens % block_size)`. If the connector returns more than the local partial tail, the partial local block is truncated and the external load covers it. Otherwise any external hit is discarded (`num_external_computed_tokens = 0`). — [SCHED l.769-811](https://github.com/vllm-project/vllm/blob/v0.27.1/vllm/v1/core/sched/scheduler.py)
- ExampleConnector's note: "in current v1 scheduler, the num_computed_tokens is aligned with the block granularity. And it expects the returned blocks and num_computed_tokens to also be aligned with the block granularity." — [EXAMPLE](https://github.com/vllm-project/vllm/blob/v0.27.1/vllm/distributed/kv_transfer/kv_connector/v1/example_connector.py)
- A local cache hit is capped at `max_cache_hit_length = request.num_tokens - 1` because at least one token must be computed to get logits. For async loads that complete the full prompt, `num_computed_tokens` is reset to `num_tokens - 1`. — [KVCM l.254-262](https://github.com/vllm-project/vllm/blob/v0.27.1/vllm/v1/core/kv_cache_manager.py); [SCHED l.2673](https://github.com/vllm-project/vllm/blob/v0.27.1/vllm/v1/core/sched/scheduler.py)
- `allocate_slots(..., num_external_computed_tokens, delay_cache_blocks)` allocates blocks covering local + external + new tokens. It raises if `num_new_tokens == 0 and num_external_computed_tokens == 0`. — [KVCM l.344-470](https://github.com/vllm-project/vllm/blob/v0.27.1/vllm/v1/core/kv_cache_manager.py)
- FlashAttention `get_kv_cache_shape` returns `(num_blocks, num_kv_heads, block_size, 2 * head_size)`, commented "K and V are packed into the content dim: logical (B, H, N, 2*D)". It raises unless `block_size % 16 == 0`. The stride order for NHD gives physical `(num_blocks, block_size, num_kv_heads, 2*head_size)`. — [FA l.134-170](https://github.com/vllm-project/vllm/blob/v0.27.1/vllm/v1/attention/backends/flash_attn.py)
- FlashAttention unpacks with `key_cache, value_cache = kv_cache.transpose(1, 2).split(self.head_size, dim=-1)`. That puts K in the first `head_size` channels and V in the last. — [FA l.905](https://github.com/vllm-project/vllm/blob/v0.27.1/vllm/v1/attention/backends/flash_attn.py)
- FlashInfer uses the same packed shape `(num_blocks, num_kv_heads, block_size, 2*head_size)`, except for nvfp4. — [FI l.398-409](https://github.com/vllm-project/vllm/blob/v0.27.1/vllm/v1/attention/backends/flashinfer.py)
- The default layout is NHD unless the connector's `get_required_kvcache_layout` or `VLLM_KV_CACHE_LAYOUT` overrides it. — [kv_connector/utils.py l.37-50](https://github.com/vllm-project/vllm/blob/v0.27.1/vllm/distributed/kv_transfer/kv_connector/utils.py); [attention/backends/utils.py l.83-108](https://github.com/vllm-project/vllm/blob/v0.27.1/vllm/v1/attention/backends/utils.py)
- RoPE runs before the cache write: in `LlamaAttention.forward`, `q, k = self.rotary_emb(positions, q, k)` comes before `attn_output = self.attn(q, k, v)` (llama.py l.228-229), with `is_neox_style = True` (l.238). `MistralAttention` subclasses `LlamaAttention` and also calls `self.rotary_emb(positions, q, k)` (mistral.py l.135). `MistralForCausalLM` is registered from `mistral.py`. — [llama.py](https://github.com/vllm-project/vllm/blob/v0.27.1/vllm/model_executor/models/llama.py); [mistral.py](https://github.com/vllm-project/vllm/blob/v0.27.1/vllm/model_executor/models/mistral.py); [registry.py l.172](https://github.com/vllm-project/vllm/blob/v0.27.1/vllm/model_executor/models/registry.py)

### Inferences
- **Write recipe for Mistral-7B on v0.27.1, FlashAttention, NHD.** Per layer, `kv = layer.kv_cache` has logical shape `[num_blocks, 8, 16, 256]`. For token slots `s`: `b = s // 16`, `o = s % 16`. Then `kv[b, :, o, :128] = K_rot[t]` (shape [T, 8, 128]) and `kv[b, :, o, 128:] = V[t]`, or in one write `kv[b, :, o] = torch.cat([K_rot, V], -1)`. The same `dst[block_idxs, :, offsets]` indexing as ExampleConnector works because it indexes the logical view. Assert the shape once in `register_kv_caches` and fail loudly if it differs, since it changed across versions.
- Because vLLM uses neox-style RoPE (rotate_half), the same convention as HF Mistral, re-rotating a key stored at position p to position p' is one more neox rotation by angle `(p'-p)·θ_i`. Mistral-7B-v0.3 uses `rope_theta = 1e6` (from the HF config, not verified in this session). Compute it in fp32 and cast to bf16.
- If the model loads in Mistral-native format (`consolidated.safetensors`, `--load-format mistral`), vLLM permutes the q/k weights to match. Use HF format for both stacks so E5's tensor comparison is apples to apples. (Unverified: whether 0.27.1 defaults to the native format for this repo.)
- **Partial block.** If the covered prefix is `L` tokens, return `floor(L/16)*16 - num_computed_tokens`. The remaining `L mod 16` tokens of the last chunk are recomputed by vLLM with full attention over the injected prefix. That differs slightly from method 1 in C2, so E5 must either use the same truncation in the HF reference or pad the header so chunk boundaries fall on multiples of 16 (padding changes tokenization, so prefer truncation).
- Do not count on block-misaligned external counts working. I found no assert forbidding them, but ExampleConnector says alignment is expected and prefix-cache hashing works on full blocks only (unverified for the partial case).

### Gaps
- I did not verify numerically that vLLM's `rotary_emb` (cos/sin cache dtype, fused kernel) matches HF's fp32-cos/sin-then-cast path bit for bit. E5's "matching sampled KV tensors" should use a tolerance, not equality.
- I did not check the attention backend that 0.27.1 picks by default on A100/H100 for Mistral. Log it at startup and pin it with the `VLLM_ATTENTION_BACKEND` env var or the `--attention-backend` flag (the exact flag name was not verified).

## Q3. Interaction with vLLM's automatic prefix caching

### Takeaway
Yes, vLLM first looks up its local GPU prefix cache, then asks the connector only for tokens beyond the block-aligned local hit. Yes, once the connector's tokens are counted as computed, vLLM hashes those blocks under the request's exact token-ID block hashes and registers them in its prefix cache. For sync loads this happens immediately in `allocate_slots`; for async loads, after the transfer completes.

So with APC on, a later request with the same token prefix (same chunks in the same order) gets a *local* hit on the approximate (re-rotated, cross-attention-free) KV. For method 1 that KV is deterministic and the same as what the connector would supply, so it is consistent, but it changes what is being measured. Isolate it with `--no-enable-prefix-caching` on connector runs, or with a per-request `cache_salt`.

### Cited Findings
- Flow at SCHED l.744-831. With a connector present, `kv_cache_manager.get_computed_blocks_for_connector(request)` runs first, then `connector.get_num_new_matched_tokens(request, block_aligned_local)`. Then `num_computed_tokens = num_new_local_computed_tokens + num_external_computed_tokens`. — [SCHED](https://github.com/vllm-project/vllm/blob/v0.27.1/vllm/v1/core/sched/scheduler.py)
- In `allocate_slots`: "if not self.enable_caching or delay_cache_blocks: return" without caching. Otherwise it caches up to `min(total_computed_tokens + num_new_tokens, request.num_tokens)` via `coordinator.cache_blocks`, where `total_computed_tokens` includes `num_external_computed_tokens`. — [KVCM l.551-563](https://github.com/vllm-project/vllm/blob/v0.27.1/vllm/v1/core/kv_cache_manager.py)
- For async loads (`delay_cache_blocks=load_kv_async`), the comment says: "_update_waiting_for_remote_kv will then cache only the successfully loaded tokens". Then `self.kv_cache_manager.cache_blocks(request, request.num_computed_tokens)` runs once the transfer is reported finished. — [SCHED l.1039, ~2650-2670](https://github.com/vllm-project/vllm/blob/v0.27.1/vllm/v1/core/sched/scheduler.py)
- `cache_salt` is mixed into the hash of the first block (`cache_salt_keys = [request.cache_salt] if start_token_idx == 0 and request.cache_salt`). — [kv_cache_utils.py l.579-580](https://github.com/vllm-project/vllm/blob/v0.27.1/vllm/v1/core/kv_cache_utils.py)
- A per-request `SamplingParams.skip_reading_prefix_cache` exists. `prefix_cache_lookup_enabled` returns `enable_caching and not request.skip_reading_prefix_cache`. It defaults to True only when `prompt_logprobs` is requested. — [KVCM l.214-216](https://github.com/vllm-project/vllm/blob/v0.27.1/vllm/v1/core/kv_cache_manager.py); [sampling_params.py l.348, 509-513](https://github.com/vllm-project/vllm/blob/v0.27.1/vllm/sampling_params.py)
- External hits are logged separately: `connector_prefix_cache_stats.record(num_tokens=..., num_hits=num_external_computed_tokens)` (SCHED l.1007-1013), and `prefill_stats` records `num_local_cached_tokens` and `num_external_cached_tokens`. — [SCHED](https://github.com/vllm-project/vllm/blob/v0.27.1/vllm/v1/core/sched/scheduler.py)
- ExampleConnector's own warning: "It does extra work which will overwrite the existing prefix-cache in GPU". — [EXAMPLE](https://github.com/vllm-project/vllm/blob/v0.27.1/vllm/distributed/kv_transfer/kv_connector/v1/example_connector.py)

### Inferences
- **Fair configurations for E3/E4:**
  - (a) Full prefill: no connector, `--no-enable-prefix-caching`.
  - (b) Prefix caching: no connector, APC on (default). This hits only when the same chunks recur in the same order after the same header.
  - (c) Connector alone: connector plus `--no-enable-prefix-caching`. Every reuse goes through the connector, so the load and re-rotation cost is measured honestly.
  - (d) Connector plus APC: report as a separate "deployable" configuration and note that repeated orderings become local hits.
  - Reset state between runs by restarting the server, or use the reset-prefix-cache endpoint (`examples/disaggregated/reset_kv` exists; I did not check the exact endpoint) and the connector `reset_cache()`.
- The contamination risk runs only one way. Approximate KV is registered under exact token hashes, and the only later readers are requests whose token prefix is the same, which C3 would have served the same KV anyway. Do not mix a connector-plus-APC run with an exact-KV correctness check on the same server.
- For E5, compare per-request external hits against the connector's own log using `num_external_cached_tokens`.

### Gaps
- I did not verify whether `--no-enable-prefix-caching` also changes `get_computed_blocks_for_connector` behaviour. It should return 0 local tokens, but I did not trace it.

## Q4. Why LMCache's CacheBlend is broken on V1, and what that implies for content-based matching

### Takeaway
On vLLM V1, LMCache's CacheBlend fails at the scheduler-side lookup. `get_num_new_matched_tokens` can report only a contiguous prefix measured from the start of the prompt, and LMCache's lookup keyed chunks by prefix-chained hashes, so a reused document after a different question got 0 hits. The in-process blend path is now unmaintained (the README patch stops applying at vLLM v0.12.0), and the newer MP-mode blend server has no client-side handshake in vLLM 0.27.1.

C3 avoids the problem by construction. It hashes each chunk independently of position, and it *places* the retrieved chunks as the prompt prefix, so the non-prefix reuse happens in C3's own lookup and re-rotation rather than in vLLM's hit-count contract. What C3 cannot do through this API is reuse a chunk that sits *after* any uncached token. One gap ends the hit, so the header and separators must also be in the store.

### Cited Findings
- LMCache #3238 (open, filed 2026-05-09, against vLLM 0.17.1/0.18.0): "CacheBlend's non-prefix KV cache reuse does not work with the vLLM V1 connector". "The scheduler-side lookup (`get_num_new_matched_tokens`) only performs prefix matching", and "the blending logic in `LMCBlender` only processes tokens marked as 'hit' by the scheduler". Repro: a 2,400-token document repeated behind a changed question got "Only 24/2451 tokens hit (the common system prompt prefix)". The original root cause was the rolling hash chain in `TokenDatabase.process_tokens`. A later maintainer comment says `SegmentTokenDatabase.process_tokens` on `dev` now hashes segments independently, so the mismatch "may be on the lookup-client side instead", but it has not been re-run. — [LMCache #3238](https://github.com/LMCache/LMCache/issues/3238)
- LMCache #4476 (open, 2026-08-10): the `examples/blend_kv_v1` README patch "stop[s] applying at vLLM v0.12.0" because `ensure_kv_transfer_initialized` moved into `Worker.initialize_from_config`. Maintainer (deng451e): "In-process CacheBlend is no longer maintained and is likely incompatible with the latest vLLM. We're developing CacheBlend support for LMCache MP mode". — [LMCache #4476](https://github.com/LMCache/LMCache/issues/4476)
- LMCache #5101 (open, 2026-09-14; image `lmcache/vllm-openai:v0.5.4` = vLLM 0.27.1 + LMCache 0.5.4):
  - The MP connector never sends `CB_REGISTER_ROPE_V3`, and `grep -c -i blend lmcache_mp_connector.py` returns 0 in v0.5.2, v0.5.4 and 0.5.5.dev114, including the copy vendored into vLLM.
  - Reordering chunks A+B+C to C+B+A took 0.709 s against a cold prefill of 0.728 s, and the external hit rate stayed near 0.
  - The reporter concludes blend V3 looks server-side only.
  - No maintainer answer had been posted as of the fetch. — [LMCache #5101](https://github.com/LMCache/LMCache/issues/5101)
- vLLM's base docstring requires the connector to report "the largest prefix of prompt-tokens for which KV cache is actually available". — [BASE](https://github.com/vllm-project/vllm/blob/v0.27.1/vllm/distributed/kv_transfer/kv_connector/v1/base.py)

### Inferences
- For C3, `get_num_new_matched_tokens` must do content matching itself. Walk the prompt from `num_computed_tokens`, match each segment's token IDs against the store (exact hash of the token-ID sequence, with collisions checked by comparing tokens), stop at the first uncovered token, and return that count, block-aligned.
- Every token before the question must be in the store: BOS, `[INST]`, system text and separators. Otherwise the hit is 0. Store the fixed header and the separator as pseudo-chunks.
- Segmenting the prompt needs either a delimiter token sequence (as CacheBlend uses `" # # "`) or exact matching against the known per-chunk token lists. The simplest reliable route is to send `prompt_token_ids` built by concatenating the per-chunk token lists, so boundaries are exact. Re-tokenizing the joined text can merge across boundaries with SentencePiece (Mistral v0.3), which breaks hashing and E5's "identical tokenization" check.
- Methods 2-4 (selective recompute) need a per-token "recompute this position" mask inside the prefix. The V1 API has no such concept, which is consistent with the proposal keeping them in Transformers.

### Gaps
- I did not confirm whether LMCache 0.5.x dev has since fixed the #3238 symptom. The latest comment says it was not re-tested.
- I did not read LMCache's own `lmcache_connector.py` internals for this note.

## Q5. Async loading, CUDA graphs and torch.compile, process placement, TP=1

### Takeaway
For C3, use **synchronous** loading: `get_num_new_matched_tokens` returns `(n, False)`, and `start_load_kv` writes every layer before the forward pass, with `wait_for_layer_load` as a no-op. That keeps the default CUDA-graph mode and avoids the `WAITING_FOR_REMOTE_KVS` state machine.

The scheduler-side connector lives in the EngineCore process, apart from the API server. The worker-side instance lives with the model runner. They share nothing except the pickled metadata, so the chunk-store index must be loadable by both.

### Cited Findings
- With async load (`load_kv_async=True`), the request gets `num_new_tokens = 0`, is put in `RequestStatus.WAITING_FOR_REMOTE_KVS`, and holds reserved blocks ("An async load holds its blocks for the whole transfer with no forward progress and isn't preemptible here"). Completion must be reported via `get_finished()` `recving` ids, and only then are blocks cached. — [SCHED l.866-1045, ~2639-2675](https://github.com/vllm-project/vllm/blob/v0.27.1/vllm/v1/core/sched/scheduler.py)
- `requires_piecewise_for_cudagraph(extra_config)` docstring: "Connectors that use asynchronous layer-by-layer operations (wait_for_layer_load/save_kv_layer) should override this method to return True ... These operations cannot be captured in CUDA graphs and will be skipped during replay, causing data races." If it returns True while full CUDA graphs are configured, `VllmConfig` overrides `cudagraph_mode` to `PIECEWISE` with a warning. — [BASE](https://github.com/vllm-project/vllm/blob/v0.27.1/vllm/distributed/kv_transfer/kv_connector/v1/base.py); [config/vllm.py l.1400-1422](https://github.com/vllm-project/vllm/blob/v0.27.1/vllm/config/vllm.py)
- `start_load_kv` is called outside the compiled model, from the runner's connector context before `forward`. — [MIXIN l.80-105](https://github.com/vllm-project/vllm/blob/v0.27.1/vllm/v1/worker/kv_connector_model_runner_mixin.py); also [v1/worker/gpu/kv_connector.py l.72-85](https://github.com/vllm-project/vllm/blob/v0.27.1/vllm/v1/worker/gpu/kv_connector.py)
- ExampleConnector iterates `forward_context.no_compile_layers` to find attention layers and their `kv_cache`. — [EXAMPLE](https://github.com/vllm-project/vllm/blob/v0.27.1/vllm/distributed/kv_transfer/kv_connector/v1/example_connector.py)
- Load failures: `get_block_ids_with_load_errors()` combined with `kv_load_failure_policy` (`"fail"` by default, or `"recompute"`). — [BASE](https://github.com/vllm-project/vllm/blob/v0.27.1/vllm/distributed/kv_transfer/kv_connector/v1/base.py); [KTC](https://github.com/vllm-project/vllm/blob/v0.27.1/vllm/config/kv_transfer.py)
- Encoder-decoder models are rejected with KV connectors (SCHED l.138), which does not affect Mistral. ExampleConnector uses `block_ids[0]`, assuming one KV cache group. — [SCHED](https://github.com/vllm-project/vllm/blob/v0.27.1/vllm/v1/core/sched/scheduler.py)

### Inferences
- Sync loading is safe without stream sync: the writes are issued on the current stream in `start_load_kv` and are ordered before the forward kernels. Layerwise async (side stream plus per-layer events in `wait_for_layer_load`) could overlap the copy with compute, but forces PIECEWISE graphs. Prefill already runs piecewise or eager in most configs, so the gain is mainly overlap. Treat it as an optimisation after E5 passes.
- Placement:
  - The API server process handles tokenization.
  - The EngineCore process holds the scheduler and the SCHEDULER-role connector.
  - The worker holds the WORKER-role connector. With TP=1 the worker is probably inside the EngineCore process (UniProcExecutor), but the two connector objects stay separate either way. Unverified for 0.27.1 defaults.
  - Design for no shared Python state. The scheduler instance loads only a lightweight index (`token-hash -> chunk_id, length`) from a path in `kv_connector_extra_config`. The worker instance holds the tensors.
- Keep the chunk store on GPU if it fits. On an A100-80GB, weights take about 15 GB and the store about 20 GB. Lower `--gpu-memory-utilization` (about 0.5-0.6) so vLLM's KV block pool does not take the memory the worker-side store needs. Otherwise keep it in pinned host memory and pay H2D (Q6).
- The TP=1 assumption makes the slot mapping and all 8 KV heads local to one rank. With TP>1, each rank would hold `8/TP` heads and the connector would slice heads by `get_tensor_model_parallel_rank()`. Out of scope.
- Preemption: if a request is preempted and resumed, ExampleConnector re-emits a load for `resumed_req_ids`. C3 should do the same, or cheaper, just let vLLM recompute.

### Gaps
- I did not confirm 0.27.1's default executor and process layout for a single GPU.
- I did not check whether the hybrid memory allocator is enabled for Mistral-7B-v0.3. It should not be, since v0.3 has no sliding window; that is a config-level claim, not verified here.

## Q6. Benchmarking: vllm bench serve, baselines, and H2D cost

### Takeaway
Use `vllm bench serve --dataset-name custom --dataset-path x.jsonl` (JSONL rows `{"prompt": ..., "output_tokens": N}`, plus `--skip-chat-template` and `--custom-output-len`). Sweep `--request-rate` and report TTFT, ITL and throughput. Compare full prefill, APC, LMCache CPU offload (prefix-keyed, which only helps exact-order repeats) and C3.

Loading costs 128 KB per token. Taking k=6 chunks of about 300 tokens as an illustration, that is roughly 230 MB per request: about 10 ms over PCIe Gen4, about 5 ms over Gen5, and well under 1 ms if the store is GPU-resident. That should be small next to a 1.8K-token prefill, so a GPU-resident store is the configuration that gives C3 its best case.

### Cited Findings
- `CustomDataset` docstring: "Loads data from a JSONL file", with example rows `{"prompt": "What is the capital of India?", "output_tokens": 10}`. "'output_tokens' column is optional and has to be provided only if 'custom-output-len' argument is None or -1". The CLI exposes `--skip-chat-template` and `--custom-output-len`, and `"custom"` is a `--dataset-name` choice. — [benchmarks/datasets/datasets.py l.1626-1672, 2496-2523](https://github.com/vllm-project/vllm/blob/v0.27.1/vllm/benchmarks/datasets/datasets.py)
- `vllm/benchmarks/` in 0.27.1 contains `serve.py`, `throughput.py`, `latency.py`, `sweep/` and `plot.py`. — [benchmarks dir](https://github.com/vllm-project/vllm/tree/v0.27.1/vllm/benchmarks)
- LMCache CPU offload baseline as shipped in vLLM examples: `KVTransferConfig(kv_connector="LMCacheConnectorV1", kv_role="kv_both")` with env `LMCACHE_CHUNK_SIZE=256`, `LMCACHE_LOCAL_CPU=True`, `LMCACHE_MAX_LOCAL_CPU_SIZE=5.0` (GB). An MP-mode script `cpu_offload_lmcache_mp.sh` also exists. — [examples/disaggregated/lmcache/cpu_offload_lmcache.py](https://github.com/vllm-project/vllm/blob/v0.27.1/examples/disaggregated/lmcache/cpu_offload_lmcache.py); [lmcache example dir](https://github.com/vllm-project/vllm/tree/v0.27.1/examples/disaggregated/lmcache)
- The reorder test in LMCache #5101 shows LMCache MP on vLLM 0.27.1 gets near-zero hits when chunks are reordered (0.709 s against 0.728 s cold). — [LMCache #5101](https://github.com/LMCache/LMCache/issues/5101)
- vLLM 0.27.1 also ships `simple_cpu_offload_connector.py` and `offloading_connector.py`, native CPU-offload connectors that could serve as a second, non-LMCache offload baseline. — [v1 dir](https://github.com/vllm-project/vllm/tree/v0.27.1/vllm/distributed/kv_transfer/kv_connector/v1)
- Model KV size: 32 layers × 8 KV heads × 128 dims × 2 (K, V) × 2 bytes = 128 KB per token, as stated in the project proposal. — [proposal §4.1](/Users/atharvamohite/dev/acads/bds/project/proposal-nonprefix-kv-reuse-agent-memory.md)

### Inferences
- **H2D arithmetic (my estimate).** These assume effective PCIe throughput of about 25 GB/s on Gen4 x16 and about 50 GB/s on Gen5, from pinned memory; measure on PACE.
  - 1,800 tokens × 128 KB is about 230 MB, which takes about 9 ms on Gen4 and about 5 ms on Gen5. From GPU HBM (about 1.5-2 TB/s) it takes about 0.15 ms.
  - A 7B prefill of 1,800 tokens is about 2 × 7e9 × 1,800, or 2.5e13 FLOPs, roughly 0.1-0.2 s on an A100 at realistic MFU.
  - The ratio favours C3 even from host memory, and decisively from GPU.
- **Re-rotation cost** per token per layer is 8 heads × 128 dims of elementwise multiply-adds. It is negligible but adds per-layer kernel launches. Batch all layers into one kernel over a `[layers, T, H, D]` tensor.
- **The `vllm bench serve` custom dataset sends text prompts** (it has a `prompt` field), and the server re-tokenizes them. If text-level concatenation does not reproduce the chunk token IDs exactly, C3 will miss. Options: verify on every request that tokenizing the joined text reproduces the chunk token lists (an E5 check), or write a small async client for C5 that posts `prompt` as a token-ID list to `/v1/completions` and records TTFT from the stream.
- **Reporting.** Collect TTFT p50/p99, ITL and request throughput at several request rates. For E4, sweep the Zipf exponent at a fixed rate. Read the server's prefix-cache and connector hit counters (`num_external_cached_tokens`) so you can separate "hit" from "fast".
- **Warm-up.** Discard the first requests after start (CUDA-graph capture and the torch.compile cache), and keep `--max-num-batched-tokens` the same across configs.

### Gaps
- I did not verify the exact `vllm bench serve` flag set in 0.27.1 beyond the dataset flags (e.g. `--request-rate`, `--burstiness`, `--percentile-metrics`, `--save-result`). Check them with `vllm bench serve --help`.
- I did not verify whether the custom dataset or `bench serve` can send pre-tokenized prompts.
- No measured PCIe numbers for PACE nodes.

## Q7. RFC #25672 "Generalized KV cache reuse" and merged support for non-prefix hits

### Takeaway
RFC #25672 was closed within five days in favour of RFC #25950. #25950 was auto-closed as stale on 2026-05-16. The implementation PR #32785 ("Segmented prefill for gapped external KV cache hits", which adds a `get_computed_token_gaps()` connector API) is still open and was last updated 2026-02-03. So v0.27.1 has no merged support for non-prefix or gapped hits, or per-token hit bitmaps. The connector contract is still "a contiguous prefix count".

### Cited Findings
- #25672 "[Feature]: Generalized KV cache reuse", opened 2025-09-25 and closed 2025-09-30 by its author with "Closing in favor of RFC issue #25950". It proposes reusing "any subset of tokens, rather than restricting reuse to prefix-complete tokens", with "compositional context" (RAG segments) as a broader goal. — [vLLM #25672](https://github.com/vllm-project/vllm/issues/25672)
- #25950 "[RFC]: Generalized KV cache reuse", opened 2025-09-30, labelled RFC and stale, closed automatically 2026-05-16. ApostaC (LMCache) replied that CacheBlend is implemented in LMCache. The author (iddo10, with IBM) proposed on 2026-01-12 a "hole-recompute" scheme that splits a prompt into growing prefixes batched as dummy requests. — [vLLM #25950](https://github.com/vllm-project/vllm/issues/25950)
- PR #32785 "Segmented prefill for gapped external KV cache hits": OPEN, created 2026-01-21, updated 2026-02-03. It adds the connector API `get_computed_token_gaps()` and scheduler-made "virtual requests (one per gap)", described as "Attention-backend agnostic", "CUDA Graphs compatible" and "Scheduler-only changes". — [vLLM PR #32785](https://github.com/vllm-project/vllm/pull/32785)
- `base.py` in v0.27.1 has no gap or bitmap method. — [BASE](https://github.com/vllm-project/vllm/blob/v0.27.1/vllm/distributed/kv_transfer/kv_connector/v1/base.py)

### Inferences
- Even if #32785 merged, it would let a connector *skip* holes so they get recomputed. It would not change the placement of KV, so it would not help method 1 (C3 already makes chunks contiguous). It could help a future "recompute the tokens at chunk boundaries" variant.
- The proposal's statement that methods 2-4 "would need scheduler changes" is consistent with the current API.

### Gaps
- I did not search for other merged PRs between v0.27.1 and v0.29 that touch non-prefix reuse. I only checked the RFC-linked PR.
