"""C3: serve re-rotated chunk KV to an unmodified vLLM (0.27.1) as the prompt prefix.

Method 1 (re-positioning only). vLLM's contract is a contiguous prefix count, so non-prefix
reuse happens here: each prompt is matched segment by segment against the chunk store, and
the matched chunks' KV is re-rotated to where they land. vLLM then computes only the rest
(the question plus at most 15 tokens of the last chunk, because hits are whole 16-token blocks).

Register with:
    --kv-transfer-config '{"kv_connector": "ChunkReuseConnector",
        "kv_connector_module_path": "kvreuse.vllm_connector", "kv_role": "kv_both",
        "kv_connector_extra_config": {"store_dir": "<dir>"}}'
Run with prefix caching off (``--no-enable-prefix-caching``): otherwise vLLM registers the
loaded (approximate) KV in its own prefix cache under exact token hashes.

Optional ``kv_connector_extra_config`` keys:
    kv_device     "cuda" (default): loaded entries stay GPU-resident. "cpu": they stay in host
                  memory and are copied to the GPU on every load, so the copy is in the
                  measured cost (E3/E4; the full ten-conversation store is ~33 GB).
    preload       load every entry at start-up (worker side), so no disk reads during a run.
    profile       log the synchronized wall time of the first 20 loads (diagnosis only; the
                  sync adds latency to those requests).
    require_seen  E4: a segment is served only after an earlier *finished* request contained
                  it, as if each chunk's KV were saved the first time it was prefilled. The
                  store itself is still built offline; saving is not timed.

The scheduler-side and worker-side instances live in different processes and share only
the metadata built here, so each loads what it needs from ``store_dir`` itself. Loads are
synchronous (written in ``start_load_kv``), so the default CUDA-graph mode is unaffected.
"""

from __future__ import annotations

import re
import time
from concurrent.futures import ThreadPoolExecutor
from itertools import accumulate
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from safetensors.torch import load_file
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.logger import init_logger

from .connector_logic import Segment, StoreIndex, assemble_packed, matched_tokens, seen_only, slot_mapping

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = init_logger(f"vllm.{__name__}")  # under "vllm" so vLLM's handler prints it
_LAYER = re.compile(r"layers\.(\d+)\.")


@dataclass
class LoadPlan:
    req_id: str
    segments: list[Segment]
    start: int  # first position to write (tokens before it were a local prefix-cache hit)
    n: int  # tokens to write
    slots: torch.Tensor  # [n] slot ids in the paged cache


@dataclass
class ChunkReuseMetadata(KVConnectorMetadata):
    loads: list[LoadPlan] = field(default_factory=list)


class ChunkReuseConnector(KVConnectorBase_V1):
    def __init__(self, vllm_config: "VllmConfig", role: KVConnectorRole, kv_cache_config: "KVCacheConfig"):
        super().__init__(vllm_config=vllm_config, role=role, kv_cache_config=kv_cache_config)
        self._block_size = vllm_config.cache_config.block_size
        store = self._kv_transfer_config.get_from_extra_config("store_dir", None)
        if store is None:
            raise ValueError("ChunkReuseConnector needs kv_connector_extra_config.store_dir")
        self._store = Path(store)
        self._index = StoreIndex.load(self._store)
        # scheduler side
        self._plans: dict[str, tuple[list[Segment], int, int]] = {}  # req_id -> (segs, start, n)
        self._need_load: set[str] = set()
        # worker side
        self._kv: dict[str, torch.Tensor] = {}
        self._inv_freq = torch.tensor(self._index.meta["inv_freq"], dtype=torch.float32)
        self._head_dim = self._index.meta["head_dim"]
        self._dump_dir = self._kv_transfer_config.get_from_extra_config("dump_dir", None)
        self._dumped = 0
        self._kv_device = self._kv_transfer_config.get_from_extra_config("kv_device", "cuda")
        self._require_seen = bool(self._kv_transfer_config.get_from_extra_config("require_seen", False))
        self._seen: set[str] = set()
        self._profile = bool(self._kv_transfer_config.get_from_extra_config("profile", False))
        self._n_loads = 0
        self.stats = {"requests": 0, "hit_tokens": 0, "prompt_tokens": 0}
        logger.info("ChunkReuseConnector role=%s store=%s entries=%d kv_device=%s require_seen=%s",
                    role, store, len(self._index.entries), self._kv_device, self._require_seen)
        if role == KVConnectorRole.WORKER and self._kv_transfer_config.get_from_extra_config("preload", False):
            t0 = time.perf_counter()
            self._preload()
            self._warm_up()
            logger.info("ChunkReuseConnector preloaded %d entries to %s in %.1f s",
                        len(self._kv), self._kv_device, time.perf_counter() - t0)

    def _preload(self) -> None:
        """Every entry, read in parallel. On cpu, into one pinned buffer: page-locking per
        entry costs ~1 s per 32 MB file; one allocation of the whole store is far cheaper."""
        keys = list(self._index.entries)
        if self._kv_device != "cpu":
            for key in keys:
                self._load(key)
            return
        m = self._index.meta
        per_tok = m["n_layers"] * m["n_kv_heads"] * 2 * m["head_dim"]
        sizes = [len(self._index.entries[k]["ids"]) * per_tok for k in keys]
        buf = torch.empty(sum(sizes), dtype=torch.bfloat16, pin_memory=True)
        offs = [0, *accumulate(sizes)][:-1]

        def read(i):
            kv = load_file(str(self._store / self._index.entries[keys[i]]["file"]), device="cpu")["kv"]
            dst = buf[offs[i] : offs[i] + sizes[i]].view(kv.shape)
            dst.copy_(kv)
            return keys[i], dst

        with ThreadPoolExecutor(8) as pool:
            self._kv.update(pool.map(read, range(len(keys))))

    def _warm_up(self) -> None:
        """One assembly on the GPU, so the first real request does not pay CUDA's lazy kernel
        loading (~0.5 s in the E3 smoke run)."""
        try:
            key, e = next((k, e) for k, e in self._index.entries.items() if e["kind"] == "chunk")
            seg = Segment(key, e["start"] + 16, len(e["ids"]), e["start"])
            kv = assemble_packed([seg], seg.dst_start, seg.length, self._get_kv, self._inv_freq, self._head_dim)
            scratch = torch.empty((1, kv.shape[2], self._block_size, kv.shape[3]), dtype=kv.dtype, device=kv.device)
            n = min(self._block_size, kv.shape[1])
            scratch[torch.zeros(n, dtype=torch.long, device=kv.device), :, torch.arange(n, device=kv.device)] = kv[0, :n]
            torch.cuda.synchronize()
        except Exception as exc:  # warm-up is an optimisation only
            logger.warning("ChunkReuseConnector warm-up skipped: %s", exc)

    # ---------------- scheduler side ----------------

    def get_num_new_matched_tokens(self, request: "Request", num_computed_tokens: int) -> tuple[int | None, bool]:
        prompt = list(request.prompt_token_ids or [])
        segs = self._index.match(prompt)  # deterministic, so repeated calls are side-effect free
        served = seen_only(segs, self._seen) if self._require_seen else segs
        n = matched_tokens(served, len(prompt), num_computed_tokens, self._block_size)
        self._plans[request.request_id] = (segs, num_computed_tokens, n)
        return n, False

    def update_state_after_alloc(self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int):
        if num_external_tokens > 0:
            self._need_load.add(request.request_id)
            segs, start, _ = self._plans[request.request_id]
            self._plans[request.request_id] = (segs, start, num_external_tokens)

    def build_connector_meta(self, scheduler_output: "SchedulerOutput") -> KVConnectorMetadata:
        meta = ChunkReuseMetadata()
        for req in scheduler_output.scheduled_new_reqs:
            if req.req_id not in self._need_load:
                continue
            segs, start, n = self._plans[req.req_id]
            slots = slot_mapping(req.block_ids[0], start, n, self._block_size)
            meta.loads.append(LoadPlan(req.req_id, segs, start, n, slots))
            self.stats["requests"] += 1
            self.stats["hit_tokens"] += n
            self.stats["prompt_tokens"] += len(req.prompt_token_ids or [])
            if self.stats["requests"] % 100 == 0:
                logger.info("ChunkReuseConnector stats %s", self.stats)
        # Preempted-and-resumed requests recompute from scratch (no reload): simple and correct.
        self._need_load.clear()
        return meta

    def request_finished(self, request: "Request", block_ids: list[int]) -> tuple[bool, dict[str, Any] | None]:
        plan = self._plans.pop(request.request_id, None)
        if plan is not None and self._require_seen:
            self._seen.update(s.key for s in plan[0])  # its KV now "exists" for later requests
        return False, None

    # ---------------- worker side ----------------

    def _load(self, key: str) -> torch.Tensor:
        kv = self._kv.get(key)
        if kv is None:
            kv = load_file(str(self._store / self._index.entries[key]["file"]), device=self._kv_device)["kv"]
            if self._kv_device == "cpu":
                kv = kv.pin_memory()  # page-locked, so the per-load copy runs at full PCIe speed
            self._kv[key] = kv  # resident on kv_device after first use
        return kv

    def _get_kv(self, key: str) -> torch.Tensor:
        return self._load(key).to("cuda", non_blocking=True)  # no copy when kv_device is cuda

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None:
        meta = self._get_connector_metadata()
        assert isinstance(meta, ChunkReuseMetadata)
        if not meta.loads:
            return
        layers = []
        for name, layer in forward_context.no_compile_layers.items():
            cache = getattr(layer, "kv_cache", None)
            m = _LAYER.search(name)
            if cache is not None and m:
                layers.append((int(m.group(1)), cache))
        for plan in meta.loads:
            prof = self._profile and self._n_loads < 20
            if prof:
                torch.cuda.synchronize()
                t0, resident = time.perf_counter(), sum(s.key in self._kv for s in plan.segments)
            kv = assemble_packed(plan.segments, plan.start, plan.n, self._get_kv, self._inv_freq, self._head_dim)
            slots = plan.slots.to(kv.device)
            blocks, offsets = slots // self._block_size, slots % self._block_size
            for idx, cache in layers:
                if cache.shape[1:] != (kv.shape[2], self._block_size, kv.shape[3]):
                    raise RuntimeError(f"unexpected KV cache layout {tuple(cache.shape)}; "
                                       "expected (blocks, kv_heads, block_size, 2*head_dim)")
                cache[blocks, :, offsets] = kv[idx]
            if prof:
                torch.cuda.synchronize()
                logger.info("ChunkReuseConnector load %d: %d tokens, %d/%d segments resident, %.1f ms",
                            self._n_loads, plan.n, resident, len(plan.segments), (time.perf_counter() - t0) * 1e3)
            self._n_loads += 1
            if self._dump_dir and self._dumped < 4:  # E5: compare against the Transformers path
                Path(self._dump_dir).mkdir(parents=True, exist_ok=True)
                torch.save({"req_id": plan.req_id, "start": plan.start, "kv": kv.cpu(),
                            "segments": [s.__dict__ for s in plan.segments]},
                           Path(self._dump_dir) / f"load_{self._dumped}.pt")
                self._dumped += 1

    def wait_for_layer_load(self, layer_name: str) -> None:
        return

    def save_kv_layer(self, layer_name: str, kv_layer: torch.Tensor, attn_metadata, **kwargs: Any) -> None:
        return  # the store is built offline by scripts/export_store.py

    def wait_for_save(self):
        return
