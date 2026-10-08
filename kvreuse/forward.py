"""Manual per-layer forward pass over an HF Mistral model.

We drive ``model.model.layers[l]`` ourselves on plain tensors instead of calling
``model(...)`` with a ``DynamicCache``, because selective recompute needs:

* an arbitrary (non-contiguous) set of query tokens per layer,
* scatter writes of fresh K/V into an assembled cache *before* attention,
* an explicit mask ``kv_pos <= q_pos``. HF's SDPA path with ``attention_mask=None``
  assumes contiguous queries and crops K/V, which is silently wrong here.

The full-prefill reference uses this same code, so kernel differences never count as
reuse error. E0(c) checks this code against ``model(ids).logits``.

KV tensors are per layer ``[1, n_kv_heads, N, head_dim]``, keys post-RoPE (same as HF and vLLM).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from .rope import apply_rope

# SDPA backends we allow. cuDNN's SDPA is left out by default: it builds a new execution plan
# for every new sequence length, and every LoCoMo prompt has a different length. Flash serves
# the causal full-prefill path, memory-efficient attention serves explicit masks, and math is
# the CPU fallback. scripts/profile_prefill.py --sdpa default measures the difference.
SDPA_BACKENDS = [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]


@dataclass
class ModelView:
    """The pieces of an HF Mistral/Llama-style model the manual forward needs."""

    model: Any  # HF *ForCausalLM; typed Any because nn.Module.__getattr__ returns Tensor | Module

    def __post_init__(self):
        m: Any = self.model.model
        cfg: Any = self.model.config
        self.layers = m.layers
        self.embed = m.embed_tokens
        self.norm = m.norm
        self.lm_head = self.model.lm_head
        self.rotary = m.rotary_emb
        self.n_layers = cfg.num_hidden_layers
        self.n_heads = cfg.num_attention_heads
        self.n_kv = cfg.num_key_value_heads
        self.head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
        self.sliding_window = getattr(cfg, "sliding_window", None)
        if getattr(self.rotary, "attention_scaling", 1.0) != 1.0:
            raise NotImplementedError("RoPE variants with attention scaling (e.g. YaRN) not supported")
        if self.rotary.rope_type not in ("default", "llama3", "linear"):
            # dynamic NTK changes frequencies with length, so delta rotation is not exact
            raise NotImplementedError(f"rope_type {self.rotary.rope_type!r} not supported")

    @property
    def inv_freq(self) -> torch.Tensor:
        return self.rotary.inv_freq  # use the model's own frequencies, never recompute theta**(-2i/d)

    @property
    def device(self) -> torch.device:
        return self.embed.weight.device

    @property
    def dtype(self) -> torch.dtype:
        return self.embed.weight.dtype

    def check_length(self, n: int):
        if self.sliding_window is not None and n > self.sliding_window:
            raise ValueError(f"sequence of {n} tokens exceeds sliding window {self.sliding_window}")

    # -- one layer, split in two so a plan can act between QKV and attention --

    def qkv(self, l: int, h: torch.Tensor, pos: torch.Tensor):
        """h: [1, s, hidden]; pos: [s] absolute positions. Returns post-RoPE q, k and v."""
        layer = self.layers[l]
        attn = layer.self_attn
        x = layer.input_layernorm(h)
        s = x.shape[1]
        q = attn.q_proj(x).view(1, s, -1, self.head_dim).transpose(1, 2)
        k = attn.k_proj(x).view(1, s, -1, self.head_dim).transpose(1, 2)
        v = attn.v_proj(x).view(1, s, -1, self.head_dim).transpose(1, 2)
        cos, sin = self.rotary(x, pos[None])  # same numerics as HF: fp32 angles, cast to model dtype
        cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
        return apply_rope(q, cos, sin), apply_rope(k, cos, sin), v

    def attn_mlp(self, l, h, q, K, V, q_pos, kv_pos, causal: bool = False):
        """Attention of queries q (at q_pos) over the full K/V (at kv_pos), then the MLP.

        ``causal=True`` is only for a contiguous q == kv sequence starting at 0 (full
        prefill): it skips the explicit [s, N] mask so long prompts can use fused kernels.
        """
        layer = self.layers[l]
        attn = layer.self_attn
        rep = self.n_heads // self.n_kv
        mask = None if causal else (kv_pos[None, :] <= q_pos[:, None])[None, None]  # True = may attend
        with sdpa_kernel(SDPA_BACKENDS):
            o = F.scaled_dot_product_attention(
                q,
                K.repeat_interleave(rep, dim=1),
                V.repeat_interleave(rep, dim=1),
                attn_mask=mask,
                is_causal=causal,
                scale=attn.scaling,
            )
        o = o.transpose(1, 2).reshape(1, q.shape[2], -1)
        h = h + attn.o_proj(o)
        return h + layer.mlp(layer.post_attention_layernorm(h))

    def logits(self, h_last: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.norm(h_last)).float()


@dataclass
class Layout:
    """Where each segment of an assembled prompt sits. Spans are [start, end)."""

    n: int
    prefix_len: int
    chunk_spans: list[tuple[int, int]]
    question_span: tuple[int, int]

    def reusable_mask(self, device=None) -> torch.Tensor:
        m = torch.zeros(self.n, dtype=torch.bool, device=device)
        for a, b in self.chunk_spans:
            m[a:b] = True
        return m

    @property
    def n_reusable(self) -> int:
        return sum(b - a for a, b in self.chunk_spans)

    def question_idx(self, device=None) -> torch.Tensor:
        return torch.arange(*self.question_span, device=device)


class Plan(Protocol):
    """A correction method. Only decides which tokens are recomputed (and, for offset
    correction, how non-recomputed KV is adjusted)."""

    def initial_set(self, layout: Layout, device) -> torch.Tensor:
        """Sorted positions computed at layer 0. Must include the whole question."""
        ...

    def on_layer(self, l, S, K, V, K0, V0, layout) -> torch.Tensor | None:
        """Called after fresh K/V for S were written into K, V (in place allowed).
        Return a sorted subset of S to continue with (S can only shrink), or None."""
        ...


@dataclass
class PrefillResult:
    K: list[torch.Tensor]
    V: list[torch.Tensor]
    last_logits: torch.Tensor  # [vocab], fp32
    qkv_tokens: list[int] = field(default_factory=list)  # reusable tokens with fresh QKV, per layer
    attn_tokens: list[int] = field(default_factory=list)  # reusable tokens through attention+MLP, per layer

    def budget(self, n_reusable: int) -> float:
        """Recompute ratio: sum_l |S_l ∩ reusable| / (L * N_reusable). Selection layers are
        charged in full (QKV for every token counts as recomputing it)."""
        if n_reusable == 0:
            return 0.0
        return sum(self.qkv_tokens) / (len(self.qkv_tokens) * n_reusable)


@torch.no_grad()
def full_prefill(view: ModelView, ids: torch.Tensor) -> PrefillResult:
    """Ordinary prefill of ``ids`` (1-D LongTensor) with this module's code path."""
    n = ids.shape[0]
    view.check_length(n)
    pos = torch.arange(n, device=view.device)
    h = view.embed(ids[None])
    Ks, Vs = [], []
    for l in range(view.n_layers):
        q, k, v = view.qkv(l, h, pos)
        h = view.attn_mlp(l, h, q, k, v, pos, pos, causal=True)
        Ks.append(k)
        Vs.append(v)
    return PrefillResult(Ks, Vs, view.logits(h[0, -1]), [n] * view.n_layers, [n] * view.n_layers)


@torch.no_grad()
def selective_prefill(
    view: ModelView,
    ids: torch.Tensor,
    K0: torch.Tensor,
    V0: torch.Tensor,
    layout: Layout,
    plan: Plan,
) -> PrefillResult:
    """Prefill with reused KV ``K0, V0`` ([L, n_kv, N, D]; question positions arbitrary),
    recomputing only the tokens ``plan`` selects. The last token must be in the question."""
    n = ids.shape[0]
    view.check_length(n)
    dev = view.device
    pos = torch.arange(n, device=dev)
    reusable = layout.reusable_mask(dev)
    S = plan.initial_set(layout, dev)
    if S[-1].item() != n - 1:
        raise ValueError("the last prompt token must be recomputed")
    h = view.embed(ids[S][None])
    Ks, Vs, qkv_tok, attn_tok = [], [], [], []
    for l in range(view.n_layers):
        q, k, v = view.qkv(l, h, pos[S])
        K = K0[l : l + 1].clone()
        V = V0[l : l + 1].clone()
        K[:, :, S] = k  # write fresh KV before attention so recomputed tokens see each other
        V[:, :, S] = v
        qkv_tok.append(int(reusable[S].sum()))
        keep = plan.on_layer(l, S, K, V, K0[l : l + 1], V0[l : l + 1], layout)
        if keep is not None:
            idx = torch.searchsorted(S, keep)
            if not torch.equal(S[idx], keep):
                raise ValueError("plan returned positions outside the current set")
            S, h, q = keep, h[:, idx], q[:, :, idx]
        attn_tok.append(int(reusable[S].sum()))
        h = view.attn_mlp(l, h, q, K, V, pos[S], pos)
        Ks.append(K)
        Vs.append(V)
    return PrefillResult(Ks, Vs, view.logits(h[0, -1]), qkv_tok, attn_tok)


@torch.no_grad()
def greedy_decode(
    view: ModelView,
    prefill: PrefillResult,
    n_prompt: int,
    max_new_tokens: int,
    stop_ids: set[int],
) -> list[int]:
    """Greedy decoding on top of a (possibly approximate) prefilled cache."""
    Ks, Vs = list(prefill.K), list(prefill.V)
    out: list[int] = []
    tok = int(prefill.last_logits.argmax())
    for i in range(max_new_tokens):
        if tok in stop_ids:
            break
        out.append(tok)
        if i == max_new_tokens - 1:
            break
        p = n_prompt + i
        pos1 = torch.tensor([p], device=view.device)
        kv_pos = torch.arange(p + 1, device=view.device)
        h = view.embed(torch.tensor([[tok]], device=view.device))
        for l in range(view.n_layers):
            q, k, v = view.qkv(l, h, pos1)
            Ks[l] = torch.cat([Ks[l], k], dim=2)
            Vs[l] = torch.cat([Vs[l], v], dim=2)
            h = view.attn_mlp(l, h, q, Ks[l], Vs[l], pos1, kv_pos)
        tok = int(view.logits(h[0, -1]).argmax())
    return out
