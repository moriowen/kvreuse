"""E0 on the real model and real LoCoMo prompts (run on the GPU node).

    python scripts/e0_real.py --n 20

fp32 gate (TF32 off): our forward == HF forward, 100% recompute == full prefill, and the
identity case == full prefill. Then the bf16 noise floor: how far bf16 full prefill is
from fp32 full prefill (KL and top-1 agreement on the first answer token). Reuse error in
E1 is read against that floor. 7B in fp32 needs ~28 GB, so both copies fit on an A100-80GB.
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from kvreuse import locomo
from kvreuse.device import pick_device
from kvreuse.forward import ModelView, full_prefill, selective_prefill
from kvreuse.methods import FullRecompute, Reposition
from kvreuse.prompt import PromptBuilder
from kvreuse.store import ChunkStore, assemble


def kl(p_logits, q_logits):
    return float(F.kl_div(F.log_softmax(q_logits, -1), F.log_softmax(p_logits, -1), log_target=True,
                          reduction="sum"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mistralai/Mistral-7B-Instruct-v0.3")
    ap.add_argument("--chunks", default="data/chunks.json")
    ap.add_argument("--retrieval", default="data/retrieval_k10.json")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--atol", type=float, default=1e-3, help="max-abs logit tolerance, fp32")
    ap.add_argument("--allow-cpu", action="store_true", help="smoke tests with a tiny model only")
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dev = pick_device(args.allow_cpu)
    tok = AutoTokenizer.from_pretrained(args.model)
    pb = PromptBuilder(tok)
    convs = json.loads(Path(args.chunks).read_text())["conversations"]
    by_id = {c["id"]: c for conv in convs.values() for c in conv["chunks"]}
    qs = json.loads(Path(args.retrieval).read_text())["questions"][: args.n]

    m32 = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).to(dev).eval()
    v32 = ModelView(m32)
    store = ChunkStore(v32, pb.chunk_context())
    worst = {"hf": 0.0, "full_recompute": 0.0, "identity": 0.0}
    full32 = []
    for q in qs:
        conv = convs[q["sample_id"]]
        system = locomo.CONV_START_PROMPT.format(*conv["speakers"])
        qq = locomo.Question(**{k: q[k] for k in ("sample_id", "qa_index", "category", "question", "answer",
                                                   "evidence")})
        chunk_ids = [by_id[c]["token_ids"] for c in q["chunk_ids"]]
        q_ids = pb.question(qq.prompt())
        x, K0, V0, layout = assemble(v32, store.prefix(pb.prefix(system)), [store.chunk(c) for c in chunk_ids], q_ids)
        ref = full_prefill(v32, x).last_logits
        full32.append((x.cpu(), ref.cpu()))
        with torch.no_grad():
            hf = m32(x[None]).logits[0, -1].float()
        fr = selective_prefill(v32, x, K0, V0, layout, FullRecompute()).last_logits
        # identity: cache the first chunk after the real prefix, place it right after the prefix
        ident_store = ChunkStore(v32, pb.prefix(system))
        xi, Ki, Vi, li = assemble(v32, ident_store.prefix(pb.prefix(system)), [ident_store.chunk(chunk_ids[0])], q_ids)
        idl = selective_prefill(v32, xi, Ki, Vi, li, Reposition()).last_logits
        idref = full_prefill(v32, xi).last_logits
        worst["hf"] = max(worst["hf"], (ref - hf).abs().max().item())
        worst["full_recompute"] = max(worst["full_recompute"], (ref - fr).abs().max().item())
        worst["identity"] = max(worst["identity"], (idref - idl).abs().max().item())
    print("fp32 worst max-abs logit error:", worst)
    gate = all(v <= args.atol for v in worst.values())
    print("E0 fp32 gate:", "PASS" if gate else "FAIL", f"(atol {args.atol})")

    del m32, v32, store
    torch.cuda.empty_cache() if dev.type == "cuda" else None
    vb = ModelView(AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to(dev).eval())
    kls, agree = [], []
    for x, ref in full32:
        lb = full_prefill(vb, x.to(dev)).last_logits.cpu()
        kls.append(kl(ref, lb))
        agree.append(float(ref.argmax() == lb.argmax()))
    print(f"bf16 noise floor over {len(kls)} prompts: mean KL {sum(kls)/len(kls):.3e}, "
          f"max KL {max(kls):.3e}, top-1 agreement {sum(agree)/len(agree):.3f}")
    if not gate:
        raise SystemExit(1)  # nonzero exit stops dependent Slurm jobs


if __name__ == "__main__":
    main()
