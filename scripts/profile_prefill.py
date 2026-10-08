"""Where does prefill time go? Profiles full prefill vs. re-positioning on one real prompt.

    python scripts/profile_prefill.py --question 0

Prints CUDA-event timings (median of --iters after warm-up) and torch.profiler tables
sorted by GPU time and by CPU self time. If GPU time is far below wall time, the loop is
bound by kernel launches or host syncs, not by the model.
"""

import argparse
import json
import statistics
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile
from transformers import AutoModelForCausalLM, AutoTokenizer

from kvreuse import forward, locomo
from kvreuse.device import pick_device
from kvreuse.forward import ModelView, full_prefill, selective_prefill
from kvreuse.methods import Reposition
from kvreuse.prompt import PromptBuilder
from kvreuse.store import ChunkStore, assemble


def cuda_ms(fn, iters):
    times = []
    for _ in range(iters):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        torch.cuda.synchronize()
        times.append(a.elapsed_time(b))
    return statistics.median(times)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mistralai/Mistral-7B-Instruct-v0.3")
    ap.add_argument("--chunks", default="data/chunks.json")
    ap.add_argument("--retrieval", default="data/retrieval_k10.json")
    ap.add_argument("--question", type=int, default=0)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--allow-cpu", action="store_true")
    ap.add_argument("--sdpa", default="no-cudnn", choices=["no-cudnn", "default"],
                    help="default = let PyTorch pick (may choose cuDNN)")
    ap.add_argument("--shapes", type=int, default=12, help="distinct prompts for the varying-shape test")
    args = ap.parse_args()

    dev = pick_device(args.allow_cpu)
    if args.sdpa == "default":
        from torch.nn.attention import SDPBackend
        forward.SDPA_BACKENDS = list(SDPBackend.__members__.values())
        forward.SDPA_BACKENDS = [b for b in forward.SDPA_BACKENDS if b not in (SDPBackend.ERROR, SDPBackend.OVERRIDEABLE)]
    print("SDPA backends:", [b.name for b in forward.SDPA_BACKENDS])
    tok = AutoTokenizer.from_pretrained(args.model)
    pb = PromptBuilder(tok)
    view = ModelView(AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to(dev).eval())
    convs = json.loads(Path(args.chunks).read_text())["conversations"]
    by_id = {c["id"]: c for conv in convs.values() for c in conv["chunks"]}
    questions = json.loads(Path(args.retrieval).read_text())["questions"]
    store = ChunkStore(view, pb.chunk_context())

    def build(q):
        conv = convs[q["sample_id"]]
        system = locomo.CONV_START_PROMPT.format(*conv["speakers"])
        qq = locomo.Question(**{k: q[k] for k in ("sample_id", "qa_index", "category", "question", "answer",
                                                   "evidence")})
        return assemble(view, store.prefix(pb.prefix(system)),
                        [store.chunk(by_id[c]["token_ids"]) for c in q["chunk_ids"]], pb.question(qq.prompt()))

    x, K0, V0, layout = build(questions[args.question])
    n_q = layout.question_span[1] - layout.question_span[0]
    print(f"prompt {layout.n} tokens, {layout.n_reusable} reusable, {n_q} question tokens, "
          f"GPU {torch.cuda.get_device_name()}")

    runs = {
        "full": lambda: full_prefill(view, x),
        "reposition": lambda: selective_prefill(view, x, K0, V0, layout, Reposition()),
        # same number of tokens as reposition, but contiguous and with no reused cache:
        "question_only_fresh": lambda: full_prefill(view, x[-n_q:]),
    }
    for name, fn in runs.items():
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        print(f"{name:22s} median {cuda_ms(fn, args.iters):8.2f} ms")

    # Varying shapes, as in run_eval: each prompt has a new length and runs once.
    built, seen = [], {layout.n}
    for q in questions[args.question + 1:]:
        b = build(q)
        if b[3].n not in seen:
            seen.add(b[3].n)
            built.append(b)
        if len(built) == args.shapes:
            break
    for name in ("full", "reposition"):
        times = []
        for xi, Ki, Vi, li in built:
            fn = (lambda: full_prefill(view, xi)) if name == "full" else (
                lambda: selective_prefill(view, xi, Ki, Vi, li, Reposition()))
            times.append(cuda_ms(fn, 1))
        print(f"{name:22s} new shape each call: median {statistics.median(times):8.2f} ms "
              f"over {len(times)} lengths {sorted(b[3].n for b in built)[:3]}...")

    for name in ("full", "reposition"):
        fn = runs[name]
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for _ in range(3):
                fn()
            torch.cuda.synchronize()
        ka = prof.key_averages()
        gpu_us = sum(e.self_device_time_total for e in ka) / 3
        n_kernels = sum(e.count for e in ka if e.self_device_time_total > 0) / 3
        print(f"\n=== {name}: GPU busy {gpu_us/1000:.2f} ms per call, ~{n_kernels:.0f} GPU ops per call ===")
        print(ka.table(sort_by="self_device_time_total", row_limit=12))
        print(ka.table(sort_by="self_cpu_time_total", row_limit=12))


if __name__ == "__main__":
    main()
