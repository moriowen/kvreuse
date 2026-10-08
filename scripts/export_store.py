"""Write the C3 chunk store: KV of every chunk and system prefix, computed by the Transformers
path and saved in vLLM's packed layout (see kvreuse/connector_logic.py for the format).

    python scripts/export_store.py --out store/ --samples conv-26      # one conversation (~2.8 GB)
    python scripts/export_store.py --out store/                         # all ten (~35 GB)

Chunks are prefilled after [BOS][INST] exactly as ChunkStore does, so the connector serves
the same bytes the Transformers re-positioning baseline uses.
"""

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from kvreuse import locomo
from kvreuse.connector_logic import segment_key, to_packed
from kvreuse.device import pick_device
from kvreuse.forward import ModelView
from kvreuse.prompt import PromptBuilder
from kvreuse.store import ChunkStore


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mistralai/Mistral-7B-Instruct-v0.3")
    ap.add_argument("--chunks", default="data/chunks.json")
    ap.add_argument("--samples", nargs="*", help="sample ids to export (default: all)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--allow-cpu", action="store_true")
    args = ap.parse_args()

    dev = pick_device(args.allow_cpu)
    tok = AutoTokenizer.from_pretrained(args.model)
    pb = PromptBuilder(tok)
    dtype = torch.bfloat16 if dev.type == "cuda" else torch.float32
    view = ModelView(AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(dev).eval())
    data = json.loads(Path(args.chunks).read_text())
    assert data["tokenizer"] == args.model, "chunks were tokenized with a different tokenizer"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    context = pb.chunk_context()
    store = ChunkStore(view, context)
    entries, total = {}, 0
    for sid, conv in data["conversations"].items():
        if args.samples and sid not in args.samples:
            continue
        system = locomo.CONV_START_PROMPT.format(*conv["speakers"])
        segs = [("prefix", pb.prefix(system))] + [("chunk", c["token_ids"]) for c in conv["chunks"]]
        for kind, ids in segs:
            key = segment_key(kind, context, ids)
            if key in entries:
                continue
            e = store.prefix(ids) if kind == "prefix" else store.chunk(ids)
            save_file({"kv": to_packed(e)}, str(out / f"{key}.safetensors"))
            entries[key] = {"kind": kind, "start": e.start, "ids": list(ids), "file": f"{key}.safetensors",
                            "sample_id": sid}
            total += len(ids)
            store._chunks.clear()  # keep GPU memory flat; entries are on disk now
            store._prefixes.clear()
        print(f"{sid}: {len(entries)} entries so far, {total} tokens")

    meta = {
        "model": args.model, "dtype": str(dtype), "n_layers": view.n_layers, "n_kv_heads": view.n_kv,
        "head_dim": view.head_dim, "inv_freq": view.inv_freq.float().cpu().tolist(),
        "chunk_context": context, "layout": "[layers, T, n_kv_heads, 2*head_dim], K post-RoPE then V",
        "entries": entries,
    }
    (out / "meta.json").write_text(json.dumps(meta))
    print(f"wrote {len(entries)} entries ({total} tokens, ~{total * 128 / 1024**2:.1f} GiB) to {out}")


if __name__ == "__main__":
    main()
