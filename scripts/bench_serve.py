"""C5: drive a running ``vllm serve`` with a token-id workload and record TTFT, latency and
throughput (E3, E4).

    python scripts/bench_serve.py --url http://localhost:8100 --workload work/e3_k6.jsonl \
        --concurrency 1 --out results/serve/e3_k6_connector_r0.jsonl --tag config=connector k=6

Prompts go to /v1/completions as lists of token ids, so the server sees exactly the ids the
chunk store was built from (vLLM's own benchmark tool sends text, which would be re-tokenized
and could break chunk matching at the joins). Responses are streamed; TTFT is the time to the
first streamed token. ``cached_tokens`` is what vLLM reports as already computed for the prompt
(needs ``--enable-prompt-tokens-details``): prefix-cache hits, or the connector's external hits.
Requests are sent in workload order by a closed loop of ``concurrency`` workers.
"""

import argparse
import asyncio
import json
import random
import statistics
import time
from pathlib import Path

import aiohttp


async def one(session, url, model, ids, max_tokens):
    body = {"model": model, "prompt": ids, "max_tokens": max_tokens, "temperature": 0.0, "stream": True,
            "stream_options": {"include_usage": True}}
    t0 = time.perf_counter()
    ttft, n_out, cached, text = None, 0, None, []
    async with session.post(f"{url}/v1/completions", json=body) as resp:
        resp.raise_for_status()
        async for raw in resp.content:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            msg = json.loads(data)
            for c in msg.get("choices") or []:
                if c.get("text"):
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    text.append(c["text"])
            usage = msg.get("usage")
            if usage:
                n_out = usage.get("completion_tokens", 0)
                cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
    end = time.perf_counter() - t0
    return {"ttft_ms": None if ttft is None else ttft * 1e3, "latency_ms": end * 1e3, "out_tokens": n_out,
            "cached_tokens": cached, "text": "".join(text)}


async def run(args, reqs, model):
    timeout = aiohttp.ClientTimeout(total=600)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        rng = random.Random(1234)  # warm-up prompts: random ids, so no cache of any kind helps
        for _ in range(args.warmup):
            await one(session, args.url, model, [1] + [rng.randrange(1000, 30000) for _ in range(512)], 8)

        queue = asyncio.Queue()
        for i, r in enumerate(reqs):
            queue.put_nowait((i, r))
        rows = [None] * len(reqs)

        async def worker():
            while not queue.empty():
                i, r = queue.get_nowait()
                t_send = time.perf_counter() - t_start
                res = await one(session, args.url, model, r["prompt_ids"], args.max_tokens)
                rows[i] = {"i": i, "id": r["id"], "prompt_tokens": len(r["prompt_ids"]),
                           "repeats": r.get("repeats"), "t_send_s": t_send, **res}

        t_start = time.perf_counter()
        await asyncio.gather(*[worker() for _ in range(args.concurrency)])
        wall = time.perf_counter() - t_start
    return rows, wall


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))] if xs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--workload", required=True)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--tag", nargs="*", default=[], help="key=value pairs stored in the summary")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    reqs = [json.loads(line) for line in Path(args.workload).read_text().splitlines() if line.strip()]
    reqs = reqs[: args.limit] if args.limit else reqs

    async def model_name():
        async with aiohttp.ClientSession() as s:
            async with s.get(f"{args.url}/v1/models") as r:
                return (await r.json())["data"][0]["id"]

    model = asyncio.run(model_name())
    rows, wall = asyncio.run(run(args, reqs, model))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    ttft = [r["ttft_ms"] for r in rows if r["ttft_ms"] is not None]
    lat = [r["latency_ms"] for r in rows]
    cached = [r["cached_tokens"] for r in rows if r["cached_tokens"] is not None]
    wl_meta = Path(args.workload).with_suffix(".meta.json")
    summary = {
        "name": out.stem, "tags": dict(t.split("=", 1) for t in args.tag), "model": model,
        "n": len(rows), "concurrency": args.concurrency, "max_tokens": args.max_tokens,
        "ttft_ms_median": statistics.median(ttft) if ttft else None,
        "ttft_ms_p90": pct(ttft, 0.9), "ttft_ms_p99": pct(ttft, 0.99),
        "latency_ms_median": statistics.median(lat), "wall_s": wall, "req_per_s": len(rows) / wall,
        "out_tok_per_s": sum(r["out_tokens"] for r in rows) / wall,
        "mean_prompt_tokens": statistics.mean(r["prompt_tokens"] for r in rows),
        "cached_fraction": sum(cached) / sum(r["prompt_tokens"] for r in rows if r["cached_tokens"] is not None)
        if cached else None,
        "workload": str(args.workload),
        "workload_meta": json.loads(wl_meta.read_text()) if wl_meta.exists() else None,
    }
    out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "workload_meta"}))


if __name__ == "__main__":
    main()
