"""T024 paired runner: whole-EN-MLX vs whole-MULTI-MLX vs ANE-chunks vs MLX-chunks.

Per doc (sequential, Jev impossible — explicit backends everywhere):
  E: whole state, backend=mlx, model=english          (production reference)
  M: whole state, backend=mlx, model=multilingual     (checkpoint control)
  A: each chunk, backend=ane, model=multilingual      (experiment arm)
  C: each chunk, backend=mlx, model=multilingual      (same chunks: isolates
     backend numeric drift from context/aggregation loss)

Aggregation (choice): hard majority (ties abstain) + pooled mean argmax.
Diagnostics per doc: vote_entropy, vote_margin, pooled_margin,
between_chunk_js, within_chunk_entropy, pooled would-escalate.
Latencies: per-call server ms + per-arm sums (k-dependent totals).

Usage: chunk_run.py --chunks datasets/agnews_chunks.jsonl
         --out results/chunks_paired.jsonl [--limit N] [--resume]
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import httpx

LABELS = ["World", "Sports", "Business", "Sci/Tech"]
INSTRUCTIONS = "Classify the news article by topic."
TIMEOUT = 90.0


def call(client, url, token, state, backend, model):
    t0 = time.perf_counter()
    try:
        r = client.post(
            url,
            json={
                "state": state,
                "questions": {
                    "q": {
                        "type": "choice",
                        "instructions": INSTRUCTIONS,
                        "criteria": {label: None for label in LABELS},
                    }
                },
                "backend": backend,
                "model": model,
            },
            headers={"Authorization": f"Bearer {token}"},
            timeout=TIMEOUT,
        )
    except httpx.HTTPError as e:
        return {"ok": False, "error": f"transport: {type(e).__name__}"}
    dt = (time.perf_counter() - t0) * 1000
    if r.status_code != 200:
        return {"ok": False, "error": f"http-{r.status_code}: {r.text[:160]}"}
    try:
        data = r.json()
        ans = data["answers"]["q"]
        probs = [float(ans["probabilities"][lab]) for lab in LABELS]
        if not all(math.isfinite(p) for p in probs):
            return {"ok": False, "error": "non-finite probs"}
        return {
            "ok": True,
            "predicted": ans["choice"],
            "probs": probs,
            "confidence": ans["confidence"],
            "margin": data.get("routing", {}).get("margin"),
            "backend": data.get("routing", {}).get("backend"),
            "checkpoint": data.get("routing", {}).get("checkpoint"),
            "token_count": data.get("routing", {}).get("token_count"),
            "latency_ms": round(dt, 1),
            "server_ms": data.get("routing", {}).get("latency_ms"),
        }
    except (KeyError, TypeError, ValueError) as e:
        return {"ok": False, "error": f"bad-payload: {type(e).__name__}"}


def entropy(ps):
    h = 0.0
    for p in ps:
        if p > 0:
            h -= p * math.log(p)
    return h


def aggregate(chunk_results):
    """Majority + pooled-mean over per-chunk prob vectors (uniform weights)."""
    ok = [c for c in chunk_results if c["ok"]]
    k = len(chunk_results)
    votes = [LABELS.index(c["predicted"]) for c in ok]
    counts = [votes.count(j) for j in range(len(LABELS))]
    top = max(counts)
    winners = [j for j, c in enumerate(counts) if c == top]
    majority = None if len(winners) != 1 else LABELS[winners[0]]
    pooled = [sum(c["probs"][j] for c in ok) / max(1, len(ok)) for j in range(len(LABELS))]
    pooled_pred = LABELS[max(range(len(LABELS)), key=lambda j: pooled[j])]
    hp = entropy(pooled) / math.log(len(LABELS))
    s = sorted(pooled, reverse=True)
    # vote shares for entropy/margin over winners
    v = [c / max(1, len(ok)) for c in counts]
    ve = entropy(v) / math.log(k) if k > 1 else 0.0
    vs = sorted(v, reverse=True)
    js = entropy(pooled) - sum(entropy(c["probs"]) for c in ok) / max(1, len(ok))
    within = sum(entropy(c["probs"]) for c in ok) / max(1, len(ok))
    conf = 1.0 - hp
    margin = s[0] - s[1]
    return {
        "majority": majority,
        "tie": len(winners) != 1,
        "pooled": pooled_pred,
        "pooled_probs": [round(p, 4) for p in pooled],
        "pooled_confidence": round(conf, 4),
        "pooled_margin": round(margin, 4),
        "vote_entropy": round(ve, 4),
        "vote_margin": round(vs[0] - vs[1], 4),
        "between_chunk_js": round(js, 4),
        "within_chunk_entropy": round(within, 4),
        "would_escalate": bool(conf < 0.75 or margin < 0.20),
        "failed_chunks": k - len(ok),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base-url", default="http://127.0.0.1:8768")
    ap.add_argument("--token", default="bench-t024")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    docs = [json.loads(l) for l in open(args.chunks, encoding="utf-8") if l.strip()]
    # Reattach whole texts from the source corpus.
    corpus = {}
    for l in open(Path(args.chunks).parent / "agnews_test.jsonl", encoding="utf-8"):
        r = json.loads(l)
        corpus[r["id"]] = r["text"]
    if args.limit:
        docs = docs[: args.limit]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if args.resume and out_path.exists():
        for l in open(out_path, encoding="utf-8"):
            try:
                done.add(json.loads(l)["id"])
            except (ValueError, KeyError):
                pass
    todo = [d for d in docs if d["id"] not in done]
    print(f"{len(docs)} docs, {len(done)} done, {len(todo)} to run", flush=True)

    url = args.base_url.rstrip("/") + "/predict"
    mode = "a" if args.resume else "w"
    t0 = time.perf_counter()
    n = 0
    with httpx.Client(http2=False) as client, open(out_path, mode, encoding="utf-8") as f:
        for doc in todo:
            text = corpus[doc["id"]]
            rec = {"id": doc["id"], "label": doc["label"], "k": doc["k"],
                   "fitting_control": doc["fitting_control"], "arms": {}}
            rec["arms"]["whole_en_mlx"] = call(client, url, args.token, text, "mlx", "english")
            rec["arms"]["whole_multi_mlx"] = call(client, url, args.token, text, "mlx", "multilingual")
            ane_chunks, mlx_chunks = [], []
            for c in doc["chunks"]:
                ane_chunks.append(call(client, url, args.token, c["text"], "ane", "multilingual"))
                mlx_chunks.append(call(client, url, args.token, c["text"], "mlx", "multilingual"))
            rec["arms"]["ane_chunks"] = ane_chunks
            rec["arms"]["mlx_chunks"] = mlx_chunks
            rec["agg_ane"] = aggregate(ane_chunks)
            rec["agg_mlx_chunks"] = aggregate(mlx_chunks)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            n += 1
            if n % 50 == 0:
                print(f"  {n}/{len(todo)}", flush=True)
    print(f"done {n} docs in {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    sys.exit(main())
