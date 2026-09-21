"""Consumer-level sweep runner: samples JSONL -> live jevalaya /predict.

Each sample becomes one single-`choice` question (dict-form criteria, bare
labels). Local-only by construction: requests carry an explicit backend
(ane|mlx) and the bench server runs with Jev disabled, so no sweep can
spend API fees. Records what WOULD have escalated (confidence < 0.75 or
margin < 0.20, the server defaults) without calling Jev.

Usage:
    sweep.py --samples datasets/agnews_test.jsonl --out results/agnews_mlx.jsonl \\
        --backend mlx --limit 100
    sweep.py --samples ... --out ... --resume   # skip ids already recorded

Output lines: {id, dataset, label, predicted, correct, backend, checkpoint,
confidence, margin, token_count, ane_eligible, escalated, would_escalate,
escalate_triggers, latency_ms (client), server_latency_ms, reason, error}.
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

CONF_THRESHOLD = 0.75
MARGIN_THRESHOLD = 0.20
RETRY_STATUSES = {429, 503}
MAX_ATTEMPTS = 4


def build_body(sample, question_id, instructions, labels, backend, model):
    body = {
        "state": sample["text"],
        "questions": {
            question_id: {
                "type": "choice",
                "instructions": instructions,
                "criteria": {label: None for label in labels},
            }
        },
        "backend": backend,
    }
    if model:
        body["model"] = model  # force checkpoint (e.g. multilingual for ANE parity)
    return body


def post_once(client, url, token, body, timeout):
    t0 = time.perf_counter()
    try:
        r = client.post(
            url,
            json=body,
            headers={"Authorization": f"Bearer {token}"},
            timeout=timeout,
        )
    except httpx.HTTPError as e:
        return None, 0, f"transport: {type(e).__name__}"
    latency_ms = (time.perf_counter() - t0) * 1000
    if r.status_code in RETRY_STATUSES:
        return None, latency_ms, f"retryable-http-{r.status_code}"
    if r.status_code != 200:
        try:
            detail = r.json()
        except ValueError:
            detail = r.text[:200]
        return None, latency_ms, f"http-{r.status_code}: {json.dumps(detail)[:200]}"
    try:
        return r.json(), latency_ms, ""
    except ValueError:
        return None, latency_ms, "unreadable-json"


def run_sample(client, url, token, sample, args, labels):
    body = build_body(sample, args.question_id, args.instructions, labels, args.backend, args.model)
    data, latency_ms, error = None, 0.0, ""
    for attempt in range(MAX_ATTEMPTS):
        data, latency_ms, error = post_once(client, url, token, body, args.timeout)
        if data is not None or not error.startswith("retryable"):
            break
        time.sleep(0.5 * (2**attempt))
    rec = {
        "id": sample["id"],
        "dataset": args.dataset,
        "label": sample["label"],
        "predicted": None,
        "correct": False,
        "backend": None,
        "checkpoint": None,
        "confidence": None,
        "margin": None,
        "token_count": None,
        "ane_eligible": None,
        "escalated": None,
        "would_escalate": None,
        "escalate_triggers": [],
        "latency_ms": round(latency_ms, 1),
        "server_latency_ms": None,
        "reason": "",
        "error": error,
    }
    if data is None:
        return rec
    try:
        answer = data["answers"][args.question_id]
        routing = data.get("routing", {})
        predicted = answer.get("choice")
        conf = routing.get("confidence")
        margin = routing.get("margin")
        triggers = []
        if isinstance(conf, (int, float)) and conf < CONF_THRESHOLD:
            triggers.append("low_confidence")
        if isinstance(margin, (int, float)) and margin < MARGIN_THRESHOLD:
            triggers.append("low_margin")
        rec.update(
            predicted=predicted,
            correct=predicted == sample["label"],
            backend=routing.get("backend"),
            checkpoint=routing.get("checkpoint"),
            confidence=conf,
            margin=margin,
            token_count=routing.get("token_count"),
            ane_eligible=routing.get("ane_eligible"),
            escalated=routing.get("escalated"),
            would_escalate=bool(triggers),
            escalate_triggers=triggers,
            server_latency_ms=routing.get("latency_ms"),
            reason=routing.get("reason", ""),
            error="",
        )
    except (KeyError, TypeError) as e:
        rec["error"] = f"bad-payload: {type(e).__name__}"
    return rec


def load_done(out_path):
    done = set()
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["id"])
                except (ValueError, KeyError):
                    pass
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dataset", default="agnews")
    ap.add_argument("--question-id", default="topic")
    ap.add_argument("--instructions", default="Classify the news article by topic.")
    ap.add_argument("--labels", nargs="+", default=["World", "Sports", "Business", "Sci/Tech"])
    ap.add_argument(
        "--labels-file",
        default="",
        help="JSON list of labels; overrides --labels (avoids shell word-splitting pitfalls)",
    )
    ap.add_argument("--backend", default="mlx", choices=["mlx", "ane", "auto"])
    ap.add_argument("--model", default="", help="force checkpoint: english|multilingual|typed-decisions")
    ap.add_argument("--base-url", default="http://127.0.0.1:8767")
    ap.add_argument("--token", default="bench-local")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    if args.labels_file:
        args.labels = json.load(open(args.labels_file, encoding="utf-8"))
        print(f"{len(args.labels)} labels from {args.labels_file}", flush=True)

    samples = [
        json.loads(line) for line in open(args.samples, encoding="utf-8") if line.strip()
    ]
    if args.limit:
        samples = samples[: args.limit]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(out_path) if args.resume else set()
    todo = [s for s in samples if s["id"] not in done]
    print(f"{len(samples)} samples, {len(done)} already recorded, {len(todo)} to run", flush=True)

    url = args.base_url.rstrip("/") + "/predict"
    load_start = os.getloadavg()[0]
    print(f"loadavg at start: {load_start:.1f}", flush=True)
    mode = "a" if args.resume else "w"
    ok = err = 0
    t0 = time.perf_counter()
    with (
        httpx.Client(http2=False) as client,
        open(out_path, mode, encoding="utf-8") as f,
        ThreadPoolExecutor(max_workers=args.concurrency) as pool,
    ):
        for rec in pool.map(
            lambda s: run_sample(client, url, args.token, s, args, args.labels), todo
        ):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            if rec["error"]:
                err += 1
            else:
                ok += 1
            if (ok + err) % 500 == 0:
                print(f"  {ok + err}/{len(todo)} ok={ok} err={err}", flush=True)
    dt = time.perf_counter() - t0
    print(f"done: ok={ok} err={err} in {dt:.0f}s ({(ok + err) / max(dt, 1):.1f} req/s)")
    load_end = os.getloadavg()[0]
    meta = {
        "samples": args.samples,
        "out": args.out,
        "dataset": args.dataset,
        "backend": args.backend,
        "model": args.model or None,
        "concurrency": args.concurrency,
        "ok": ok,
        "errors": err,
        "seconds": round(dt, 1),
        "loadavg_start": round(load_start, 2),
        "loadavg_end": round(load_end, 2),
        # Accuracy is load-independent; latency under load is the worst-case
        # tail, not the budget number. Flag it so reports can't be misread.
        "contaminated": bool(load_start > 4 or load_end > 4),
    }
    meta_path = str(out_path) + ".meta.json"
    json.dump(meta, open(meta_path, "w"), indent=2)
    print(f"loadavg {load_start:.1f} -> {load_end:.1f} contaminated={meta['contaminated']}")
    return 0 if err == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
