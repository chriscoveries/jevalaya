# Consumer handover — ask jevalaya anything (well, almost)

Laissez les bons temps rouler, cher — this here's the doc you hand another project when it wants answers outta jevalaya. Short version: jevalaya is a **generic local decision router**. It knows nothin' about your app — no browser, no selectors, no business logic. You bring a `state` (what's goin' on) and `questions` (what you need decided); it sends each one to the right pot — Neural Engine, local MLX, or Jev — and hands you back typed answers plus a routing receipt. What the questions *mean* is entirely your business, and that's the point.

## Run it

```bash
cargo build --workspace
cp config/jevalaya.example.toml jevalaya.toml   # point [models] at your dirs
export JEVALAYA_TOKEN=pick-somethin'-secret      # bearer for /predict
export TYPESAFE_API_KEY=...                      # only if you use Jev
./target/debug/jevalaya serve --config jevalaya.toml   # listens 127.0.0.1:8767
```

Details (build flags, model deps, launchd) live in `README.md` under Install. One warnin': keep it on loopback unless you've thought it through — LAN needs explicit config, and the bearer token is the whole lock on the door.

## The contract: POST /predict

One endpoint does the work. Everything is JSON; bodies over 256 KiB get a `400`.

```json
{
  "state": "string, object, or list — your world, verbatim",
  "questions": {
    "topic": {
      "type": "choice | score | noul",
      "instructions": "string, or a JSON value if you think in structures",
      "criteria": "depends on the type, see below"
    }
  },
  "backend": "auto | ane | mlx | jev (default auto)",
  "model": "english | multilingual | typed-decisions (optional override)",
  "request_id": "your id, echoed back in logs (optional)"
}
```

Only `state` and `questions` are required. Unknown routing fields are rejected in strict mode (the default) — we don't guess what you meant.

### The three question types

**choice** — *this or that.* `criteria` is a map of label → description, or a plain list of labels. A `null`/empty description means the bare label. Keep your label sets modest, cher — every option costs tokens, and a 150-label question won't fit the head budget at all (you'll get a `500`, honest).

```json
"topic": {"type": "choice",
  "instructions": "what is this about?",
  "criteria": {"billing": "money, charges, refunds", "shipping": null}}
```

Answer: `{"type": "choice", "choice": "billing", "probabilities": {"billing": 0.91, ...}, "confidence": 0.83, "action": {"act_probability": 1.0}}`. The `choice` is always the top-probability label; `probabilities` sum to ~1.

**score** — *how much.* `criteria` is a list of level descriptions, index 0 up.

```json
"sev": {"type": "score",
  "instructions": "how severe?",
  "criteria": ["trivial", "annoying", "unusable"]}
```

Answer: `{"type": "score", "score": 1.36, "legend": {"0": "trivial", ...}, "probabilities": {"0": 0.04, ...}, "confidence": ..., "action": {...}}`. The `score` is the probability-weighted level — a dial, not a label.

**noul** — *yes or no* (well, *true or false*). `criteria` is optional: `{"false": "...", "true": "..."}` descriptions, or omit it entirely for the defaults.

```json
"refund": {"type": "noul", "instructions": "should we refund?"}
```

Answer: `{"type": "noul", "noul": 0.21, "confidence": 0.79, "action": {...}}`. `noul` is P(true); `confidence` is `max(p, 1-p)` — direction-free certainty.

Every answer carries `action.act_probability` — the model's own sense of whether to act at all. It's an input to your gate, not a permission slip. Outcomes still need independent verification on your side; a DONE-flavored answer ain't proof.

### The routing receipt

Every response includes `routing` — the receipt that says what happened and why:

```json
"routing": {"checkpoint": "english", "backend": "mlx",
  "model_id": "/models/laya-english (or endpoint)",
  "reason": "not_multilingual: English Latin text",
  "token_count": 34, "ane_eligible": false, "fallback": false,
  "latency_ms": 45.6, "escalated": false, "degraded": false,
  "confidence": 0.79, "margin": 0.57,
  "cost": {"input_tokens": 34, "output_tokens": 0, "remote_usd": null}}
```

Read it like this: `backend` is who answered; `reason` is a stable prefix plus human detail (greppable — build dashboards on the prefix); `token_count` is the pre-truncation rendered count; `confidence`/`margin` are minimums across your questions; `escalated` + `jev_trigger` mean Jev weighed in; `fallback` + `fallback_from` mean ANE stumbled and MLX caught it; `degraded` means somethin' upstream didn't go to plan but the answer stands. `usage` (`input_tokens`, `output_tokens` always 0 locally) sits next to `answers` like laya always did.

### Overrides

- `POST /v1/systemone` is an alias of `/predict` — existing Jev clients that post `{model, state, questions}` can repoint their `base_url` at jevalaya unchanged. `jev-*` model ids are accepted and don't pin a checkpoint; only `english`/`multilingual`/`typed-decisions` pin weights.
- `"backend": "ane"` asks for the Neural Engine but doesn't skip the safety gates — short single multilingual only (≤96 rendered tokens), or you'll ride MLX with an honest reason. `"backend": "mlx"` pins local. `"backend": "jev"` goes straight to TypeSafe (needs the key; `token_count` stays null — Jev tokenizes server-side).
- `"model": "multilingual"` pins the checkpoint; otherwise script/language detection picks (explicit beats detected, always).
- `"compare": ["mlx", "jev"]` (or `true`) fans out and returns every backend's answer side by side under `compare`, primary first. Handy for calibratin' your own thresholds.

### Worked example

```bash
curl -s http://127.0.0.1:8767/predict \
  -H "Authorization: Bearer $JEVALAYA_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"state": "The order arrived broken",
       "questions": {
         "sev": {"type": "score", "instructions": "How severe?",
                 "criteria": ["trivial", "annoying", "unusable"]},
         "refund": {"type": "noul", "instructions": "Should we refund?"}}}' | python3 -m json.tool
```

Minimal Python client:

```python
import httpx

def decide(state, questions, backend="auto", base="http://127.0.0.1:8767", token="..."):
    r = httpx.post(base + "/predict",
                   json={"state": state, "questions": questions, "backend": backend},
                   headers={"Authorization": f"Bearer {token}"}, timeout=60)
    r.raise_for_status()
    body = r.json()
    return body["answers"], body["routing"]   # answers + receipt, always together
```

Check `r.status_code` first: `401` bad/missing token, `400` malformed body (or over 256 KiB), `404` wrong path, `409` you asked for a backend that ain't configured, `429` everybody's busy (honor `Retry-After`), `502` Jev stumbled, `503` a backend's still loadin', `500` inference failed hard, `504` your request outran the timeout. All errors are `{"error": {"code": ..., "message": ...}}` — no stack traces, no secrets, no echo of your state.

## Writin' your own questions

A few house rules from folks who've burned themselves:

- **Labels are your API.** Short, stable, unique strings — they come back verbatim as `choice`. Descriptions after the colon help the model; the label is what your code switches on.
- **Instructions carry the goal.** One sentence of what "good" means beats three paragraphs of context. Non-string instructions (objects, lists) are fine — they travel as compact JSON.
- **States are verbatim.** Strings pass through untouched; dicts/lists become compact JSON. Mask-token-lookin' text (`[MASK]`) gets scrubbed to spaces before tokenizin' — don't rely on it survivin'.
- **Mind the option budget.** Each option eats tokens out of a fixed head budget (~192 english); past ~20 long options you're truncatin' descriptions. The ceiling is deliberate, not accidental: if your options don't fit, you get a `400` with `too_many_options: at most {max} options fit the token budget, received {n}` — shrink the set or split it into a coarse question then a fine one. Two cheap calls beat one rejected call, and a 400 beats a silent wrong answer every time.
- **Multi-question is one round trip.** Send all your questions at once; the minimum confidence/margin across them lands in `routing` so one shaky answer can't hide behind confident siblings.
- **Jev is opt-in spend.** Auto mode escalates low-confidence answers to TypeSafe — real money, real latency (hundreds of ms). Set `backend` explicit or disable Jev in config if you want purely local answers; watch `escalated` in the receipt to audit what you'd have spent.

## Feedback — tell us how the answers did

Here's the part that makes the whole thing smarter, and there's two ways to send it, cher — pick whichever fits your setup:

**1. `POST /feedback`** (same bearer token as `/predict`):

```bash
curl -s http://127.0.0.1:8767/feedback \
  -H "Authorization: Bearer $JEVALAYA_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"consumer": "my-app", "request_id": "abc-123",
       "chosen_answer_ok": false, "expected": "billing",
       "notes": "refund was a close second at 0.44"}' 
# → {"ok": true} with HTTP 202
```

Required: `consumer` (who you are), `request_id` (echo it from your request so we can join), `chosen_answer_ok` (did the top answer do right by you). Optional: `expected` (the label you wanted), `notes` (free text — close calls, weird confidences, all welcome), `ts` (your judgment timestamp; the server stamps `received_unix` regardless). Missing fields get a `400`; no token a `401`; a server without a feedback sink answers `503`. The record lands in the operator's configured `[feedback] sink` JSONL — ask them where it lives if you're curious.

**2. Append the file yourself** — same schema, one JSON line per judgment, to whatever path you and the operator agreed on (default convention `~/.jevalaya/feedback.jsonl`). Same fields, same effect; handy when you're batchin' judgments offline.

Truthful negatives are the lagniappe — a hundred "this was wrong and here's why" beats a thousand silent successes for tunin' thresholds. And `request_id` only shows up in our logs if you send it — so send it, cher, otherwise we can't find your run in a crowd.

## Jev, money, and leavin' the machine

Local backends (ANE, MLX) cost nothin' but electricity and never leave your Mac. Jev is the only call that leaves the machine: it needs `TYPESAFE_API_KEY` in the environment (never in a file), runs ~300–900ms, and bills per call. In auto mode it fires on low confidence/margin or retry; explicit `backend=jev` always calls it; `compare` with jev in the set calls it once per request. Budget accordingly, audit with the `escalated` flag and the `cost` block, and remember: when Jev is unreachable in auto mode you keep the validated local answer marked `degraded` — the show goes on.
