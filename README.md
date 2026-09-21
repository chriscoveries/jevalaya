# jevalaya

Laissez les bons temps rouler, cher — this here's a fast little router for your Mac.

`jevalaya` serves one simple endpoint and answers typed questions — *this or that, how much, yes or no* — by sending each request to the right pot: a tiny CoreML model on the Neural Engine when the ask is short, a local MLX Laya when it needs more room, and Jev (TypeSafe) when the answer's too close to call. You bring your own models and keys; jevalaya just does the routin', quick and honest.

## What it does

- `POST /predict` — the drop-in laya/jev predict contract: `{state, questions}` in, `{model, answers, usage, routing}` out.
- Routes by content: checkpoint family (english / multilingual / typed-decisions), rendered token count, and confidence — with a fallback hop from ANE to MLX and an escalation hop from local to Jev.
- Degrades graceful: runs fine on three backends, two, or just one — whatever's standin' is what you get, always with the full routing receipt.
- Every request writes a structured event (latency, backend, confidence, cost) to a JSONL stream — that's the lagniappe a future lil' app can visualize.
- Compare mode: ask for `compare=["ane","mlx","jev"]` and get every backend's answer side by side.

## Quick start

```bash
cargo build --workspace
cp config/jevalaya.example.toml jevalaya.toml   # point it at your model dirs
export JEVALAYA_TOKEN=local-dev                  # bearer for /predict
export TYPESAFE_API_KEY=...                      # only if you use Jev

./target/debug/jevalaya serve --config jevalaya.toml
```

```bash
curl -s http://127.0.0.1:8767/predict \
  -H "Authorization: Bearer $JEVALAYA_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "state": "Mein Konto wurde zweimal belastet",
    "questions": {"topic": {"type": "choice",
      "instructions": "what is this about?",
      "criteria": {"billing": "...", "shipping": "..."}}}
  }'
```

## The rules of the house

- ANE only gets short, single, multilingual questions (≤96 rendered tokens). English and typed-decisions never ride the Neural Engine.
- An ANE capacity/shape failure retries exactly once on MLX. A hard failure stays visible — we don't dress it up as an answer.
- Jev fires when you ask for it, when confidence runs too close, or on retry. It's the only path that leaves the machine.
- Keys live in your environment (Infisical-style), never in this repo.

## Layout

```
crates/
  render/            # canonical prompt + tokenizer parity port (the gate's ruler)
  router-core/       # decision table, dispatch, escalation, compare, events
  backend-mlx-pyo3/  # laya-mlx in-process via PyO3
  backend-jev/       # TypeSafe HTTP client
  backend-coreml/    # native ANE (Apple Silicon)
  jevalaya-server/   # axum + `jevalaya` CLI
docs/                # DESIGN.md (authoritative), smoke transcript
config/            # example TOML — copy and point at your models
```

## Consumers

Any project on the machine can ask jevalaya a question — define your `state` and your `questions`, POST to `/predict`, read the typed answer and the routing receipt. See `docs/CONSUMER-HANDOVER.md` (coming with first release) and drop run feedback in the configured feedback sink so we can tune the thresholds togetha.

## Status

Milestone build: routing core, MLX bridge, and Jev client verified end-to-end; CoreML ANE adapter in progress. See `docs/DESIGN.md` for the full contract.
