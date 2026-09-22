<p align="center"><img src="docs/assets/logo.svg" alt="jevalaya" width="320"></p>

# "But we have Jev at home."

*The same Jev-like decision making and more, on your Mac — but the Neural Engine fires 20–30 answers inside one Jev roundtrip, and Jev is still there for the 1-in-50 call that earns it.*

**What is Jev?** TypeSafe's cloud decision API — you send state and typed questions, it sends back answers from inside your own answer set, with confidence.

<p align="center"><img src="docs/assets/hero.svg" alt="jevalaya routes one /predict endpoint to ANE, MLX, or Jev" width="900"></p>


### 1. SPEEEEED

`backend: ane / mlx / jev` `ms: server-reported` `reason: explicit_backend`

<p align="center"><a href="docs/assets/race-demo.mp4"><img src="docs/assets/race-demo-preview.gif" alt="Burst race: ANE and MLX markers at their server-reported timings, both choosing Sci/Tech (click for full MP4)" width="540"></a></p>

<details>
<summary>Inside the burst</summary>

**Watch the backends race:** [the three-lane demo](docs/assets/race-demo.mp4) fires the same headline at all three backends concurrently — a 4-call burst each at ANE and MLX plus one real paid Jev call, every marker at its server-measured latency on one shared scale. ANE drains in a tight cluster; MLX stair-steps as calls serialize on the bridge; the API call lands a few hundred milliseconds later, having crossed the internet. The axis always runs to the slowest answer of the session. Every marker stays on the row you asked for, in that row's color; a rerouted call shows as a hollow ring labeled with where it actually went (e.g. `→MLX` on the ANE row). Source: `race.html`/`race.js` in the same directory.

| Measurement | Context |
| --- | --- |
| 8.8ms p50 | ANE, 30 solo calls, M1 Max (low power off) |
| 10.3ms p50 | MLX, 30 solo calls, M1 Max (low power off) |
| ~8ms | upstream laya-coreml on M3 Max |
| ~93% | local AG News |

[Full chart: local backends at full power](docs/assets/local-backends.png)


</details>


## ELI12

Every request goes to one of three places. jevalaya picks the cheapest one that can handle it:

- **ANE — the Apple Neural Engine.** It's in every Mac since the M1 and almost nothing uses it. Lots of tiny cores, tiny context, super efficient, super fast. If your question fits in a handful of tokens, this is where it goes — and it answers in milliseconds, on the chip, for free.
- **MLX — your GPU.** Bigger, heavier, holds much more context. When a question is too long for the Neural Engine, it stays on your Mac anyway and runs locally through MLX.
- **API — Jev, the cloud.** That means sending a packet from you, past your wifi, through all the internet, to Jev... and back again. For maybe a 1-in-50 chance of a better answer. Oh, and paying for it. jevalaya only spends that call when the local models genuinely can't call it — and tells you exactly why in the receipt.

**You don't know you need it until** your Jev bill lands and you realise most of those calls could've been answered on the chip you already own. Or until you're on a plane, offline, and the app still works. Or until you look at a latency graph and notice the internet is the slow part.

## What is Jev

*TypeSafe's cloud decision API — you send state and typed questions, it sends back answers from inside your own answer set, with confidence.*

Jev is not a chat LLM — it doesn't generate text. You send it a `state` (any text or JSON: an email, a market snapshot, a tool call an agent wants to run, a whole inbox) plus one or more typed questions, and [TypeSafe](https://typesafe.ai)'s System One model returns calibrated probabilities for every question in a single ~100ms round trip. Your code, not the model, makes the final decision by thresholding on those numbers.

Three primitives, picked by the shape of the question: **noul** — yes/no, returns one probability ("is this urgent?"). **Choice** — pick from known options, returns probabilities over them ("which team handles this?"). **Score** — grade on a rubric, returns a score plus level distribution ("how risky is this?"). No generated JSON to parse, no free-text to regex — it's the piece that turns "ask the model" into "ask a question your code can switch on." There's an interactive playground at [jev-explained](https://jev-explained-repo.vercel.app/) — bring your own TypeSafe or Vercel AI Gateway key.

Laya is the open-weight sibling of the same idea — the same typed-decision contract, running as a local encoder instead of a paid endpoint. jevalaya sits between the two: same `/predict` shape in, Laya on your silicon when it'll do, Jev when it won't.

<p align="center"><img src="docs/assets/jev-bell.png" alt="LLM vs Jev: a chat model talks with its hands; Jev just rings the bell" width="420"></p>

## How do I use it

You POST `state` and `questions`, jevalaya returns typed answers plus the routing receipt — so anything that can speak the predict contract works. Folks are already building on Jev:

- [jev](https://github.com/anilsenay/jev) (Go, unofficial) — ask Jev questions and get answers back as your own Go types; the compiler checks your `switch`, not just the JSON.
- [jev-router](https://github.com/Akashdb5/jev-router) (Python) — Jev-powered security screening and cost-aware routing across OpenAI, Anthropic, and OpenRouter models.
- [Predict-With-Jev](https://github.com/Protocol-Lattice/Predict-With-Jev) — a crypto market research dashboard that runs forecasts and walk-forward evaluation through Jev's System One API.
- [pi-jev-router](https://pi.dev/packages/pi-jev-router) — lets Jev pick a model and reasoning effort for the Pi coding agent.
- [pi-typesafe-router](https://pi.dev/packages/pi-typesafe-router) — Jev classifies Pi requests and routes them across TypeSafe, Cloudflare, Vercel, and OpenRouter backends.

Point your own project at `http://127.0.0.1:8767/predict` instead of Jev's endpoint and the same calls get answered locally first — your agent code doesn't change, your bill does. The full contract is in `docs/CONSUMER-HANDOVER.md`.

## Quick start

Grab the prebuilt binary from [Releases](https://github.com/chriscoveries/jevalaya/releases) (`jevalaya-v0.1.0-macos-aarch64.tar.gz` — Apple Silicon), or build from source:

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

<p align="center"><img src="docs/assets/escalation.svg" alt="local when it's sure — Jev when it isn't" width="700"></p>

## The rules of the house

- ANE only gets short, single questions that fit the exported CoreML head (≤96 rendered tokens) — english and multilingual both ride the Neural Engine when the fit gate passes. Fit decides, not the language detector.
- An ANE capacity/shape failure retries exactly once on MLX. A hard failure stays visible — we don't dress it up as an answer.
- Jev (the paid cloud API) only gets called in three cases: you explicitly ask for it (`backend: "jev"`), the local answer isn't confident enough to trust, or a local attempt fails and the router retries upward. It's the only path that ever leaves your Mac — and the receipt on every response says which backend answered and why, so a Jev call can never happen silently.
- Keys live in your environment variables, never in this repo.

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

## Install

From source, the whole workspace (default features: MLX + ANE):

```bash
export PYO3_PYTHON=/path/to/laya-mlx/.venv/bin/python  # the interpreter with mlx + laya_mlx
cargo build --workspace                                 # ./target/debug/jevalaya
```

No Python toolchain on the box? `cargo build -p jevalaya-server --no-default-features` gives you a Jev-only binary, no libpython needed. The ANE adapter compiles everywhere but only serves on Apple Silicon — off-target it registers unavailable instead of lyin' about it.

What you need besides the binary — the whole thing, step by step (an agent can run this verbatim):

```bash
# 1. Python env for the MLX backend (Python 3.11+; the reference venv is 3.13)
python3 -m venv .venv && . .venv/bin/activate
pip install laya-mlx            # gives you the laya_mlx package + mlx

# 2. Model weights — four HF repos, into the standard hub cache
hf download aac6fef/laya-mlx                        # english checkpoint
hf download aac6fef/laya-multilingual-mlx           # multilingual checkpoint
hf download aac6fef/laya-typed-decisions-mlx        # typed-decisions checkpoint
hf download aac6fef/laya-multilingual-coreml-ane    # CoreML/ANE bundle

# 3. Config — the example already points at the HF cache layout
cp config/jevalaya.example.toml jevalaya.toml       # fix the four paths if your cache differs
export JEVALAYA_TOKEN=local-dev                     # bearer for /predict
export TYPESAFE_API_KEY=...                         # only if you enable Jev

# 4. Validate without loading weights, then serve
./target/release/jevalaya check --config jevalaya.toml
./target/release/jevalaya serve --config jevalaya.toml
```

Notes that matter:

- **`PYO3_PYTHON` is a build-time binding.** The interpreter PyO3 embeds is chosen when you compile, not when you run — set it to your venv python *before* `cargo build` or MLX will pick up the wrong site-packages. (Prebuilt releases were built against Python 3.13.)
- **`[mlx] python_path`** in the config is a list of `sys.path` entries: the directory containing `laya_mlx/` (a laya-mlx checkout, or your site-packages if pip-installed) plus your venv's `site-packages`.
- **`[ane] model`** is the CoreML bundle dir (`laya-multilingual-coreml-ane`), separate from the MLX checkpoints.
- **`offline = true`** keeps hub resolution fully local once the snapshots exist.
- **launchd** for always-on: wrap `serve` in a LaunchAgent plist (program args + `KeepAlive`), logs to a file you rotate. Bind stays loopback unless your config says otherwise, on purpose.

## For agents building apps

Got your own state and your own questions? POST 'em to `/predict`, read the typed answers plus the routing receipt, and tell us how it went — the full deal (contract shapes, overrides, error codes, feedback schema, a curl and a Python snippet) is in `docs/CONSUMER-HANDOVER.md`. jevalaya don't know your domain and don't need to: define questions in your own words, watch `confidence`/`margin`/`escalated` in the receipt, and append judgments to the feedback sink so the thresholds learn your world, togetha.

## Consumers

Any project on the machine can ask jevalaya a question — define your `state` and your `questions`, POST to `/predict`, read the typed answer and the routing receipt. Start at `docs/CONSUMER-HANDOVER.md` and drop run feedback in the configured feedback sink so we can tune the thresholds togetha.

## Status

Routing core, CoreML ANE adapter, MLX bridge, and Jev client verified end-to-end — all three backends answer live in the demos above. See `docs/DESIGN.md` for the full contract.

## Credits

jevalaya is a router — the brains belong to other people:

- **[Laya](https://github.com/NandhaKishorM/laya)** — the open-weight typed-decision models and original implementation, by [Convai Innovations](https://huggingface.co/convaiinnovations) and contributors (Apache-2.0). English, multilingual, and typed-decisions checkpoints.
- **[laya-mlx](https://github.com/mizorewww/laya-mlx)** — the native MLX port for Apple silicon (`aac6fef/*-mlx` on the hub). Our GPU lane is this package embedded over PyO3.
- **[laya-coreml](https://github.com/mizorewww/laya-coreml)** — the Core ML / Neural Engine bundles (`aac6fef/*-coreml*`). Our ANE lane runs the `-ane` variant.
- **[TypeSafe](https://typesafe.ai)** — Jev and the System One API jevalaya escalates to.

Both Apple conversions are independent ports, not official Convai or Apple releases — all credit to the porters, all model behavior to the original checkpoints.

## More from the workshop

- [CCCC-Workflows](https://github.com/chriscoveries/CCCC-Workflows) — multi-agent collaboration workflows.
