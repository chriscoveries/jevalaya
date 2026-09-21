# jevalaya Rust service design

Status: implementation design for the jevalaya Rust HTTP service (github.com/eafire15/jevalaya), reviewed against the current laya-mlx and jev-ultrafast sources and the post-draft board decisions.

This document defines a model-agnostic router service. The service ships routing, prompt/parity logic, HTTP, lifecycle, and backend adapters; users supply model directories and remote endpoint configuration. It does not vendor laya-mlx, laya-coreml, model weights, or TypeSafe credentials.

## Consumers and product surface

jevalaya is a simple local router for Macs. Its `/predict` endpoint is the product surface: any local project can POST `{state, questions}` and receive typed `choice`, `score`, or `noul` answers. Consumers own their state, question definitions, and actions taken from the answers; jevalaya owns routing and the shared prediction contract. For example, a consumer may define a choice question to decide whether a page needs VLM or OCR processing. Those labels are request data, never built-in branches.

Keep consumer-specific logic and project names out of the implementation. The README should teach the generic pattern: define your question, choose its type, POST the state and questions, and read the routed typed answer and routing metadata.

## Goals and invariants

- Expose a drop-in local HTTP prediction endpoint for the laya-mlx server contract, while accepting the Jev/TypeSafe request shape where it is compatible.
- Route among native CoreML ANE, native MLX Laya through an in-process PyO3 bridge for v1, and Jev through an HTTP client.
- Keep the fast local route free of network calls. Jev is the only inference path that deliberately leaves the machine.
- English and typed-decisions checkpoints never execute on ANE.
- ANE is only an optimization: a multilingual, one-question request whose fully rendered, model-tokenized input fits the aligned 96-token limit.
- A capacity/shape failure may trigger exactly one MLX retry. A hard failure must be visible; it must not be disguised as a different answer.
- No PyTorch or Transformers inference runtime is used. The only embedded Python is a thin PyO3 bridge to the existing laya-mlx package; the primary MLX path has no process or socket hop. Native mlx-rs may replace that bridge later.
- Local model directories work offline. A path-like missing directory is an error, not a reason to silently contact Hugging Face.
- Default bind is loopback. Authentication, body limits, timeouts, and redacted logs are part of the service contract.

The public package is macOS/Apple Silicon oriented because native ANE and MLX are the target runtime. router-core and endpoint tests should compile and run with mocked backends on other hosts; an actual ANE backend is compile-time and run-time gated.

## Source verification and endpoint decision

The current laya-mlx server at laya-mlx/server/laya_server.py is a small HTTP server:

- GET /health returns { "ok": true, "model": ... }.
- POST /predict requires a Bearer token, a JSON body with state (string/object/list) and questions (object), and returns the laya answer payload.
- It caps bodies at 256 KiB and maps bad auth to 401, bad input to 400, busy semaphore to 429, and inference errors to 500.
- laya-mlx answers are {model, answers, usage}; its Router adds routing without changing answers.

The current jev-ultrafast repository has no local HTTP /predict endpoint. Its model.py choose() builds a browser-specific state/questions body and sends it to https://api.typesafe.ai/v1/systemone with Authorization: Bearer TYPESAFE_API_KEY. The response contains model, answers, and optional usage. The client retries 429, 529, and 503 up to three total attempts with 0.5/1.0 second backoff, and validates selected choice probabilities before execution.

This difference is material and is fixed in this design: jevalaya exposes /predict as the public compatibility surface and uses /v1/systemone only as the configurable outbound Jev adapter default. The public Jev-compatible payload is accepted unchanged when a caller supplies the same state/questions shape. jevalaya does not execute browser mutations or invent selectors/actions.

## Workspace and crate layout

Use a Cargo workspace with policy-only core and separately gated backends:

~~~
jevalaya/
  Cargo.toml
  crates/
    router-core/
      Cargo.toml
      src/
        lib.rs
        config.rs
        schema.rs
        decision.rs
        policy.rs
        prompt.rs
        tokenization.rs
        language.rs
        lifecycle.rs
        errors.rs
    backend-coreml/
      Cargo.toml
      src/lib.rs
    backend-mlx-pyo3/
      Cargo.toml
      src/lib.rs
    backend-jev/
      Cargo.toml
      src/lib.rs
    jevalaya-server/
      Cargo.toml
      src/main.rs
      src/bin/jevalaya.rs
  sidecar-fallback/README.md
  config/jevalaya.example.toml
  tests/fixtures/{prompt-parity,answer-parity,routing}
~~~

router-core owns no HTTP client, embedded Python, or CoreML symbols. Every backend implements one trait, so a future native mlx-rs implementation can replace backend-mlx-pyo3 without changing policy or the wire contract.

## Wire contract

POST /predict accepts:

~~~
{
  "state": "string, object, or list",
  "questions": {
    "question_id": {
      "type": "choice | score | noul",
      "instructions": "string or JSON value",
      "criteria": "choice map/list, score list, or noul map"
    }
  },
  "backend": "auto | ane | mlx | jev",
  "model": "english | multilingual | typed-decisions",
  "task": "typed_decisions",
  "lang": "en | de | ...",
  "request_id": "caller supplied id",
  "compare": "false | true | [\"ane\", \"mlx\", \"jev\"]"
}
~~~

Only state and questions are required. Unknown routing fields should be rejected in strict mode, which is the default, rather than silently changing backend.

A successful response preserves the laya-mlx schema:

~~~
{
  "model": "laya-rl-agent | jev-latest | configured model",
  "answers": { "...": { "type": "...", "...": "..." } },
  "usage": { "input_tokens": 0, "output_tokens": 0 },
  "routing": {
    "checkpoint": "english | multilingual | typed-decisions | null",
    "backend": "ane | mlx | jev",
    "model_id": "configured path, id, or endpoint",
    "reason": "stable reason prefix plus detail",
    "token_count": 42,
    "ane_eligible": true,
    "fallback": false,
    "latency_ms": 7.4,
    "escalated": false,
    "degraded": false,
    "confidence": 0.82,
    "margin": 0.31,
    "cost": { "input_tokens": 42, "output_tokens": 0, "remote_usd": null }
  }
}
~~~

The required routing keys are checkpoint, backend, model_id, reason, token_count, ane_eligible, fallback, and latency_ms. token_count is the raw, pre-model-truncation gate count when a local prompt was rendered; it may be null for explicit Jev. `execution_token_count` may additionally report the model-bounded sequence sent to MLX. latency_ms is null from route-only code and end-to-end inference time in an HTTP response. backend describes the backend that produced returned answers. After ANE failure and MLX retry, backend is mlx, fallback is true, fallback_from is ane, and ane_eligible remains true. Jev escalation is represented by escalated=true and jev_trigger; it is not a fallback unless a later policy explicitly defines it as one. Confidence, margin, degraded, escalation_error, and cost are optional until a backend result exists.

For a single question, `confidence` and `margin` are that answer's confidence and top-two probability margin. For multiple questions, the request-level values are the minimum confidence and minimum margin across answers (the event may also carry a per-question map). A low-margin trigger fires when any question crosses the configured threshold; this avoids hiding one ambiguous answer behind a confident aggregate.

Every backend implements one normalized adapter contract: the router passes the validated `{state, questions}` request (plus model/task/lang metadata), and a successful adapter returns the same `{model, answers, usage}` payload. CoreML and MLX execute locally; Jev serializes that same body as `POST <base_url>/v1/systemone` and normalizes the provider response. The browser-oriented `jev_ultrafast.choose()` wrapper is not this contract: it constructs a specialized question set and returns a derived action decision, which remains outside jevalaya.

When `compare` is present, the response keeps the normal primary `model`/`answers`/`usage` and adds a `compare` object containing one entry per requested backend, each with its normalized result (when available), `RouteDecision`, latency, cost, or a structured error. `compare=true` uses the configured comparison set; an explicit list is filtered through the health snapshot. A compare set has two or three distinct backends before health filtering; a one-backend set is ordinary routing, not compare mode.

Expose GET /health (preserving ok/model fields), GET /ready (readiness without forcing all lazy weights), and POST /predict. A future /route may return RouteDecision without running inference.

Status/error mapping is stable JSON with code and no stack trace:

- 400 invalid_request: malformed JSON, missing state/questions, invalid question schema, unknown backend, or body over 256 KiB.
- 401 unauthorized: public bearer token missing/incorrect.
- 404 not_found: unknown path.
- 409 backend_unavailable: explicitly requested backend is disabled/not configured.
- 429 busy: request semaphore or backend capacity exhausted; include Retry-After when known.
- 502 upstream_error: Jev network/provider failure or backend protocol failure after bounded retry.
- 503 not_ready: lazy backend is starting or a required local model is unavailable.
- 500 inference_failed: classified hard inference failure. In auto mode, a failed Jev escalation is recorded and the validated local answer may remain the best available result; an explicit Jev request is a 502/503.

## Core types and backend trait

router-core defines PredictRequest, AnswerPayload, Checkpoint (English, Multilingual, TypedDecisions), BackendKind (Ane, Mlx, Jev), RouteDecision, RenderedPrompt, BackendCapabilities, and typed errors.

RouteDecision is an immutable serializable value:

~~~
{
  "checkpoint": "english | multilingual | typed-decisions | null",
  "backend": "ane | mlx | jev",
  "model_id": "configured model directory, checkpoint id, or endpoint",
  "reason": "stable reason prefix plus human detail",
  "token_count": "raw rendered count or null",
  "ane_eligible": "bool",
  "fallback": "bool",
  "latency_ms": "number or null",
  "execution_token_count": "optional model-bounded count",
  "fallback_from": "optional backend",
  "fallback_reason": "optional stable error code",
  "escalated": "bool",
  "jev_trigger": "optional stable trigger",
  "confidence": "optional number",
  "margin": "optional top-two probability margin",
  "degraded": "bool",
  "escalation_error": "optional stable error code",
  "cost": "optional usage/cost object"
}
~~~

Use an async trait boundary and permit blocking implementations:

~~~
trait PredictBackend: Send + Sync {
    fn kind(&self) -> BackendKind;
    fn capabilities(&self) -> BackendCapabilities;
    async fn preload(&self, request: PreloadRequest) -> Result<(), BackendError>;
    async fn unload(&self, target: UnloadTarget) -> Result<(), BackendError>;
    async fn predict(
        &self,
        request: &PredictRequest,
        rendered: Option<&RenderedPrompt>,
    ) -> Result<BackendResult, BackendError>;
}
~~~

CoreML prediction runs in bounded spawn_blocking. The embedded PyO3 bridge runs blocking Python/MLX work behind a bounded executor and a GIL-aware bridge; Jev uses async I/O. Backend adapters, not policy, translate native exceptions into typed errors.

## Model source resolution and config

Every model reference is a ModelSource:

- Existing local directory: use as-is and never download.
- Hugging Face id/revision/subfolder: resolve through a pinned hf-hub client into the configured cache.
- Path-like but missing: clear missing-local-model error; do not reinterpret it as a hub id.
- Offline mode: permit cached hub content and reject all network resolution.

Validate tokenizer/tokenizer.json, tokenizer config/special tokens, model configuration/calibration metadata, and backend-specific weights/model package before advertising availability. No checkpoint or conversion artifact belongs in this repository.

Example config:

~~~toml
[server]
listen = "127.0.0.1:8767"
auth_token_env = "JEVALAYA_TOKEN"
max_body_bytes = 262144
max_inflight = 2
offline = false

[reload]
watch = true
debounce_ms = 200

[models]
english = "/models/laya-english"
multilingual = "/models/laya-multilingual"
typed_decisions = "/models/laya-typed-decisions"

[ane]
enabled = true
model = "/models/laya-multilingual-coreml-ane"
max_tokens = 96
alignment = 1
# false: cold ANE warms in background on first eligible request — that
#   request and any while loading fail over to MLX with ane_warming.
# true: block readiness until the model is resident (production default).
preload = false

[mlx]
enabled = true
python = "/path/to/venv/bin/python"
pyo3_module = "laya_mlx"
max_loaded = 1
preload = false
compile = false
cache_prompts = false
pad_to_multiple = 1
startup_timeout_ms = 30000
request_timeout_ms = 30000
# Optional socket sidecar fallback, never the v1 primary path.
sidecar_fallback = false
sidecar_socket = "/tmp/jevalaya-mlx.sock"

[jev]
enabled = true
base_url = "https://api.typesafe.ai"
predict_path = "/v1/systemone"
api_key_env = "TYPESAFE_API_KEY"
model = "jev-latest"
request_timeout_ms = 25000
retries = 2
confidence_threshold = 0.75
margin_threshold = 0.20
target_confidence_threshold = 0.85

[observability]
sink = "jsonl:/var/tmp/jevalaya-events.jsonl"
queue_capacity = 1024

[feedback]
path = "~/.jevalaya/feedback.jsonl"
http_enabled = false
~~~

The key is always read from the environment named by api_key_env. Neither TOML nor logs may contain a secret.

## Canonical prompt and tokenizer parity

This is a Rust port of laya-mlx behavior, not a new prompt format. Parity is a release gate.

### tokenizer.json

Use the Rust tokenizers crate and load tokenizer.json in the selected model directory. Read cls/sep/pad/mask token strings and ids from tokenizer config and fail closed when a required special token is absent. The tokenizer used for the ANE decision must be the ANE model bundled tokenizer, or the exact tokenizer capability exposed by the native ANE adapter; do not estimate with Unicode, bytes, or words. Load tokenizer.json and tokenizer_config.json from each model directory. English commonly resolves CLS/SEP as [CLS]/[SEP]; multilingual resolves them as <bos>/<eos>. Never hard-code special ids across families. The PyO3 bridge may use its selected MLX tokenizer for execution, but the ANE gate is based on the model that receives the ANE input.

### JSON and question rendering

Port these distinctions exactly:

1. String state is unchanged. Object/list state is JSON encoded with non-ASCII preserved, Python-style comma-space and colon-space separators, and insertion order from the request.
2. Non-string instructions are JSON encoded with the laya-mlx Agent behavior, including Python default escaping where applicable.
3. String criteria pass through. Structured criteria use non-ASCII-preserving JSON with comma-space/colon-space separators.
4. Choice options preserve map order and render as label when value is null/empty, otherwise label + colon-space + criterion.
5. Score options render as level N + colon-space + criterion.
6. Noul options are exactly false/true with the existing default sentences when descriptions are absent.
7. Replace the tokenizer mask token with a space inside instructions, options, and serialized state before tokenization.

The canonical per-question sequence is:

~~~
[CLS] <type> question: <instructions> [SEP]
[MASK] option_0 [MASK] option_1 ... [SEP]
<serialized state> [SEP]
~~~

Question prefix and marker positions follow laya-mlx build_prefix behavior. The invariant layout is [CLS] head [SEP] ( [MASK] opt )* [SEP] state [SEP], where head is exactly "<type> question: <ins>" and each opt is [MASK] followed by tok(" " + opt) truncated to 48 tokens. Read head_max_len from each rl_agent_config.json (192 for English and 256 for multilingual in the shipped references). State consumes the remaining model budget. Multilingual special tokens are <bos>/<eos>, so per-model-directory loading is mandatory. Apply semantic option/head budgets exactly, but do not final-truncate state merely to force an ANE request under 96; count token ids first and route an over-limit request to MLX. Compare token ids in parity tests, not only rendered JSON strings.

For one question, keep two lengths and never substitute one for the other:

- Build the canonical prefix and tokenize the complete serialized state with the tokenizer of the model that would receive the request. For the ANE gate that is the configured CoreML/ANE model tokenizer; for MLX execution it is the selected MLX checkpoint tokenizer. `raw_count` is the full semantic sequence length (prefix + complete state + final separator) before the model's `max_len` truncation. If ANE and MLX tokenizers differ, count and parity-test each path separately and fail closed on missing ANE metadata.
- `aligned_count` is `round_up(raw_count, model alignment)`, with alignment 1 when unspecified. The gate uses this count, not a batcher's dynamic padding length.
- `execution_token_count` is the length of the model-bounded sequence actually sent to MLX. The ordinary execution renderer may truncate state to `max_len`; that truncation must not make an over-limit prompt appear ANE-eligible.
- ANE is eligible only when `aligned_count <=` the configured `ane.max_tokens` (default 96, never above the native fixed input length), and the eligible ids/mask are then padded to the CoreML fixed input length. Padding is allowed; truncating or dropping state to get below 96 is not.

The renderer API should expose both `raw_count`/`aligned_count` and the model-bounded execution ids. In particular, do not derive the ANE gate from `build_sequence(...).len()` after that function has applied `max_len` truncation. This is the parity-sensitive boundary between routing and backend execution.

For multiple questions, ANE is ineligible. Report maximum per-question raw count, or null if rendering failed before a valid count. max_tokens must not exceed native input shape. pad_to_multiple applies to MLX batching and, when compatible, ANE alignment; it cannot increase the ANE ceiling.

### Language and typed-workflow routing

Port laya-mlx script ranges and stopword/diacritic heuristics into Rust with golden cases for English Latin, non-English Latin, non-Latin, no-letter, and mixed state. Preserve the source precedence exactly: explicit `model` > explicit `task` > exact typed-workflow match when opt-in detection is enabled > explicit `lang` > detected script/language > configured default.

Typed workflow detection is exact id-set matching for known workflows, not loose key search. Even when automatic typed selection is disabled for laya compatibility, a typed marker blocks ANE. English and typed-decisions are hard ANE exclusions.

## Ordered routing and execution

The router has a pure decision phase followed by execution/escalation. No weight engine is required for policy.

1. Backend override:
   - backend=jev selects Jev directly and skips local ANE/MLX; checkpoint and token_count may be null.
   - backend=mlx selects a checkpoint through the Rust checkpoint router and hard-disables ANE.
   - backend=ane expresses intent but does not bypass safety gates.
   - backend=auto runs the local route and may escalate to Jev after the local answer.
2. Checkpoint router:
   - explicit `model` wins, then explicit `task`;
   - exact typed-decisions workflow selection is next only when opt-in task detection is enabled;
   - explicit `lang` then wins over script/language detection;
   - otherwise port laya-mlx language/script detection: non-Latin or confidently non-English Latin selects Multilingual; English Latin selects English; unknown/no-letter uses configured default.
3. Render/token count:
   - canonical prompt through prompt.rs/tokenization.rs;
   - raw and aligned count from tokenizer.json;
   - never use MLX dynamic batch padding for this decision.
4. ANE eligibility requires all of:
   - ane enabled/configured and macOS arm64;
   - native ANE backend, model, tokenizer, and capabilities available;
   - selected checkpoint Multilingual and request not typed;
   - exactly one question;
   - prefer_ane_for_short or explicit ANE intent;
   - aligned_count <= configured `ane.max_tokens` (default 96) and native fixed input length.
   Any failed gate produces MLX with stable reason prefixes such as ane_disabled, platform_unsupported, ane_unavailable, not_multilingual, typed_decisions, question_count, or token_count_over_limit.
5. MLX default sends the request to the in-process PyO3 bridge with the already selected checkpoint/model. The bridge must not reroute it differently. A loopback sidecar is a documented opt-in fallback only.
6. ANE runtime fallback invokes one CoreML agent. Only ANECapacity or ANEShape releases/resets ANE and retries once on MLX using the same Multilingual checkpoint. Returned metadata has backend=mlx, fallback=true, fallback_from=ane, MLX model_id, original token_count, and ane_eligible=true. Hard ANE errors never retry.
7. Jev escalation occurs after local output:
   - auto mode uses configurable confidence and margin thresholds; the primary trigger is top_p - second_p below the configured margin, with optional confidence and target-confidence gates;
   - escalation also fires after a local backend retry when jev.escalate_on_retry=true;
   - explicit backend=jev always calls Jev;
   - successful escalation returns Jev answers with backend=jev, escalated=true, jev_trigger, and total latency;
   - compare mode fans out to two or all three selected backends and returns every normalized answer plus per-backend routing;
   - when Jev is unreachable in auto mode, retain the validated local answer with degraded=true and escalation_error. An explicit Jev request remains a 502/503.

## Backend implementations

### Native CoreML ANE

backend-coreml is target_os=macos and uses native CoreML bindings. It loads exactly one configured ANE model per service, validates input names/shapes/dtypes, and exposes tokenizer/capability metadata without a second model copy. Synchronous prediction is behind a bounded blocking executor.

Native errors translate at the boundary:

- ANECapacity: transient memory pressure, ANE compute-unit/queue exhaustion, or known resource-capacity code.
- ANEShape: fixed length, alignment, dtype, mask, or tensor shape mismatch identified by the adapter.
- ANEHard: corrupt/incompatible model, unsupported operation, permission failure, non-finite output, malformed response, or unknown native error.

Do not classify every CoreML exception by broad string matching. Use native error types/codes where available and a small adapter-owned mapping; unknown exceptions are ANEHard. Shape errors usually indicate deterministic configuration/model bugs, so a process-local ANE circuit breaker may remain active after the one permitted retry.

Normalize logits/probabilities, calibration, confidence, action probability, and usage to the laya answer schema. A CoreML model that cannot expose required outputs is unavailable, not a reason to invent a new wire format.

### Native MLX through PyO3 (v1)

The laya-mlx implementation already exists and is the v1 inference engine. Embed its Python module in-process through PyO3 behind the backend trait: initialize one interpreter/runtime per service, import laya_mlx, construct the configured Router/Agents, and call predict with the already selected checkpoint. There is no loopback socket, JSON hop, or duplicate prompt renderer in the hot path. The bridge owns up to max_loaded MLX checkpoints using laya-mlx LRU semantics and returns the exact laya answer payload. Keep all Python objects behind a narrow Rust adapter so native mlx-rs can replace it later.

PyO3 startup, GIL acquisition, model loading, and Python exceptions are measured separately from warm request overhead. The bridge must never import torch or Transformers for inference. A process sidecar can be enabled only as an explicitly documented fallback for environments where PyO3 cannot initialize; it is not the v1 default and does not change routing policy.

### Documented fallback: socket Python sidecar

A managed Python child plus loopback JSON protocol is not the v1 default because local Laya is sub-100 ms and a process/socket hop consumes the latency budget. Document its health, restart, and protocol rules in sidecar-fallback/README.md so it can be enabled for environments where PyO3 cannot initialize without changing PredictBackend.

### Jev / TypeSafe HTTP client

backend-jev uses a pooled reqwest client, configured base URL/path, bounded timeout (25 seconds default), and the key from TYPESAFE_API_KEY or configured environment:

~~~
POST <base_url>/v1/systemone
Authorization: Bearer <TYPESAFE_API_KEY>
Content-Type: application/json

{"model":"jev-latest","state":<state>,"questions":<questions>}
~~~

Preserve current jev-ultrafast retry behavior: only 429, 529, and 503, up to three total attempts, with 0.5/1.0 second backoff. Make it configurable but never retry arbitrary 4xx, malformed responses, or cancellation. Validate returned answers with choice probability invariants where selected choices matter. Normalize model/answers/usage and retain latency. Never include Authorization in errors/logs.

## Engine lifecycle and limits

- Service construction and pure route phase load no weights.
- CoreML owns at most one ANE model; its residency is independent of MLX max_loaded.
- The embedded PyO3 bridge owns up to max_loaded checkpoints through laya-mlx LRU; Rust owns no duplicate MLX weights.
- Jev is stateless except for the HTTP connection pool.
- preload=false means no model engine. preload=true loads configured MLX checkpoints after the embedded bridge is ready; ANE remains lazy unless ane.preload=true. Explicit lifecycle calls may preload/unload by backend/checkpoint. `max_loaded` is a hard resident-checkpoint cap: a preload set larger than it is a configuration error (or must explicitly raise the configured cap before startup), never silent load/evict churn.
- ANE residency warms in the background, never inside a request. The first eligible request on a cold ANE (or any request while its ~50–90 s verify+compile+load is in flight) returns `not_ready`/`ane_warming` immediately and the router serves MLX in the same request; once resident, eligible requests take ANE. ane.preload=true blocks readiness until resident — the production default for latency-critical deployments, since a cold ANE otherwise makes every early eligible request take the MLX path.
- unload(ane) releases CoreML; unload(mlx/checkpoint) drops the corresponding Python Agent; unload(all) releases both. Later requests may lazy-load again.
- Capacity fallback releases ANE before MLX retry and never invokes ANE twice in one request. Deterministic shape failure may circuit-break ANE until reload.
- Semaphores bound CoreML, the embedded MLX bridge/GIL work, and total request concurrency. Busy requests return 429; do not queue unboundedly.

## CLI and operations

~~~
jevalaya serve --config config/jevalaya.toml
jevalaya predict --config ... --state-file state.json --questions questions.json [--backend auto|ane|mlx|jev]
jevalaya check-config --config ...
jevalaya health --url http://127.0.0.1:8767
~~~

predict uses the same core path as HTTP and prints only JSON. check-config validates paths, tokenizer metadata, native capabilities, embedded Python/PyO3 initialization, and Jev endpoint configuration without making inference. Credentials are checked for environment presence only.

Metrics/log fields are request id, route reason, checkpoint, backend, raw/aligned count, fallback/escalation flags, queue wait, backend latency, and status. Never log full state/questions, auth headers, TYPESAFE_API_KEY, or model secrets. Bind loopback by default; LAN requires explicit config.

## Parity and test plan

All tests are offline unless marked live and gated by local model paths.

1. Prompt golden fixtures generated by a pinned laya-mlx reference: string/object/list states, Unicode and non-Latin text, insertion order, structured instructions/criteria, choice/score/noul defaults, mask replacement, option/head caps, exact token ids, marker positions, raw and aligned counts.
2. Routing unit tests with fake tokenizer/capability/backend: short multilingual => ANE; 95/96/97 raw or aligned boundary; alignment crossing 96 => MLX; three questions => MLX; English/typed always MLX; explicit mlx/ane/jev; unavailable ANE route-time MLX; Jev confidence/margin escalation.
3. Backend contract tests: fake CoreML capacity/shape exactly one MLX retry; hard CoreML no retry; PyO3 bridge readiness/crash/protocol mismatch (and the sidecar only in its fallback fixture); Jev retry statuses and response validation without a real key; output normalization.
4. Endpoint tests: /health, /ready, auth, body cap, malformed schema, 404, busy 429, JSON round-trip, and log redaction.
5. Optional Apple Silicon integration: local ANE/MLX directories only, selected-answer/probability tolerance fixture, one PyO3 parity run, no network, no vendored weights.
6. Degradation/compare tests: each 3/2/1-backend health snapshot, zero-backend 503, partial compare success, explicit unavailable backend, low-confidence Jev outage retaining the local answer, and event emission for exclusion/recovery.
7. Reload/observability tests: atomic config snapshots between requests, debounce of invalid intermediate files, append-only JSONL schema, bounded sink backpressure, and secret/state redaction.

Parity acceptance requires selected-answer equality on the reference fixture plus bounded probability/confidence deltas for dtype/backend. Valid JSON alone is insufficient: token ids, markers, answer fields, and usage must agree with laya-mlx.

## Implementation-time confirmations

- Select and pin the native CoreML Rust binding crate/API after checking exact model I/O metadata; keep it target-gated.
- Define the narrow PyO3 module bridge and keep laya-mlx external. Native mlx-rs remains the replacement seam.
- Keep the socket sidecar as an opt-in fallback; do not make it part of the v1 hot path.
- Confirm user-supplied CoreML fixed shape/alignment/output tensor names; defaults assume 96 input tokens and one multilingual model.
- Keep the verified Jev distinction explicit: current jev-ultrafast has no local HTTP endpoint, so /predict is laya compatibility and /v1/systemone is the configurable outbound TypeSafe path.



## Latency budget

The local target is sub-100 ms end to end for a warm one-question request. Keep routing overhead visible instead of hiding it inside backend latency:

- request parse/validation and zero-copy JSON handoff: target p95 <= 0.5 ms;
- language/script and typed-workflow route decision: target p95 <= 0.2 ms;
- warm tokenizer/render/count: target p95 <= 3 ms for one short question;
- dispatch/lock acquisition: target p95 <= 0.5 ms;
- router overhead (tokenize + route + dispatch, excluding model execution): target p95 <= 5 ms and p99 <= 10 ms;
- warm local backend execution: target p95 <= 20 ms for native MLX and <= 10 ms for ANE where the supplied model meets that capability; cold load/preload is reported separately;
- Jev is remote and uses its configured timeout; network latency is reported separately.

Preload tokenizer/config objects, keep model-directory metadata in memory, avoid serializing state more than once, and measure GIL/PyO3 overhead independently. These are budgets and instrumentation gates, not claims about every model or machine.

## Compare mode and shared parallel dispatch

The same dispatch/comparator mechanism powers low-confidence escalation and an explicit compare request. A request may set compare=true or compare=["ane", "mlx", "jev"] (the exact JSON shape is versioned in schema.rs). The router validates availability, removes down/unconfigured backends, renders/tokenizes once per required model family, fans out with bounded concurrency, normalizes every result to the shared answers/usage shape, then computes per-question confidence, top-two margin, selected-label agreement, latency, cost, fallback, and error.

The response keeps the normal primary answers and adds compare containing every backend result and its RouteDecision. A partial compare is useful: unavailable backends appear with a structured error while available answers remain intact. Compare mode never leaks API keys or raw prompts. In auto mode a local low-margin result may invoke this same comparator with Jev as second opinion; if Jev is unavailable, retain the local answer and mark degraded/escalation_error.

## Graceful degradation

Backend availability is a first-class capability snapshot, refreshed at startup and by bounded runtime health checks. The decision table is evaluated against currently available backends:

- with three backends, normal ANE/MLX routing plus optional Jev escalation applies;
- with two, missing ANE never kills MLX and a Jev outage never kills local routing;
- with one, the remaining backend is a validated passthrough and still emits full routing metadata, confidence, token count where available, latency, and degraded status.

An explicitly requested but unavailable backend returns backend_unavailable. Auto mode excludes unavailable backends and chooses the safest available local path; when only Jev remains it uses Jev, and when only local remains it does not attempt a dead remote escalation. Health recovery re-enables a backend without restarting the service. Every exclusion and recovery emits an event.

If the health snapshot has no usable backend for an automatic request, return `503 not_ready` with a stable `backend_unavailable` detail; never fabricate an answer. Compare mode may return a partial result when at least one selected backend succeeds, but a compare with zero successful backends is the same 503 outcome.

For an auto low-confidence result, a failed Jev escalation does not erase a valid local answer. Return the local answer with escalated=true, escalation_error, degraded=true, and the original confidence/margin. An explicit backend=jev request remains a hard upstream failure because no local answer was requested.

## Observability and live tuning

Every routed request emits one append-only structured event to JSONL or a local Unix socket. The event schema is useful to a future native Mac app:

~~~json
{
  "ts": "RFC3339",
  "request_id": "opaque",
  "backend": "ane | mlx | jev",
  "checkpoint": "english | multilingual | typed-decisions | null",
  "reason": "stable code",
  "token_count": 42,
  "confidence": 0.82,
  "margin": 0.31,
  "latency_ms": 8.1,
  "queue_ms": 0.4,
  "fallback": false,
  "escalated": false,
  "degraded": false,
  "cost": { "input_tokens": 42, "output_tokens": 0, "remote_usd": null },
  "available_backends": ["mlx", "jev"],
  "status": 200
}
~~~

Never include state, questions, Authorization, TYPESAFE_API_KEY, model weights, or raw provider payloads in this stream. The event sink must be bounded/non-blocking so observability cannot stall inference; failed writes are counted and do not alter answers.

Thresholds and routing knobs are file-watched or reloadable atomically between requests: confidence margin, optional confidence gate, retry escalation, `prefer_ane`, `ane.max_tokens`, compare defaults, backend enablement, and concurrency. A request sees one immutable config snapshot. A future Mac app may mutate this surface, but the app is out of scope for v1.

## Consumer feedback

Consumers can report how routed answers performed through a configured local JSONL file, defaulting to `~/.jevalaya/feedback.jsonl`. Expand the home-directory prefix when resolving this config path. Each line is a standalone record:

~~~json
{
  "consumer": "local-client",
  "request_id": "opaque-id-from-predict",
  "ts": "2026-09-22T00:00:00Z",
  "answer_ok": false,
  "expected": { "question_id": { "choice": "expected-label" } },
  "notes": "Optional description of the observed outcome."
}
~~~

`consumer` and `request_id` are required nonempty strings, `ts` is an RFC3339 timestamp, and `answer_ok` is a required boolean. `expected` is optional arbitrary JSON (prefer a map keyed by question id for multiple questions); `notes` is an optional string. Consumer names are opaque data. Return the effective request id in `/predict` routing metadata and the routing event, generating one when absent, so consumers can join feedback to backend, checkpoint, latency, confidence, and config version. Include `config_version` in routing events to identify the policy used for each result.

Service writes append one complete JSON record per line with serialized access to the configured file. Consumers may also append directly or edit the file while the writer is stopped; offline readers report malformed lines and continue with valid records. This remains a plain local file, without a database or automatic threshold updates. Feedback is consumer-reported evidence for later evaluation and threshold tuning; preserve conflicting or repeated reports rather than assuming each is an independent verified label.

Optionally enable `POST /feedback` to validate and append the same record, using the prediction endpoint's bearer authentication and body limit. Return `201 {"ok":true}` only after the append succeeds; invalid records return 400 and sink failures return 503. An unknown request id may still be recorded because routing logs can rotate. When disabled, the endpoint returns 404. Feedback write failures do not change prediction results. The service never inserts prompt state or credentials into feedback; `expected` and `notes` contain only what the consumer supplies.

## Chunked inference strategy

Status: proposed, offline/shadow experiment for T024; not a relaxation of the production routing rules above. Fixed-length ANE calls have no continuation/KV state: repeating a question on several slices and aggregating is a new predictor, not equivalent whole-context inference. Chunking must not silently turn an over-limit request into ordinary `ane_eligible=true` or substitute an English/typed request into the multilingual model. Family-specific artifacts and routing capabilities are a separate prerequisite; see [ANE export feasibility](ANE-EXPORT-FEASIBILITY.md).

### Recommendation and ranked options

1. Keep the whole state on a compatible MLX checkpoint, or a validated longer ANE profile when available. This is the correctness baseline and the preferred automatic route; a larger shape preserves cross-section attention that voting cannot recover.
2. Measure non-overlapping token windows with the unchanged question as the simplest baseline: choice majority, score mean, noul any-hit. These are experimental hypotheses, not generally valid semantics. Then compare sentence/paragraph packing within the same token budget; this is the first production candidate for explicitly declared local-evidence tasks.
3. Add modest overlap (start with 25% of the state window) only if boundary-error reduction pays for extra calls. Combine with abstention and whole-context escalation, rather than forcing every document into a voted answer.
4. Test confidence/margin-weighted pooling, rank fusion, and head/tail weighting as ablations after the unweighted baseline. They add assumptions that must earn their complexity on held-out labels.
5. Leave hierarchical meta-questions last. They change the task twice, cost another call, and cannot restore discarded evidence.

The reported service baseline is roughly 77 ms per warm serial ANE call, not a new measurement in this design. Two/four chunks therefore spend roughly 154/308 ms in backend work alone before routing or escalation. Chunking is not a way to meet the current sub-100 ms common-path target on those timings. Its plausible value is a measured energy/resource tradeoff or a disagreement signal; neither is established by making all slices fit. Record actual elapsed latency, not `k × single-call p95` presented as a measured percentile.

### Geometry and exact prompt budget

Let `L` be the selected bundle's fixed length, `a` its alignment, `P` the complete canonical question prefix length, and `K` the number of option markers. The state allowance is `B = a * floor(L / a) - P - 1`, reserving the final separator. At L96/a1, `B = 95 - P`; a fixed 60-token state window only fits when P <= 35. Build P with the bundle tokenizer and the existing option/head caps, not a special shortened question. Reject `B <= 0`, excess K, lost markers, or any fully re-rendered chunk whose aligned count exceeds L. Do not truncate options/instructions further to manufacture capacity.

- Naive windows: contiguous, complete coverage in original order. Compute offsets on the canonical mask-sanitized state text, retaining an original-text mapping if offsets are exposed. Use tokenizer offsets to choose valid text boundaries, then tokenize each actual chunk again; decoding an arbitrary BPE slice and re-encoding is not guaranteed to preserve ids or length. Verify exact ids/masks/markers and zero backend truncation. Preserve the original full state for any later escalation.
- Sentence/paragraph packing: greedily pack whole spans up to B. Split an oversized span at a safe text boundary and flag it. Boundary detection is a cheap language-dependent heuristic, not a guarantee against splitting negation, pronouns, or linked facts. Compare at matched call budgets and report when packing increases k.
- Sliding windows: stride `B - overlap`, with `0 <= overlap < B`. Cover the tail, avoid identical duplicate windows, and record source offsets. Overlap repeats evidence; votes are correlated and must not be interpreted as independent witnesses. Use unique-source-coverage weights as an ablation, not a claim that correlation has been eliminated.
- Head/tail bias: either weight first/last windows while still examining all spans, or explicitly call the operation evidence selection when middle spans are omitted. It can help position-biased tasks but has no generic justification for arbitrary state. Keep it off by default and test evidence placed only in the middle.
- Structured state: do not chop serialized JSON and assume fragments retain their key/path semantics. Start with string state. A later generic record-aware adapter may repeat necessary field/path context, charging those tokens against B and requiring caller-declared independent records; nested relational state otherwise goes whole-context. Never add consumer-specific field names to the router.

The first experiment uses one question. Supporting several questions requires separate head budgets and normally `sum(k_q)` calls, not k calls shared across different heads; splitting questions also remains an explicit strategy change. Pin checkpoint, tokenizer, bundle, policy/config generation, and question/option order across all calls.

### Aggregation semantics by question type

The source contract matters: `laya_mlx/agent.py` returns choice probabilities, score as `sum(level_index * p_level)`, and noul as `p(true)`; its `action.act_probability` is a separately learned head. None of those types specifies how facts across documents compose.

| Type | First baselines | Conditions and failure modes |
| --- | --- | --- |
| choice | Hard majority; uniform mean probability vector with argmax | Plausible for a declared document-level topic-vote task; a single decisive span can be outvoted by irrelevant spans. Ties abstain. A confidence-weighted sum can amplify confidently irrelevant chunks. |
| score | Mean per-chunk score; weighted mean of level distributions | A mean is appropriate only when the requested quantity really decomposes into independent, equally weighted units, or caller-defined weights. Unequal chunks and ordinal/global rubrics do not automatically satisfy that assumption. Counts, maxima, totals, and overall quality are different operations. |
| noul | Any positive chunk / max p(true); compare mean p(true) separately | Any-hit applies only to an explicitly existential predicate with self-contained evidence. It is wrong for universal conditions, absence, consistency, or document-wide truth. Mean probability is not an existential probability either. |

For choice/score, retain each `p_i` in the same label order and test pooled `p_bar = sum(w_i * p_i) / sum(w_i)`, requiring finite nonnegative weights with positive total; an all-zero weight set abstains. Uniform weights are the baseline. Margin/entropy-confidence weights are candidates, not calibrated correctness weights; compare with novelty/length weights only where the task's unit of evidence warrants them. Reciprocal-rank fusion `R_j = sum(w_i / (c + rank_i(j)))`, with fixed positive c, tests reliance on ordering instead of probability magnitudes, but loses uncertainty information and still rewards repeated irrelevant evidence. It is lower priority than simple pooling plus abstention.

For existential noul, declare the per-chunk positive threshold and test document-level false positives as k grows. Even under an illustrative independence assumption, per-chunk false-positive rate alpha yields document false-positive rate `1 - (1 - alpha)^k`; actual overlapping windows are dependent. Neither noisy-OR nor multiplied probabilities is justified without a dependence/calibration model. A negative answer requires full coverage and adequate evidence detection; “no slice voted true” is not proof of absence. Early stopping is allowed only for an explicitly validated existential-witness policy, never merely because the current majority appears stable.

### Disagreement as an escalation signal

Measure evidence sensitivity separately from each call's uncertainty. For normalized nonnegative weights summing to one, let `v_j = sum_i w_i * 1[argmax(p_i) = j]`. With K >= 2, record:

~~~
vote_entropy = -sum_j v_j * log(v_j) / log(K)       # 0 log 0 = 0
vote_margin = largest(v) - second_largest(v)
pooled_margin = largest(p_bar) - second_largest(p_bar)
between_chunk_js = H(p_bar) - sum_i w_i * H(p_i)
within_chunk_entropy = sum_i w_i * H(p_i)
~~~

Vote entropy/margin detect changing winners; the Jensen-Shannon quantity detects differing distributions even with the same winner. Within-chunk entropy detects uniformly uncertain calls that a unanimous vote hides. For score, additionally report weighted variance/range of per-chunk expected levels; adjacent levels are not the same disagreement as opposite ends of the rubric. For noul, use `[1 - p(true), p(true)]`. Renormalize finite rounded probability vectors before calculating diagnostics; reject invalid vectors, and define K=1 diagnostics as degenerate rather than dividing by log(1).

Candidate escalation policy: any semantic/budget/coverage gate fails, an aggregate tie occurs, vote entropy exceeds a learned threshold, pooled margin falls below a learned threshold, or within-chunk uncertainty exceeds a learned threshold => use the original state on the whole-context path. Disagreement and uncertainty are complementary OR features to evaluate, not mandatory universal thresholds. Calibrate by checkpoint, question type, K and k where data supports it: vote entropy's attainable range changes with the number of chunks. Do not adopt the ordinary single-call confidence threshold unchanged for a pooled result.

This signal is a hypothesis about evidence dependence, not an accuracy certificate. High divergence can mean valid locally different facts, not model failure; unanimous chunks can all be wrong because every slice lost a necessary relationship. Compare disagreement-only, current margin-only, and combined triggers at the same escalation rate/accepted coverage. Prefer whole-context MLX as the next local opinion; only the existing consent/configured policy may then call Jev. Offline sweeps log `would_escalate` and make no additional paid requests.

### Semantic pre-gates: what not to chunk

Default to whole-context for undeclared semantics. A generic caller/experiment can explicitly declare modes such as `document_vote`, `independent_unit_mean`, or `existential_local_evidence`; these are proposed policy metadata, not accepted wire fields yet. A type (`choice`, `score`, `noul`) or keyword alone is never positive proof that one of these modes is safe.

Cheap conservative rejection cues include count/total, compare across sections, sequence/latest, contradiction/consistency, all/every, none/absence, global optimum, linked records, or instructions requiring multiple facts together. They are language-dependent deny signals, not a complete classifier. For example, “does any pair conflict?” contains “any” but still needs cross-chunk reasoning. When classification is uncertain, or structured-state relationships are unknown, bypass chunking. Preserve checkpoint and option-count exclusions; chunking state does not solve a head that consumes L or K above the artifact's output capacity.

### Hierarchical and iterative variants

Laya can technically accept a second typed question over a serialized list of first-stage winners/probabilities or extracted text. That is a new task requiring independent evaluation, not continuation of the original request. Winner labels alone discard negative evidence, contradictions, and context; asking a model to vote on them cannot reconstruct what it never saw. Adding evidence windows may itself exceed L, and Laya's typed head does not generate faithful free-text summaries/rationales.

If explored, use deterministic extracts with source-span ids, bound the experiment to k first-stage calls plus one meta-call, and expose both stages in results. Re-render and count the meta-question under its own budget. Do not recursively summarize until something fits. Compare against simple pooling and whole-context MLX; do not use meta-model agreement as ground truth. This stays behind the basic geometry/disagreement experiments in priority.

### Failure taxonomy, lifecycle, and answer integrity

| Failure | Required behavior / stable reason candidate |
| --- | --- |
| Unknown/global semantics, relational structured state | Bypass before inference: `chunk_semantics_unsupported`. |
| No state room, excessive options, marker loss | Bypass: `chunk_head_over_capacity` / `chunk_options_over_capacity`; never trim the question or drop choices. |
| Uncovered spans, unsafe boundaries, overflow after re-tokenization | Do not aggregate an incomplete request: `chunk_coverage_invalid` / `chunk_render_overflow`. |
| Conflicting votes, tie, uncertain aggregate | Whole-state escalation: `chunk_disagreement` / `chunk_uncertain`; semantic escalation is not a native ANE error. |
| Native capacity/shape error mid-series | Abort the series; apply the existing narrow taxonomy with at most one MLX retry of the original full request. Do not retry each slice and quietly mix different backends into an ANE vote. |
| Native hard error or malformed/non-finite output | Surface the existing hard error; do not disguise it as disagreement or silently discard the failed slice. |
| Chunk cap/deadline/queue exhausted | Preflight bypass when possible: `chunk_budget_exceeded`; otherwise stop scheduling. Whole-state escalation needs remaining budget; never return partial aggregation as a complete answer. |

Use the shared dispatch executor, bounded queues and a per-request `max_chunks`/total deadline. Reserve fallback budget before starting if fallback is promised. No unbounded chunk fan-out or nested retry loops. One immutable capability/config snapshot governs the series; a model lease and native execution permit survive cancellation until actual native work completes. Include queue time, repeated question tokens, all attempts, and later escalation in total latency/usage/cost. Profile serial execution first; running competing ANE calls concurrently is not assumed to reduce latency.

Keep production `answers` unchanged during the experiment: offline reports or opt-in shadow records contain aggregates, while `/predict` still returns the normal whole-context primary result. Store the original full raw count plus per-chunk counts; do not replace the routing gate count with the largest small slice. Do not manufacture an `action.act_probability` by averaging learned action heads, or call vote concentration the original model's confidence. Promotion to live answers requires a versioned aggregate schema/calibration decision (including the action field) and an explicit strategy identifier; same-shaped JSON alone is insufficient compatibility.

One request event may contain bounded chunk summaries: strategy/version, geometry, k, source offsets, coverage/overlap fraction, raw/full and per-chunk token counts, checkpoint/profile, aggregate method, disagreement statistics, trigger, per-stage timings and total cost. Keep state/evidence text out of logs. Any future runtime strategy config is off by default and reloads atomically under the existing config rules.

### What T024 should measure first

1. Paired baseline, same labeled documents/questions and checkpoint family: whole-context MLX; ANE chunks; MLX on those exact same chunks with the same aggregator. The latter isolates chunk/aggregation loss from CoreML numeric/backend drift. Add a whole-context larger ANE profile when available. If the “whole-context” request exceeds MLX's own max_len, flag truncation and report it separately; it is not a full-evidence reference. Forced multilingual experiments on English data are labeled as such, never silently compared as equivalent to the English checkpoint.
2. Start with k=2/3/4 and the measured dynamic head budget. Establish naive non-overlap, then sentence packing, then 25% overlap; record actual k and unique coverage. Do not tune geometry and aggregation simultaneously. Stratify by question type, family, head size, K, state length, and evidence location. Keep single-chunk fitting controls.
3. Use labels for accuracy; MLX agreement is only a diagnostic. Report choice accuracy/macro-F1, score error and rubric-appropriate ordinal metrics, and noul precision/recall/false positives. Include rare decisive evidence among irrelevant chunks, cross-boundary negation/coreference, contradictions, reordered facts, counts/universal/absence questions, duplicated overlap, long options, Unicode, and structured keys as stress/rejection cases.
4. Fit disagreement/margin thresholds on a validation split; freeze them before held-out testing. Keep chunks from one document in the same split, and group related documents to avoid leakage. Report accepted-set error versus coverage, would-escalate rate, escalations that correct versus regress an answer, overall routed accuracy and remaining undetected errors. Compare trigger families at equal remote/escalation budgets; bootstrap by document, not by chunk.
5. Report end-to-end p50/p95/p99, per-call/queue/render time, k-dependent total tokens, rejection rates, completed/aborted series, warm/cold conditions and memory. Include the latency of the whole-state escalation in the policy result. Energy/power is a separate measurement prerequisite for any efficiency claim, not an inference from using ANE. Keep Jev disabled in sweeps unless a separately authorized, capped evaluation requests it.

Promotion requires a predeclared quality tolerance and a measured benefit against whole-context MLX on the intended semantic subset, plus bounded failure behavior and answer-contract tests. Reject the strategy when it is dominated on quality/latency/resource use. ANE coverage percentage by itself is not a success metric.

## Voice and naming

The package, Cargo crates, CLI, and repository are jevalaya, the fusion of Jev and Laya. User-facing prose, README copy, CLI help, and tasteful log headings may carry a warm Deep South/Cajun register: an occasional "cher" (shah), "lagniappe", r-dropped phrasing such as "togetha", and "Laissez les bons temps rouler!" as an opener. Technical identifiers, JSON keys, error codes, thresholds, and machine-readable logs stay plain and stable. Flavor is seasoning, never an obstacle to precise operations or safe failure.
