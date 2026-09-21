Design handoff for `laya-hybrid` (implementation target: devin-max)

The package should be a thin orchestrator over `laya-mlx` plus an optional `laya-coreml` ANE adapter. It must not vendor either package and must not import torch/transformers during inference. A local path is always resolved before any hub lookup; a path-like missing directory is an error, so offline local directories work without network access.

## Module map

### `router.py`

Public `Router` and `RouteDecision`. The constructor mirrors the requested API:

`Router(dtype="float16", max_loaded=1, preload=False, compile=False, cache_prompts=False, pad_to_multiple=None, ane_model="aac6fef/laya-multilingual-coreml-ane", ane_enabled=True, max_ane_tokens=96, prefer_ane_for_short=True, **mlx_kwargs)`

It owns policy only: checkpoint-router invocation, prompt-token metadata, ordered route selection, backend invocation, and ANE-capacity/shape retry. It does not know CoreML call details or MLX tensor details.

Use a serializable mapping (dict-like, like `laya_mlx.router.RouteDecision`) with this required shape:

```text
{
  checkpoint: "english" | "multilingual" | "typed-decisions",
  backend: "mlx" | "ane",       # backend actually selected/executed
  model_id: str,                # configured/resolved id for that backend
  reason: str,                  # stable prefix/reason code plus useful detail
  token_count: int | None,      # raw rendered-token count, before padding
  ane_eligible: bool,
  fallback: bool,
  latency_ms: float | None      # None from route(); end-to-end inference time from predict()
}
```

Optional fallback diagnostics may add `fallback_from="ane"` and `fallback_reason`; do not make callers parse an exception string. Keep compatibility properties `.model` (alias of `checkpoint`) and `.repo` if useful, but the JSON routing contract is the fields above.

`predict(state, questions, *, backend=None, model=None, task=None, lang=None)` returns exactly the laya-mlx answer payload (`model`, `answers`, `usage`) plus `routing`. Normalize ANE output to the same `choice`/`score`/`noul` answer shapes, confidence, action probability, and `usage.output_tokens == 0`. `system_one = predict` is a useful drop-in alias.

### `tokenize.py`

The only canonical prompt renderer/token counter. It returns a small immutable `RenderedPrompt` containing token ids, raw count, padded count, question count, and any typed-workflow marker needed by policy. Do not count characters/words and do not count an already padded CoreML tensor.

Use the tokenizer bundled with the configured ANE checkpoint for ANE eligibility (Rust `tokenizers` tokenizer loaded from `tokenizer.json`/config, or the tokenizer exposed by laya-coreml). Reuse laya-mlx's public/common rendering primitives rather than inventing a second template: `serialize_state` (JSON for dict/list, unchanged string), `render_options`, and the sequence layout `[CLS] <type> question: <instructions> [SEP] [MASK] option... [MASK] ... [SEP] state [SEP]`. Normalize question definitions with the same choice/score/noul validation. If the ANE adapter exposes a canonical renderer, call it; otherwise the above layout is the fallback.

Render before truncation. Count special tokens, masks, options, and state tokens exactly as they will be submitted. Never truncate a request to make it fit ANE: if raw or aligned/padded length exceeds `max_ane_tokens`, route MLX. For one question let `padded_count = ceil(raw/alignment) * alignment` (alignment is the ANE/model alignment, or 1 when none is advertised); eligibility requires `padded_count <= max_ane_tokens`. The ANE adapter pads its eligible input to the model's fixed sequence shape (normally 96). `token_count` in routing is always raw count. For multiple questions, ANE is ineligible; report the maximum per-question rendered count (or None when no valid prompt can be rendered) rather than a misleading sum.

### `backends.py`

Backend protocols/adapters, capability checks, answer normalization, and typed error translation. Suggested pieces:

- `MLXBackend`: wraps the existing `laya_mlx.Router`/Agent; its LRU is the source of truth for MLX checkpoints.
- `ANEBackend`: lazy wrapper around laya-coreml/CoreML; owns one loaded CoreML agent and exposes tokenizer/capabilities (family multilingual, fixed input length, alignment). It accepts a local model directory and never imports PyTorch.
- `BackendUnavailable`/capability checks for route-time decisions.
- Native CoreML errors are translated here to `ANECapacityError`, `ANEShapeError`, or `ANEHardError`. Router must catch only the first two.

No backend module decides english vs multilingual or whether a question is “short”; that belongs in router/tokenize.

## Ordered routing (must be deterministic)

1. **Backend override.** Validate `backend in {auto, mlx, ane}`. `backend="mlx"` is a hard no-ANE request; it still asks the checkpoint router for the checkpoint family. `backend="ane"` is an intent, not a safety bypass: it continues through all ANE gates below. If a gate fails, return MLX with a reason such as `backend_override:ane_rejected:<gate>`; english and typed-decisions can never be forced onto ANE.
2. **Existing checkpoint router.** Call `laya_mlx.Router.route(state, questions, model=model, task=task, lang=lang)` without loading weights. Preserve its precedence for explicit model/task/lang and language detection. Normalize the result to `checkpoint`. Treat a typed workflow as ANE-blocking even if the underlying router was not configured for automatic typed detection (explicit `task=typed_decisions`, checkpoint `typed-decisions`, or a matching workflow signature).
3. **Rendered token count.** Build the canonical prompt through `tokenize.py`; this is the count used for the decision, not the MLX batch's dynamic padding. If rendering/tokenizer capability is unavailable, mark ANE ineligible and use MLX (unless the request was explicitly malformed, in which case propagate the validation error).
4. **ANE eligibility.** All of these must hold: `ane_enabled`; macOS + arm64; ANE dependency/model/tokenizer available; checkpoint is `multilingual`; not typed; exactly one question; `prefer_ane_for_short` (unless an explicit ANE intent); and aligned/padded rendered length <= `max_ane_tokens` (default 96). English and typed-decisions are unconditional MLX. A disabled/unavailable ANE is a normal MLX route-time decision, not a runtime fallback.
5. **MLX default.** If any gate fails, select MLX with the checkpoint chosen in step 2 and a stable reason (examples: `checkpoint:english`, `not_multilingual`, `question_count:3`, `token_count:97>96`, `typed_decisions`, `ane_disabled`, `platform_unsupported`, `ane_unavailable`). `ane_eligible=false` for these decisions.
6. **ANE-error fallback (during predict only).** On an eligible ANE decision, load/invoke the single ANE agent. If the adapter raises `ANECapacityError` or `ANEShapeError`, release/reset the ANE instance as appropriate, retry exactly once through MLX on the same `multilingual` checkpoint, and set `fallback=true`. The returned routing metadata describes what actually produced the answer (`backend="mlx"`, MLX `model_id`, `fallback=true`, `fallback_from="ane"`); retain `ane_eligible=true` so the original decision is observable. Measure `latency_ms` end-to-end including retry. Never retry an ANE hard failure.

Error taxonomy is deliberately narrow: capacity = transient memory/compute-unit/ANE queue pressure; shape = fixed sequence length, alignment, dtype, or input-shape mismatch that the adapter can identify. Missing/corrupt model files, tokenizer mismatch, permission/offline download failure, unsupported operation, non-finite outputs, invalid input/schema, cancellation, and unexpected programming errors are hard failures (or route-time unavailability where capability can be checked before invocation); surface them instead of hiding them behind MLX.

## Engine lifecycle

- Constructor and `route()` do not load weights. Tokenizer/config reads are allowed and lazy.
- Exactly one ANE agent is held per Router. Guard first-load with a lock; `max_loaded` never counts/evicts ANE.
- MLX loading delegates to `laya_mlx.Router(max_loaded=...)`, preserving its LRU. Eviction drops the Agent and releases MLX cache.
- `preload=False` means no engines. `preload=True` follows laya-mlx semantics for configured MLX checkpoints; keep ANE lazy unless an explicit `preload(..., include_ane=True)`/equivalent is requested. Expose `preload(names=None, include_ane=False)` and `unload(name=None)`; `unload("ane")` releases only CoreML, and `unload()` releases both. An ANE capacity/shape fallback must not trigger a second ANE attempt in the same request; a later request may lazy-reload after capacity release, while a deterministic shape mismatch may be circuit-broken for the Router instance.
- `ane_model` and MLX model values may be local directories. Do not vendor checkpoints or silently require online access.

## CLI

Expose `laya-hybrid = laya_hybrid.cli:main` and a `predict` subcommand:

```text
laya-hybrid predict (--state TEXT | --state-file PATH) --questions PATH
  [--backend auto|mlx|ane] [--model MODEL_OR_PATH] [--task TASK] [--lang LANG]
  [--dtype float16|float32|bfloat16] [--max-loaded N] [--preload]
  [--compile] [--cache-prompts] [--pad-to-multiple N]
  [--ane-model MODEL_OR_PATH] [--no-ane] [--max-ane-tokens 96]
  [--no-prefer-ane-for-short]
```

Print only UTF-8 JSON (same payload as `Router.predict`) to stdout; diagnostics/errors go to stderr with non-zero exit. `--state-file` and `--questions` are JSON files; local model paths must work with no network.
