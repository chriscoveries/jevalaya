//! Axum surface for the router (DESIGN.md §Wire contract):
//! `POST /predict`, `POST /route`, `GET /health`, `GET /ready`,
//! `POST /feedback` (consumer run judgments → configured JSONL sink).
//! Bearer auth on /predict + /route + /feedback when `auth_token_env` is
//! configured, 256 KiB body cap, inflight semaphore → 429, per-request
//! timeout → 504. Error bodies are always `{ "error": { "code", "message" } }`.

use std::sync::Arc;
use std::time::Duration;

use axum::body::Bytes;
use axum::extract::{DefaultBodyLimit, Request, State};
use axum::http::{header, HeaderMap, StatusCode};
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router as AxumRouter};
use serde_json::{json, Map, Value};
use tokio::sync::Semaphore;
use tower_http::trace::TraceLayer;

use jevalaya_router_core::{BackendKind, PredictBackend, PredictBody, RouteError, Router};

/// Shared handler state.
pub struct AppState {
    pub router: Router,
    /// Bearer token; None = auth disabled (config had no auth_token_env).
    pub auth_token: Option<String>,
    /// Configured default checkpoint's model_id for the health `model`
    /// field (laya-mlx compatibility: `{ok, model}`).
    pub health_model: String,
    pub inflight: Semaphore,
    pub request_timeout: Duration,
    /// Consumer feedback sink (`[feedback] sink = "jsonl:PATH"`); None =
    /// unconfigured and `POST /feedback` answers 503.
    pub feedback_sink: Option<std::path::PathBuf>,
}

fn err(status: u16, code: &'static str, message: impl Into<String>) -> Response {
    (
        StatusCode::from_u16(status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
        Json(json!({"error": {"code": code, "message": message.into()}})),
    )
        .into_response()
}

fn route_err(e: RouteError) -> Response {
    (
        StatusCode::from_u16(e.status()).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
        Json(e.body()),
    )
        .into_response()
}

/// Constant-time-ish bearer compare (local service; avoids early-exit
/// length/mismatch leaks without pulling in `subtle`).
fn token_eq(a: &str, b: &str) -> bool {
    let (a, b) = (a.as_bytes(), b.as_bytes());
    let mut diff = a.len() ^ b.len();
    for i in 0..a.len().max(b.len()) {
        diff |= a.get(i).copied().unwrap_or(0) as usize ^ b.get(i).copied().unwrap_or(0) as usize;
    }
    diff == 0
}

async fn auth(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    request: Request,
    next: Next,
) -> Response {
    let Some(expected) = &state.auth_token else {
        return next.run(request).await;
    };
    let ok = headers
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))
        .is_some_and(|tok| token_eq(tok, expected));
    if ok {
        next.run(request).await
    } else {
        err(401, "unauthorized", "missing or incorrect bearer token")
    }
}

/// `{state, questions}` with strict-field parsing → PredictBody, or the
/// structured 400. Bytes already honored the body limit layer.
fn parse_body(bytes: &Bytes) -> Result<PredictBody, RouteError> {
    serde_json::from_slice::<PredictBody>(bytes)
        .map_err(|e| RouteError::InvalidRequest(format!("malformed body: {e}")))
}

async fn predict(
    State(state): State<Arc<AppState>>,
    body: Result<Bytes, axum::extract::rejection::BytesRejection>,
) -> Response {
    let _permit = match state.inflight.try_acquire() {
        Ok(p) => p,
        Err(_) => {
            return err(429, "busy", "inflight request limit reached");
        }
    };
    let bytes = match body {
        Ok(b) => b,
        Err(_) => {
            return err(
                413,
                "body_too_large",
                "request body exceeds the configured limit",
            )
        }
    };
    let parsed = match parse_body(&bytes) {
        Ok(b) => b,
        Err(e) => return route_err(e),
    };
    match tokio::time::timeout(state.request_timeout, state.router.predict(parsed)).await {
        Ok(Ok(resp)) => (
            StatusCode::OK,
            Json(serde_json::to_value(resp).unwrap_or(Value::Null)),
        )
            .into_response(),
        Ok(Err(e)) => route_err(e),
        Err(_) => err(504, "timeout", "request exceeded the configured timeout"),
    }
}

async fn route_only(
    State(state): State<Arc<AppState>>,
    body: Result<Bytes, axum::extract::rejection::BytesRejection>,
) -> Response {
    let bytes = match body {
        Ok(b) => b,
        Err(_) => {
            return err(
                413,
                "body_too_large",
                "request body exceeds the configured limit",
            )
        }
    };
    match parse_body(&bytes).and_then(|b| state.router.route_only(b)) {
        Ok(decision) => (StatusCode::OK, Json(decision)).into_response(),
        Err(e) => route_err(e),
    }
}

/// One backend's `GET /health` entry. Shape contract — dashboards read
/// this, so keys are additive-only once shipped:
/// - `available`: last-known capability flag, i.e. "may serve". It is NOT
///   a reachability guarantee (Jev reports API-key presence, never endpoint
///   liveness) and NOT dispatch readiness (ANE reports available while
///   warming — an explicit pin still falls back until resident; the only
///   machine-read readiness signal today is the "; warming" suffix in
///   `detail`, as `wait_ane_ready` parses it).
/// - `configured`: the adapter is constructed and registered. Absent
///   backends render as null (not configured at all).
/// - `reachable`: null — health never probes the network and never fetches
///   keys, so reachability is untested by construction. A future prober may
///   fill true/false; until then, never read `available` as "will serve".
fn backend_entry(backend: Option<&Arc<dyn PredictBackend>>) -> Value {
    match backend {
        None => Value::Null,
        Some(b) => {
            let c = b.capabilities();
            json!({
                "available": c.available,
                "detail": c.detail,
                "configured": true,
                "reachable": Value::Null,
            })
        }
    }
}

async fn health(State(state): State<Arc<AppState>>) -> Response {
    let avail = state.router.availability();
    let backends = |k: BackendKind| backend_entry(state.router.backend(k).as_ref());
    let any = avail.list();
    Json(json!({
        "ok": !any.is_empty(),
        "model": state.health_model,
        "backends": {
            "ane": backends(BackendKind::Ane),
            "mlx": backends(BackendKind::Mlx),
            "jev": backends(BackendKind::Jev),
        },
        "available_backends": avail.list(),
        "degraded": state.router.degraded(),
    }))
    .into_response()
}

async fn ready(State(state): State<Arc<AppState>>) -> Response {
    let avail = state.router.availability();
    let list = avail.list();
    if list.is_empty() {
        err(503, "not_ready", "no usable backend")
    } else {
        Json(json!({"ready": true, "available_backends": list})).into_response()
    }
}

/// Validate a feedback record per docs/CONSUMER-HANDOVER.md. Required:
/// `consumer` + `request_id` (strings), `chosen_answer_ok` (bool).
/// `expected`/`notes`/`ts` optional; unknown fields pass through untouched.
fn validate_feedback(value: &Value) -> Result<Map<String, Value>, String> {
    let obj = value.as_object().ok_or("body must be a JSON object")?;
    let req_str = |key: &str| {
        obj.get(key)
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
            .ok_or(format!("missing required string field {key:?}"))
            .map(str::to_string)
    };
    let consumer = req_str("consumer")?;
    let request_id = req_str("request_id")?;
    let chosen = obj
        .get("chosen_answer_ok")
        .and_then(Value::as_bool)
        .ok_or("missing required boolean field \"chosen_answer_ok\"")?;
    let mut rec = Map::new();
    rec.insert("consumer".to_string(), Value::String(consumer));
    rec.insert("request_id".to_string(), Value::String(request_id));
    rec.insert("chosen_answer_ok".to_string(), Value::Bool(chosen));
    for key in ["expected", "notes", "ts"] {
        if let Some(v) = obj.get(key) {
            rec.insert(key.to_string(), v.clone());
        }
    }
    // Pass through anything else the consumer attached (forward-compat).
    for (k, v) in obj {
        if !rec.contains_key(k) {
            rec.insert(k.clone(), v.clone());
        }
    }
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    rec.insert("received_unix".to_string(), Value::from(now));
    Ok(rec)
}

/// `POST /feedback`: auth-gated consumer judgments → JSONL sink.
/// Validated records append as one line each; 202 on success, 400 on bad
/// records, 503 when no sink is configured, 500 when the sink won't write.
/// Deliberately outside the inference semaphore — recording must never
/// compete with predictions.
async fn feedback(
    State(state): State<Arc<AppState>>,
    body: Result<Bytes, axum::extract::rejection::BytesRejection>,
) -> Response {
    let Some(sink) = &state.feedback_sink else {
        return err(503, "not_ready", "feedback sink not configured");
    };
    let bytes = match body {
        Ok(b) => b,
        Err(_) => {
            return err(
                413,
                "body_too_large",
                "request body exceeds the configured limit",
            )
        }
    };
    let value: Value = match serde_json::from_slice(&bytes) {
        Ok(v) => v,
        Err(e) => return err(400, "invalid_request", format!("malformed body: {e}")),
    };
    let rec = match validate_feedback(&value) {
        Ok(r) => r,
        Err(e) => return err(400, "invalid_request", e),
    };
    let mut line = serde_json::to_string(&rec).unwrap_or_else(|_| "{}".to_string());
    line.push('\n');
    match std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(sink)
        .and_then(|mut f| {
            use std::io::Write;
            f.write_all(line.as_bytes())
        }) {
        Ok(()) => (StatusCode::ACCEPTED, Json(json!({"ok": true}))).into_response(),
        Err(e) => err(500, "feedback_failed", format!("cannot append feedback: {e}")),
    }
}

/// Build the axum app. `max_body` feeds axum's request-body limit so
/// oversized payloads are rejected before buffering past the cap.
pub fn app(state: Arc<AppState>, max_body: usize) -> AxumRouter {
    let authed = AxumRouter::new()
        .route("/predict", post(predict))
        .route("/v1/systemone", post(predict))
        .route("/route", post(route_only))
        .route("/feedback", post(feedback))
        .route_layer(middleware::from_fn_with_state(state.clone(), auth));
    AxumRouter::new()
        .merge(authed)
        .route("/health", get(health))
        .route("/ready", get(ready))
        .layer(DefaultBodyLimit::max(max_body))
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rec(v: Value) -> Result<Map<String, Value>, String> {
        validate_feedback(&v)
    }

    #[test]
    fn feedback_validation_accepts_full_and_minimal() {
        let full = rec(json!({
            "consumer": "my-app", "request_id": "abc",
            "chosen_answer_ok": false, "expected": "billing",
            "notes": "close call", "ts": "2026-01-01T00:00:00Z",
        }))
        .unwrap();
        assert_eq!(full["consumer"], json!("my-app"));
        assert_eq!(full["expected"], json!("billing"));
        assert!(full["received_unix"].is_number());

        let minimal = rec(json!({
            "consumer": "c", "request_id": "r", "chosen_answer_ok": true,
        }))
        .unwrap();
        assert_eq!(minimal["chosen_answer_ok"], json!(true));
    }

    #[test]
    fn feedback_validation_rejects_missing_and_mistyped() {
        assert!(rec(json!({"request_id": "r", "chosen_answer_ok": true})).is_err());
        assert!(rec(json!({"consumer": "c", "request_id": "r"})).is_err());
        assert!(
            rec(json!({"consumer": "c", "request_id": "r", "chosen_answer_ok": "yes"}))
                .is_err()
        );
        assert!(rec(json!(["not", "an", "object"])).is_err());
    }

    struct ScriptedBackend {
        kind: BackendKind,
        available: bool,
        detail: String,
    }

    #[async_trait::async_trait]
    impl jevalaya_router_core::PredictBackend for ScriptedBackend {
        fn kind(&self) -> BackendKind {
            self.kind
        }
        fn capabilities(&self) -> jevalaya_router_core::BackendCapabilities {
            jevalaya_router_core::BackendCapabilities {
                kind: self.kind,
                available: self.available,
                detail: self.detail.clone(),
            }
        }
        async fn preload(
            &self,
            _request: jevalaya_router_core::PreloadRequest,
        ) -> Result<(), jevalaya_router_core::BackendError> {
            Ok(())
        }
        async fn unload(
            &self,
            _target: jevalaya_router_core::UnloadTarget,
        ) -> Result<(), jevalaya_router_core::BackendError> {
            Ok(())
        }
        async fn predict(
            &self,
            _request: &jevalaya_router_core::PredictRequest,
            _rendered: Option<&jevalaya_router_core::RenderedPrompt>,
        ) -> Result<jevalaya_router_core::BackendResult, jevalaya_router_core::BackendError> {
            Err(jevalaya_router_core::BackendError::Inference(
                "never called".to_string(),
            ))
        }
    }

    fn scripted(kind: BackendKind, available: bool, detail: &str) -> Arc<dyn PredictBackend> {
        Arc::new(ScriptedBackend {
            kind,
            available,
            detail: detail.to_string(),
        })
    }

    #[test]
    fn health_entry_marks_capability_not_reachability() {
        // T054: `available` is capability ("may serve"), never liveness.
        // `reachable` is null because health never probes the network.
        let v = backend_entry(Some(
            &scripted(BackendKind::Jev, true, "typesafe https://api.typesafe.ai"),
        ));
        assert_eq!(v["available"], json!(true));
        assert_eq!(v["configured"], json!(true));
        assert!(v["reachable"].is_null());
        assert_eq!(v["detail"], json!("typesafe https://api.typesafe.ai"));

        // Keyless Jev still reports configured (adapter built) but not
        // available — a dashboard can tell "no credential" from "down".
        let v = backend_entry(Some(&scripted(BackendKind::Jev, false, "typesafe unset")));
        assert_eq!(v["available"], json!(false));
        assert_eq!(v["configured"], json!(true));
        assert!(v["reachable"].is_null());

        // Warming detail passes through untouched: it is the only
        // machine-read dispatch-readiness signal today.
        let v = backend_entry(Some(&scripted(
            BackendKind::Ane,
            true,
            "laya-coreml-ane B1 L96 K32 fp16 (cpu_ne); warming",
        )));
        assert_eq!(v["available"], json!(true));
        assert!(v["detail"].as_str().unwrap().contains("warming"));

        // Absent backends render null, never a fabricated entry.
        assert_eq!(backend_entry(None), Value::Null);
    }
}
