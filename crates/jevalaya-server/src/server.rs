//! Axum surface for the router (DESIGN.md §Wire contract):
//! `POST /predict`, `POST /route`, `GET /health`, `GET /ready`.
//! Bearer auth on /predict + /route when `auth_token_env` is configured,
//! 256 KiB body cap, inflight semaphore → 429, per-request timeout → 504.
//! Error bodies are always `{ "error": { "code", "message" } }`.

use std::sync::Arc;
use std::time::Duration;

use axum::body::Bytes;
use axum::extract::{DefaultBodyLimit, Request, State};
use axum::http::{header, HeaderMap, StatusCode};
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router as AxumRouter};
use serde_json::{json, Value};
use tokio::sync::Semaphore;
use tower_http::trace::TraceLayer;

use jevalaya_router_core::{BackendKind, PredictBody, RouteError, Router};

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

async fn health(State(state): State<Arc<AppState>>) -> Response {
    let avail = state.router.availability();
    let backends = |k: BackendKind| {
        state.router.backend(k).map(|b| {
            let c = b.capabilities();
            json!({"available": c.available, "detail": c.detail})
        })
    };
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

/// Build the axum app. `max_body` feeds axum's request-body limit so
/// oversized payloads are rejected before buffering past the cap.
pub fn app(state: Arc<AppState>, max_body: usize) -> AxumRouter {
    let authed = AxumRouter::new()
        .route("/predict", post(predict))
        .route("/route", post(route_only))
        .route_layer(middleware::from_fn_with_state(state.clone(), auth));
    AxumRouter::new()
        .merge(authed)
        .route("/health", get(health))
        .route("/ready", get(ready))
        .layer(DefaultBodyLimit::max(max_body))
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}
