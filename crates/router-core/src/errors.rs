//! Typed router errors mapped to the stable wire status/code table
//! (DESIGN.md §Wire contract). Messages never carry secrets or raw prompts.

use serde_json::{json, Value};
use thiserror::Error;

use crate::types::{BackendError, BackendKind};

#[derive(Debug, Error)]
pub enum RouteError {
    /// 400 invalid_request: malformed JSON, missing state/questions, invalid
    /// question schema, unknown backend/model/task name.
    #[error("invalid request: {0}")]
    InvalidRequest(String),
    /// 409 backend_unavailable: explicitly requested backend is
    /// disabled/not configured/unavailable.
    #[error("backend unavailable: {0}")]
    BackendUnavailable(BackendKind),
    /// 429 busy: request semaphore or backend capacity exhausted.
    #[error("busy: {0}")]
    Busy(String),
    /// 502 upstream_error: Jev network/provider failure or backend protocol
    /// failure after bounded retry.
    #[error("upstream error: {0}")]
    Upstream(String),
    /// 503 not_ready: lazy backend starting or required local model
    /// unavailable, or no usable backend for an automatic request.
    #[error("not ready: {0}")]
    NotReady(String),
    /// 500 inference_failed: classified hard inference failure.
    #[error("inference failed: {0}")]
    Inference(String),
}

impl RouteError {
    pub fn status(&self) -> u16 {
        match self {
            RouteError::InvalidRequest(_) => 400,
            RouteError::BackendUnavailable(_) => 409,
            RouteError::Busy(_) => 429,
            RouteError::Upstream(_) => 502,
            RouteError::NotReady(_) => 503,
            RouteError::Inference(_) => 500,
        }
    }

    pub fn code(&self) -> &'static str {
        match self {
            RouteError::InvalidRequest(_) => "invalid_request",
            RouteError::BackendUnavailable(_) => "backend_unavailable",
            RouteError::Busy(_) => "busy",
            RouteError::Upstream(_) => "upstream_error",
            RouteError::NotReady(_) => "not_ready",
            RouteError::Inference(_) => "inference_failed",
        }
    }

    /// Stable JSON error body: `{ "error": { "code", "message" } }`, no
    /// stack trace.
    pub fn body(&self) -> Value {
        json!({
            "error": {
                "code": self.code(),
                "message": self.to_string(),
            }
        })
    }

    /// Map an adapter failure to the wire error for the *primary* dispatch.
    /// Explicit Jev failures stay 502/503 per DESIGN.md §Wire contract.
    pub fn from_backend(err: BackendError) -> RouteError {
        match err {
            BackendError::Unavailable(m) => RouteError::NotReady(m),
            BackendError::NotReady(m) => RouteError::NotReady(m),
            BackendError::MissingCredential(m) => RouteError::NotReady(m),
            BackendError::Timeout(ms) => RouteError::Upstream(format!("timed out after {ms}ms")),
            BackendError::Transport(m) => RouteError::Upstream(m),
            BackendError::Upstream(_, m) => RouteError::Upstream(m),
            BackendError::InvalidResponse(m) => RouteError::Upstream(m),
            BackendError::Inference(m) => RouteError::Inference(m),
            BackendError::Capacity(m) => RouteError::Inference(format!("ane capacity: {m}")),
            BackendError::Shape(m) => RouteError::Inference(format!("ane shape: {m}")),
        }
    }
}
