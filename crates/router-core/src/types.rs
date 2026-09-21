//! Core types shared by every backend adapter and the router itself.
//!
//! Error strings here must never contain secrets (API keys, auth headers) or
//! raw prompts; adapters translate native errors into [`BackendError`].

use std::fmt;

use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::{Map, Value};
use thiserror::Error;

/// Which backend produced an answer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BackendKind {
    Ane,
    Mlx,
    Jev,
}

impl BackendKind {
    pub fn as_str(self) -> &'static str {
        match self {
            BackendKind::Ane => "ane",
            BackendKind::Mlx => "mlx",
            BackendKind::Jev => "jev",
        }
    }

    pub fn from_name(name: &str) -> Option<BackendKind> {
        match name.trim().to_lowercase().as_str() {
            "ane" | "coreml" => Some(BackendKind::Ane),
            "mlx" | "local" => Some(BackendKind::Mlx),
            "jev" | "typesafe" => Some(BackendKind::Jev),
            _ => None,
        }
    }

    pub const ALL: [BackendKind; 3] = [BackendKind::Ane, BackendKind::Mlx, BackendKind::Jev];
}

impl fmt::Display for BackendKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl Serialize for BackendKind {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for BackendKind {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let s = String::deserialize(d)?;
        BackendKind::from_name(&s)
            .ok_or_else(|| serde::de::Error::custom(format!("unknown backend {s:?}")))
    }
}

/// Which Laya checkpoint a request selects.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Checkpoint {
    English,
    Multilingual,
    TypedDecisions,
}

impl Checkpoint {
    /// Canonical [`laya_mlx` router names](https://github.com/convaiinnovations/laya):
    /// `english`, `multilingual`, `typed-decisions`.
    pub fn as_str(self) -> &'static str {
        match self {
            Checkpoint::English => "english",
            Checkpoint::Multilingual => "multilingual",
            Checkpoint::TypedDecisions => "typed-decisions",
        }
    }

    pub fn from_name(name: &str) -> Option<Checkpoint> {
        match name.trim().to_lowercase().as_str() {
            "english" | "en" | "laya" | "default" => Some(Checkpoint::English),
            "multilingual" | "multi" | "ml" | "laya-multilingual" => Some(Checkpoint::Multilingual),
            "typed-decisions"
            | "typed_decisions"
            | "typed"
            | "decisions"
            | "laya-typed-decisions" => Some(Checkpoint::TypedDecisions),
            _ => None,
        }
    }
}

impl fmt::Display for Checkpoint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl Serialize for Checkpoint {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for Checkpoint {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let s = String::deserialize(d)?;
        Checkpoint::from_name(&s)
            .ok_or_else(|| serde::de::Error::custom(format!("unknown checkpoint {s:?}")))
    }
}

/// Validated `{state, questions}` request plus routing metadata. The router
/// passes this to exactly one backend; adapters never reroute.
#[derive(Debug, Clone)]
pub struct PredictRequest {
    pub state: Value,
    pub questions: Map<String, Value>,
    /// Selected checkpoint. Unused by Jev; the router fills the routed or
    /// configured default value so adapters can rely on it being present.
    pub checkpoint: Checkpoint,
    pub request_id: String,
}

/// Rendered prompt for backends that consume token ids directly (CoreML).
/// MLX-via-PyO3 and Jev ignore this — Python re-renders internally and Jev
/// serializes `{state, questions}` — so it travels as `Option`.
pub type RenderedPrompt = jevalaya_render::Rendered;

/// What a backend can do right now. Detail stays coarse here; the ANE
/// adapter owns fixed-shape/alignment specifics.
#[derive(Debug, Clone)]
pub struct BackendCapabilities {
    pub kind: BackendKind,
    pub available: bool,
    pub detail: String,
}

/// Which checkpoints to make resident.
#[derive(Debug, Clone, Default)]
pub struct PreloadRequest {
    pub checkpoints: Vec<Checkpoint>,
}

/// What to evict from a backend.
#[derive(Debug, Clone)]
pub enum UnloadTarget {
    All,
    Checkpoint(Checkpoint),
}

/// Normalized adapter output: the shared `{model, answers, usage}` shape plus
/// the backend's own latency. Routing metadata is the router's job, not the
/// adapter's.
#[derive(Debug, Clone)]
pub struct BackendResult {
    pub model: String,
    pub answers: Value,
    pub usage: Value,
    pub latency_ms: u64,
}

/// Typed adapter failures. Adapters translate native errors here; policy
/// decides retry/escalation. No variant carries secrets or raw prompts.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum BackendError {
    #[error("backend unavailable: {0}")]
    Unavailable(String),
    #[error("backend not ready: {0}")]
    NotReady(String),
    #[error("request timed out after {0}ms")]
    Timeout(u64),
    #[error("transport error: {0}")]
    Transport(String),
    #[error("upstream provider error{0:?}: {1}")]
    Upstream(Option<u16>, String),
    #[error("invalid backend response: {0}")]
    InvalidResponse(String),
    #[error("inference failed: {0}")]
    Inference(String),
    #[error("missing credential: {0}")]
    MissingCredential(String),
    /// ANE capacity/shape classes (DESIGN.md §Native CoreML ANE): transient
    /// memory pressure / compute-unit exhaustion. Permits exactly one MLX
    /// retry.
    #[error("ane capacity: {0}")]
    Capacity(String),
    /// Fixed length, alignment, dtype, mask, or tensor shape mismatch.
    /// Permits exactly one MLX retry; may trip a process-local circuit.
    #[error("ane shape: {0}")]
    Shape(String),
}

impl BackendError {
    /// Stable snake-case code for `escalation_error` / `fallback_reason`
    /// fields and event payloads. Never carries the message body.
    pub fn stable_code(&self) -> &'static str {
        match self {
            BackendError::Unavailable(_) => "backend_unavailable",
            BackendError::NotReady(_) => "not_ready",
            BackendError::Timeout(_) => "timeout",
            BackendError::Transport(_) => "transport",
            BackendError::Upstream(Some(429), _) => "upstream_429",
            BackendError::Upstream(Some(529), _) => "upstream_529",
            BackendError::Upstream(Some(503), _) => "upstream_503",
            BackendError::Upstream(Some(_), _) => "upstream",
            BackendError::Upstream(None, _) => "upstream",
            BackendError::InvalidResponse(_) => "invalid_response",
            BackendError::Inference(_) => "inference_failed",
            BackendError::MissingCredential(_) => "missing_credential",
            BackendError::Capacity(_) => "ane_capacity",
            BackendError::Shape(_) => "ane_shape",
        }
    }
}

/// One normalized adapter contract (DESIGN.md §Core types). Async boundary;
/// blocking implementations (CoreML, PyO3 bridge) run behind a bounded
/// executor owned by the server layer — mark them [`Self::is_blocking`] so
/// the router dispatches them via `spawn_blocking`.
#[async_trait::async_trait]
pub trait PredictBackend: Send + Sync {
    fn kind(&self) -> BackendKind;
    fn capabilities(&self) -> BackendCapabilities;
    /// True when `predict` performs blocking work inline (Python/MLX,
    /// CoreML). The router runs such calls on the blocking pool so they
    /// never stall the async executor.
    fn is_blocking(&self) -> bool {
        false
    }
    async fn preload(&self, request: PreloadRequest) -> Result<(), BackendError>;
    async fn unload(&self, target: UnloadTarget) -> Result<(), BackendError>;
    async fn predict(
        &self,
        request: &PredictRequest,
        rendered: Option<&RenderedPrompt>,
    ) -> Result<BackendResult, BackendError>;
}
