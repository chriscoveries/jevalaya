//! `RouteDecision`: the immutable serializable routing record (DESIGN.md
//! §Core types). Required keys are always present; detail fields are
//! `skip_serializing_if` until a backend result exists.

use serde::Serialize;
use serde_json::Value;

use crate::types::{BackendKind, Checkpoint};

/// Stable reason prefixes for the ordered ANE gate (DESIGN.md §Ordered
/// routing) and the rest of the decision table. Reason strings are
/// `"<prefix>: <human detail>"`; the prefix before `:` is contract-stable.
pub mod reason {
    pub const EXPLICIT_BACKEND: &str = "explicit_backend";
    pub const ANE_DISABLED: &str = "ane_disabled";
    pub const PLATFORM_UNSUPPORTED: &str = "platform_unsupported";
    pub const ANE_UNAVAILABLE: &str = "ane_unavailable";
    pub const NOT_MULTILINGUAL: &str = "not_multilingual";
    pub const TYPED_DECISIONS: &str = "typed_decisions";
    pub const QUESTION_COUNT: &str = "question_count";
    pub const ANE_NOT_PREFERRED: &str = "ane_not_preferred";
    pub const TOKEN_COUNT_OVER_LIMIT: &str = "token_count_over_limit";
    /// ANE won every gate: short single-question multilingual prompt.
    pub const ANE_SHORT_PATH: &str = "ane_short_path";
    /// T023: the request fit the ANE bundle's tokenizer budget, so the
    /// multilingual checkpoint was selected regardless of detected
    /// language. The detector's verdict is preserved in the detail.
    pub const ANE_FIT_MULTILINGUAL: &str = "ane_fit_multilingual";
    /// Auto mode with no usable local backend; Jev is the passthrough.
    pub const NO_LOCAL_BACKEND: &str = "no_local_backend";
    /// Entry inside a `compare` fan-out.
    pub const COMPARE: &str = "compare";
}

/// Stable `jev_trigger` values (DESIGN.md §Ordered routing step 7).
pub mod trigger {
    /// `top_p - second_p` below `jev.margin_threshold` (primary trigger).
    pub const LOW_MARGIN: &str = "low_margin";
    /// Answer confidence below `jev.confidence_threshold`.
    pub const LOW_CONFIDENCE: &str = "low_confidence";
    /// Any answer's `target_confidence` below
    /// `jev.target_confidence_threshold`.
    pub const LOW_TARGET_CONFIDENCE: &str = "low_target_confidence";
    /// Local answer needed the one permitted ANE->MLX retry and
    /// `jev.escalate_on_retry` is set.
    pub const RETRY: &str = "retry";
}

/// The routing outcome for one dispatched request (or one compare entry).
/// Required keys per DESIGN.md: `checkpoint`, `backend`, `model_id`,
/// `reason`, `token_count`, `ane_eligible`, `fallback`, `latency_ms`.
#[derive(Debug, Clone, Serialize)]
pub struct RouteDecision {
    /// `english | multilingual | typed-decisions | null` (null for
    /// requests that never touched the checkpoint router, e.g. explicit
    /// Jev).
    pub checkpoint: Option<Checkpoint>,
    pub backend: BackendKind,
    /// Configured model directory, checkpoint id, or endpoint.
    pub model_id: String,
    /// `"<stable prefix>: <human detail>"`.
    pub reason: String,
    /// Raw, pre-truncation gate count when a local prompt was rendered;
    /// null for explicit Jev. For ANE-candidate requests this is the ANE
    /// model tokenizer's count; otherwise the selected checkpoint's.
    pub token_count: Option<u64>,
    /// The request satisfied every ANE content/shape gate (checkpoint,
    /// question count, token limit, enabled, platform) — independent of
    /// whether ANE was available or preferred. Stays true after an
    /// ANE->MLX runtime fallback.
    pub ane_eligible: bool,
    /// True when the answer came from the one permitted ANE->MLX retry.
    pub fallback: bool,
    /// Null from route-only code; end-to-end inference time in an HTTP
    /// response (ms, includes escalation time).
    pub latency_ms: Option<f64>,
    /// Model-bounded sequence length actually sent to MLX, when computed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub execution_token_count: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fallback_from: Option<BackendKind>,
    /// Stable error code (`ane_capacity`, `ane_shape`, ...).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fallback_reason: Option<String>,
    /// True when Jev escalation was attempted — whether or not Jev
    /// answered (`escalation_error` distinguishes).
    pub escalated: bool,
    /// Stable trigger name (`low_margin`, `low_confidence`, ...).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub jev_trigger: Option<String>,
    /// Min per-question confidence once a backend result exists.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f64>,
    /// Min per-question top-two probability margin once a result exists.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub margin: Option<f64>,
    /// A configured backend was unavailable, or escalation failed and the
    /// local answer was retained.
    pub degraded: bool,
    /// Stable code when a Jev escalation attempt failed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub escalation_error: Option<String>,
    /// `{input_tokens, output_tokens, remote_usd}` once a result exists.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cost: Option<Value>,
}

impl RouteDecision {
    pub fn new(backend: BackendKind, model_id: impl Into<String>, reason: String) -> Self {
        RouteDecision {
            checkpoint: None,
            backend,
            model_id: model_id.into(),
            reason,
            token_count: None,
            ane_eligible: false,
            fallback: false,
            latency_ms: None,
            execution_token_count: None,
            fallback_from: None,
            fallback_reason: None,
            escalated: false,
            jev_trigger: None,
            confidence: None,
            margin: None,
            degraded: false,
            escalation_error: None,
            cost: None,
        }
    }

    /// The stable prefix of `reason` (text before the first `:`).
    pub fn reason_prefix(&self) -> &str {
        match self.reason.find(':') {
            Some(i) => &self.reason[..i],
            None => self.reason.as_str(),
        }
    }
}
