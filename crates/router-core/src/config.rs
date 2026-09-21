//! Policy configuration — the routing knobs DESIGN.md §Observability
//! makes reloadable. Every field has a default so a partial config file
//! works; the server layer owns TOML parsing.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::types::{BackendKind, Checkpoint};

fn default_max_ane_tokens() -> usize {
    jevalaya_render::DEFAULT_MAX_ANE_TOKENS
}

fn default_alignment() -> usize {
    1
}

fn default_confidence_threshold() -> Option<f64> {
    Some(0.75)
}

fn default_margin_threshold() -> Option<f64> {
    Some(0.20)
}

fn default_target_confidence_threshold() -> Option<f64> {
    Some(0.85)
}

fn default_jev_model() -> String {
    "jev-latest".to_string()
}

fn default_compare_backends() -> Vec<BackendKind> {
    vec![BackendKind::Mlx, BackendKind::Jev]
}

fn default_true() -> bool {
    true
}

/// ANE policy knobs (`[ane]` in config). `configured` is set by the
/// server when an ANE model directory was supplied; `enabled` is the
/// operator kill-switch. Both must hold for the gate to run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnePolicy {
    #[serde(default)]
    pub enabled: bool,
    /// Server-populated: an ANE model dir + tokenizer actually exist.
    #[serde(default)]
    pub configured: bool,
    /// Configured model dir/id reported as `model_id` on ANE decisions.
    #[serde(default)]
    pub model_id: String,
    /// Fixed input token budget (default 96, never above the native
    /// fixed input length).
    #[serde(default = "default_max_ane_tokens")]
    pub max_tokens: usize,
    /// Alignment the gate rounds raw counts up to (1 = none).
    #[serde(default = "default_alignment")]
    pub alignment: usize,
    /// Auto mode may pick ANE for eligible short prompts without an
    /// explicit `backend="ane"` hint.
    #[serde(default = "default_true")]
    pub prefer_ane_for_short: bool,
}

impl Default for AnePolicy {
    fn default() -> Self {
        AnePolicy {
            enabled: false,
            configured: false,
            model_id: String::new(),
            max_tokens: default_max_ane_tokens(),
            alignment: 1,
            prefer_ane_for_short: true,
        }
    }
}

/// Jev escalation triggers (`[jev]` policy subset — connection details
/// live in the server config and in backend-jev).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JevPolicy {
    /// `model_id` reported on Jev decisions.
    #[serde(default = "default_jev_model")]
    pub model_id: String,
    /// Escalate when min answer confidence is below this (None disables).
    #[serde(default = "default_confidence_threshold")]
    pub confidence_threshold: Option<f64>,
    /// Primary trigger: escalate when min top-two margin is below this.
    #[serde(default = "default_margin_threshold")]
    pub margin_threshold: Option<f64>,
    /// Escalate when any answer's `target_confidence` is below this.
    #[serde(default = "default_target_confidence_threshold")]
    pub target_confidence_threshold: Option<f64>,
    /// Escalate to Jev whenever the local answer needed its ANE retry.
    #[serde(default)]
    pub escalate_on_retry: bool,
}

impl Default for JevPolicy {
    fn default() -> Self {
        JevPolicy {
            model_id: default_jev_model(),
            confidence_threshold: default_confidence_threshold(),
            margin_threshold: default_margin_threshold(),
            target_confidence_threshold: default_target_confidence_threshold(),
            escalate_on_retry: false,
        }
    }
}

/// Immutable routing snapshot: a request sees exactly one of these.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyConfig {
    /// Checkpoint for unknown/no-letter states (Python `default=`).
    #[serde(default = "default_checkpoint")]
    pub default_checkpoint: Checkpoint,
    /// Opt-in typed-workflow auto-selection (laya-compatible default off).
    #[serde(default)]
    pub auto_task_detection: bool,
    /// Per-checkpoint `model_id` strings for reporting (dir, id, or name).
    #[serde(default)]
    pub models: HashMap<Checkpoint, String>,
    #[serde(default)]
    pub ane: AnePolicy,
    #[serde(default)]
    pub jev: JevPolicy,
    /// Backend set for `compare=true` (explicit lists bypass this).
    #[serde(default = "default_compare_backends")]
    pub compare_backends: Vec<BackendKind>,
}

fn default_checkpoint() -> Checkpoint {
    Checkpoint::English
}

impl Default for PolicyConfig {
    fn default() -> Self {
        PolicyConfig {
            default_checkpoint: default_checkpoint(),
            auto_task_detection: false,
            models: HashMap::new(),
            ane: AnePolicy::default(),
            jev: JevPolicy::default(),
            compare_backends: default_compare_backends(),
        }
    }
}

impl PolicyConfig {
    /// `model_id` for a checkpoint decision; falls back to the name.
    pub fn model_id(&self, cp: Checkpoint) -> String {
        self.models
            .get(&cp)
            .cloned()
            .unwrap_or_else(|| cp.as_str().to_string())
    }
}
