//! Server config: `jevalaya.toml` → `PolicyConfig` + backend wiring.
//! Field names mirror the DESIGN.md example; secrets come only from the
//! named env vars (`auth_token_env`, `api_key_env`) — never from TOML.

use std::path::{Path, PathBuf};

use serde::Deserialize;

use jevalaya_router_core::{BackendKind, Checkpoint, LayaPromptEngine, PolicyConfig, RouteError};

#[derive(Debug, Deserialize)]
pub struct ServerConfig {
    #[serde(default)]
    pub server: ServerSection,
    #[serde(default)]
    pub models: ModelsSection,
    #[serde(default)]
    pub ane: AneSection,
    #[serde(default)]
    pub mlx: MlxSection,
    #[serde(default)]
    pub jev: JevSection,
    #[serde(default)]
    pub observability: ObservabilitySection,
    /// Checkpoint router knobs (`default_checkpoint`, `auto_task_detection`,
    /// `compare_backends`) — parsed straight into `PolicyConfig`.
    #[serde(default)]
    pub policy: PolicyOverrides,
}

#[derive(Debug, Deserialize)]
pub struct ServerSection {
    /// Loopback-only default per DESIGN.md.
    #[serde(default = "default_listen")]
    pub listen: String,
    /// Env var holding the bearer token. Absent = auth disabled (warned at
    /// startup); set but empty/missing env = refuse to start (fail closed).
    #[serde(default)]
    pub auth_token_env: Option<String>,
    #[serde(default = "default_max_body")]
    pub max_body_bytes: usize,
    #[serde(default = "default_max_inflight")]
    pub max_inflight: usize,
    /// Per-request wall clock bound on /predict.
    #[serde(default = "default_request_timeout_ms")]
    pub request_timeout_ms: u64,
    #[serde(default)]
    pub offline: bool,
}

#[derive(Debug, Default, Deserialize)]
pub struct ModelsSection {
    pub english: Option<String>,
    pub multilingual: Option<String>,
    pub typed_decisions: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
pub struct AneSection {
    #[serde(default)]
    pub enabled: bool,
    /// Directory holding the CoreML/ANE model + bundled tokenizer.
    pub model: Option<String>,
    #[serde(default)]
    pub max_tokens: Option<usize>,
    #[serde(default)]
    pub alignment: Option<usize>,
    #[serde(default)]
    pub prefer_ane_for_short: Option<bool>,
    #[serde(default)]
    pub preload: bool,
}

#[derive(Debug, Default, Deserialize)]
pub struct MlxSection {
    #[serde(default)]
    pub enabled: bool,
    /// `sys.path` inserts for the embedded interpreter: the laya-mlx
    /// checkout dir and the venv site-packages.
    #[serde(default)]
    pub python_path: Vec<PathBuf>,
    #[serde(default = "default_mlx_module")]
    pub pyo3_module: String,
    #[serde(default = "default_dtype")]
    pub dtype: String,
    #[serde(default = "default_max_loaded")]
    pub max_loaded: usize,
    #[serde(default)]
    pub preload: bool,
}

#[derive(Debug, Deserialize)]
pub struct JevSection {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default = "default_jev_base")]
    pub base_url: String,
    #[serde(default = "default_jev_path")]
    pub predict_path: String,
    #[serde(default = "default_jev_key_env")]
    pub api_key_env: String,
    #[serde(default = "default_jev_model")]
    pub model: String,
    #[serde(default = "default_jev_timeout_ms")]
    pub request_timeout_ms: u64,
    /// Total attempts incl. the first (design default 3).
    #[serde(default = "default_jev_retries")]
    pub retries: u32,
    pub confidence_threshold: Option<f64>,
    pub margin_threshold: Option<f64>,
    pub target_confidence_threshold: Option<f64>,
    #[serde(default)]
    pub escalate_on_retry: bool,
}

#[derive(Debug, Default, Deserialize)]
pub struct ObservabilitySection {
    /// `jsonl:/path/to/events.jsonl` — anything else is rejected at startup.
    pub sink: Option<String>,
    #[serde(default = "default_queue_capacity")]
    pub queue_capacity: usize,
}

#[derive(Debug, Default, Deserialize)]
pub struct PolicyOverrides {
    pub default_checkpoint: Option<String>,
    #[serde(default)]
    pub auto_task_detection: bool,
    /// `compare=true` fan-out set (names like "mlx", "jev").
    #[serde(default)]
    pub compare_backends: Vec<String>,
}

fn default_listen() -> String {
    "127.0.0.1:8767".to_string()
}
fn default_max_body() -> usize {
    256 * 1024
}
fn default_max_inflight() -> usize {
    2
}
fn default_request_timeout_ms() -> u64 {
    30_000
}
fn default_mlx_module() -> String {
    "laya_mlx".to_string()
}
fn default_dtype() -> String {
    "float16".to_string()
}
fn default_max_loaded() -> usize {
    1
}
fn default_jev_base() -> String {
    "https://api.typesafe.ai".to_string()
}
fn default_jev_path() -> String {
    "/v1/systemone".to_string()
}
fn default_jev_key_env() -> String {
    "TYPESAFE_API_KEY".to_string()
}
fn default_jev_model() -> String {
    "jev-latest".to_string()
}
fn default_jev_timeout_ms() -> u64 {
    25_000
}
fn default_jev_retries() -> u32 {
    3
}
fn default_queue_capacity() -> usize {
    1024
}

impl Default for ServerSection {
    fn default() -> Self {
        Self {
            listen: default_listen(),
            auth_token_env: None,
            max_body_bytes: default_max_body(),
            max_inflight: default_max_inflight(),
            request_timeout_ms: default_request_timeout_ms(),
            offline: false,
        }
    }
}

impl Default for JevSection {
    fn default() -> Self {
        Self {
            enabled: false,
            base_url: default_jev_base(),
            predict_path: default_jev_path(),
            api_key_env: default_jev_key_env(),
            model: default_jev_model(),
            request_timeout_ms: default_jev_timeout_ms(),
            retries: default_jev_retries(),
            confidence_threshold: None,
            margin_threshold: None,
            target_confidence_threshold: None,
            escalate_on_retry: false,
        }
    }
}

/// Result of [`ServerConfig::build_engine`]. `ane_ready` is false when
/// `[ane].model` was set but its tokenizer/budgets failed to load — the
/// caller must force `policy.ane.configured = false` so the gate reports
/// honestly.
pub struct EngineBuild {
    pub engine: LayaPromptEngine,
    pub warnings: Vec<String>,
    pub ane_ready: bool,
}

impl ServerConfig {
    pub fn load(path: &Path) -> Result<Self, String> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| format!("cannot read {}: {e}", path.display()))?;
        toml::from_str(&text).map_err(|e| format!("cannot parse {}: {e}", path.display()))
    }

    /// The router's immutable policy snapshot.
    pub fn policy(&self) -> Result<PolicyConfig, RouteError> {
        let mut policy = PolicyConfig::default();
        if let Some(name) = &self.policy.default_checkpoint {
            policy.default_checkpoint = Checkpoint::from_name(name).ok_or_else(|| {
                RouteError::InvalidRequest(format!(
                    "policy.default_checkpoint {name:?} is not a checkpoint"
                ))
            })?;
        }
        policy.auto_task_detection = self.policy.auto_task_detection;
        for (cp, dir) in self.model_dirs() {
            policy.models.insert(cp, dir.display().to_string());
        }
        if !self.policy.compare_backends.is_empty() {
            let mut kinds = Vec::new();
            for n in &self.policy.compare_backends {
                let k = BackendKind::from_name(n).ok_or_else(|| {
                    RouteError::InvalidRequest(format!(
                        "policy.compare_backends entry {n:?} is not a backend"
                    ))
                })?;
                if !kinds.contains(&k) {
                    kinds.push(k);
                }
            }
            policy.compare_backends = kinds;
        }
        policy.ane.enabled = self.ane.enabled;
        policy.ane.configured = self.ane.model.is_some();
        if let Some(m) = &self.ane.model {
            policy.ane.model_id = m.clone();
        }
        if let Some(t) = self.ane.max_tokens {
            policy.ane.max_tokens = t;
        }
        if let Some(a) = self.ane.alignment {
            policy.ane.alignment = a;
        }
        if let Some(p) = self.ane.prefer_ane_for_short {
            policy.ane.prefer_ane_for_short = p;
        }
        policy.jev.model_id = self.jev.model.clone();
        for (dst, src) in [
            (
                &mut policy.jev.confidence_threshold,
                self.jev.confidence_threshold,
            ),
            (&mut policy.jev.margin_threshold, self.jev.margin_threshold),
            (
                &mut policy.jev.target_confidence_threshold,
                self.jev.target_confidence_threshold,
            ),
        ] {
            if src.is_some() {
                *dst = src;
            }
        }
        policy.jev.escalate_on_retry = self.jev.escalate_on_retry;
        Ok(policy)
    }

    /// Checkpoint → model directory pairs from `[models]`.
    pub fn model_dirs(&self) -> Vec<(Checkpoint, PathBuf)> {
        [
            (Checkpoint::English, &self.models.english),
            (Checkpoint::Multilingual, &self.models.multilingual),
            (Checkpoint::TypedDecisions, &self.models.typed_decisions),
        ]
        .into_iter()
        .filter_map(|(cp, d)| d.as_ref().map(|d| (cp, PathBuf::from(d))))
        .collect()
    }

    /// The prompt engine: per-checkpoint tokenizer+budgets plus the ANE
    /// model's bundled tokenizer when `[ane].model` is configured.
    /// Per-model load failures degrade to warnings (that checkpoint reports
    /// `token_count: null` at request time; a failed ANE tokenizer means
    /// the ANE gate reports itself unconfigured) instead of killing boot —
    /// graceful degradation with fewer usable parts is the contract.
    pub fn build_engine(&self) -> EngineBuild {
        let mut engine = LayaPromptEngine::new();
        let mut warnings = Vec::new();
        for (cp, dir) in self.model_dirs() {
            if let Err(e) = engine.load_checkpoint(cp, &dir) {
                warnings.push(format!(
                    "{} model {}: {e} — token counts will report null",
                    cp.as_str(),
                    dir.display()
                ));
            }
        }
        let mut ane_ready = false;
        if let Some(dir) = &self.ane.model {
            let alignment = self.ane.alignment.unwrap_or(1);
            match engine.load_ane(Path::new(dir), alignment) {
                Ok(()) => ane_ready = true,
                Err(e) => warnings.push(format!(
                    "ane model {dir}: {e} — ANE gate reports unconfigured"
                )),
            }
        }
        EngineBuild {
            engine,
            warnings,
            ane_ready,
        }
    }

    /// Backends present in config — the `degraded` denominator. A backend
    /// counts as configured when its section is enabled, even if its
    /// adapter could not be constructed (missing credential, pending
    /// implementation): those report unavailable through capabilities.
    pub fn configured_backends(&self, built: &[BackendKind]) -> Vec<BackendKind> {
        let mut out = Vec::new();
        if self.ane.enabled && built.contains(&BackendKind::Ane) {
            out.push(BackendKind::Ane);
        }
        if self.mlx.enabled || built.contains(&BackendKind::Mlx) {
            out.push(BackendKind::Mlx);
        }
        if self.jev.enabled {
            out.push(BackendKind::Jev);
        }
        out
    }

    /// Bearer token resolution: `auth_token_env` unset → no auth; set but
    /// empty/missing → startup error (fail closed). Returns the token.
    pub fn auth_token(&self) -> Result<Option<String>, String> {
        match &self.server.auth_token_env {
            None => Ok(None),
            Some(env) => {
                let token = std::env::var(env).unwrap_or_default();
                if token.is_empty() {
                    Err(format!(
                        "auth_token_env {env:?} is configured but the environment variable is empty or unset"
                    ))
                } else {
                    Ok(Some(token))
                }
            }
        }
    }
}
