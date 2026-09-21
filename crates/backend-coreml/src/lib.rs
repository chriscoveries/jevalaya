//! `jevalaya-backend-coreml` — native CoreML/ANE backend (T004).
//!
//! Loads exactly one `laya-coreml-ane` bundle per service (fixed B1 /
//! L96 / K32 fp16 encoder + fp32 host action head), serves the single
//! multilingual short-question path the router's ANE gate selects, and
//! maps native failures onto `BackendError::Capacity` / `::Shape` /
//! hard `::Inference` per DESIGN.md §Native CoreML ANE:
//!
//! * `Capacity` — ANE daemon/compiler or POSIX resource exhaustion;
//!   releases the resident model and permits the router's one MLX retry.
//! * `Shape` — adapter-identified fixed-length/options/dtype/mask
//!   violations; also trips a process-local circuit breaker (these are
//!   deterministic configuration bugs) until `unload`/`preload`.
//! * hard — corrupt model, missing outputs, non-finite results, or any
//!   unrecognized native error; never retried.
//!
//! The crate compiles everywhere: the Objective-C surface is target-
//! gated to macOS arm64, and off-target construction succeeds but
//! reports `available: false` through capabilities.

mod bundle;
mod host;

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
mod native;

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Mutex;
use std::time::Instant;

use jevalaya_render::{to_internal, Question};
use jevalaya_router_core::types::{
    BackendCapabilities, BackendError, BackendKind, BackendResult, PredictBackend, PredictRequest,
    PreloadRequest, RenderedPrompt, UnloadTarget,
};
use serde::Deserialize;

/// True only where the native backend can run.
pub const ANE_TARGET: bool = cfg!(all(target_os = "macos", target_arch = "aarch64"));

#[derive(Debug, Clone)]
pub struct AneConfig {
    /// Directory of the `laya-coreml-ane` bundle (manifest, mlpackage,
    /// host_weights.safetensors, tokenizer/, encoder/, rl_agent_config).
    pub model_dir: PathBuf,
    /// `cpu_ne` (default) / `cpu` / `cpu_gpu` / `all`.
    pub compute_units: String,
    /// Where the symlink-free mlpackage copy is materialized.
    /// Default: `$JEVALAYA_ANE_CACHE` or `~/Library/Caches/jevalaya/ane`.
    pub cache_dir: Option<PathBuf>,
}

impl Default for AneConfig {
    fn default() -> Self {
        Self {
            model_dir: PathBuf::new(),
            compute_units: "cpu_ne".into(),
            cache_dir: None,
        }
    }
}

#[derive(Debug, Deserialize)]
struct EncoderCfg {
    hidden_size: usize,
    #[serde(default = "default_local_attention")]
    local_attention: usize,
}

fn default_local_attention() -> usize {
    128
}

#[derive(Debug, Deserialize)]
struct AgentCfg {
    #[serde(default)]
    temperature: Option<Vec<f64>>,
    #[serde(default)]
    temperature_by_options: HashMap<String, f64>,
}

struct Inner {
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    model: Option<native::NativeModel>,
    circuit_open: bool,
}

pub struct AneBackend {
    bundle: bundle::Bundle,
    weights: host::HostWeights,
    act_head: host::ActHead,
    pad_id: u32,
    hidden: usize,
    local_attention: usize,
    temperature: [f64; 3],
    temperature_by_options: HashMap<String, f64>,
    compute_units_name: String,
    cache_dir: PathBuf,
    inner: Mutex<Inner>,
    detail: String,
    os_ok: bool,
}

impl AneBackend {
    /// Validate the bundle and stage everything except the CoreML
    /// residency (lazy: `preload` or first predict). Construction fails
    /// only on a missing/invalid bundle — an off-target or ANE-less host
    /// still constructs and reports itself unavailable.
    pub fn new(cfg: AneConfig) -> Result<Self, String> {
        let bundle = bundle::Bundle::open(&cfg.model_dir).map_err(|e| e.to_string())?;
        let encoder: EncoderCfg = serde_json::from_str(
            &std::fs::read_to_string(cfg.model_dir.join("encoder/config.json"))
                .map_err(|e| format!("encoder/config.json: {e}"))?,
        )
        .map_err(|e| format!("encoder/config.json: {e}"))?;
        let agent: AgentCfg = serde_json::from_str(
            &std::fs::read_to_string(cfg.model_dir.join("rl_agent_config.json"))
                .map_err(|e| format!("rl_agent_config.json: {e}"))?,
        )
        .map_err(|e| format!("rl_agent_config.json: {e}"))?;
        let temperature: [f64; 3] = agent
            .temperature
            .unwrap_or_else(|| vec![1.0, 1.0, 1.0])
            .try_into()
            .map_err(|_| "rl_agent_config.json temperature must have 3 entries".to_string())?;
        if temperature.iter().any(|t| !t.is_finite() || *t <= 0.0)
            || agent
                .temperature_by_options
                .values()
                .any(|t| !t.is_finite() || *t <= 0.0)
        {
            return Err("calibration temperatures must be finite and positive".into());
        }

        let tok = jevalaya_render::LayaTokenizer::from_model_dir(&cfg.model_dir)
            .map_err(|e| format!("ane tokenizer: {e}"))?;
        let weights = host::HostWeights::open(&cfg.model_dir.join("host_weights.safetensors"))
            .map_err(|e| e.to_string())?;
        let act_head = host::ActHead::load(&weights).map_err(|e| e.to_string())?;

        let cache_dir = cfg.cache_dir.clone().unwrap_or_else(|| {
            std::env::var("JEVALAYA_ANE_CACHE")
                .map(PathBuf::from)
                .unwrap_or_else(|_| {
                    PathBuf::from(std::env::var("HOME").unwrap_or_else(|_| "/tmp".into()))
                        .join("Library/Caches/jevalaya/ane")
                })
        });

        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        native::parse_compute_units(&cfg.compute_units)?;
        let compute_units_name = cfg.compute_units.clone();

        let os_ok = ANE_TARGET && {
            #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
            {
                native::host_supports(bundle.config.minimum_deployment_target.as_deref())
            }
            #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
            {
                false
            }
        };

        let s = &bundle.config.shape;
        let detail = format!(
            "laya-coreml-ane B{} L{} K{} fp16 ({}{}{})",
            s.batch_size,
            s.max_length,
            s.max_options,
            compute_units_name,
            if ANE_TARGET { "" } else { ", off-target stub" },
            if ANE_TARGET && !os_ok {
                format!(
                    ", requires {}",
                    bundle
                        .config
                        .minimum_deployment_target
                        .as_deref()
                        .unwrap_or("newer macOS")
                )
            } else {
                String::new()
            }
        );

        Ok(Self {
            bundle,
            weights,
            act_head,
            pad_id: tok.pad_id(),
            hidden: encoder.hidden_size,
            local_attention: encoder.local_attention,
            temperature,
            temperature_by_options: agent.temperature_by_options,
            compute_units_name,
            cache_dir,
            inner: Mutex::new(Inner {
                #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
                model: None,
                circuit_open: false,
            }),
            detail,
            os_ok,
        })
    }

    /// Capability/log line detail (shapes, compute units, stub marker).
    pub fn detail_line(&self) -> &str {
        &self.detail
    }

    fn cache_or_bundle_package(&self) -> Result<PathBuf, BackendError> {
        self.bundle
            .verify_files()
            .map_err(|e| BackendError::Unavailable(format!("bundle integrity: {e}")))?;
        self.bundle
            .materialized_package(&self.cache_dir)
            .map_err(|e| BackendError::Unavailable(format!("package materialize: {e}")))
    }
}

#[async_trait::async_trait]
impl PredictBackend for AneBackend {
    fn kind(&self) -> BackendKind {
        BackendKind::Ane
    }

    fn is_blocking(&self) -> bool {
        true
    }

    fn capabilities(&self) -> BackendCapabilities {
        let circuit = self.inner.lock().map(|i| i.circuit_open).unwrap_or(true);
        BackendCapabilities {
            kind: BackendKind::Ane,
            available: ANE_TARGET && self.os_ok && !circuit,
            detail: if circuit {
                format!("{}; circuit open", self.detail)
            } else {
                self.detail.clone()
            },
        }
    }

    async fn preload(&self, _request: PreloadRequest) -> Result<(), BackendError> {
        self.ensure_model().map(|_| ())
    }

    async fn unload(&self, _target: UnloadTarget) -> Result<(), BackendError> {
        let mut inner = self.inner.lock().unwrap();
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            inner.model = None;
        }
        inner.circuit_open = false;
        Ok(())
    }

    async fn predict(
        &self,
        request: &PredictRequest,
        rendered: Option<&RenderedPrompt>,
    ) -> Result<BackendResult, BackendError> {
        let t0 = Instant::now();
        if !ANE_TARGET {
            return Err(BackendError::Unavailable(
                "ane backend requires macOS arm64".into(),
            ));
        }
        if !self.os_ok {
            return Err(BackendError::Unavailable(format!(
                "ane bundle requires {}",
                self.bundle
                    .config
                    .minimum_deployment_target
                    .as_deref()
                    .unwrap_or("a newer macOS")
            )));
        }
        if self.inner.lock().unwrap().circuit_open {
            return Err(BackendError::Unavailable(
                "ane circuit open after deterministic failure".into(),
            ));
        }
        if request.questions.len() != 1 {
            return Err(self.shape_fault(format!(
                "ane serves exactly one question (got {})",
                request.questions.len()
            )));
        }
        let rendered = rendered.ok_or_else(|| {
            self.shape_fault("ane predict requires a rendered prompt".to_string())
        })?;
        let (qid, qdef) = request.questions.iter().next().unwrap();
        let question: Question =
            to_internal(qdef).map_err(|e| BackendError::Inference(format!("question: {e}")))?;
        if rendered.markers.len() != question.options.len() {
            return Err(self.shape_fault(format!(
                "question {qid:?} has too many options for the token budget"
            )));
        }

        self.ensure_model()?;

        let inputs = host::build_inputs(
            &self.weights,
            &rendered.ids,
            &rendered.markers,
            rendered.qtype,
            self.pad_id,
            self.bundle.fixed_len(),
            self.bundle.max_options(),
            self.hidden,
            self.local_attention,
        )
        .map_err(|e| self.shape_fault(e.to_string()))?;

        let (logits, pooled) = self.run_model(&inputs)?;

        let answer = host::decode_answer(
            &logits,
            &pooled,
            inputs.k,
            rendered.qtype,
            qdef,
            &question,
            &self.temperature,
            &self.temperature_by_options,
            &self.act_head,
        )
        .map_err(|e| match e {
            host::HostError::NonFinite => BackendError::Inference("non-finite output".into()),
            other => BackendError::Inference(other.to_string()),
        })?;

        let mut answers = serde_json::Map::new();
        answers.insert(qid.clone(), answer);
        Ok(BackendResult {
            model: "laya-rl-agent".into(),
            answers: serde_json::Value::Object(answers),
            usage: serde_json::json!({
                "input_tokens": inputs.input_len,
                "output_tokens": 0,
            }),
            latency_ms: t0.elapsed().as_millis() as u64,
        })
    }
}

// ---- platform split -----------------------------------------------------

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
impl AneBackend {
    /// Lazy residency: load + signature-validate on first use.
    fn ensure_model(&self) -> Result<(), BackendError> {
        let mut inner = self.inner.lock().unwrap();
        if inner.model.is_some() {
            return Ok(());
        }
        let package = self.cache_or_bundle_package()?;
        let compiled = self.bundle.compiled_model_dir(&self.cache_dir);
        native::compile_package(&package, &compiled).map_err(|e| match e {
            native::NativeError::Capacity(m) => BackendError::Capacity(m),
            native::NativeError::Inference(m) => BackendError::Inference(m),
        })?;
        let units = native::parse_compute_units(&self.compute_units_name)
            .map_err(BackendError::Inference)?;
        let model = native::NativeModel::load(
            &compiled,
            self.hidden,
            self.bundle.fixed_len(),
            self.bundle.max_options(),
            units,
        )
        .map_err(|e| match e {
            native::NativeError::Capacity(m) => BackendError::Capacity(m),
            native::NativeError::Inference(m) => BackendError::Inference(m),
        })?;
        inner.model = Some(model);
        Ok(())
    }

    fn run_model(&self, inputs: &host::AneInputs) -> Result<(Vec<f32>, Vec<f32>), BackendError> {
        let mut inner = self.inner.lock().unwrap();
        let model = inner
            .model
            .as_ref()
            .ok_or_else(|| BackendError::NotReady("ane model not resident".into()))?;
        match model.predict(
            inputs,
            self.bundle.fixed_len(),
            self.hidden,
            self.bundle.max_options(),
        ) {
            Ok(v) => Ok(v),
            Err(native::NativeError::Capacity(m)) => {
                // DESIGN.md: capacity fallback releases ANE before the
                // router's one permitted MLX retry.
                inner.model = None;
                Err(BackendError::Capacity(m))
            }
            Err(native::NativeError::Inference(m)) => Err(BackendError::Inference(m)),
        }
    }

    /// Shape faults are deterministic bugs: record + open the circuit.
    fn shape_fault(&self, msg: String) -> BackendError {
        self.inner.lock().unwrap().circuit_open = true;
        BackendError::Shape(msg)
    }
}

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
impl AneBackend {
    fn ensure_model(&self) -> Result<(), BackendError> {
        Err(BackendError::Unavailable(
            "ane backend requires macOS arm64".into(),
        ))
    }

    fn run_model(&self, _inputs: &host::AneInputs) -> Result<(Vec<f32>, Vec<f32>), BackendError> {
        Err(BackendError::Unavailable(
            "ane backend requires macOS arm64".into(),
        ))
    }

    fn shape_fault(&self, msg: String) -> BackendError {
        self.inner.lock().unwrap().circuit_open = true;
        BackendError::Shape(msg)
    }
}
