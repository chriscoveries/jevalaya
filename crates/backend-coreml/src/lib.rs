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
use std::sync::{Arc, Condvar, Mutex};
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
    /// A background residency load is in flight.
    loading: bool,
    /// Latched load failure: `(capacity_class, message)`. Hard failures
    /// latch until `unload`; capacity-class failures are retried on the
    /// next predict.
    load_error: Option<(bool, String)>,
    /// Bumped by `unload` so an in-flight load discards its result.
    generation: u64,
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
    /// `(Mutex, Condvar)` pair: the condvar lets `preload` wait for an
    /// in-flight background load to reach a terminal state.
    state: Arc<(Mutex<Inner>, Condvar)>,
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
            state: Arc::new((
                Mutex::new(Inner {
                    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
                    model: None,
                    loading: false,
                    load_error: None,
                    generation: 0,
                    circuit_open: false,
                }),
                Condvar::new(),
            )),
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
        let inner = self.state.0.lock().unwrap();
        BackendCapabilities {
            kind: BackendKind::Ane,
            available: ANE_TARGET && self.os_ok && !inner.circuit_open,
            detail: if inner.circuit_open {
                format!("{}; circuit open", self.detail)
            } else if inner.loading {
                format!("{}; warming", self.detail)
            } else {
                self.detail.clone()
            },
        }
    }

    async fn preload(&self, _request: PreloadRequest) -> Result<(), BackendError> {
        // Block (in the caller's spawn_blocking context) until the model
        // is resident or loading reaches a terminal failure. When no load
        // is in flight this performs it inline.
        let (mu, cv) = &*self.state;
        loop {
            let generation = {
                let mut inner = mu.lock().unwrap();
                #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
                if inner.model.is_some() {
                    return Ok(());
                }
                if inner.loading {
                    inner = cv.wait(inner).unwrap();
                    continue;
                }
                if let Some((_, msg)) = inner.load_error.clone() {
                    return Err(BackendError::Inference(msg));
                }
                #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
                return Err(BackendError::Unavailable(
                    "ane backend requires macOS arm64".into(),
                ));
                #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
                {
                    inner.loading = true;
                    inner.load_error = None;
                    inner.generation
                }
            };
            #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
            {
                let r = self.load_native();
                let mut inner = mu.lock().unwrap();
                inner.loading = false;
                match r {
                    Ok(model) => {
                        if inner.generation == generation {
                            inner.model = Some(model);
                        }
                    }
                    Err(e) => inner.load_error = Some(latched(&e)),
                }
                cv.notify_all();
                if inner.model.is_some() {
                    return Ok(());
                }
                if let Some((_, m)) = &inner.load_error {
                    return Err(BackendError::Inference(m.clone()));
                }
                // Generation changed mid-load (unload raced us): retry.
            }
        }
    }

    async fn unload(&self, _target: UnloadTarget) -> Result<(), BackendError> {
        let mut inner = self.state.0.lock().unwrap();
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            inner.model = None;
        }
        inner.load_error = None;
        inner.circuit_open = false;
        // Invalidate any in-flight background load; its result is
        // discarded on completion.
        inner.generation += 1;
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
        if self.state.0.lock().unwrap().circuit_open {
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
    /// The full cold-load path: bundle verify → package materialize →
    /// mlpackage compile → MLModel load + signature check. Runs inside
    /// `preload`; the request path uses the background-thread variant in
    /// `ensure_model`.
    fn load_native(&self) -> Result<native::NativeModel, BackendError> {
        let package = self.cache_or_bundle_package()?;
        let compiled = self.bundle.compiled_model_dir(&self.cache_dir);
        native::compile_package(&package, &compiled).map_err(|e| match e {
            native::NativeError::Capacity(m) => BackendError::Capacity(m),
            native::NativeError::Inference(m) => BackendError::Inference(m),
        })?;
        let units = native::parse_compute_units(&self.compute_units_name)
            .map_err(BackendError::Inference)?;
        native::NativeModel::load(
            &compiled,
            self.hidden,
            self.bundle.fixed_len(),
            self.bundle.max_options(),
            units,
        )
        .map_err(|e| match e {
            native::NativeError::Capacity(m) => BackendError::Capacity(m),
            native::NativeError::Inference(m) => BackendError::Inference(m),
        })
    }

    /// Residency gate for the request path: `Ok` when the model is
    /// resident. Otherwise kicks off (or observes) a background load and
    /// returns `NotReady("ane_warming…")` fast so the router fails over
    /// to MLX instead of blocking 50–90 s inside the request.
    fn ensure_model(&self) -> Result<(), BackendError> {
        let (mu, _cv) = &*self.state;
        let mut inner = mu.lock().unwrap();
        if inner.model.is_some() {
            return Ok(());
        }
        if inner.loading {
            return Err(BackendError::NotReady("ane_warming: model loading".into()));
        }
        if let Some((capacity, msg)) = &inner.load_error {
            let (capacity, msg) = (*capacity, msg.clone());
            if !capacity {
                return Err(BackendError::Inference(msg));
            }
            // Transient capacity failure: clear and retry once more.
            inner.load_error = None;
        }
        inner.loading = true;
        let generation = inner.generation;
        // The loader thread cannot borrow `self`, so it re-opens the
        // bundle from owned context.
        let ctx = LoadCtx {
            bundle_dir: self.bundle.dir.clone(),
            cache_dir: self.cache_dir.clone(),
            compute_units: self.compute_units_name.clone(),
            hidden: self.hidden,
            fixed_len: self.bundle.fixed_len(),
            max_options: self.bundle.max_options(),
        };
        let state = self.state.clone();
        std::thread::spawn(move || {
            let r = ctx.load();
            let (mu, cv) = &*state;
            let mut inner = mu.lock().unwrap();
            inner.loading = false;
            match r {
                Ok(model) => {
                    if inner.generation == generation {
                        inner.model = Some(model);
                    }
                }
                Err(e) => inner.load_error = Some(latched(&e)),
            }
            cv.notify_all();
        });
        Err(BackendError::NotReady("ane_warming: model loading".into()))
    }

    fn run_model(&self, inputs: &host::AneInputs) -> Result<(Vec<f32>, Vec<f32>), BackendError> {
        let mut inner = self.state.0.lock().unwrap();
        let model = inner
            .model
            .as_ref()
            .ok_or_else(|| BackendError::NotReady("ane_warming: model loading".into()))?;
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
        self.state.0.lock().unwrap().circuit_open = true;
        BackendError::Shape(msg)
    }
}

/// Latchable load-failure record: `(capacity_class, message)`.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn latched(e: &BackendError) -> (bool, String) {
    match e {
        BackendError::Capacity(m) => (true, m.clone()),
        other => (false, other.to_string()),
    }
}

/// Owned context the background residency thread loads with — the
/// thread cannot borrow `&AneBackend`, so it re-opens the bundle and
/// re-derives the cache paths itself.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
struct LoadCtx {
    bundle_dir: PathBuf,
    cache_dir: PathBuf,
    compute_units: String,
    hidden: usize,
    fixed_len: usize,
    max_options: usize,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
impl LoadCtx {
    fn load(&self) -> Result<native::NativeModel, BackendError> {
        let bundle = bundle::Bundle::open(&self.bundle_dir)
            .map_err(|e| BackendError::Unavailable(format!("bundle: {e}")))?;
        bundle
            .verify_files()
            .map_err(|e| BackendError::Unavailable(format!("bundle integrity: {e}")))?;
        let package = bundle
            .materialized_package(&self.cache_dir)
            .map_err(|e| BackendError::Unavailable(format!("package materialize: {e}")))?;
        let compiled = bundle.compiled_model_dir(&self.cache_dir);
        native::compile_package(&package, &compiled).map_err(|e| match e {
            native::NativeError::Capacity(m) => BackendError::Capacity(m),
            native::NativeError::Inference(m) => BackendError::Inference(m),
        })?;
        let units =
            native::parse_compute_units(&self.compute_units).map_err(BackendError::Inference)?;
        native::NativeModel::load(
            &compiled,
            self.hidden,
            self.fixed_len,
            self.max_options,
            units,
        )
        .map_err(|e| match e {
            native::NativeError::Capacity(m) => BackendError::Capacity(m),
            native::NativeError::Inference(m) => BackendError::Inference(m),
        })
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
        self.state.0.lock().unwrap().circuit_open = true;
        BackendError::Shape(msg)
    }
}
