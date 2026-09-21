//! MLX backend: thin in-process PyO3 bridge reusing the existing laya-mlx
//! package. Rust owns routing/HTTP/tokenizer; Python hosts only the proven
//! forward pass behind [`PredictBackend`].
//!
//! BUILD REQUIREMENT: compiling (and running) this crate needs the Python
//! interpreter that owns mlx + laya_mlx. Point PyO3 at it:
//!
//! ```sh
//! export PYO3_PYTHON=/Users/chrisd/PROJECTS/laya-mlx/.venv/bin/python
//! cargo test -p jevalaya-backend-mlx-pyo3
//! ```
//!
//! The `laya_mlx` module itself is found via [`MlxConfig::python_path`]
//! (source checkout dir), so it is never vendored here. Concurrency: one
//! router behind a mutex — Python calls are mutually exclusive, matching
//! laya-mlx's own bounded semaphore. The server layer adds its own
//! `spawn_blocking` + semaphore around [`MlxBridge::predict_json`].

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Mutex;
use std::time::Instant;

use jevalaya_router_core::{
    BackendCapabilities, BackendError, BackendKind, BackendResult, Checkpoint, PredictBackend,
    PredictRequest, PreloadRequest, RenderedPrompt, UnloadTarget,
};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use serde_json::{Map, Value};

#[derive(Debug, Clone)]
pub struct MlxConfig {
    /// Directories prepended to `sys.path` (front) so `module` imports.
    /// The embedded interpreter is the venv's own libpython (see header docs),
    /// but a venv's `site-packages` is NOT on its default path — list it here
    /// along with the dir CONTAINING `laya_mlx/` for a source checkout, e.g.
    /// `["/path/to/laya-mlx", "/path/to/laya-mlx/.venv/lib/python3.13/site-packages"]`.
    pub python_path: Vec<PathBuf>,
    /// Python module to import (default `laya_mlx`).
    pub module: String,
    /// Router model specs: canonical checkpoint name → HF id or local dir.
    /// Empty means laya-mlx defaults. Local dirs keep this fully offline.
    pub models: HashMap<String, String>,
    pub dtype: String,
    pub max_loaded: usize,
}

impl Default for MlxConfig {
    fn default() -> Self {
        Self {
            python_path: Vec::new(),
            module: "laya_mlx".to_string(),
            models: HashMap::new(),
            dtype: "float16".to_string(),
            max_loaded: 1,
        }
    }
}

pub struct MlxBridge {
    config: MlxConfig,
    router: Mutex<Py<PyAny>>,
}

impl MlxBridge {
    pub fn new(config: MlxConfig) -> Result<Self, BackendError> {
        let router = Python::attach(|py| -> PyResult<Py<PyAny>> {
            let sys = PyModule::import(py, "sys")?;
            let path = sys.getattr("path")?;
            for dir in &config.python_path {
                let dir = dir.to_string_lossy().to_string();
                path.call_method1("insert", (0, dir))?;
            }
            // The bridge must never pull in a torch/Transformers runtime.
            let torch = PyModule::import(py, "torch");
            let transformers = PyModule::import(py, "transformers");
            if torch.is_ok() || transformers.is_ok() {
                return Err(pyo3::exceptions::PyImportError::new_err(
                    "refusing bridge with torch/transformers importable",
                ));
            }
            let module = PyModule::import(py, config.module.as_str())?;
            let cls = module.getattr("Router")?;
            let kwargs = PyDict::new(py);
            if !config.models.is_empty() {
                let models = PyDict::new(py);
                for (k, v) in &config.models {
                    models.set_item(k, v)?;
                }
                kwargs.set_item("models", models)?;
            }
            kwargs.set_item("dtype", config.dtype.as_str())?;
            kwargs.set_item("max_loaded", config.max_loaded)?;
            Ok(cls.call((), Some(&kwargs))?.into())
        })
        .map_err(|e| BackendError::NotReady(format!("python bridge init: {e}")))?;
        Ok(Self {
            config,
            router: Mutex::new(router),
        })
    }

    pub fn config(&self) -> &MlxConfig {
        &self.config
    }

    /// Blocking predict through the embedded `Router.predict` with the
    /// already-selected checkpoint. `rendered` is ignored by design: Python
    /// renders internally, so there is exactly one renderer in the hot path.
    /// Callers must run this on a bounded blocking executor.
    pub fn predict_json(
        &self,
        state: &Value,
        questions: &Map<String, Value>,
        checkpoint: Checkpoint,
    ) -> Result<BackendResult, BackendError> {
        let state_json = serde_json::to_string(state)
            .map_err(|e| BackendError::InvalidResponse(format!("state not JSON: {e}")))?;
        let questions_json = serde_json::to_string(questions)
            .map_err(|e| BackendError::InvalidResponse(format!("questions not JSON: {e}")))?;
        let name = checkpoint.as_str().to_string();
        let started = Instant::now();
        let out_json = Python::attach(|py| -> PyResult<String> {
            let router = self
                .router
                .lock()
                .map_err(|_| pyo3::exceptions::PyRuntimeError::new_err("bridge mutex poisoned"))?;
            let json_mod = PyModule::import(py, "json")?;
            let state_obj = json_mod.call_method1("loads", (state_json,))?;
            let questions_obj = json_mod.call_method1("loads", (questions_json,))?;
            let kwargs = PyDict::new(py);
            kwargs.set_item("model", name)?;
            let result =
                router.call_method(py, "predict", (state_obj, questions_obj), Some(&kwargs))?;
            let dumped: String = json_mod.call_method1("dumps", (result,))?.extract()?;
            Ok(dumped)
        })
        .map_err(|e| BackendError::Inference(format!("mlx predict: {e}")))?;
        let latency_ms = started.elapsed().as_millis() as u64;
        let response: Value = serde_json::from_str(&out_json)
            .map_err(|e| BackendError::InvalidResponse(format!("bridge protocol: {e}")))?;
        let answers = response.get("answers").cloned().ok_or_else(|| {
            BackendError::InvalidResponse("bridge payload missing answers".to_string())
        })?;
        if !answers.is_object() {
            return Err(BackendError::InvalidResponse(
                "bridge payload answers not an object".to_string(),
            ));
        }
        Ok(BackendResult {
            model: response
                .get("model")
                .and_then(Value::as_str)
                .unwrap_or("laya-rl-agent")
                .to_string(),
            answers,
            usage: response
                .get("usage")
                .cloned()
                .unwrap_or_else(|| serde_json::json!({})),
            latency_ms,
        })
    }
}

#[async_trait::async_trait]
impl PredictBackend for MlxBridge {
    fn kind(&self) -> BackendKind {
        BackendKind::Mlx
    }

    /// The bridge's `predict` runs Python/MLX inline — dispatch via the
    /// blocking pool (DESIGN.md: "blocking Python/MLX work behind a
    /// bounded executor").
    fn is_blocking(&self) -> bool {
        true
    }

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            kind: BackendKind::Mlx,
            available: true,
            detail: format!("pyo3 {}", self.config.module),
        }
    }

    async fn preload(&self, request: PreloadRequest) -> Result<(), BackendError> {
        let names: Vec<String> = request
            .checkpoints
            .iter()
            .map(|c| c.as_str().to_string())
            .collect();
        Python::attach(|py| -> PyResult<()> {
            let router = self
                .router
                .lock()
                .map_err(|_| pyo3::exceptions::PyRuntimeError::new_err("bridge mutex"))?;
            if names.is_empty() {
                router.call_method0(py, "preload")?;
            } else {
                router.call_method1(py, "preload", (names,))?;
            }
            Ok(())
        })
        .map_err(|e| BackendError::Inference(format!("mlx preload: {e}")))?;
        Ok(())
    }

    async fn unload(&self, target: UnloadTarget) -> Result<(), BackendError> {
        Python::attach(|py| -> PyResult<()> {
            let router = self
                .router
                .lock()
                .map_err(|_| pyo3::exceptions::PyRuntimeError::new_err("bridge mutex"))?;
            match target {
                UnloadTarget::All => {
                    router.call_method0(py, "unload")?;
                }
                UnloadTarget::Checkpoint(cp) => {
                    router.call_method1(py, "unload", (cp.as_str(),))?;
                }
            }
            Ok(())
        })
        .map_err(|e| BackendError::Inference(format!("mlx unload: {e}")))?;
        Ok(())
    }

    async fn predict(
        &self,
        request: &PredictRequest,
        _rendered: Option<&RenderedPrompt>,
    ) -> Result<BackendResult, BackendError> {
        // Same-thread blocking call; the server layer owns spawn_blocking.
        self.predict_json(&request.state, &request.questions, request.checkpoint)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checkpoint_names_match_router_specs() {
        assert_eq!(Checkpoint::English.as_str(), "english");
        assert_eq!(Checkpoint::Multilingual.as_str(), "multilingual");
        assert_eq!(Checkpoint::TypedDecisions.as_str(), "typed-decisions");
        assert_eq!(MlxConfig::default().module, "laya_mlx");
    }

    /// Live bridge test: real laya-mlx through the embedded interpreter.
    /// Needs `LAYA_MLX_CHECKOUT` (dir containing `laya_mlx/`) and a local
    /// model snapshot; runs fully offline (`HF_HUB_OFFLINE=1`).
    /// Build the crate with `PYO3_PYTHON=<venv python>` (see header docs).
    #[test]
    fn bridge_predict_matches_python_answers() {
        let checkout = match std::env::var("LAYA_MLX_CHECKOUT") {
            Ok(v) => PathBuf::from(v),
            Err(_) => {
                eprintln!("SKIP bridge_predict: set LAYA_MLX_CHECKOUT=<laya-mlx dir>");
                return;
            }
        };
        let snapshot = match std::env::var("LAYA_MLX_ENGLISH_SNAPSHOT") {
            Ok(v) => v,
            Err(_) => {
                eprintln!("SKIP bridge_predict: set LAYA_MLX_ENGLISH_SNAPSHOT=<snapshot dir>");
                return;
            }
        };
        std::env::set_var("HF_HUB_OFFLINE", "1");
        let site_packages = match std::env::var("LAYA_MLX_SITE_PACKAGES") {
            Ok(v) => PathBuf::from(v),
            Err(_) => {
                eprintln!("SKIP bridge_predict: set LAYA_MLX_SITE_PACKAGES=<venv site-packages>");
                return;
            }
        };
        let mut models = HashMap::new();
        models.insert("english".to_string(), snapshot);
        let bridge = MlxBridge::new(MlxConfig {
            python_path: vec![checkout, site_packages],
            models,
            ..MlxConfig::default()
        })
        .expect("bridge init");
        // laya-mlx must stay torch/Transformers-free inside the bridge too.
        let uses_torch = Python::attach(|py| -> PyResult<bool> {
            let sys = PyModule::import(py, "sys")?;
            let modules = sys.getattr("modules")?;
            Ok(modules.contains("torch")? || modules.contains("transformers")?)
        })
        .expect("gil");
        assert!(!uses_torch, "bridge imported torch/transformers");

        let state = serde_json::json!("I was charged twice and want a refund");
        let questions = serde_json::json!({
            "topic": {"type": "choice", "instructions": "Choose the topic",
                      "criteria": ["billing", "refund", "shipping"]},
        })
        .as_object()
        .unwrap()
        .clone();
        let out = bridge
            .predict_json(&state, &questions, Checkpoint::English)
            .expect("predict");
        let choice = out.answers["topic"]["choice"].as_str().expect("choice");
        assert!(
            ["billing", "refund", "shipping"].contains(&choice),
            "unexpected choice {choice}"
        );
        assert!(out.answers["topic"]["confidence"].as_f64().unwrap() > 0.0);
        assert!(out.usage["input_tokens"].as_u64().unwrap() > 0);
        assert_eq!(out.usage["output_tokens"], serde_json::json!(0));
    }
}
