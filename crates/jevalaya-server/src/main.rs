//! `jevalaya` — the routing service binary (T002).
//!
//! ```text
//! jevalaya serve [--config PATH] [--listen ADDR]
//! jevalaya check [--config PATH]     # config + engine + adapters, no bind
//! ```
//!
//! Config defaults to `jevalaya.toml` in cwd or `$JEVALAYA_CONFIG`.
//! Secrets are env-only (`auth_token_env`, `api_key_env`).

mod config;
mod server;

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use jevalaya_router_core::{BackendKind, EventSink, JsonlSink, NullSink, PredictBackend, Router};
use tracing::{info, warn};

use config::ServerConfig;

const USAGE: &str =
    "usage: jevalaya serve [--config PATH] [--listen ADDR]\n       jevalaya check [--config PATH]";

struct Args {
    command: String,
    config: PathBuf,
    listen: Option<String>,
}

fn parse_args() -> Result<Args, String> {
    let mut args = std::env::args().skip(1);
    let command = args.next().ok_or_else(|| USAGE.to_string())?;
    if command != "serve" && command != "check" {
        return Err(format!("unknown command {command:?}\n{USAGE}"));
    }
    let mut config = std::env::var("JEVALAYA_CONFIG")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("jevalaya.toml"));
    let mut listen = None;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--config" => {
                config = PathBuf::from(args.next().ok_or("--config requires a path")?);
            }
            "--listen" => {
                listen = Some(args.next().ok_or("--listen requires an address")?);
            }
            other => return Err(format!("unknown flag {other:?}\n{USAGE}")),
        }
    }
    Ok(Args {
        command,
        config,
        listen,
    })
}

/// MLX via the in-process PyO3 bridge (needs the crate feature).
#[cfg(feature = "mlx")]
fn maybe_mlx(cfg: &ServerConfig) -> Option<Arc<dyn PredictBackend>> {
    use jevalaya_backend_mlx_pyo3::{MlxBridge, MlxConfig};
    if !cfg.mlx.enabled {
        return None;
    }
    let models: HashMap<String, String> = cfg
        .model_dirs()
        .into_iter()
        .map(|(cp, d)| (cp.as_str().to_string(), d.display().to_string()))
        .collect();
    let mcfg = MlxConfig {
        python_path: cfg.mlx.python_path.clone(),
        module: cfg.mlx.pyo3_module.clone(),
        models,
        dtype: cfg.mlx.dtype.clone(),
        max_loaded: cfg.mlx.max_loaded,
    };
    match MlxBridge::new(mcfg) {
        Ok(b) => {
            info!(
                "mlx backend up (pyo3 bridge, module {})",
                cfg.mlx.pyo3_module
            );
            Some(Arc::new(b))
        }
        Err(e) => {
            warn!("mlx enabled but bridge failed: {e}");
            None
        }
    }
}

#[cfg(not(feature = "mlx"))]
fn maybe_mlx(_cfg: &ServerConfig) -> Option<Arc<dyn PredictBackend>> {
    None
}

/// Native CoreML/ANE adapter (T004). Off-target builds register nothing;
/// a configured-but-broken bundle is a warning, not an exit.
#[cfg(feature = "ane")]
fn maybe_ane(cfg: &ServerConfig) -> Option<Arc<dyn PredictBackend>> {
    use jevalaya_backend_coreml::{AneBackend, AneConfig};
    if !cfg.ane.enabled {
        return None;
    }
    let model = cfg.ane.model.as_ref()?;
    let acfg = AneConfig {
        model_dir: PathBuf::from(model),
        compute_units: cfg.ane.compute_units.clone(),
        cache_dir: cfg.ane.cache_dir.clone().map(PathBuf::from),
    };
    match AneBackend::new(acfg) {
        Ok(b) => {
            info!("ane backend registered: {}", b.detail_line());
            Some(Arc::new(b))
        }
        Err(e) => {
            warn!("ane enabled but bundle invalid: {e}");
            None
        }
    }
}

#[cfg(not(feature = "ane"))]
fn maybe_ane(_cfg: &ServerConfig) -> Option<Arc<dyn PredictBackend>> {
    None
}

/// Construct backend adapters from config. Failures are warnings, not
/// exits: the service boots with what it has and `degraded`/`/health`
/// tell the truth.
fn build_backends(cfg: &ServerConfig) -> HashMap<BackendKind, Arc<dyn PredictBackend>> {
    let mut backends: HashMap<BackendKind, Arc<dyn PredictBackend>> = HashMap::new();

    if let Some(b) = maybe_ane(cfg) {
        backends.insert(BackendKind::Ane, b);
    }

    if let Some(b) = maybe_mlx(cfg) {
        backends.insert(BackendKind::Mlx, b);
    }

    if cfg.jev.enabled {
        use jevalaya_backend_jev::{JevBackend, JevConfig};
        let jcfg = JevConfig {
            base_url: cfg.jev.base_url.clone(),
            predict_path: cfg.jev.predict_path.clone(),
            api_key_env: cfg.jev.api_key_env.clone(),
            model: cfg.jev.model.clone(),
            request_timeout: Duration::from_millis(cfg.jev.request_timeout_ms),
            max_attempts: cfg.jev.retries,
            backoff: vec![Duration::from_millis(500), Duration::from_millis(1000)],
        };
        match JevBackend::from_env(jcfg) {
            Ok(b) => {
                info!("jev backend up ({})", b.config().endpoint());
                backends.insert(BackendKind::Jev, Arc::new(b));
            }
            Err(e) => warn!("jev enabled but unavailable: {e}"),
        }
    }

    backends
}

fn build_sink(cfg: &ServerConfig) -> Result<Arc<dyn EventSink>, String> {
    match &cfg.observability.sink {
        None => Ok(Arc::new(NullSink)),
        Some(spec) => match spec.strip_prefix("jsonl:") {
            Some(path) => {
                let sink =
                    JsonlSink::open(std::path::Path::new(path), cfg.observability.queue_capacity)
                        .map_err(|e| format!("cannot open event sink {path}: {e}"))?;
                info!("routing events → {path}");
                Ok(Arc::new(sink))
            }
            None => Err(format!(
                "observability.sink {spec:?}: only jsonl:PATH is supported"
            )),
        },
    }
}

fn assemble(cfg: &ServerConfig) -> Result<(Router, Vec<String>), String> {
    if cfg.server.offline {
        // Offline mode: cached hub content only, no network resolution.
        std::env::set_var("HF_HUB_OFFLINE", "1");
    }
    let mut policy = cfg.policy().map_err(|e| e.to_string())?;
    let build = cfg.build_engine();
    for w in &build.warnings {
        warn!("{w}");
    }
    if !build.ane_ready && policy.ane.configured {
        warn!("ane model configured but its tokenizer/budgets failed to load — treating ANE as unconfigured");
        policy.ane.configured = false;
    }
    let backends = build_backends(cfg);
    let built: Vec<BackendKind> = backends.keys().copied().collect();
    let configured = cfg.configured_backends(&built);
    info!("configured backends: {configured:?}; built: {built:?}");
    let sink = build_sink(cfg)?;
    Ok((
        Router::new(policy, Arc::new(build.engine), backends, configured, sink),
        build.warnings,
    ))
}

#[tokio::main(flavor = "multi_thread")]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "jevalaya=info,tower_http=info".into()),
        )
        .init();

    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(2);
        }
    };
    let cfg = match ServerConfig::load(&args.config) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(2);
        }
    };
    let auth_token = match cfg.auth_token() {
        Ok(t) => t,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(2);
        }
    };
    match (&auth_token, &cfg.server.auth_token_env) {
        (Some(_), Some(env)) => info!("bearer auth via env {env}"),
        (None, Some(env)) => unreachable!("auth_token() fails closed on {env}"),
        _ => warn!("no auth_token_env configured — /predict and /route are unauthenticated"),
    }

    let (router, _warnings) = match assemble(&cfg) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(2);
        }
    };

    if args.command == "check" {
        let avail = router.availability();
        let backends = avail.list();
        println!(
            "config ok — available: [{}]{}",
            backends
                .iter()
                .map(|b| b.as_str())
                .collect::<Vec<_>>()
                .join(", "),
            if router.degraded() { " (degraded)" } else { "" }
        );
        return;
    }

    let listen = args
        .listen
        .clone()
        .unwrap_or_else(|| cfg.server.listen.clone());
    let policy = router.config();
    let health_model = policy.model_id(policy.default_checkpoint);
    drop(policy);
    let state = Arc::new(server::AppState {
        router,
        auth_token,
        health_model,
        inflight: tokio::sync::Semaphore::new(cfg.server.max_inflight.max(1)),
        request_timeout: Duration::from_millis(cfg.server.request_timeout_ms),
    });
    // Preload-on-boot runs in the background so /health answers
    // immediately while weights warm (DESIGN.md: readiness without
    // forcing lazy weights).
    {
        let mut preloads: Vec<Arc<dyn PredictBackend>> = Vec::new();
        if cfg.mlx.preload {
            if let Some(b) = state.router.backend(BackendKind::Mlx) {
                preloads.push(b);
            }
        }
        if cfg.ane.preload {
            if let Some(b) = state.router.backend(BackendKind::Ane) {
                preloads.push(b);
            }
        }
        if !preloads.is_empty() {
            tokio::spawn(async move {
                for b in preloads {
                    let kind = b.kind();
                    let res = if b.is_blocking() {
                        tokio::task::spawn_blocking(move || {
                            futures::executor::block_on(b.preload(Default::default()))
                        })
                        .await
                        .unwrap_or_else(|e| {
                            Err(jevalaya_router_core::BackendError::Inference(format!(
                                "preload join: {e}"
                            )))
                        })
                    } else {
                        b.preload(Default::default()).await
                    };
                    match res {
                        Ok(()) => info!("{kind} preload complete"),
                        Err(e) => warn!("{kind} preload failed: {e}"),
                    }
                }
            });
        }
    }

    let app = server::app(state, cfg.server.max_body_bytes);
    let listener = match tokio::net::TcpListener::bind(&listen).await {
        Ok(l) => l,
        Err(e) => {
            eprintln!("cannot bind {listen}: {e}");
            std::process::exit(1);
        }
    };
    info!(
        "jevalaya listening on http://{}",
        listener.local_addr().unwrap()
    );
    if let Err(e) = axum::serve(listener, app)
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
            info!("shutting down");
        })
        .await
    {
        eprintln!("server error: {e}");
        std::process::exit(1);
    }
}
