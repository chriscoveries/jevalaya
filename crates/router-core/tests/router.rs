//! Decision-table tests with mock backends + a scripted prompt engine
//! (DESIGN.md §Parity and test plan §2/§6): the six spec routing cases
//! plus escalation and degradation paths. The prompt-parity burden stays
//! in `jevalaya-render`; here the engine is scripted so gate boundaries
//! are exact.

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use jevalaya_render::{GateCounts, Question};
use jevalaya_router_core::*;
use serde_json::{json, Map, Value};

// ---------------------------------------------------------------- fakes

struct ScriptedEngine {
    ane: GateCounts,
    ckpt: GateCounts,
    ane_available: bool,
}

impl ScriptedEngine {
    fn new(raw: usize, aligned: usize) -> Self {
        Self {
            ane: GateCounts {
                raw_count: raw,
                aligned_count: aligned,
            },
            ckpt: GateCounts {
                raw_count: raw,
                aligned_count: aligned,
            },
            ane_available: true,
        }
    }
}

impl PromptEngine for ScriptedEngine {
    fn ane_counts(&self, _s: &Value, _q: &Question) -> Result<GateCounts, RouteError> {
        if !self.ane_available {
            return Err(RouteError::NotReady("no ane tokenizer".into()));
        }
        Ok(self.ane)
    }
    fn ane_render(&self, _s: &Value, q: &Question) -> Result<RenderedPrompt, RouteError> {
        Ok(RenderedPrompt {
            ids: vec![1, 2, 3],
            markers: vec![1],
            qtype: q.qtype.id(),
        })
    }
    fn counts(&self, _c: Checkpoint, _s: &Value, _q: &Question) -> Result<GateCounts, RouteError> {
        Ok(self.ckpt)
    }
    fn execution_len(
        &self,
        _c: Checkpoint,
        _s: &Value,
        _q: &Question,
    ) -> Result<usize, RouteError> {
        Ok(self.ckpt.raw_count.min(64))
    }
}

struct MockBackend {
    kind: BackendKind,
    available: AtomicBool,
    blocking: bool,
    calls: AtomicUsize,
    script: Mutex<VecDeque<Result<BackendResult, BackendError>>>,
    last_rendered: Mutex<Option<RenderedPrompt>>,
}

impl MockBackend {
    fn ok(kind: BackendKind, conf: f64, margin: (f64, f64)) -> Arc<Self> {
        let r = BackendResult {
            model: format!("{kind}-model"),
            answers: json!({
                "q": {
                    "type": "choice", "choice": "a", "confidence": conf,
                    "probabilities": {"a": margin.0, "b": margin.1},
                }
            }),
            usage: json!({"input_tokens": 42, "output_tokens": 0}),
            latency_ms: 3,
        };
        Arc::new(Self {
            kind,
            available: AtomicBool::new(true),
            blocking: false,
            calls: AtomicUsize::new(0),
            script: Mutex::new(VecDeque::from([Ok(r)])),
            last_rendered: Mutex::new(None),
        })
    }

    fn scripted(kind: BackendKind, results: Vec<Result<BackendResult, BackendError>>) -> Arc<Self> {
        Arc::new(Self {
            kind,
            available: AtomicBool::new(true),
            blocking: false,
            calls: AtomicUsize::new(0),
            script: Mutex::new(VecDeque::from(results)),
            last_rendered: Mutex::new(None),
        })
    }

    fn set_available(&self, v: bool) {
        self.available.store(v, Ordering::SeqCst);
    }

    fn calls(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }
}

#[async_trait::async_trait]
impl PredictBackend for MockBackend {
    fn kind(&self) -> BackendKind {
        self.kind
    }
    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            kind: self.kind,
            available: self.available.load(Ordering::SeqCst),
            detail: "mock".to_string(),
        }
    }
    fn is_blocking(&self) -> bool {
        self.blocking
    }
    async fn preload(&self, _r: PreloadRequest) -> Result<(), BackendError> {
        Ok(())
    }
    async fn unload(&self, _t: UnloadTarget) -> Result<(), BackendError> {
        Ok(())
    }
    async fn predict(
        &self,
        _req: &PredictRequest,
        rendered: Option<&RenderedPrompt>,
    ) -> Result<BackendResult, BackendError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        *self.last_rendered.lock().unwrap() = rendered.cloned();
        self.script
            .lock()
            .unwrap()
            .pop_front()
            .unwrap_or_else(|| Err(BackendError::Inference("script exhausted".into())))
    }
}

// ---------------------------------------------------------------- setup

fn question() -> Map<String, Value> {
    json!({"q": {"type": "choice", "instructions": "pick", "criteria": ["a", "b"]}})
        .as_object()
        .unwrap()
        .clone()
}

fn body(state: Value, backend: &str) -> PredictBody {
    serde_json::from_value(json!({
        "state": state,
        "questions": question(),
        "backend": backend,
    }))
    .unwrap()
}

fn cfg_with_ane() -> PolicyConfig {
    let mut cfg = PolicyConfig {
        ane: AnePolicy {
            enabled: true,
            configured: true,
            model_id: "/models/laya-ane".into(),
            max_tokens: 96,
            alignment: 1,
            prefer_ane_for_short: true,
        },
        ..PolicyConfig::default()
    };
    cfg.models.insert(Checkpoint::English, "/models/en".into());
    cfg.models
        .insert(Checkpoint::Multilingual, "/models/multi".into());
    cfg.models
        .insert(Checkpoint::TypedDecisions, "/models/typed".into());
    cfg
}

struct Rig {
    router: Router,
    ane: Arc<MockBackend>,
    mlx: Arc<MockBackend>,
    jev: Arc<MockBackend>,
}

fn rig(engine: ScriptedEngine, cfg: PolicyConfig, ane_up: bool, mlx_up: bool, jev_up: bool) -> Rig {
    let ane = MockBackend::ok(BackendKind::Ane, 0.9, (0.9, 0.1));
    let mlx = MockBackend::ok(BackendKind::Mlx, 0.9, (0.9, 0.1));
    let jev = MockBackend::ok(BackendKind::Jev, 0.9, (0.9, 0.1));
    ane.set_available(ane_up);
    mlx.set_available(mlx_up);
    jev.set_available(jev_up);
    let mut backends: HashMap<BackendKind, Arc<dyn PredictBackend>> = HashMap::new();
    backends.insert(BackendKind::Ane, ane.clone());
    backends.insert(BackendKind::Mlx, mlx.clone());
    backends.insert(BackendKind::Jev, jev.clone());
    let configured = vec![BackendKind::Ane, BackendKind::Mlx, BackendKind::Jev];
    Rig {
        router: Router::for_test(cfg, Arc::new(engine), backends, configured),
        ane,
        mlx,
        jev,
    }
}

/// German state → multilingual checkpoint.
const DE: &str = "Mein Konto wurde zweimal belastet und ich möchte eine Rückerstattung";
const EN: &str = "I was charged twice and I want a refund for this order";

// ---------------------------------------------------------------- tests

#[tokio::test(flavor = "current_thread")]
async fn short_multilingual_auto_routes_ane() {
    let r = rig(
        ScriptedEngine::new(50, 50),
        cfg_with_ane(),
        true,
        true,
        true,
    );
    let out = r.router.predict(body(json!(DE), "auto")).await.unwrap();
    let d = &out.routing;
    assert_eq!(d.backend, BackendKind::Ane);
    assert_eq!(d.checkpoint, Some(Checkpoint::Multilingual));
    assert_eq!(d.reason_prefix(), reason::ANE_SHORT_PATH);
    assert_eq!(d.token_count, Some(50));
    assert!(d.ane_eligible);
    assert_eq!(d.model_id, "/models/laya-ane");
    // The ANE adapter got the pre-rendered prompt.
    assert!(r.ane.last_rendered.lock().unwrap().is_some());
    assert_eq!(r.mlx.calls(), 0);
    assert_eq!(r.jev.calls(), 0);
}

#[tokio::test(flavor = "current_thread")]
async fn aligned_boundary_96_is_ane_97_is_mlx() {
    for (raw, aligned, want) in [
        (95, 95, BackendKind::Ane),
        (96, 96, BackendKind::Ane),
        (97, 97, BackendKind::Mlx),
    ] {
        let r = rig(
            ScriptedEngine::new(raw, aligned),
            cfg_with_ane(),
            true,
            true,
            true,
        );
        let out = r.router.predict(body(json!(DE), "auto")).await.unwrap();
        assert_eq!(
            out.routing.backend, want,
            "raw={raw} aligned={aligned} want {want}"
        );
        if want == BackendKind::Mlx {
            assert_eq!(out.routing.reason_prefix(), reason::TOKEN_COUNT_OVER_LIMIT);
            assert!(!out.routing.ane_eligible);
            assert_eq!(out.routing.token_count, Some(97));
            assert!(out.routing.execution_token_count.is_some());
        }
    }
}

#[tokio::test(flavor = "current_thread")]
async fn alignment_crossing_96_routes_mlx() {
    // alignment 16: raw 96 aligns to 96 (ANE), raw 97 aligns to 112 (MLX).
    let mut cfg = cfg_with_ane();
    cfg.ane.alignment = 16;
    let r = rig(ScriptedEngine::new(96, 96), cfg.clone(), true, true, true);
    let out = r.router.predict(body(json!(DE), "auto")).await.unwrap();
    assert_eq!(out.routing.backend, BackendKind::Ane);

    let r = rig(ScriptedEngine::new(97, 112), cfg, true, true, true);
    let out = r.router.predict(body(json!(DE), "auto")).await.unwrap();
    assert_eq!(out.routing.backend, BackendKind::Mlx);
    assert_eq!(out.routing.reason_prefix(), reason::TOKEN_COUNT_OVER_LIMIT);
}

#[tokio::test(flavor = "current_thread")]
async fn three_questions_route_mlx() {
    let questions = json!({
        "a": {"type": "choice", "instructions": "p", "criteria": ["x", "y"]},
        "b": {"type": "choice", "instructions": "p", "criteria": ["x", "y"]},
        "c": {"type": "choice", "instructions": "p", "criteria": ["x", "y"]},
    });
    let body = serde_json::from_value(json!({
        "state": DE, "questions": questions, "backend": "auto"
    }))
    .unwrap();
    let r = rig(
        ScriptedEngine::new(20, 20),
        cfg_with_ane(),
        true,
        true,
        true,
    );
    let out = r.router.predict(body).await.unwrap();
    assert_eq!(out.routing.backend, BackendKind::Mlx);
    assert_eq!(out.routing.reason_prefix(), reason::QUESTION_COUNT);
    assert!(!out.routing.ane_eligible);
    // multi-question: max per-question raw count reported
    assert_eq!(out.routing.token_count, Some(20));
}

#[tokio::test(flavor = "current_thread")]
async fn ane_fit_overrides_detected_language() {
    // T023: an English single-question request that fits the ANE bundle's
    // tokenizer routes checkpoint=multilingual → ANE — the detector no
    // longer gates ANE, the bundle's own budget does.
    let r = rig(
        ScriptedEngine::new(20, 20),
        cfg_with_ane(),
        true,
        true,
        true,
    );
    let out = r.router.predict(body(json!(EN), "auto")).await.unwrap();
    let d = &out.routing;
    assert_eq!(d.backend, BackendKind::Ane);
    assert_eq!(d.checkpoint, Some(Checkpoint::Multilingual));
    assert_eq!(d.reason_prefix(), reason::ANE_FIT_MULTILINGUAL);
    // the detector's verdict stays visible in the detail
    assert!(d.reason.contains("English Latin text"), "{}", d.reason);
    assert!(d.ane_eligible);
    assert_eq!(r.ane.calls(), 1);
    assert_eq!(r.mlx.calls(), 0);

    // ANE down → same checkpoint decision, served by multilingual MLX.
    let r = rig(
        ScriptedEngine::new(20, 20),
        cfg_with_ane(),
        false,
        true,
        true,
    );
    let out = r.router.predict(body(json!(EN), "auto")).await.unwrap();
    let d = &out.routing;
    assert_eq!(d.backend, BackendKind::Mlx);
    assert_eq!(d.checkpoint, Some(Checkpoint::Multilingual));
    assert_eq!(d.reason_prefix(), reason::ANE_UNAVAILABLE);
    assert!(d.reason.contains(reason::ANE_FIT_MULTILINGUAL), "{}", d.reason);
    assert!(d.ane_eligible);
    assert!(d.degraded);

    // prefer_ane_for_short=false: fit still selects the checkpoint, MLX
    // executes it on multilingual weights.
    let mut cfg = cfg_with_ane();
    cfg.ane.prefer_ane_for_short = false;
    let r = rig(ScriptedEngine::new(20, 20), cfg, true, true, true);
    let out = r.router.predict(body(json!(EN), "auto")).await.unwrap();
    let d = &out.routing;
    assert_eq!(d.backend, BackendKind::Mlx);
    assert_eq!(d.checkpoint, Some(Checkpoint::Multilingual));
    assert_eq!(d.reason_prefix(), reason::ANE_NOT_PREFERRED);
    assert!(d.ane_eligible);
}

#[tokio::test(flavor = "current_thread")]
async fn explicit_pins_and_mlx_backend_beat_fit() {
    // model= pin names weights ANE doesn't have → not_multilingual stands.
    let r = rig(
        ScriptedEngine::new(20, 20),
        cfg_with_ane(),
        true,
        true,
        true,
    );
    let b: PredictBody = serde_json::from_value(json!({
        "state": EN, "questions": question(), "model": "english",
    }))
    .unwrap();
    let out = r.router.predict(b).await.unwrap();
    assert_eq!(out.routing.backend, BackendKind::Mlx);
    assert_eq!(out.routing.checkpoint, Some(Checkpoint::English));
    assert_eq!(out.routing.reason_prefix(), reason::NOT_MULTILINGUAL);
    assert!(!out.routing.ane_eligible);
    assert_eq!(r.ane.calls(), 0);

    // lang= is a hint, not a pin: EN text + explicit lang=en still fits.
    let r = rig(
        ScriptedEngine::new(20, 20),
        cfg_with_ane(),
        true,
        true,
        true,
    );
    let b: PredictBody = serde_json::from_value(json!({
        "state": EN, "questions": question(), "lang": "en",
    }))
    .unwrap();
    let out = r.router.predict(b).await.unwrap();
    assert_eq!(out.routing.backend, BackendKind::Ane);
    assert_eq!(out.routing.checkpoint, Some(Checkpoint::Multilingual));
    assert_eq!(out.routing.reason_prefix(), reason::ANE_FIT_MULTILINGUAL);

    // backend=mlx keeps the detected checkpoint — fit only informs the
    // ane_eligible report; ANE is never dispatched.
    let r = rig(
        ScriptedEngine::new(20, 20),
        cfg_with_ane(),
        true,
        true,
        true,
    );
    let out = r.router.predict(body(json!(EN), "mlx")).await.unwrap();
    let d = &out.routing;
    assert_eq!(d.backend, BackendKind::Mlx);
    assert_eq!(d.checkpoint, Some(Checkpoint::English));
    assert_eq!(d.reason_prefix(), reason::EXPLICIT_BACKEND);
    assert!(d.ane_eligible); // it WOULD fit — honest report
    assert_eq!(r.ane.calls(), 0);
}

#[tokio::test(flavor = "current_thread")]
async fn typed_decisions_never_routes_ane() {
    // Explicit typed-decisions task → typed_decisions gate
    let b: PredictBody = serde_json::from_value(json!({
        "state": EN, "questions": question(), "task": "typed_decisions",
    }))
    .unwrap();
    let r = rig(
        ScriptedEngine::new(20, 20),
        cfg_with_ane(),
        true,
        true,
        true,
    );
    let out = r.router.predict(b).await.unwrap();
    assert_eq!(out.routing.backend, BackendKind::Mlx);
    assert_eq!(out.routing.checkpoint, Some(Checkpoint::TypedDecisions));
    assert_eq!(out.routing.reason_prefix(), reason::TYPED_DECISIONS);

    // A typed workflow id-set blocks ANE even with detection off
    let wf = json!({
        "action": {"type":"choice","instructions":"p","criteria":["x","y"]},
        "category": {"type":"choice","instructions":"p","criteria":["x","y"]},
        "churn_risk": {"type":"score","instructions":"p","criteria":["a","b","c"]},
        "needs_human": {"type":"noul","instructions":"p"},
        "urgency": {"type":"score","instructions":"p","criteria":["a","b","c"]},
    });
    let b: PredictBody = serde_json::from_value(json!({
        "state": DE, "questions": wf,
    }))
    .unwrap();
    let r = rig(
        ScriptedEngine::new(20, 20),
        cfg_with_ane(),
        true,
        true,
        true,
    );
    let out = r.router.predict(b).await.unwrap();
    assert_eq!(out.routing.backend, BackendKind::Mlx);
    assert_eq!(out.routing.reason_prefix(), reason::TYPED_DECISIONS);
}

#[tokio::test(flavor = "current_thread")]
async fn explicit_backend_overrides() {
    // explicit mlx hard-disables ANE
    let r = rig(
        ScriptedEngine::new(20, 20),
        cfg_with_ane(),
        true,
        true,
        true,
    );
    let out = r.router.predict(body(json!(DE), "mlx")).await.unwrap();
    assert_eq!(out.routing.backend, BackendKind::Mlx);
    assert_eq!(out.routing.reason_prefix(), reason::EXPLICIT_BACKEND);
    assert_eq!(r.ane.calls(), 0);

    // explicit jev → jev, checkpoint/token_count null
    let r = rig(
        ScriptedEngine::new(20, 20),
        cfg_with_ane(),
        true,
        true,
        true,
    );
    let out = r.router.predict(body(json!(DE), "jev")).await.unwrap();
    assert_eq!(out.routing.backend, BackendKind::Jev);
    assert_eq!(out.routing.checkpoint, None);
    assert_eq!(out.routing.token_count, None);
    assert_eq!(r.ane.calls(), 0);
    assert_eq!(r.mlx.calls(), 0);

    // explicit ane + ane up → ANE even with prefer_ane_for_short off
    let mut cfg = cfg_with_ane();
    cfg.ane.prefer_ane_for_short = false;
    let r = rig(ScriptedEngine::new(20, 20), cfg, true, true, true);
    let out = r.router.predict(body(json!(DE), "ane")).await.unwrap();
    assert_eq!(out.routing.backend, BackendKind::Ane);

    // explicit ane + ane down → 409 backend_unavailable
    let r = rig(
        ScriptedEngine::new(20, 20),
        cfg_with_ane(),
        false,
        true,
        true,
    );
    let err = r.router.predict(body(json!(DE), "ane")).await.unwrap_err();
    assert_eq!(err.status(), 409);
    assert_eq!(err.code(), "backend_unavailable");

    // auto + ane down → mlx with ane_unavailable
    let r = rig(
        ScriptedEngine::new(20, 20),
        cfg_with_ane(),
        false,
        true,
        true,
    );
    let out = r.router.predict(body(json!(DE), "auto")).await.unwrap();
    assert_eq!(out.routing.backend, BackendKind::Mlx);
    assert_eq!(out.routing.reason_prefix(), reason::ANE_UNAVAILABLE);
    assert!(out.routing.ane_eligible);
    assert!(out.routing.degraded);

    // explicit mlx + mlx down → 409
    let r = rig(
        ScriptedEngine::new(20, 20),
        cfg_with_ane(),
        true,
        false,
        true,
    );
    let err = r.router.predict(body(json!(EN), "mlx")).await.unwrap_err();
    assert_eq!(err.status(), 409);
}

#[tokio::test(flavor = "current_thread")]
async fn prefer_ane_off_auto_routes_mlx() {
    let mut cfg = cfg_with_ane();
    cfg.ane.prefer_ane_for_short = false;
    let r = rig(ScriptedEngine::new(20, 20), cfg, true, true, true);
    let out = r.router.predict(body(json!(DE), "auto")).await.unwrap();
    assert_eq!(out.routing.backend, BackendKind::Mlx);
    assert_eq!(out.routing.reason_prefix(), reason::ANE_NOT_PREFERRED);
    assert!(out.routing.ane_eligible); // eligible but not preferred
}

#[tokio::test(flavor = "current_thread")]
async fn ane_capacity_retries_once_on_mlx() {
    let ane = MockBackend::scripted(
        BackendKind::Ane,
        vec![Err(BackendError::Capacity("memory pressure".into()))],
    );
    let mlx = MockBackend::ok(BackendKind::Mlx, 0.9, (0.9, 0.1));
    let jev = MockBackend::ok(BackendKind::Jev, 0.9, (0.9, 0.1));
    let mut backends: HashMap<BackendKind, Arc<dyn PredictBackend>> = HashMap::new();
    backends.insert(BackendKind::Ane, ane.clone());
    backends.insert(BackendKind::Mlx, mlx.clone());
    backends.insert(BackendKind::Jev, jev.clone());
    let r = Router::for_test(
        cfg_with_ane(),
        Arc::new(ScriptedEngine::new(50, 50)),
        backends,
        vec![BackendKind::Ane, BackendKind::Mlx, BackendKind::Jev],
    );
    let out = r.predict(body(json!(DE), "auto")).await.unwrap();
    let d = &out.routing;
    assert_eq!(d.backend, BackendKind::Mlx);
    assert!(d.fallback);
    assert_eq!(d.fallback_from, Some(BackendKind::Ane));
    assert_eq!(d.fallback_reason.as_deref(), Some("ane_capacity"));
    assert!(d.ane_eligible); // remains true after fallback
    assert_eq!(d.token_count, Some(50));
    assert_eq!(ane.calls(), 1); // ANE never twice in one request
    assert_eq!(mlx.calls(), 1);
}

#[tokio::test(flavor = "current_thread")]
async fn ane_warming_retries_once_on_mlx() {
    // Cold ANE residency reports a fast NotReady("ane_warming: …"); the
    // router fails over to MLX in the same request rather than blocking
    // on the 50–90 s load.
    let ane = MockBackend::scripted(
        BackendKind::Ane,
        vec![Err(BackendError::NotReady(
            "ane_warming: model loading".into(),
        ))],
    );
    let mlx = MockBackend::ok(BackendKind::Mlx, 0.9, (0.9, 0.1));
    let mut backends: HashMap<BackendKind, Arc<dyn PredictBackend>> = HashMap::new();
    backends.insert(BackendKind::Ane, ane.clone());
    backends.insert(BackendKind::Mlx, mlx.clone());
    let r = Router::for_test(
        cfg_with_ane(),
        Arc::new(ScriptedEngine::new(50, 50)),
        backends,
        vec![BackendKind::Ane, BackendKind::Mlx],
    );
    let out = r.predict(body(json!(DE), "auto")).await.unwrap();
    let d = &out.routing;
    assert_eq!(d.backend, BackendKind::Mlx);
    assert!(d.fallback);
    assert_eq!(d.fallback_from, Some(BackendKind::Ane));
    assert_eq!(d.fallback_reason.as_deref(), Some("not_ready"));
    assert!(d.reason.contains("ane_warming"), "reason: {}", d.reason);
    assert_eq!(ane.calls(), 1);
    assert_eq!(mlx.calls(), 1);
}

#[tokio::test(flavor = "current_thread")]
async fn ane_hard_error_never_retries() {
    let ane = MockBackend::scripted(
        BackendKind::Ane,
        vec![Err(BackendError::Inference("corrupt model".into()))],
    );
    let mlx = MockBackend::ok(BackendKind::Mlx, 0.9, (0.9, 0.1));
    let mut backends: HashMap<BackendKind, Arc<dyn PredictBackend>> = HashMap::new();
    backends.insert(BackendKind::Ane, ane.clone());
    backends.insert(BackendKind::Mlx, mlx.clone());
    let r = Router::for_test(
        cfg_with_ane(),
        Arc::new(ScriptedEngine::new(50, 50)),
        backends,
        vec![BackendKind::Ane, BackendKind::Mlx],
    );
    let err = r.predict(body(json!(DE), "auto")).await.unwrap_err();
    assert_eq!(err.status(), 500);
    assert_eq!(ane.calls(), 1);
    assert_eq!(mlx.calls(), 0); // no retry on hard failures
}

#[tokio::test(flavor = "current_thread")]
async fn low_margin_escalates_to_jev() {
    let mlx = MockBackend::ok(BackendKind::Mlx, 0.6, (0.55, 0.45)); // margin .10 < .20
    let jev = MockBackend::ok(BackendKind::Jev, 0.95, (0.95, 0.05));
    let mut backends: HashMap<BackendKind, Arc<dyn PredictBackend>> = HashMap::new();
    backends.insert(BackendKind::Mlx, mlx.clone());
    backends.insert(BackendKind::Jev, jev.clone());
    let r = Router::for_test(
        cfg_with_ane(),
        Arc::new(ScriptedEngine::new(20, 20)),
        backends,
        vec![BackendKind::Mlx, BackendKind::Jev],
    );
    let out = r.predict(body(json!(EN), "auto")).await.unwrap();
    let d = &out.routing;
    assert_eq!(d.backend, BackendKind::Jev);
    assert!(d.escalated);
    assert_eq!(d.jev_trigger.as_deref(), Some(trigger::LOW_MARGIN));
    assert_eq!(d.confidence, Some(0.95)); // jev's answer wins
    assert_eq!(out.model, "jev-model");
}

#[tokio::test(flavor = "current_thread")]
async fn low_confidence_escalates() {
    let mlx = MockBackend::ok(BackendKind::Mlx, 0.5, (0.9, 0.1)); // conf .5 < .75, margin fine
    let jev = MockBackend::ok(BackendKind::Jev, 0.95, (0.95, 0.05));
    let mut backends: HashMap<BackendKind, Arc<dyn PredictBackend>> = HashMap::new();
    backends.insert(BackendKind::Mlx, mlx);
    backends.insert(BackendKind::Jev, jev);
    let r = Router::for_test(
        cfg_with_ane(),
        Arc::new(ScriptedEngine::new(20, 20)),
        backends,
        vec![BackendKind::Mlx, BackendKind::Jev],
    );
    let out = r.predict(body(json!(EN), "auto")).await.unwrap();
    assert_eq!(out.routing.backend, BackendKind::Jev);
    assert_eq!(
        out.routing.jev_trigger.as_deref(),
        Some(trigger::LOW_CONFIDENCE)
    );
}

#[tokio::test(flavor = "current_thread")]
async fn jev_escalation_failure_keeps_local_answer() {
    let mlx = MockBackend::ok(BackendKind::Mlx, 0.6, (0.55, 0.45));
    let jev = MockBackend::scripted(
        BackendKind::Jev,
        vec![Err(BackendError::Upstream(
            Some(503),
            "provider down".into(),
        ))],
    );
    let mut backends: HashMap<BackendKind, Arc<dyn PredictBackend>> = HashMap::new();
    backends.insert(BackendKind::Mlx, mlx);
    backends.insert(BackendKind::Jev, jev);
    let r = Router::for_test(
        cfg_with_ane(),
        Arc::new(ScriptedEngine::new(20, 20)),
        backends,
        vec![BackendKind::Mlx, BackendKind::Jev],
    );
    let out = r.predict(body(json!(EN), "auto")).await.unwrap();
    let d = &out.routing;
    assert_eq!(d.backend, BackendKind::Mlx); // local answer retained
    assert!(d.escalated);
    assert_eq!(d.escalation_error.as_deref(), Some("upstream_503"));
    assert!(d.degraded);
    assert_eq!(d.confidence, Some(0.6)); // original confidence kept
    assert_eq!(out.model, "mlx-model");
}

#[tokio::test(flavor = "current_thread")]
async fn per_backend_thresholds_judge_the_serving_backend() {
    // T022: identical answer (conf .65, margin .90) — under ANE gates
    // (conf .6 / margin .2) it is kept; under MLX gates (conf .7 / margin
    // .3) it escalates. Confidence scales differ per runtime; the serving
    // backend owns the calibration.
    let mut cfg = cfg_with_ane();
    cfg.thresholds.ane = Some(BackendThresholds {
        confidence: Some(0.6),
        margin: Some(0.2),
        ..Default::default()
    });
    cfg.thresholds.mlx = Some(BackendThresholds {
        confidence: Some(0.7),
        margin: Some(0.3),
        ..Default::default()
    });

    // ANE-served (DE + fit): kept, no escalation.
    let ane = MockBackend::ok(BackendKind::Ane, 0.65, (0.95, 0.05));
    let mlx = MockBackend::ok(BackendKind::Mlx, 0.65, (0.95, 0.05));
    let jev = MockBackend::ok(BackendKind::Jev, 0.95, (0.95, 0.05));
    let mut backends: HashMap<BackendKind, Arc<dyn PredictBackend>> = HashMap::new();
    backends.insert(BackendKind::Ane, ane.clone());
    backends.insert(BackendKind::Mlx, mlx.clone());
    backends.insert(BackendKind::Jev, jev.clone());
    let r = Router::for_test(
        cfg.clone(),
        Arc::new(ScriptedEngine::new(20, 20)),
        backends,
        vec![BackendKind::Ane, BackendKind::Mlx, BackendKind::Jev],
    );
    let out = r.predict(body(json!(DE), "auto")).await.unwrap();
    assert_eq!(out.routing.backend, BackendKind::Ane);
    assert!(!out.routing.escalated);
    assert_eq!(jev.calls(), 0);

    // MLX-served (model=english pin → ANE never in play): same answer
    // crosses the MLX confidence gate → escalates.
    let mlx = MockBackend::ok(BackendKind::Mlx, 0.65, (0.95, 0.05));
    let jev = MockBackend::ok(BackendKind::Jev, 0.95, (0.95, 0.05));
    let mut backends: HashMap<BackendKind, Arc<dyn PredictBackend>> = HashMap::new();
    backends.insert(BackendKind::Ane, MockBackend::ok(BackendKind::Ane, 0.9, (0.9, 0.1)));
    backends.insert(BackendKind::Mlx, mlx.clone());
    backends.insert(BackendKind::Jev, jev.clone());
    let r = Router::for_test(
        cfg,
        Arc::new(ScriptedEngine::new(20, 20)),
        backends,
        vec![BackendKind::Ane, BackendKind::Mlx, BackendKind::Jev],
    );
    let b: PredictBody = serde_json::from_value(json!({
        "state": EN, "questions": question(), "model": "english",
    }))
    .unwrap();
    let out = r.predict(b).await.unwrap();
    assert_eq!(out.routing.backend, BackendKind::Jev);
    assert!(out.routing.escalated);
    assert_eq!(
        out.routing.jev_trigger.as_deref(),
        Some(trigger::LOW_CONFIDENCE)
    );
    assert_eq!(jev.calls(), 1);
}

#[tokio::test(flavor = "current_thread")]
async fn missing_threshold_table_falls_back_to_global() {
    // No [policy.thresholds.*] tables → the [jev] global keys (0.75 /
    // 0.20 / 0.85 defaults) judge every backend, exactly as pre-T022.
    let ane = MockBackend::ok(BackendKind::Ane, 0.7, (0.9, 0.1)); // conf .7 < .75
    let mlx = MockBackend::ok(BackendKind::Mlx, 0.9, (0.9, 0.1));
    let jev = MockBackend::ok(BackendKind::Jev, 0.95, (0.95, 0.05));
    let mut backends: HashMap<BackendKind, Arc<dyn PredictBackend>> = HashMap::new();
    backends.insert(BackendKind::Ane, ane);
    backends.insert(BackendKind::Mlx, mlx);
    backends.insert(BackendKind::Jev, jev.clone());
    let r = Router::for_test(
        cfg_with_ane(),
        Arc::new(ScriptedEngine::new(20, 20)),
        backends,
        vec![BackendKind::Ane, BackendKind::Mlx, BackendKind::Jev],
    );
    let out = r.predict(body(json!(DE), "auto")).await.unwrap();
    assert_eq!(out.routing.backend, BackendKind::Jev);
    assert_eq!(
        out.routing.jev_trigger.as_deref(),
        Some(trigger::LOW_CONFIDENCE)
    );
    assert_eq!(jev.calls(), 1);
}

#[tokio::test(flavor = "current_thread")]
async fn ane_mlx_fallback_is_judged_by_mlx_gates() {
    // ANE margin gate is .2, MLX's is .3. ANE fails → MLX answers with
    // margin .25: kept under ANE calibration, escalated under MLX's —
    // the serving backend's gates win.
    let mut cfg = cfg_with_ane();
    cfg.thresholds.ane = Some(BackendThresholds {
        margin: Some(0.2),
        ..Default::default()
    });
    cfg.thresholds.mlx = Some(BackendThresholds {
        margin: Some(0.3),
        ..Default::default()
    });
    let ane = MockBackend::scripted(
        BackendKind::Ane,
        vec![Err(BackendError::Capacity("pressure".into()))],
    );
    let mlx = MockBackend::ok(BackendKind::Mlx, 0.9, (0.625, 0.375)); // margin .25
    let jev = MockBackend::ok(BackendKind::Jev, 0.95, (0.95, 0.05));
    let mut backends: HashMap<BackendKind, Arc<dyn PredictBackend>> = HashMap::new();
    backends.insert(BackendKind::Ane, ane);
    backends.insert(BackendKind::Mlx, mlx);
    backends.insert(BackendKind::Jev, jev.clone());
    let r = Router::for_test(
        cfg,
        Arc::new(ScriptedEngine::new(50, 50)),
        backends,
        vec![BackendKind::Ane, BackendKind::Mlx, BackendKind::Jev],
    );
    let out = r.predict(body(json!(DE), "auto")).await.unwrap();
    let d = &out.routing;
    assert!(d.fallback);
    assert_eq!(d.fallback_from, Some(BackendKind::Ane));
    assert_eq!(d.backend, BackendKind::Jev);
    assert!(d.escalated);
    assert_eq!(d.jev_trigger.as_deref(), Some(trigger::LOW_MARGIN));
    assert_eq!(jev.calls(), 1);
}

#[tokio::test(flavor = "current_thread")]
async fn escalate_on_retry_fires_jev() {
    let mut cfg = cfg_with_ane();
    cfg.jev.escalate_on_retry = true;
    let ane = MockBackend::scripted(
        BackendKind::Ane,
        vec![Err(BackendError::Shape("fixed len 96".into()))],
    );
    let mlx = MockBackend::ok(BackendKind::Mlx, 0.9, (0.9, 0.1)); // confident — still escalates
    let jev = MockBackend::ok(BackendKind::Jev, 0.95, (0.95, 0.05));
    let mut backends: HashMap<BackendKind, Arc<dyn PredictBackend>> = HashMap::new();
    backends.insert(BackendKind::Ane, ane);
    backends.insert(BackendKind::Mlx, mlx);
    backends.insert(BackendKind::Jev, jev);
    let r = Router::for_test(
        cfg,
        Arc::new(ScriptedEngine::new(50, 50)),
        backends,
        vec![BackendKind::Ane, BackendKind::Mlx, BackendKind::Jev],
    );
    let out = r.predict(body(json!(DE), "auto")).await.unwrap();
    assert_eq!(out.routing.backend, BackendKind::Jev);
    assert!(out.routing.fallback);
    assert_eq!(out.routing.jev_trigger.as_deref(), Some(trigger::RETRY));
}

#[tokio::test(flavor = "current_thread")]
async fn degradation_matrix() {
    // Only jev → auto passes through, degraded
    let jev = MockBackend::ok(BackendKind::Jev, 0.9, (0.9, 0.1));
    let mut backends: HashMap<BackendKind, Arc<dyn PredictBackend>> = HashMap::new();
    backends.insert(BackendKind::Jev, jev);
    let r = Router::for_test(
        cfg_with_ane(),
        Arc::new(ScriptedEngine::new(20, 20)),
        backends,
        vec![BackendKind::Ane, BackendKind::Mlx, BackendKind::Jev],
    );
    let out = r.predict(body(json!(EN), "auto")).await.unwrap();
    assert_eq!(out.routing.backend, BackendKind::Jev);
    assert!(out.routing.degraded);
    assert_eq!(out.routing.reason_prefix(), reason::NO_LOCAL_BACKEND);

    // Only mlx → everything local, degraded
    let mlx = MockBackend::ok(BackendKind::Mlx, 0.9, (0.9, 0.1));
    let mut backends: HashMap<BackendKind, Arc<dyn PredictBackend>> = HashMap::new();
    backends.insert(BackendKind::Mlx, mlx);
    let r = Router::for_test(
        cfg_with_ane(),
        Arc::new(ScriptedEngine::new(20, 20)),
        backends,
        vec![BackendKind::Ane, BackendKind::Mlx, BackendKind::Jev],
    );
    let out = r.predict(body(json!(DE), "auto")).await.unwrap();
    assert_eq!(out.routing.backend, BackendKind::Mlx);
    assert!(out.routing.degraded);
    // jev not configured→absent: no escalation machinery engaged either way

    // Nothing usable → 503 not_ready
    let r = Router::for_test(
        cfg_with_ane(),
        Arc::new(ScriptedEngine::new(20, 20)),
        HashMap::new(),
        vec![BackendKind::Ane, BackendKind::Mlx, BackendKind::Jev],
    );
    let err = r.predict(body(json!(EN), "auto")).await.unwrap_err();
    assert_eq!(err.status(), 503);
}

#[tokio::test(flavor = "current_thread")]
async fn explicit_jev_failure_is_upstream() {
    let jev = MockBackend::scripted(
        BackendKind::Jev,
        vec![Err(BackendError::Upstream(Some(503), "down".into()))],
    );
    let mut backends: HashMap<BackendKind, Arc<dyn PredictBackend>> = HashMap::new();
    backends.insert(BackendKind::Jev, jev);
    let r = Router::for_test(
        cfg_with_ane(),
        Arc::new(ScriptedEngine::new(20, 20)),
        backends,
        vec![BackendKind::Jev],
    );
    let err = r.predict(body(json!(EN), "jev")).await.unwrap_err();
    assert_eq!(err.status(), 502);
}

#[tokio::test(flavor = "current_thread")]
async fn compare_fans_out_and_reports_partial() {
    // Primary routes mlx (english); compare mlx+jev both succeed.
    let mlx = MockBackend::ok(BackendKind::Mlx, 0.9, (0.9, 0.1));
    let jev = MockBackend::ok(BackendKind::Jev, 0.8, (0.8, 0.2));
    let mut backends: HashMap<BackendKind, Arc<dyn PredictBackend>> = HashMap::new();
    backends.insert(BackendKind::Mlx, mlx.clone());
    backends.insert(BackendKind::Jev, jev.clone());
    let r = Router::for_test(
        cfg_with_ane(),
        Arc::new(ScriptedEngine::new(20, 20)),
        backends,
        vec![BackendKind::Mlx, BackendKind::Jev],
    );
    let mut b = body(json!(EN), "auto");
    b.compare = Some(CompareSpec::List(vec!["mlx".into(), "jev".into()]));
    let out = r.predict(b).await.unwrap();
    assert_eq!(out.routing.backend, BackendKind::Mlx);
    assert!(out.compare.contains_key("mlx"));
    assert!(out.compare.contains_key("jev"));
    assert_eq!(out.compare["jev"]["model"], json!("jev-model"));
    // primary result reused — mlx dispatched once
    assert_eq!(mlx.calls(), 1);

    // ane requested in compare but request ineligible → structured error.
    // Three questions fail the question_count gate even though the text
    // would tokenize fine.
    let ane = MockBackend::ok(BackendKind::Ane, 0.9, (0.9, 0.1));
    let mlx = MockBackend::ok(BackendKind::Mlx, 0.9, (0.9, 0.1));
    let mut backends: HashMap<BackendKind, Arc<dyn PredictBackend>> = HashMap::new();
    backends.insert(BackendKind::Ane, ane.clone());
    backends.insert(BackendKind::Mlx, mlx);
    let r = Router::for_test(
        cfg_with_ane(),
        Arc::new(ScriptedEngine::new(20, 20)),
        backends,
        vec![BackendKind::Ane, BackendKind::Mlx],
    );
    let mut b: PredictBody = serde_json::from_value(json!({
        "state": EN,
        "questions": {
            "a": {"type":"choice","instructions":"p","criteria":["x","y"]},
            "b": {"type":"choice","instructions":"p","criteria":["x","y"]},
            "c": {"type":"choice","instructions":"p","criteria":["x","y"]},
        },
        "backend": "auto",
    }))
    .unwrap();
    b.compare = Some(CompareSpec::List(vec!["ane".into(), "mlx".into()]));
    let out = r.predict(b).await.unwrap();
    assert_eq!(out.compare["ane"]["error"]["code"], json!("ane_ineligible"));
    assert_eq!(ane.calls(), 0); // never executed
}

#[tokio::test(flavor = "current_thread")]
async fn compare_true_uses_configured_set() {
    let mlx = MockBackend::ok(BackendKind::Mlx, 0.9, (0.9, 0.1));
    let jev = MockBackend::ok(BackendKind::Jev, 0.8, (0.8, 0.2));
    let mut backends: HashMap<BackendKind, Arc<dyn PredictBackend>> = HashMap::new();
    backends.insert(BackendKind::Mlx, mlx);
    backends.insert(BackendKind::Jev, jev);
    let r = Router::for_test(
        cfg_with_ane(),
        Arc::new(ScriptedEngine::new(20, 20)),
        backends,
        vec![BackendKind::Mlx, BackendKind::Jev],
    );
    let mut b = body(json!(EN), "auto");
    b.compare = Some(CompareSpec::Flag(true));
    let out = r.predict(b).await.unwrap();
    assert!(out.compare.contains_key("mlx"));
    assert!(out.compare.contains_key("jev"));
}

#[tokio::test(flavor = "current_thread")]
async fn multi_question_min_confidence_margin() {
    // one ambiguous answer must not hide behind a confident one
    let mlx = MockBackend::scripted(
        BackendKind::Mlx,
        vec![Ok(BackendResult {
            model: "mlx-model".into(),
            answers: json!({
                "a": {"type":"choice","choice":"x","confidence":0.99,
                      "probabilities":{"x":0.99,"y":0.01}},
                "b": {"type":"choice","choice":"x","confidence":0.60,
                      "probabilities":{"x":0.55,"y":0.45}},
            }),
            usage: json!({}),
            latency_ms: 5,
        })],
    );
    let jev = MockBackend::ok(BackendKind::Jev, 0.95, (0.95, 0.05));
    let mut backends: HashMap<BackendKind, Arc<dyn PredictBackend>> = HashMap::new();
    backends.insert(BackendKind::Mlx, mlx);
    backends.insert(BackendKind::Jev, jev);
    let r = Router::for_test(
        cfg_with_ane(),
        Arc::new(ScriptedEngine::new(20, 20)),
        backends,
        vec![BackendKind::Mlx, BackendKind::Jev],
    );
    let b: PredictBody = serde_json::from_value(json!({
        "state": EN,
        "questions": {
            "a": {"type":"choice","instructions":"p","criteria":["x","y"]},
            "b": {"type":"choice","instructions":"p","criteria":["x","y"]},
        },
    }))
    .unwrap();
    let out = r.predict(b).await.unwrap();
    // min margin .10 < .20 → escalated
    assert_eq!(out.routing.backend, BackendKind::Jev);
    assert_eq!(
        out.routing.jev_trigger.as_deref(),
        Some(trigger::LOW_MARGIN)
    );
}

#[tokio::test(flavor = "current_thread")]
async fn routedecision_wire_shape() {
    let r = rig(
        ScriptedEngine::new(50, 50),
        cfg_with_ane(),
        true,
        true,
        true,
    );
    let out = r.router.predict(body(json!(DE), "auto")).await.unwrap();
    let v = serde_json::to_value(&out).unwrap();
    let routing = &v["routing"];
    // required keys always present
    for k in [
        "checkpoint",
        "backend",
        "model_id",
        "reason",
        "token_count",
        "ane_eligible",
        "fallback",
        "latency_ms",
    ] {
        assert!(routing.get(k).is_some(), "missing routing key {k}");
    }
    assert_eq!(routing["checkpoint"], json!("multilingual"));
    assert_eq!(routing["backend"], json!("ane"));
    assert!(routing["latency_ms"].is_number());

    // route-only: latency null, no backend executed
    let d = r.router.route_only(body(json!(DE), "auto")).unwrap();
    assert_eq!(d.backend, BackendKind::Ane);
    assert_eq!(d.latency_ms, None);
    assert_eq!(r.ane.calls(), 1); // only the predict call
}

#[tokio::test(flavor = "current_thread")]
async fn event_emitted_per_request() {
    let sink = Arc::new(VecSink::default());
    let mlx = MockBackend::ok(BackendKind::Mlx, 0.9, (0.9, 0.1));
    let mut backends: HashMap<BackendKind, Arc<dyn PredictBackend>> = HashMap::new();
    backends.insert(BackendKind::Mlx, mlx);
    let r = Router::new(
        cfg_with_ane(),
        Arc::new(ScriptedEngine::new(20, 20)),
        backends,
        vec![BackendKind::Ane, BackendKind::Mlx],
        sink.clone(),
    );
    r.predict(body(json!(EN), "auto")).await.unwrap();
    let events = sink.events.lock().unwrap();
    assert_eq!(events.len(), 1);
    let ev = &events[0];
    assert_eq!(ev.status, 200);
    assert_eq!(ev.backend, Some(BackendKind::Mlx));
    // T023: English text that fits the ANE budget selects the
    // multilingual checkpoint; ANE absent → ane_unavailable explains MLX.
    assert_eq!(ev.checkpoint, Some(Checkpoint::Multilingual));
    assert_eq!(ev.reason, reason::ANE_UNAVAILABLE);
    assert_eq!(ev.token_count, Some(20));
    assert!(ev.degraded); // ane configured but absent
    let v = serde_json::to_value(ev).unwrap();
    assert!(v["ts"].is_string());
    assert_eq!(v["available_backends"], json!(["mlx"]));
}

#[tokio::test(flavor = "current_thread")]
async fn invalid_question_schema_is_400() {
    let r = rig(
        ScriptedEngine::new(20, 20),
        cfg_with_ane(),
        true,
        true,
        true,
    );
    let b: PredictBody = serde_json::from_value(json!({
        "state": "s",
        "questions": {"q": {"type": "bogus", "instructions": "x"}},
    }))
    .unwrap();
    let err = r.router.predict(b).await.unwrap_err();
    assert_eq!(err.status(), 400);
    assert_eq!(err.code(), "invalid_request");
}
