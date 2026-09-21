//! The `Router`: wraps the pure decision table (`policy::route`) with
//! execution — dispatch, the one permitted ANE→MLX retry, Jev escalation
//! on confidence/margin/retry triggers, compare fan-out, and per-request
//! observability events.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::Instant;

use futures::future::join_all;
use serde_json::{json, Map, Value};

use crate::config::{JevPolicy, PolicyConfig};
use crate::decision::{reason, trigger, RouteDecision};
use crate::engine::PromptEngine;
use crate::errors::RouteError;
use crate::events::{rfc3339_now, EventSink, NullSink, RoutingEvent};
use crate::policy::{route, Availability, RoutePlan, ValidatedRequest};
use crate::schema::{BackendSpec, CompareEntry, PredictBody, PredictResponse};
use crate::types::{
    BackendError, BackendKind, BackendResult, Checkpoint, PredictBackend, PredictRequest,
    PreloadRequest, RenderedPrompt, UnloadTarget,
};

/// Per-question confidence/margin extraction (DESIGN.md §Wire contract):
/// request-level values are the MIN across answers so one ambiguous answer
/// is never hidden behind a confident aggregate.
pub fn confidence_margin(
    questions: &Map<String, Value>,
    answers: &Value,
) -> (Option<f64>, Option<f64>) {
    let mut conf_min: Option<f64> = None;
    let mut margin_min: Option<f64> = None;
    for qid in questions.keys() {
        let Some(ans) = answers.get(qid) else {
            continue;
        };
        if let Some(c) = ans.get("confidence").and_then(Value::as_f64) {
            conf_min = Some(conf_min.map_or(c, |m: f64| m.min(c)));
        }
        if let Some(m) = answer_margin(ans) {
            margin_min = Some(margin_min.map_or(m, |x: f64| x.min(m)));
        }
    }
    (conf_min, margin_min)
}

fn answer_margin(ans: &Value) -> Option<f64> {
    if let Some(probs) = ans.get("probabilities").and_then(Value::as_object) {
        let mut vals: Vec<f64> = probs.values().filter_map(Value::as_f64).collect();
        vals.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        return match vals.len() {
            0 => None,
            1 => Some(vals[0]),
            _ => Some(vals[0] - vals[1]),
        };
    }
    // noul answers carry `noul` = p(true); the binary margin is |p - (1-p)|.
    ans.get("noul")
        .and_then(Value::as_f64)
        .map(|p| (2.0 * p - 1.0).abs())
}

/// Min of a numeric per-answer field (e.g. `target_confidence`).
fn min_answer_field(questions: &Map<String, Value>, answers: &Value, field: &str) -> Option<f64> {
    let mut out: Option<f64> = None;
    for qid in questions.keys() {
        if let Some(v) = answers
            .get(qid)
            .and_then(|a| a.get(field))
            .and_then(Value::as_f64)
        {
            out = Some(out.map_or(v, |m: f64| m.min(v)));
        }
    }
    out
}

/// Escalation triggers in stable priority order (margin is primary).
fn jev_trigger_for(
    conf: Option<f64>,
    margin: Option<f64>,
    target_conf: Option<f64>,
    retried: bool,
    jev: &JevPolicy,
) -> Option<&'static str> {
    if let (Some(m), Some(t)) = (margin, jev.margin_threshold) {
        if m < t {
            return Some(trigger::LOW_MARGIN);
        }
    }
    if let (Some(c), Some(t)) = (conf, jev.confidence_threshold) {
        if c < t {
            return Some(trigger::LOW_CONFIDENCE);
        }
    }
    if let (Some(tc), Some(t)) = (target_conf, jev.target_confidence_threshold) {
        if tc < t {
            return Some(trigger::LOW_TARGET_CONFIDENCE);
        }
    }
    if retried && jev.escalate_on_retry {
        return Some(trigger::RETRY);
    }
    None
}

fn cost_of(usage: &Value) -> Value {
    json!({
        "input_tokens": usage.get("input_tokens").cloned().unwrap_or(Value::Null),
        "output_tokens": usage.get("output_tokens").cloned().unwrap_or(Value::Null),
        "remote_usd": Value::Null,
    })
}

/// Dispatch through the backend trait; blocking adapters run on the
/// blocking pool so they never stall the async executor.
async fn dispatch(
    backend: &Arc<dyn PredictBackend>,
    req: &PredictRequest,
    rendered: Option<RenderedPrompt>,
) -> Result<BackendResult, BackendError> {
    if backend.is_blocking() {
        let b = backend.clone();
        let r = req.clone();
        return tokio::task::spawn_blocking(move || {
            futures::executor::block_on(b.predict(&r, rendered.as_ref()))
        })
        .await
        .map_err(|e| BackendError::Inference(format!("dispatch join: {e}")))?;
    }
    backend.predict(req, rendered.as_ref()).await
}

/// The routing engine: a [`PolicyConfig`] snapshot, a [`PromptEngine`],
/// the configured backend adapters, and an event sink.
pub struct Router {
    cfg: RwLock<PolicyConfig>,
    engine: Arc<dyn PromptEngine>,
    backends: HashMap<BackendKind, Arc<dyn PredictBackend>>,
    /// Backends present in config (whether or not currently healthy) —
    /// the denominator for `degraded`.
    configured: Vec<BackendKind>,
    sink: Arc<dyn EventSink>,
}

impl Router {
    pub fn new(
        config: PolicyConfig,
        engine: Arc<dyn PromptEngine>,
        backends: HashMap<BackendKind, Arc<dyn PredictBackend>>,
        configured: Vec<BackendKind>,
        sink: Arc<dyn EventSink>,
    ) -> Self {
        Self {
            cfg: RwLock::new(config),
            engine,
            backends,
            configured,
            sink,
        }
    }

    /// Minimal constructor for tests: no event emission.
    pub fn for_test(
        config: PolicyConfig,
        engine: Arc<dyn PromptEngine>,
        backends: HashMap<BackendKind, Arc<dyn PredictBackend>>,
        configured: Vec<BackendKind>,
    ) -> Self {
        Self::new(config, engine, backends, configured, Arc::new(NullSink))
    }

    /// Atomic config swap — a request sees exactly one snapshot.
    pub fn update_config(&self, cfg: PolicyConfig) {
        *self.cfg.write().unwrap() = cfg;
    }

    pub fn config(&self) -> PolicyConfig {
        self.cfg.read().unwrap().clone()
    }

    /// Live availability from adapter capabilities — health recovery is
    /// automatic because this is re-read per request.
    pub fn availability(&self) -> Availability {
        let get = |k: BackendKind| {
            self.backends
                .get(&k)
                .map(|b| b.capabilities().available)
                .unwrap_or(false)
        };
        Availability {
            ane: get(BackendKind::Ane),
            mlx: get(BackendKind::Mlx),
            jev: get(BackendKind::Jev),
        }
    }

    pub fn backend(&self, kind: BackendKind) -> Option<Arc<dyn PredictBackend>> {
        self.backends.get(&kind).cloned()
    }

    /// True when any configured backend is currently unavailable —
    /// `degraded` on decisions/events comes from the same source.
    pub fn degraded(&self) -> bool {
        self.availability().degraded(&self.configured)
    }

    /// The `/route` surface: the decision without running inference.
    /// `latency_ms` stays null.
    pub fn route_only(&self, body: PredictBody) -> Result<RouteDecision, RouteError> {
        let cfg = self.config();
        let avail = self.availability();
        let req = ValidatedRequest::from_body(body)?;
        let plan = route(&req, &*self.engine, &avail, &cfg, &self.configured)?;
        Ok(plan.decision)
    }

    /// Full predict path: validate → decide → dispatch (+ANE retry) →
    /// escalation → compare → event. Returns the wire response.
    pub async fn predict(&self, body: PredictBody) -> Result<PredictResponse, RouteError> {
        let cfg = self.config();
        let avail = self.availability();
        let t0 = Instant::now();
        let fallback_id = body.request_id.clone();
        let req = match ValidatedRequest::from_body(body) {
            Ok(r) => r,
            Err(e) => {
                self.emit(
                    fallback_id.as_deref().unwrap_or("req"),
                    &avail,
                    e.status(),
                    None,
                    Some(e.code()),
                    t0,
                );
                return Err(e);
            }
        };
        let out = self.predict_validated(&req, &cfg, &avail, t0).await;
        let (status, decision, code) = match &out {
            Ok(r) => (200, Some(&r.routing), None),
            Err(e) => (e.status(), None, Some(e.code())),
        };
        self.emit(&req.request_id, &avail, status, decision, code, t0);
        out
    }

    async fn predict_validated(
        &self,
        req: &ValidatedRequest,
        cfg: &PolicyConfig,
        avail: &Availability,
        t0: Instant,
    ) -> Result<PredictResponse, RouteError> {
        let mut plan = route(req, &*self.engine, avail, cfg, &self.configured)?;
        let preq = PredictRequest {
            state: req.state.clone(),
            questions: req.questions.clone(),
            checkpoint: plan.checkpoint.unwrap_or(cfg.default_checkpoint),
            request_id: req.request_id.clone(),
        };

        // ---- primary dispatch (+ the one ANE→MLX retry) -------------
        let mut result = self.execute_primary(&mut plan, &preq, cfg, avail).await?;

        // ---- confidence/margin on the (possibly retried) answer ------
        let (conf, margin) = confidence_margin(&preq.questions, &result.answers);
        plan.decision.confidence = conf;
        plan.decision.margin = margin;
        plan.decision.cost = Some(cost_of(&result.usage));

        // ---- Jev escalation (auto mode, local answer only) ----------
        let target_conf = min_answer_field(&preq.questions, &result.answers, "target_confidence");
        if req.backend == BackendSpec::Auto
            && matches!(plan.decision.backend, BackendKind::Ane | BackendKind::Mlx)
            && self.configured.contains(&BackendKind::Jev)
        {
            if let Some(trig) =
                jev_trigger_for(conf, margin, target_conf, plan.decision.fallback, &cfg.jev)
            {
                plan.decision.escalated = true;
                plan.decision.jev_trigger = Some(trig.to_string());
                if avail.jev {
                    let jev = self.backend(BackendKind::Jev).unwrap();
                    match dispatch(&jev, &preq, None).await {
                        Ok(jev_result) => {
                            plan.decision.backend = BackendKind::Jev;
                            plan.decision.model_id = cfg.jev.model_id.clone();
                            plan.decision.reason =
                                format!("{}: escalated to jev", plan.decision.reason);
                            let (c, m) = confidence_margin(&preq.questions, &jev_result.answers);
                            plan.decision.confidence = c;
                            plan.decision.margin = m;
                            plan.decision.cost = Some(cost_of(&jev_result.usage));
                            result = jev_result;
                        }
                        Err(e) => {
                            // Retain the validated local answer.
                            plan.decision.escalation_error = Some(e.stable_code().to_string());
                            plan.decision.degraded = true;
                        }
                    }
                } else {
                    plan.decision.escalation_error = Some("backend_unavailable".to_string());
                    plan.decision.degraded = true;
                }
            }
        }

        // ---- compare fan-out ----------------------------------------
        let compare = self
            .run_compare(req, &preq, &plan, &result, cfg, avail)
            .await;

        plan.decision.latency_ms = Some(t0.elapsed().as_secs_f64() * 1000.0);

        Ok(PredictResponse {
            model: result.model,
            answers: result.answers,
            usage: result.usage,
            routing: plan.decision,
            compare,
        })
    }

    /// Primary dispatch with the one permitted ANE→MLX retry on
    /// capacity/shape failures (DESIGN.md §Ordered routing step 6). Hard
    /// ANE errors never retry; ANE is never invoked twice per request.
    async fn execute_primary(
        &self,
        plan: &mut RoutePlan,
        preq: &PredictRequest,
        cfg: &PolicyConfig,
        avail: &Availability,
    ) -> Result<BackendResult, RouteError> {
        let kind = plan.decision.backend;
        let backend = self
            .backend(kind)
            .ok_or_else(|| RouteError::NotReady(format!("{kind} backend missing")))?;
        match kind {
            BackendKind::Ane => match dispatch(&backend, preq, plan.rendered.clone()).await {
                Ok(r) => Ok(r),
                Err(e @ (BackendError::Capacity(_) | BackendError::Shape(_))) => {
                    let mlx = self.backend(BackendKind::Mlx).filter(|_| avail.mlx);
                    match mlx {
                        Some(mlx) => {
                            let r = dispatch(&mlx, preq, None)
                                .await
                                .map_err(RouteError::from_backend)?;
                            plan.decision.backend = BackendKind::Mlx;
                            plan.decision.model_id =
                                cfg.model_id(plan.checkpoint.unwrap_or(Checkpoint::Multilingual));
                            plan.decision.fallback = true;
                            plan.decision.fallback_from = Some(BackendKind::Ane);
                            plan.decision.fallback_reason = Some(e.stable_code().to_string());
                            plan.decision.reason = format!(
                                "{}: ane dispatch failed, retried once on mlx",
                                plan.decision.reason
                            );
                            Ok(r)
                        }
                        None => Err(RouteError::Inference(format!(
                            "ane {} and no mlx for the one permitted retry",
                            e.stable_code()
                        ))),
                    }
                }
                Err(e) => Err(RouteError::from_backend(e)),
            },
            BackendKind::Mlx | BackendKind::Jev => dispatch(&backend, preq, None)
                .await
                .map_err(RouteError::from_backend),
        }
    }

    /// Compare mode: `compare=true` (configured set) or an explicit list.
    /// Compare-ness is decided on the requested set BEFORE health
    /// filtering; a ≤1-backend set is ordinary routing. The primary
    /// result is reused for its own entry — never double-charged.
    async fn run_compare(
        &self,
        req: &ValidatedRequest,
        preq: &PredictRequest,
        plan: &RoutePlan,
        primary: &BackendResult,
        cfg: &PolicyConfig,
        avail: &Availability,
    ) -> Map<String, Value> {
        let set: Vec<BackendKind> = if req.compare_configured {
            cfg.compare_backends.clone()
        } else {
            req.compare.clone().unwrap_or_default()
        };
        // Dedup preserving order; compare requires ≥2 distinct pre-filter.
        let mut set_dedup: Vec<BackendKind> = Vec::new();
        for k in &set {
            if !set_dedup.contains(k) {
                set_dedup.push(*k);
            }
        }
        if set_dedup.len() < 2 {
            return Map::new();
        }

        let primary_kind = plan.decision.backend;
        let futures = set_dedup.iter().map(|kind| {
            self.compare_entry(*kind, req, preq, plan, primary, primary_kind, cfg, avail)
        });
        let entries = join_all(futures).await;
        let mut out = Map::new();
        for (kind, entry) in set_dedup.iter().zip(entries) {
            out.insert(
                kind.as_str().to_string(),
                serde_json::to_value(entry).unwrap_or(Value::Null),
            );
        }
        out
    }

    async fn compare_entry(
        &self,
        kind: BackendKind,
        req: &ValidatedRequest,
        preq: &PredictRequest,
        plan: &RoutePlan,
        primary: &BackendResult,
        primary_kind: BackendKind,
        cfg: &PolicyConfig,
        avail: &Availability,
    ) -> CompareEntry {
        let mut routing = RouteDecision::new(
            kind,
            match kind {
                BackendKind::Ane => cfg.ane.model_id.clone(),
                BackendKind::Mlx => cfg.model_id(plan.checkpoint.unwrap_or(cfg.default_checkpoint)),
                BackendKind::Jev => cfg.jev.model_id.clone(),
            },
            format!("{}: fan-out", reason::COMPARE),
        );
        routing.checkpoint = plan.checkpoint;
        routing.ane_eligible = plan.decision.ane_eligible;

        // Reuse the primary result for its own backend.
        if kind == primary_kind {
            routing.confidence = plan.decision.confidence;
            routing.margin = plan.decision.margin;
            routing.cost = plan.decision.cost.clone();
            routing.latency_ms = Some(primary.latency_ms as f64);
            return CompareEntry {
                model: Some(primary.model.clone()),
                answers: Some(primary.answers.clone()),
                usage: Some(primary.usage.clone()),
                routing,
                error: None,
            };
        }

        if !avail.get(kind) {
            return CompareEntry {
                model: None,
                answers: None,
                usage: None,
                routing,
                error: Some(json!({
                    "code": "backend_unavailable",
                    "message": format!("{kind} is not available"),
                })),
            };
        }

        // ANE never bypasses the eligibility gates, even in compare.
        let rendered = match kind {
            BackendKind::Ane => {
                if !plan.decision.ane_eligible {
                    return CompareEntry {
                        model: None,
                        answers: None,
                        usage: None,
                        routing,
                        error: Some(json!({
                            "code": "ane_ineligible",
                            "message": "request does not satisfy the ANE gates",
                        })),
                    };
                }
                match plan.rendered.clone() {
                    Some(r) => Some(r),
                    None => match self.engine.ane_render(&req.state, &req.parsed[0].1) {
                        Ok(r) => Some(r),
                        Err(e) => {
                            return CompareEntry {
                                model: None,
                                answers: None,
                                usage: None,
                                routing,
                                error: Some(json!({
                                    "code": e.code(),
                                    "message": e.to_string(),
                                })),
                            }
                        }
                    },
                }
            }
            _ => None,
        };

        let backend = match self.backend(kind) {
            Some(b) => b,
            None => {
                return CompareEntry {
                    model: None,
                    answers: None,
                    usage: None,
                    routing,
                    error: Some(json!({
                        "code": "backend_unavailable",
                        "message": format!("{kind} is not configured"),
                    })),
                }
            }
        };
        let t0 = Instant::now();
        match dispatch(&backend, preq, rendered).await {
            Ok(r) => {
                routing.latency_ms = Some(t0.elapsed().as_secs_f64() * 1000.0);
                let (c, m) = confidence_margin(&preq.questions, &r.answers);
                routing.confidence = c;
                routing.margin = m;
                routing.cost = Some(cost_of(&r.usage));
                CompareEntry {
                    model: Some(r.model),
                    answers: Some(r.answers),
                    usage: Some(r.usage),
                    routing,
                    error: None,
                }
            }
            Err(e) => {
                routing.latency_ms = Some(t0.elapsed().as_secs_f64() * 1000.0);
                CompareEntry {
                    model: None,
                    answers: None,
                    usage: None,
                    routing,
                    error: Some(json!({
                        "code": e.stable_code(),
                        "message": e.to_string(),
                    })),
                }
            }
        }
    }

    /// Forward preload to all configured backends.
    pub async fn preload(
        &self,
        request: PreloadRequest,
    ) -> Vec<(BackendKind, Result<(), BackendError>)> {
        let mut out = Vec::new();
        for (k, b) in &self.backends {
            out.push((*k, b.preload(request.clone()).await));
        }
        out
    }

    /// Forward unload to all configured backends.
    pub async fn unload(
        &self,
        target: UnloadTarget,
    ) -> Vec<(BackendKind, Result<(), BackendError>)> {
        let mut out = Vec::new();
        for (k, b) in &self.backends {
            out.push((*k, b.unload(target.clone()).await));
        }
        out
    }

    fn emit(
        &self,
        request_id: &str,
        avail: &Availability,
        status: u16,
        decision: Option<&RouteDecision>,
        err_code: Option<&'static str>,
        t0: Instant,
    ) {
        let latency = t0.elapsed().as_secs_f64() * 1000.0;
        let ev = match decision {
            Some(d) => RoutingEvent {
                ts: rfc3339_now(),
                request_id: request_id.to_string(),
                backend: Some(d.backend),
                checkpoint: d.checkpoint,
                reason: d.reason_prefix().to_string(),
                token_count: d.token_count,
                confidence: d.confidence,
                margin: d.margin,
                latency_ms: Some(latency),
                queue_ms: None,
                fallback: d.fallback,
                escalated: d.escalated,
                degraded: d.degraded,
                cost: d.cost.clone(),
                available_backends: avail.list(),
                status,
            },
            None => RoutingEvent {
                ts: rfc3339_now(),
                request_id: request_id.to_string(),
                backend: None,
                checkpoint: None,
                reason: err_code.unwrap_or("error").to_string(),
                token_count: None,
                confidence: None,
                margin: None,
                latency_ms: Some(latency),
                queue_ms: None,
                fallback: false,
                escalated: false,
                degraded: avail.degraded(&self.configured),
                cost: None,
                available_backends: avail.list(),
                status,
            },
        };
        self.sink.emit(ev);
    }
}
