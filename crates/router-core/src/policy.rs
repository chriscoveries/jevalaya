//! The ordered decision table (DESIGN.md §Ordered routing): a pure
//! decision phase producing a [`RoutePlan`]; execution/escalation lives in
//! `router.rs`. No weight engine is required for policy.

use serde_json::{Map, Value};

use crate::checkpoint::{choose_checkpoint, CheckpointChoice};
use crate::config::PolicyConfig;
use crate::decision::{reason, RouteDecision};
use crate::engine::PromptEngine;
use crate::errors::RouteError;
use crate::schema::{BackendSpec, CompareSpec, PredictBody};
use crate::types::{BackendKind, Checkpoint, RenderedPrompt};
use jevalaya_render::{to_internal, GateCounts, Question};

/// A request after validation: raw `{state, questions}` plus the parsed
/// questions in map order and resolved routing fields.
#[derive(Debug)]
pub struct ValidatedRequest {
    pub state: Value,
    pub questions: Map<String, Value>,
    /// `to_internal`-validated questions in map (insertion) order.
    pub parsed: Vec<(String, Question)>,
    pub backend: BackendSpec,
    pub model: Option<String>,
    pub task: Option<String>,
    pub lang: Option<String>,
    pub request_id: String,
    /// `compare=true` — fan out to the configured comparison set.
    pub compare_configured: bool,
    /// Resolved explicit compare list (`compare=["ane","jev"]`).
    pub compare: Option<Vec<BackendKind>>,
}

static REQUEST_COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

impl ValidatedRequest {
    pub fn from_body(body: PredictBody) -> Result<Self, RouteError> {
        let mut parsed = Vec::with_capacity(body.questions.len());
        for (qid, qdef) in &body.questions {
            let q = to_internal(qdef)
                .map_err(|e| RouteError::InvalidRequest(format!("question {qid:?}: {e}")))?;
            parsed.push((qid.clone(), q));
        }
        let compare_configured = matches!(body.compare, Some(CompareSpec::Flag(true)));
        let compare = match &body.compare {
            None | Some(CompareSpec::Flag(_)) => None,
            Some(CompareSpec::List(names)) => {
                let mut kinds: Vec<BackendKind> = Vec::new();
                for n in names {
                    let k = BackendKind::from_name(n).ok_or_else(|| {
                        RouteError::InvalidRequest(format!("unknown backend {n:?} in compare"))
                    })?;
                    if !kinds.contains(&k) {
                        kinds.push(k);
                    }
                }
                Some(kinds)
            }
        };
        let request_id = body.request_id.clone().unwrap_or_else(|| {
            let n = REQUEST_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let millis = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis())
                .unwrap_or(0);
            format!("req-{millis:x}-{n:04x}")
        });
        Ok(ValidatedRequest {
            state: body.state,
            questions: body.questions,
            parsed,
            backend: body.backend,
            model: body.model,
            task: body.task,
            lang: body.lang,
            request_id,
            compare_configured,
            compare,
        })
    }
}

/// Backend availability snapshot — refreshed per request from
/// `capabilities()` so health recovery needs no restart.
#[derive(Debug, Clone, Copy, Default)]
pub struct Availability {
    pub ane: bool,
    pub mlx: bool,
    pub jev: bool,
}

impl Availability {
    pub fn get(&self, kind: BackendKind) -> bool {
        match kind {
            BackendKind::Ane => self.ane,
            BackendKind::Mlx => self.mlx,
            BackendKind::Jev => self.jev,
        }
    }

    pub fn list(&self) -> Vec<BackendKind> {
        BackendKind::ALL
            .iter()
            .copied()
            .filter(|k| self.get(*k))
            .collect()
    }

    /// Any *configured* backend currently unavailable → degraded service.
    pub fn degraded(&self, configured: &[BackendKind]) -> bool {
        configured.iter().any(|k| !self.get(*k))
    }
}

/// What the decision table concluded: the serializable decision plus the
/// execution inputs (selected checkpoint, pre-rendered ANE prompt).
#[derive(Debug)]
pub struct RoutePlan {
    pub decision: RouteDecision,
    /// Selected checkpoint (None only when no local routing happened).
    pub checkpoint: Option<Checkpoint>,
    /// Pre-rendered prompt for ANE dispatch; None for MLX (Python
    /// re-renders) and Jev (serializes state/questions).
    pub rendered: Option<RenderedPrompt>,
}

/// True on the only platform ANE runs on.
fn ane_platform_supported() -> bool {
    cfg!(all(target_os = "macos", target_arch = "aarch64"))
}

/// Ordered decision table. Pure: reads `req`, `engine`, `avail`, `cfg`;
/// never calls a backend.
///
/// Order (DESIGN.md §Ordered routing):
/// 1. backend override (jev direct; mlx hard-disables ANE; ane is intent)
/// 2. checkpoint router (model > task > workflow(opt-in) > lang > detect > default)
/// 3. render/token count via the model-family tokenizer
/// 4. ANE eligibility gates in order → MLX reason prefix on failure
/// 5. MLX default
/// Availability-aware: auto excludes down backends; explicit+down → 409;
/// nothing usable → 503.
pub fn route(
    req: &ValidatedRequest,
    engine: &dyn PromptEngine,
    avail: &Availability,
    cfg: &PolicyConfig,
    configured: &[BackendKind],
) -> Result<RoutePlan, RouteError> {
    let degraded = avail.degraded(configured);

    // ---- 1. Backend override --------------------------------------
    if req.backend == BackendSpec::Jev {
        if !avail.jev {
            return Err(RouteError::BackendUnavailable(BackendKind::Jev));
        }
        let mut d = RouteDecision::new(
            BackendKind::Jev,
            cfg.jev.model_id.clone(),
            format!("{}: backend=jev", reason::EXPLICIT_BACKEND),
        );
        d.degraded = degraded;
        return Ok(RoutePlan {
            decision: d,
            checkpoint: None,
            rendered: None,
        });
    }

    // ---- 2. Checkpoint router -------------------------------------
    let choice: CheckpointChoice = choose_checkpoint(
        req.model.as_deref(),
        req.task.as_deref(),
        req.lang.as_deref(),
        &req.state,
        &req.questions,
        cfg.auto_task_detection,
        cfg.default_checkpoint,
    )?;
    let cp = choice.checkpoint;
    let single = req.parsed.len() == 1;

    // ---- 3/4. ANE gates -------------------------------------------
    // Content/shape eligibility (excludes availability + preference, so
    // `ane_eligible` reports "would ANE serve this" even when it's down).
    let mut eligible = true;
    let mut first_fail: Option<&'static str> = None;
    let fail = |code: &'static str,
                affects_eligibility: bool,
                first_fail: &mut Option<&'static str>,
                eligible: &mut bool| {
        if first_fail.is_none() {
            *first_fail = Some(code);
        }
        if affects_eligibility {
            *eligible = false;
        }
    };

    // Content gates first: the reason prefix should name the request's own
    // failure (not_multilingual, question_count, ...) before service-side
    // gates like availability — a multi-question request on a down-ANE
    // host is explained by question_count, not ane_unavailable.
    if !(cfg.ane.enabled && cfg.ane.configured) {
        fail(reason::ANE_DISABLED, true, &mut first_fail, &mut eligible);
    }
    if !ane_platform_supported() {
        fail(
            reason::PLATFORM_UNSUPPORTED,
            true,
            &mut first_fail,
            &mut eligible,
        );
    }
    if cp == Checkpoint::TypedDecisions || choice.workflow.is_some() {
        fail(
            reason::TYPED_DECISIONS,
            true,
            &mut first_fail,
            &mut eligible,
        );
    } else if cp != Checkpoint::Multilingual {
        fail(
            reason::NOT_MULTILINGUAL,
            true,
            &mut first_fail,
            &mut eligible,
        );
    }
    if !single {
        fail(reason::QUESTION_COUNT, true, &mut first_fail, &mut eligible);
    }

    // Token-count gate runs only when the content gates so far hold —
    // counting is the expensive step and meaningless for e.g. English.
    // When ANE is up the gate is mandatory (fail closed on missing ANE
    // metadata); when ANE is down the count is informational only.
    let mut ane_gate_counts: Option<GateCounts> = None;
    if eligible {
        let q = &req.parsed[0].1;
        match engine.ane_counts(&req.state, q) {
            Ok(counts) => {
                ane_gate_counts = Some(counts);
                if counts.aligned_count > cfg.ane.max_tokens {
                    fail(
                        reason::TOKEN_COUNT_OVER_LIMIT,
                        true,
                        &mut first_fail,
                        &mut eligible,
                    );
                }
            }
            Err(e @ RouteError::NotReady(_)) if !avail.ane => {
                let _ = e; // ANE is down anyway; the count is moot.
            }
            Err(e) => return Err(e),
        }
    }

    // Service gates last — they never touch `eligible`, only the reason
    // and the dispatch choice.
    if !avail.ane {
        fail(
            reason::ANE_UNAVAILABLE,
            false,
            &mut first_fail,
            &mut eligible,
        );
    }
    let prefer_ok = cfg.ane.prefer_ane_for_short || req.backend == BackendSpec::Ane;
    if !prefer_ok {
        fail(
            reason::ANE_NOT_PREFERRED,
            false,
            &mut first_fail,
            &mut eligible,
        );
    }

    // ---- explicit-backend availability -----------------------------
    if req.backend == BackendSpec::Ane && !avail.ane {
        return Err(RouteError::BackendUnavailable(BackendKind::Ane));
    }
    if req.backend == BackendSpec::Mlx && !avail.mlx {
        return Err(RouteError::BackendUnavailable(BackendKind::Mlx));
    }

    // ---- 5. Dispatch choice ----------------------------------------
    let want_ane = eligible
        && avail.ane
        && match req.backend {
            BackendSpec::Ane => true,
            BackendSpec::Auto => cfg.ane.prefer_ane_for_short,
            BackendSpec::Mlx => false,
            BackendSpec::Jev => unreachable!(),
        };

    if want_ane {
        let q = &req.parsed[0].1;
        let rendered = engine.ane_render(&req.state, q)?;
        let mut d = RouteDecision::new(
            BackendKind::Ane,
            cfg.ane.model_id.clone(),
            format!(
                "{}: {}",
                reason::ANE_SHORT_PATH,
                "single multilingual prompt fits the ANE token budget"
            ),
        );
        d.checkpoint = Some(cp);
        d.token_count = ane_gate_counts.map(|c| c.raw_count as u64);
        d.ane_eligible = true;
        d.degraded = degraded;
        return Ok(RoutePlan {
            decision: d,
            checkpoint: Some(cp),
            rendered: Some(rendered),
        });
    }

    if avail.mlx {
        let prefix = match req.backend {
            BackendSpec::Mlx => reason::EXPLICIT_BACKEND,
            _ => first_fail.unwrap_or(reason::ANE_UNAVAILABLE),
        };
        let detail = match prefix {
            p if p == reason::EXPLICIT_BACKEND => "backend=mlx".to_string(),
            _ => choice.reason.clone(),
        };
        // token_count: the gate count when ANE was evaluated (ANE model
        // tokenizer), else the executing checkpoint's tokenizer. Multiple
        // questions report the maximum per-question raw count. A missing
        // checkpoint tokenizer (NotReady) degrades the report to null —
        // DESIGN.md: "or null if rendering failed before a valid count" —
        // while request-shape errors still fail the request.
        let mut token_count = ane_gate_counts.map(|c| c.raw_count as u64);
        let mut exec_count: Option<u64> = None;
        if token_count.is_none() {
            let mut max_raw = 0u64;
            let mut counted = false;
            for (_, q) in &req.parsed {
                match engine.counts(cp, &req.state, q) {
                    Ok(c) => {
                        max_raw = max_raw.max(c.raw_count as u64);
                        counted = true;
                    }
                    Err(RouteError::NotReady(_)) => {}
                    Err(e) => return Err(e),
                }
            }
            token_count = counted.then_some(max_raw);
        }
        for (_, q) in &req.parsed {
            match engine.execution_len(cp, &req.state, q) {
                Ok(n) => {
                    exec_count = Some(exec_count.map_or(n as u64, |m: u64| m.max(n as u64)));
                }
                Err(RouteError::NotReady(_)) => {}
                Err(e) => return Err(e),
            }
        }
        let mut d = RouteDecision::new(
            BackendKind::Mlx,
            cfg.model_id(cp),
            format!("{prefix}: {detail}"),
        );
        d.checkpoint = Some(cp);
        d.token_count = token_count;
        d.execution_token_count = exec_count;
        d.ane_eligible = eligible;
        d.degraded = degraded;
        return Ok(RoutePlan {
            decision: d,
            checkpoint: Some(cp),
            rendered: None,
        });
    }

    // No MLX. Auto may pass through to Jev; explicit local requests fail.
    match req.backend {
        BackendSpec::Auto if avail.jev => {
            let mut d = RouteDecision::new(
                BackendKind::Jev,
                cfg.jev.model_id.clone(),
                format!(
                    "{}: no usable local backend (ane={}, mlx=false)",
                    reason::NO_LOCAL_BACKEND,
                    avail.ane
                ),
            );
            d.degraded = true;
            Ok(RoutePlan {
                decision: d,
                checkpoint: None,
                rendered: None,
            })
        }
        BackendSpec::Auto => Err(RouteError::NotReady(
            "no usable backend for automatic request".to_string(),
        )),
        BackendSpec::Ane => Err(RouteError::NotReady(
            "request is not ANE-eligible and mlx is unavailable".to_string(),
        )),
        BackendSpec::Mlx => unreachable!(),
        BackendSpec::Jev => unreachable!(),
    }
}
