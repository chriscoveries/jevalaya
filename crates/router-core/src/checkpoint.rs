//! Checkpoint router: which Laya checkpoint a request selects. Exact port
//! of `Router.route` precedence (laya-mlx `router.py`):
//!
//! ```text
//! explicit `model` > explicit `task` > detected workflow (opt-in) >
//! explicit `lang` > detected script/language > configured default
//! ```

use serde_json::{Map, Value};

use crate::errors::RouteError;
use crate::lang::{analyse, Detection};
use crate::types::Checkpoint;

/// Question-id signatures of the four typed-decisions workflows
/// (`router.py::_TYPED_DECISION_WORKFLOWS`). Exact id-set match only.
const TYPED_DECISION_WORKFLOWS: &[(&str, &[&str])] = &[
    (
        "agent_trace_observability",
        &["action", "needs_review", "outcome", "risk", "urgency"],
    ),
    (
        "customer_service",
        &["action", "category", "churn_risk", "needs_human", "urgency"],
    ),
    (
        "invoice_processing",
        &[
            "discrepancy_severity",
            "disposition",
            "duplicate",
            "matches_order",
            "urgency",
        ],
    ),
    (
        "security_incidents",
        &[
            "credential_compromise",
            "disposition",
            "severity",
            "true_positive",
            "urgency",
        ],
    ),
];

/// Name of the typed-decisions workflow whose question ids these are,
/// else None. Requires an exact id-set match, so an unrelated schema that
/// happens to contain `urgency` is never captured.
pub fn match_typed_decisions_workflow(questions: &Map<String, Value>) -> Option<&'static str> {
    for (wf, sig) in TYPED_DECISION_WORKFLOWS {
        if questions.len() == sig.len() && sig.iter().all(|id| questions.contains_key(*id)) {
            return Some(wf);
        }
    }
    None
}

/// Outcome of the checkpoint router.
#[derive(Debug, Clone)]
pub struct CheckpointChoice {
    pub checkpoint: Checkpoint,
    /// Human detail for the RouteDecision reason (mirrors Python phrasing).
    pub reason: String,
    /// Matched typed-decisions workflow id, if the question ids hit a
    /// signature — present even when `auto_task_detection` is off, because
    /// a typed marker still blocks ANE.
    pub workflow: Option<&'static str>,
    /// Script/language detection detail when the detection branch ran.
    pub detection: Option<Detection>,
}

/// Source precedence, exactly `Router.route`:
/// `model` > `task` > detected workflow (opt-in) > `lang` > detected
/// script/language > default.
pub fn choose_checkpoint(
    model: Option<&str>,
    task: Option<&str>,
    lang: Option<&str>,
    state: &Value,
    questions: &Map<String, Value>,
    auto_task_detection: bool,
    default: Checkpoint,
) -> Result<CheckpointChoice, RouteError> {
    let workflow = match_typed_decisions_workflow(questions);

    if let Some(model) = model {
        if let Some(cp) = Checkpoint::from_name(model) {
            return Ok(CheckpointChoice {
                checkpoint: cp,
                reason: format!("explicit model={model:?}"),
                workflow,
                detection: None,
            });
        }
        // `jev-*` ids name the remote TypeSafe model, not local weights —
        // drop-in Jev clients send them unconditionally, so don't pin.
        if !model.to_lowercase().starts_with("jev") {
            return Err(RouteError::InvalidRequest(format!("unknown model {model:?}")));
        }
    }

    if let Some(task) = task {
        let key = if task.to_lowercase().replace('-', "_") == "typed_decisions" {
            Checkpoint::TypedDecisions
        } else {
            Checkpoint::from_name(task)
                .ok_or_else(|| RouteError::InvalidRequest(format!("unknown task {task:?}")))?
        };
        return Ok(CheckpointChoice {
            checkpoint: key,
            reason: format!("explicit task={task:?}"),
            workflow,
            detection: None,
        });
    }

    if let (Some(wf), true) = (workflow, auto_task_detection) {
        return Ok(CheckpointChoice {
            checkpoint: Checkpoint::TypedDecisions,
            reason: format!("question ids match the {wf:?} typed-decisions workflow"),
            workflow,
            detection: None,
        });
    }

    if let Some(lang) = lang {
        let en = matches!(
            lang.to_lowercase().split('-').next().unwrap_or(""),
            "en" | "eng" | "english"
        );
        return Ok(CheckpointChoice {
            checkpoint: if en {
                Checkpoint::English
            } else {
                Checkpoint::Multilingual
            },
            reason: format!("explicit lang={lang:?}"),
            workflow,
            detection: None,
        });
    }

    let det = analyse(state);
    let (cp, reason) = if det.script == "unknown" {
        (
            default,
            format!(
                "no letters detected in state; using default ({})",
                default.as_str()
            ),
        )
    } else if det.script != "latin" {
        (
            Checkpoint::Multilingual,
            format!(
                "non-Latin script ({}, {:.0}% of letters); the English checkpoint cannot read it",
                det.script,
                100.0 * det.non_latin_fraction
            ),
        )
    } else if !det.is_english {
        (
            Checkpoint::Multilingual,
            format!(
                "Latin script but language looks like {:?}, not English",
                det.language
            ),
        )
    } else {
        (Checkpoint::English, "English Latin text".to_string())
    };
    Ok(CheckpointChoice {
        checkpoint: cp,
        reason,
        workflow,
        detection: Some(det),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn qs(ids: &[&str]) -> Map<String, Value> {
        ids.iter()
            .map(|i| (i.to_string(), json!({"type":"noul","instructions":"x"})))
            .collect()
    }

    #[test]
    fn precedence_model_over_everything() {
        let c = choose_checkpoint(
            Some("multilingual"),
            Some("typed_decisions"),
            Some("de"),
            &json!("hello"),
            &qs(&["a"]),
            true,
            Checkpoint::English,
        )
        .unwrap();
        assert_eq!(c.checkpoint, Checkpoint::Multilingual);
        assert!(c.reason.contains("explicit model"));
    }

    #[test]
    fn task_typed_decisions_alias() {
        for t in ["typed_decisions", "typed-decisions", "Typed_Decisions"] {
            let c = choose_checkpoint(
                None,
                Some(t),
                None,
                &json!("hi"),
                &qs(&["a"]),
                false,
                Checkpoint::English,
            )
            .unwrap();
            assert_eq!(c.checkpoint, Checkpoint::TypedDecisions, "task {t}");
        }
    }

    #[test]
    fn workflow_needs_exact_id_set_and_opt_in() {
        let wf_qs = qs(&["action", "category", "churn_risk", "needs_human", "urgency"]);
        assert_eq!(
            match_typed_decisions_workflow(&wf_qs),
            Some("customer_service")
        );
        // extra id breaks the match
        let mut extra = wf_qs.clone();
        extra.insert(
            "other".to_string(),
            json!({"type":"noul","instructions":"x"}),
        );
        assert_eq!(match_typed_decisions_workflow(&extra), None);
        // detection only selects when opted in, but the marker is reported either way
        let off = choose_checkpoint(
            None,
            None,
            None,
            &json!("English text here and more words to see"),
            &wf_qs,
            false,
            Checkpoint::English,
        )
        .unwrap();
        assert_eq!(off.checkpoint, Checkpoint::English);
        assert_eq!(off.workflow, Some("customer_service"));
        let on = choose_checkpoint(
            None,
            None,
            None,
            &json!("English text here and more words to see"),
            &wf_qs,
            true,
            Checkpoint::English,
        )
        .unwrap();
        assert_eq!(on.checkpoint, Checkpoint::TypedDecisions);
    }

    #[test]
    fn lang_overrides_detection() {
        let c = choose_checkpoint(
            None,
            None,
            Some("de"),
            &json!("this is plainly English text with the and of"),
            &qs(&["a"]),
            false,
            Checkpoint::English,
        )
        .unwrap();
        assert_eq!(c.checkpoint, Checkpoint::Multilingual);
        let c = choose_checkpoint(
            None,
            None,
            Some("en-US"),
            &json!("我想退款"),
            &qs(&["a"]),
            false,
            Checkpoint::English,
        )
        .unwrap();
        assert_eq!(c.checkpoint, Checkpoint::English);
    }

    #[test]
    fn unknown_names_are_invalid_request() {
        assert!(matches!(
            choose_checkpoint(
                Some("nope"),
                None,
                None,
                &json!("x"),
                &qs(&["a"]),
                false,
                Checkpoint::English
            ),
            Err(RouteError::InvalidRequest(_))
        ));
        assert!(matches!(
            choose_checkpoint(
                None,
                Some("nope"),
                None,
                &json!("x"),
                &qs(&["a"]),
                false,
                Checkpoint::English
            ),
            Err(RouteError::InvalidRequest(_))
        ));
    }
}
