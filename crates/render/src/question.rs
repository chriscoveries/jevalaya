//! User-facing question definitions → validated internal questions.
//!
//! Exact port of `Agent._to_internal` (agent.py) plus `render_criterion` and
//! `render_options` (common.py). Validation messages mirror the Python ones.

use serde_json::{Map, Value};
use thiserror::Error;

use crate::pyjson::{py_dumps, serialize_state};

#[derive(Debug, Error, PartialEq)]
pub enum QuestionError {
    #[error("Each question must be a dictionary")]
    NotADict,
    #[error("Unknown question type {0:?}; expected choice, score, or noul")]
    UnknownType(String),
    #[error("Question is missing instructions")]
    MissingInstructions,
    #[error("Choice labels must be strings")]
    ChoiceLabelNotString,
    #[error("Choice labels must be unique")]
    ChoiceLabelNotUnique,
    #[error("Choice criteria must be a nonempty dictionary or list")]
    ChoiceCriteriaEmpty,
    #[error("Score criteria must be a nonempty list")]
    ScoreCriteriaInvalid,
    #[error("Noul criteria must be a dictionary with false/true descriptions")]
    NoulCriteriaInvalid,
}

/// Numeric question type ids, matching `QTYPES` (`choice=0, score=1, noul=2`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum QType {
    Choice = 0,
    Score = 1,
    Noul = 2,
}

impl QType {
    pub fn name(self) -> &'static str {
        match self {
            QType::Choice => "choice",
            QType::Score => "score",
            QType::Noul => "noul",
        }
    }

    pub fn id(self) -> u8 {
        self as u8
    }

    pub fn from_name(name: &str) -> Option<QType> {
        match name {
            "choice" => Some(QType::Choice),
            "score" => Some(QType::Score),
            "noul" => Some(QType::Noul),
            _ => None,
        }
    }
}

/// A validated question: type, rendered instruction string, rendered options.
///
/// `options` are the exact strings `build_prefix` encodes (one `[MASK]` each);
/// `choice_labels` preserves label order for answer mapping (insertion order,
/// matching Python dict order — never sorted).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Question {
    pub qtype: QType,
    pub instructions: String,
    pub options: Vec<String>,
    pub choice_labels: Vec<String>,
}

/// Parse one user-facing question definition (`{"type", "instructions",
/// "criteria"}`) into a validated [`Question`].
pub fn to_internal(qdef: &Value) -> Result<Question, QuestionError> {
    let map = qdef.as_object().ok_or(QuestionError::NotADict)?;
    let kind = map.get("type").and_then(Value::as_str).unwrap_or_default();
    let qtype =
        QType::from_name(kind).ok_or_else(|| QuestionError::UnknownType(kind.to_string()))?;
    let ins_value = map
        .get("instructions")
        .ok_or(QuestionError::MissingInstructions)?;
    // Non-string instructions become compact Python-style JSON (json.dumps).
    let instructions = match ins_value {
        Value::String(s) => s.clone(),
        other => py_dumps(other),
    };
    let criteria = map.get("criteria");

    match qtype {
        QType::Choice => {
            let pairs = choice_pairs(criteria)?;
            let options = pairs
                .iter()
                .map(|(k, v)| match v {
                    None => k.clone(),
                    Some(crit) if crit_is_blank(crit) => k.clone(),
                    Some(crit) => format!("{k}: {}", render_criterion(crit)),
                })
                .collect::<Vec<_>>();
            let choice_labels = pairs.into_iter().map(|(k, _)| k).collect();
            Ok(Question {
                qtype,
                instructions,
                options,
                choice_labels,
            })
        }
        QType::Score => {
            let crit = match criteria {
                Some(Value::Array(items)) if !items.is_empty() => items,
                _ => return Err(QuestionError::ScoreCriteriaInvalid),
            };
            let options = crit
                .iter()
                .enumerate()
                .map(|(i, c)| format!("level {i}: {}", render_criterion(c)))
                .collect();
            Ok(Question {
                qtype,
                instructions,
                options,
                choice_labels: Vec::new(),
            })
        }
        QType::Noul => {
            let crit = match criteria {
                None | Some(Value::Null) => Map::new(),
                Some(Value::Object(m)) => m.clone(),
                _ => return Err(QuestionError::NoulCriteriaInvalid),
            };
            let render_side = |key: &str, default: &str| match crit.get(key) {
                None | Some(Value::Null) => default.to_string(),
                Some(v) if crit_is_blank(v) => default.to_string(),
                Some(v) => render_criterion(v),
            };
            let options = vec![
                format!(
                    "false: {}",
                    render_side("false", "no, the statement does not hold")
                ),
                format!("true: {}", render_side("true", "yes, the statement holds")),
            ];
            Ok(Question {
                qtype,
                instructions,
                options,
                choice_labels: Vec::new(),
            })
        }
    }
}

/// Choice `(label, criterion)` pairs in definition order. A JSON list of
/// labels becomes `(label, None)` pairs (mirrors `dict.fromkeys`); a JSON
/// object keeps insertion order with its values.
fn choice_pairs(criteria: Option<&Value>) -> Result<Vec<(String, Option<Value>)>, QuestionError> {
    match criteria {
        Some(Value::Array(labels)) => {
            if labels.is_empty() {
                return Err(QuestionError::ChoiceCriteriaEmpty);
            }
            let mut out = Vec::with_capacity(labels.len());
            for label in labels {
                let s = label.as_str().ok_or(QuestionError::ChoiceLabelNotString)?;
                if out.iter().any(|(k, _): &(String, _)| k == s) {
                    return Err(QuestionError::ChoiceLabelNotUnique);
                }
                out.push((s.to_string(), None));
            }
            Ok(out)
        }
        Some(Value::Object(map)) => {
            if map.is_empty() {
                return Err(QuestionError::ChoiceCriteriaEmpty);
            }
            // JSON null is Python None ("no description"); keep it as None so
            // `null` and missing criteria render the bare label.
            Ok(map
                .iter()
                .map(|(k, v)| {
                    let opt = match v {
                        Value::Null => None,
                        other => Some(other.clone()),
                    };
                    (k.clone(), opt)
                })
                .collect())
        }
        _ => Err(QuestionError::ChoiceCriteriaEmpty),
    }
}

/// Only `None`/null and `""` mean "no description"; `0` and `false` are
/// legitimate criterion values (mirrors the Python comment in render_options).
fn crit_is_blank(v: &Value) -> bool {
    matches!(v, Value::String(s) if s.is_empty())
}

/// Render one criterion value: strings pass through, structured values become
/// `py_dumps` (mirrors `render_criterion`).
pub fn render_criterion(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        other => py_dumps(other),
    }
}

/// Re-export for the render path: state serialization shared with fixtures.
pub fn serialize_state_value(state: &Value) -> String {
    serialize_state(state)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn rejects_invalid_questions_like_python() {
        assert_eq!(
            to_internal(&json!({"type": "invalid", "instructions": "x"})),
            Err(QuestionError::UnknownType("invalid".into()))
        );
        assert_eq!(
            to_internal(&json!({"type": "choice", "instructions": "x", "criteria": []})),
            Err(QuestionError::ChoiceCriteriaEmpty)
        );
        assert_eq!(
            to_internal(&json!({"type": "choice", "instructions": "x", "criteria": ["a", "a"]})),
            Err(QuestionError::ChoiceLabelNotUnique)
        );
        assert_eq!(
            to_internal(&json!({"type": "score", "instructions": "x", "criteria": {}})),
            Err(QuestionError::ScoreCriteriaInvalid)
        );
        assert_eq!(
            to_internal(&json!({"type": "noul", "instructions": "x", "criteria": ["a"]})),
            Err(QuestionError::NoulCriteriaInvalid)
        );
        assert_eq!(
            to_internal(&json!({"type": "noul"})),
            Err(QuestionError::MissingInstructions)
        );
    }

    #[test]
    fn structured_criteria_render_python_style() {
        let q = to_internal(&json!({
            "type": "noul",
            "instructions": {"task": "verify"},
            "criteria": {"false": {"reason": "no"}, "true": {"reason": "yes"}},
        }))
        .unwrap();
        assert_eq!(q.instructions, r#"{"task": "verify"}"#);
        assert_eq!(
            q.options,
            vec![
                r#"false: {"reason": "no"}"#.to_string(),
                r#"true: {"reason": "yes"}"#.to_string(),
            ]
        );
        let q = to_internal(&json!({
            "type": "choice", "instructions": "c",
            "criteria": {"zero": 0, "no": false},
        }))
        .unwrap();
        assert_eq!(
            q.options,
            vec!["zero: 0".to_string(), "no: false".to_string()]
        );
    }
}
