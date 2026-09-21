//! Wire schema for `POST /predict` (DESIGN.md §Wire contract). Strict mode
//! is the default: unknown routing fields are rejected rather than
//! silently changing backend.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::decision::RouteDecision;

/// `backend` routing field: `auto | ane | mlx | jev` (default `auto`).
/// Unknown strings are rejected at deserialization → 400.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum BackendSpec {
    #[default]
    Auto,
    Ane,
    Mlx,
    Jev,
}

/// `compare` field: `true` (configured set), `false`, or an explicit
/// backend list `["ane","mlx","jev"]`. Unknown names are rejected during
/// validation → 400.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(untagged)]
pub enum CompareSpec {
    Flag(bool),
    List(Vec<String>),
}

/// Request body for `POST /predict`. Only `state` and `questions` are
/// required; unknown fields are rejected (strict mode default).
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PredictBody {
    /// String, object, or list.
    pub state: Value,
    /// `{question_id: {type, instructions, criteria}}`.
    pub questions: Map<String, Value>,
    #[serde(default)]
    pub backend: BackendSpec,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub task: Option<String>,
    #[serde(default)]
    pub lang: Option<String>,
    #[serde(default)]
    pub request_id: Option<String>,
    #[serde(default)]
    pub compare: Option<CompareSpec>,
}

/// One entry under the response `compare` key: the normalized backend
/// result plus its `RouteDecision`, or a structured error.
#[derive(Debug, Clone, Serialize)]
pub struct CompareEntry {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub answers: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Value>,
    pub routing: RouteDecision,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<Value>,
}

/// Successful `/predict` response — the laya-mlx schema plus `routing`
/// and optional `compare`.
#[derive(Debug, Clone, Serialize)]
pub struct PredictResponse {
    pub model: String,
    pub answers: Value,
    pub usage: Value,
    pub routing: RouteDecision,
    #[serde(skip_serializing_if = "Map::is_empty")]
    pub compare: Map<String, Value>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn strict_mode_rejects_unknown_fields() {
        let body = json!({"state": "s", "questions": {}, "typo_field": 1});
        assert!(serde_json::from_value::<PredictBody>(body).is_err());
    }

    #[test]
    fn backend_spec_and_compare_parse() {
        let b: PredictBody = serde_json::from_value(json!({
            "state": "s", "questions": {}, "backend": "jev",
            "compare": ["ane", "mlx"],
        }))
        .unwrap();
        assert_eq!(b.backend, BackendSpec::Jev);
        assert_eq!(
            b.compare,
            Some(CompareSpec::List(vec!["ane".into(), "mlx".into()]))
        );
        assert!(serde_json::from_value::<PredictBody>(json!({
            "state": "s", "questions": {}, "backend": "gpu"
        }))
        .is_err());
        let b: PredictBody = serde_json::from_value(json!({
            "state": "s", "questions": {}, "compare": true
        }))
        .unwrap();
        assert_eq!(b.compare, Some(CompareSpec::Flag(true)));
    }

    #[test]
    fn missing_required_fields_fail() {
        assert!(serde_json::from_value::<PredictBody>(json!({"questions": {}})).is_err());
        assert!(serde_json::from_value::<PredictBody>(json!({"state": "s"})).is_err());
        assert!(
            serde_json::from_value::<PredictBody>(json!({"state": "s", "questions": []})).is_err()
        );
    }
}
