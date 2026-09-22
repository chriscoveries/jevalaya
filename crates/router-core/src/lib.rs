//! jevalaya router-core: policy-only routing for the shared laya/jev
//! predict schema (docs/DESIGN.md).
//!
//! * [`schema`] — the `POST /predict` wire contract (strict by default)
//! * [`checkpoint`] — checkpoint router (`model` > `task` > workflow >
//!   `lang` > detected > default), ported from laya-mlx `router.py`
//! * [`lang`] — script/language detection, ported from `lang.py`
//! * [`engine`] — prompt rendering + gate counting surface over
//!   `jevalaya-render` (the ANE gate uses the ANE model's own tokenizer)
//! * [`policy`] — the ordered decision table → [`RouteDecision`]
//! * [`router`] — dispatch, ANE→MLX retry, Jev escalation, compare
//!   fan-out, observability events
//!
//! No HTTP client, embedded Python, or CoreML symbols live here — those
//! are the backend crates. Everything is unit-testable with mock
//! [`PredictBackend`]s and a scripted [`engine::PromptEngine`].

pub mod checkpoint;
pub mod config;
pub mod decision;
pub mod engine;
pub mod errors;
pub mod events;
pub mod lang;
pub mod policy;
pub mod router;
pub mod schema;
pub mod types;

pub use checkpoint::{choose_checkpoint, match_typed_decisions_workflow, CheckpointChoice};
pub use config::{AnePolicy, BackendThresholds, JevPolicy, PolicyConfig, ThresholdsPolicy};
pub use decision::{reason, trigger, RouteDecision};
pub use engine::{LayaPromptEngine, PromptEngine};
pub use errors::RouteError;
pub use events::{EventSink, JsonlSink, NullSink, RoutingEvent, VecSink};
pub use lang::{
    analyse, detect_script, guess_latin_language, script_profile, state_text, Detection,
};
pub use policy::{route, Availability, RoutePlan, ValidatedRequest};
pub use router::{confidence_margin, Router};
pub use schema::{BackendSpec, CompareEntry, CompareSpec, PredictBody, PredictResponse};
pub use types::{
    BackendCapabilities, BackendError, BackendKind, BackendResult, Checkpoint, PredictBackend,
    PredictRequest, PreloadRequest, RenderedPrompt, UnloadTarget,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checkpoint_names_round_trip() {
        for name in [
            "english",
            "multilingual",
            "typed-decisions",
            "typed_decisions",
            "en",
        ] {
            let cp = Checkpoint::from_name(name).unwrap();
            assert_eq!(Checkpoint::from_name(cp.as_str()), Some(cp));
        }
        assert_eq!(Checkpoint::from_name("nope"), None);
    }

    #[test]
    fn backend_kind_serde_lowercase() {
        assert_eq!(
            serde_json::to_value(BackendKind::Ane).unwrap(),
            serde_json::json!("ane")
        );
        assert_eq!(
            serde_json::from_value::<BackendKind>(serde_json::json!("jev")).unwrap(),
            BackendKind::Jev
        );
        assert!(serde_json::from_value::<BackendKind>(serde_json::json!("gpu")).is_err());
    }
}
