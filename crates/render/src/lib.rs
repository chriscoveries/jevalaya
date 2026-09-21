//! Canonical Laya prompt rendering + token counting.
//!
//! Rust port of the laya-mlx prompt path (`common.py`, `prepared.py`,
//! `tokenizer.py`, and the `Agent.prepare` / `_to_internal` methods of
//! `agent.py`). The rendered id sequence is what the ANE eligibility gate
//! counts, so parity with the Python output on fixed fixtures is the
//! correctness criterion — see `tests/parity.rs` and
//! `fixtures/MANIFEST.json`.
//!
//! Two formatting facts carry the whole port (both are covered by fixtures,
//! both fail silently as one-token gate drift if regressed):
//!
//! * structured values are serialised with CPython `json.dumps` spacing
//!   (`{"a": 1}`, see [`pyjson`]), not `serde_json::to_string` spacing;
//! * choice-label order is definition order, never sorted (requires
//!   `serde_json/preserve_order`).

pub mod config;
pub mod pyjson;
pub mod question;
pub mod render;
pub mod tokenizer;

pub use config::ModelBudgets;
pub use question::{to_internal, QType, Question};
pub use render::{
    ane_eligible, build_prefix, build_sequence, gate_counts, prepare, rendered_len, GateCounts,
    Rendered, DEFAULT_MAX_ANE_TOKENS,
};
pub use tokenizer::LayaTokenizer;
