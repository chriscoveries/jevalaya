//! Prompt engine: the router's view of rendering + token counting.
//!
//! The production impl [`LayaPromptEngine`] wraps `jevalaya-render`
//! tokenizers loaded per model directory — the ANE gate counts with the
//! ANE model's bundled tokenizer, MLX-side reporting counts with the
//! selected checkpoint's tokenizer (DESIGN.md §Canonical prompt). Tests
//! substitute scripted fakes so decision-table boundaries are exact.

use std::collections::HashMap;
use std::path::Path;

use jevalaya_render::render::RenderError;
use jevalaya_render::{
    build_sequence, gate_counts, GateCounts, LayaTokenizer, ModelBudgets, Question, Rendered,
};

use crate::errors::RouteError;
use crate::types::{Checkpoint, RenderedPrompt};

fn render_err(e: RenderError) -> RouteError {
    // All RenderError variants are request-shape failures (question schema,
    // option budgets) → invalid_request.
    RouteError::InvalidRequest(e.to_string())
}

/// Rendering/counting surface the decision table needs. Production:
/// [`LayaPromptEngine`]. Tests: scripted fakes.
pub trait PromptEngine: Send + Sync {
    /// Pre-truncation gate counts with the ANE model's bundled tokenizer
    /// and alignment — the only counts the ANE gate may consult.
    fn ane_counts(&self, state: &serde_json::Value, q: &Question)
        -> Result<GateCounts, RouteError>;
    /// Model-bounded render for ANE dispatch (unpadded ids — the adapter
    /// owns fixed-shape padding).
    fn ane_render(
        &self,
        state: &serde_json::Value,
        q: &Question,
    ) -> Result<RenderedPrompt, RouteError>;
    /// Pre-truncation counts with the selected checkpoint's tokenizer.
    fn counts(
        &self,
        checkpoint: Checkpoint,
        state: &serde_json::Value,
        q: &Question,
    ) -> Result<GateCounts, RouteError>;
    /// Length of the model-bounded sequence MLX would execute
    /// (`execution_token_count` reporting).
    fn execution_len(
        &self,
        checkpoint: Checkpoint,
        state: &serde_json::Value,
        q: &Question,
    ) -> Result<usize, RouteError>;
}

/// Real engine over `jevalaya-render`: per-checkpoint tokenizers/budgets
/// plus the ANE model's bundled tokenizer for the gate.
pub struct LayaPromptEngine {
    ane: Option<AneEngine>,
    checkpoints: HashMap<Checkpoint, CkptEngine>,
}

struct AneEngine {
    tok: LayaTokenizer,
    budgets: ModelBudgets,
    alignment: usize,
}

struct CkptEngine {
    tok: LayaTokenizer,
    budgets: ModelBudgets,
}

impl LayaPromptEngine {
    pub fn new() -> Self {
        Self {
            ane: None,
            checkpoints: HashMap::new(),
        }
    }

    /// Register a checkpoint's tokenizer + budgets (from its model dir).
    pub fn with_checkpoint(
        mut self,
        checkpoint: Checkpoint,
        tok: LayaTokenizer,
        budgets: ModelBudgets,
    ) -> Self {
        self.checkpoints
            .insert(checkpoint, CkptEngine { tok, budgets });
        self
    }

    /// Register the ANE model's bundled tokenizer + budgets + alignment.
    /// Without this, `ane_counts`/`ane_render` fail closed as `not_ready`.
    pub fn with_ane(mut self, tok: LayaTokenizer, budgets: ModelBudgets, alignment: usize) -> Self {
        self.ane = Some(AneEngine {
            tok,
            budgets,
            alignment,
        });
        self
    }

    /// Convenience: load tokenizer + budgets for a checkpoint from its
    /// model directory (`tokenizer/` + `rl_agent_config.json` inside).
    pub fn load_checkpoint(
        &mut self,
        checkpoint: Checkpoint,
        model_dir: &Path,
    ) -> Result<(), RouteError> {
        let tok = LayaTokenizer::from_model_dir(model_dir)
            .map_err(|e| RouteError::NotReady(e.to_string()))?;
        let budgets = ModelBudgets::from_model_dir(model_dir)
            .map_err(|e| RouteError::NotReady(e.to_string()))?;
        self.checkpoints
            .insert(checkpoint, CkptEngine { tok, budgets });
        Ok(())
    }

    /// Convenience: load the ANE model's bundled tokenizer + budgets.
    pub fn load_ane(&mut self, model_dir: &Path, alignment: usize) -> Result<(), RouteError> {
        let tok = LayaTokenizer::from_model_dir(model_dir)
            .map_err(|e| RouteError::NotReady(e.to_string()))?;
        let budgets = ModelBudgets::from_model_dir(model_dir)
            .map_err(|e| RouteError::NotReady(e.to_string()))?;
        self.ane = Some(AneEngine {
            tok,
            budgets,
            alignment,
        });
        Ok(())
    }
}

impl Default for LayaPromptEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl PromptEngine for LayaPromptEngine {
    fn ane_counts(
        &self,
        state: &serde_json::Value,
        q: &Question,
    ) -> Result<GateCounts, RouteError> {
        let ane = self
            .ane
            .as_ref()
            .ok_or_else(|| RouteError::NotReady("ane tokenizer unavailable".to_string()))?;
        gate_counts(&ane.tok, state, q, ane.budgets.head_max_len, ane.alignment).map_err(render_err)
    }

    fn ane_render(
        &self,
        state: &serde_json::Value,
        q: &Question,
    ) -> Result<RenderedPrompt, RouteError> {
        let ane = self
            .ane
            .as_ref()
            .ok_or_else(|| RouteError::NotReady("ane tokenizer unavailable".to_string()))?;
        let (ids, markers) = build_sequence(
            &ane.tok,
            state,
            q,
            ane.budgets.max_len,
            ane.budgets.head_max_len,
            None,
            false,
        )
        .map_err(render_err)?;
        Ok(Rendered {
            ids,
            markers,
            qtype: q.qtype.id(),
        })
    }

    fn counts(
        &self,
        checkpoint: Checkpoint,
        state: &serde_json::Value,
        q: &Question,
    ) -> Result<GateCounts, RouteError> {
        let e = self.checkpoints.get(&checkpoint).ok_or_else(|| {
            RouteError::NotReady(format!("{} tokenizer unavailable", checkpoint.as_str()))
        })?;
        gate_counts(&e.tok, state, q, e.budgets.head_max_len, 1).map_err(render_err)
    }

    fn execution_len(
        &self,
        checkpoint: Checkpoint,
        state: &serde_json::Value,
        q: &Question,
    ) -> Result<usize, RouteError> {
        let e = self.checkpoints.get(&checkpoint).ok_or_else(|| {
            RouteError::NotReady(format!("{} tokenizer unavailable", checkpoint.as_str()))
        })?;
        let (ids, _) = build_sequence(
            &e.tok,
            state,
            q,
            e.budgets.max_len,
            e.budgets.head_max_len,
            None,
            false,
        )
        .map_err(render_err)?;
        Ok(ids.len())
    }
}
