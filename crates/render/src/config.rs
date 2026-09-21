//! Per-model length budgets from `rl_agent_config.json`.
//!
//! `max_len` / `head_max_len` are per-checkpoint values (english 512/192,
//! multilingual 1024/256) — never globals. Mirrors the `Agent` constructor's
//! `4 < head_max_len < max_len <= max_position_embeddings` validation in
//! simplified form (position cap is encoder config, owned by the backend).

use std::path::Path;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("cannot read {0}: {1}")]
    Read(String, String),
    #[error("invalid budgets in {0}: need 4 < head_max_len < max_len, got {1}/{2}")]
    Invalid(String, usize, usize),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelBudgets {
    pub max_len: usize,
    pub head_max_len: usize,
}

impl ModelBudgets {
    /// Defaults matching `Agent` (`max_len=512`, `head_max_len=192`).
    pub fn defaults() -> Self {
        Self {
            max_len: 512,
            head_max_len: 192,
        }
    }

    pub fn from_model_dir(model_dir: &Path) -> Result<Self, ConfigError> {
        let path = model_dir.join("rl_agent_config.json");
        let text = std::fs::read_to_string(&path)
            .map_err(|e| ConfigError::Read(path.display().to_string(), e.to_string()))?;
        let cfg: serde_json::Value = serde_json::from_str(&text)
            .map_err(|e| ConfigError::Read(path.display().to_string(), e.to_string()))?;
        let max_len = cfg
            .get("max_len")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(512) as usize;
        let head_max_len = cfg
            .get("head_max_len")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(192) as usize;
        if !(4 < head_max_len && head_max_len < max_len) {
            return Err(ConfigError::Invalid(
                path.display().to_string(),
                head_max_len,
                max_len,
            ));
        }
        Ok(Self {
            max_len,
            head_max_len,
        })
    }
}
