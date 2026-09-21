//! Per-model tokenizer loading: `tokenizer.json` + special ids from
//! `tokenizer_config.json`. Exact port of `laya_mlx/tokenizer.py`.
//!
//! Special-token *names* vary per model directory (english uses
//! `[CLS]`/`[SEP]`, multilingual uses `<bos>`/`<eos>`), so ids are always
//! resolved per directory — never hardcoded. Like the Python loader, padding
//! and truncation stay disabled and a missing/invalid special token is an
//! error.

use std::path::Path;
use thiserror::Error;
use tokenizers::Tokenizer as Backend;

#[derive(Debug, Error)]
pub enum TokenizerError {
    #[error("cannot load tokenizer.json from {0}: {1}")]
    LoadFile(String, String),
    #[error("cannot read tokenizer_config.json from {0}: {1}")]
    ConfigFile(String, String),
    #[error("tokenizer_config.json in {0} is not a JSON object")]
    ConfigNotObject(String),
    #[error("Tokenizer is missing a valid {0}")]
    MissingSpecial(&'static str),
}

#[derive(Debug)]
pub struct LayaTokenizer {
    backend: Backend,
    cls_token: String,
    sep_token: String,
    pad_token: String,
    mask_token: String,
    cls_id: u32,
    sep_id: u32,
    pad_id: u32,
    mask_id: u32,
}

impl LayaTokenizer {
    /// Load from a `tokenizer/` directory (`tokenizer.json` +
    /// `tokenizer_config.json` inside).
    pub fn from_tokenizer_dir(dir: &Path) -> Result<Self, TokenizerError> {
        let file = dir.join("tokenizer.json");
        let mut backend = Backend::from_file(file.to_str().unwrap_or(""))
            .map_err(|e| TokenizerError::LoadFile(dir.display().to_string(), e.to_string()))?;
        // Mirror `.no_padding()` / `.no_truncation()`: the Rust crate leaves
        // both unset by default; clear them explicitly in case a saved
        // tokenizer.json carries truncation/padding settings.
        backend
            .with_truncation(None)
            .map_err(|e| TokenizerError::LoadFile(dir.display().to_string(), e.to_string()))?;
        backend.with_padding(None);

        let cfg_text = std::fs::read_to_string(dir.join("tokenizer_config.json"))
            .map_err(|e| TokenizerError::ConfigFile(dir.display().to_string(), e.to_string()))?;
        let cfg: serde_json::Value = serde_json::from_str(&cfg_text)
            .map_err(|e| TokenizerError::ConfigFile(dir.display().to_string(), e.to_string()))?;
        let cfg = cfg
            .as_object()
            .ok_or_else(|| TokenizerError::ConfigNotObject(dir.display().to_string()))?;

        let resolve = |name: &'static str| -> Result<(String, u32), TokenizerError> {
            let raw = cfg.get(name).ok_or(TokenizerError::MissingSpecial(name))?;
            // Values may be a plain string or {"content": "..."}.
            let token = match raw {
                serde_json::Value::String(s) => s.clone(),
                serde_json::Value::Object(m) => m
                    .get("content")
                    .and_then(|v| v.as_str())
                    .ok_or(TokenizerError::MissingSpecial(name))?
                    .to_string(),
                _ => return Err(TokenizerError::MissingSpecial(name)),
            };
            let id = backend
                .token_to_id(&token)
                .ok_or(TokenizerError::MissingSpecial(name))?;
            Ok((token, id))
        };

        let (cls_token, cls_id) = resolve("cls_token")?;
        let (sep_token, sep_id) = resolve("sep_token")?;
        let (pad_token, pad_id) = resolve("pad_token")?;
        let (mask_token, mask_id) = resolve("mask_token")?;
        Ok(Self {
            backend,
            cls_token,
            sep_token,
            pad_token,
            mask_token,
            cls_id,
            sep_id,
            pad_id,
            mask_id,
        })
    }

    /// Load from a model directory (expects `tokenizer/` inside, mirroring
    /// `Agent`, which passes `model_dir / "tokenizer"`).
    pub fn from_model_dir(model_dir: &Path) -> Result<Self, TokenizerError> {
        Self::from_tokenizer_dir(&model_dir.join("tokenizer"))
    }

    /// Encode with no special tokens added (mirrors
    /// `__call__(text, add_special_tokens=False)`).
    pub fn encode(&self, text: &str) -> Vec<u32> {
        self.backend
            .encode(text, false)
            .map(|e| e.get_ids().to_vec())
            .unwrap_or_default()
    }

    pub fn cls_id(&self) -> u32 {
        self.cls_id
    }
    pub fn sep_id(&self) -> u32 {
        self.sep_id
    }
    pub fn pad_id(&self) -> u32 {
        self.pad_id
    }
    pub fn mask_id(&self) -> u32 {
        self.mask_id
    }
    pub fn mask_token(&self) -> &str {
        &self.mask_token
    }
    pub fn cls_token(&self) -> &str {
        &self.cls_token
    }
    pub fn sep_token(&self) -> &str {
        &self.sep_token
    }
    pub fn pad_token(&self) -> &str {
        &self.pad_token
    }
}
