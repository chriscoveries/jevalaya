//! `laya-coreml-ane` bundle validation, file verification, and the
//! symlink-free package materialization CoreML compilation requires
//! (Hub snapshots store `model.mlpackage` members as blob symlinks; the
//! native compiler can copy them into broken paths — mirrors
//! `laya_coreml.artifacts.package_for_coreml`).

use std::collections::BTreeMap;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use serde::Deserialize;
use sha2::{Digest, Sha256};

#[derive(Debug, thiserror::Error)]
pub enum BundleError {
    #[error("bundle io: {0}")]
    Io(#[from] io::Error),
    #[error("bundle manifest: {0}")]
    Manifest(String),
    #[error("bundle file {0} failed its manifest sha256")]
    Digest(String),
    #[error("bundle file {0} is missing")]
    Missing(String),
}

#[derive(Debug, Deserialize)]
pub struct CoremlConfig {
    pub format: String,
    pub format_version: u32,
    pub shape: ShapeSection,
    #[serde(default)]
    pub files: BTreeMap<String, FileEntry>,
    #[serde(default)]
    pub package_sha256: Option<String>,
    #[serde(default)]
    pub minimum_deployment_target: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct ShapeSection {
    pub batch_size: usize,
    pub max_length: usize,
    pub min_length: usize,
    pub max_options: usize,
    pub flexible: bool,
}

#[derive(Debug, Deserialize)]
pub struct FileEntry {
    pub sha256: String,
}

/// A validated bundle directory: paths plus the fixed exported shape.
pub struct Bundle {
    pub dir: PathBuf,
    pub config: CoremlConfig,
}

pub fn sha256_file(path: &Path) -> Result<String, io::Error> {
    let mut f = fs::File::open(path)?;
    let mut h = Sha256::new();
    io::copy(&mut f, &mut h)?;
    Ok(hex_lower(&h.finalize()))
}

fn hex_lower(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

impl Bundle {
    /// Parse + structurally validate `coreml_config.json`. Cheap: file
    /// existence only; content digests are checked by
    /// [`Bundle::verify_files`] during materialization/first load.
    pub fn open(dir: &Path) -> Result<Self, BundleError> {
        let text = fs::read_to_string(dir.join("coreml_config.json")).map_err(|e| {
            BundleError::Manifest(format!(
                "cannot read {}/coreml_config.json: {e}",
                dir.display()
            ))
        })?;
        let config: CoremlConfig = serde_json::from_str(&text)
            .map_err(|e| BundleError::Manifest(format!("coreml_config.json: {e}")))?;
        if (config.format.as_str(), config.format_version) != ("laya-coreml-ane", 1) {
            return Err(BundleError::Manifest(format!(
                "unsupported format {}/v{} (need laya-coreml-ane/1)",
                config.format, config.format_version
            )));
        }
        let s = &config.shape;
        if s.batch_size != 1 || s.min_length != s.max_length || s.flexible || s.max_options == 0 {
            return Err(BundleError::Manifest(format!(
                "ANE runtime requires a fixed B1/K>0 bundle, got batch={} len={}..{} options={} flexible={}",
                s.batch_size, s.min_length, s.max_length, s.max_options, s.flexible
            )));
        }
        Ok(Self {
            dir: dir.to_path_buf(),
            config,
        })
    }

    pub fn fixed_len(&self) -> usize {
        self.config.shape.max_length
    }

    pub fn max_options(&self) -> usize {
        self.config.shape.max_options
    }

    /// sha256-verify every manifest-listed file under the bundle dir.
    /// Runs once on first model load, not at startup.
    pub fn verify_files(&self) -> Result<(), BundleError> {
        for name in self.config.files.keys() {
            let rel = Path::new(name);
            if rel.is_absolute() || name.contains("..") || name.contains('\\') {
                return Err(BundleError::Manifest(format!(
                    "manifest file name {name:?} escapes the bundle"
                )));
            }
            let path = self.dir.join(rel);
            if !path.exists() {
                return Err(BundleError::Missing(name.clone()));
            }
            let got = sha256_file(&path)?;
            if got != self.config.files[name].sha256 {
                return Err(BundleError::Digest(name.clone()));
            }
        }
        Ok(())
    }

    /// A CoreML-loadable `model.mlpackage` path: the bundle's own package
    /// when its tree contains no symlinks, else a verified symlink-free
    /// copy under `cache_root/<package-key>/`.
    pub fn materialized_package(&self, cache_root: &Path) -> Result<PathBuf, BundleError> {
        let package = self.dir.join("model.mlpackage");
        if !package.is_dir() {
            return Err(BundleError::Missing("model.mlpackage".into()));
        }
        if !tree_has_symlink(&package)? {
            return Ok(package);
        }
        let key = self
            .config
            .package_sha256
            .clone()
            .unwrap_or_else(|| "unknown".to_string());
        let dest_dir = cache_root.join(&key);
        let dest = dest_dir.join("model.mlpackage");
        if dest.is_dir() {
            self.verify_materialized(&dest)?;
            return Ok(dest);
        }
        fs::create_dir_all(&dest_dir)?;
        let tmp = dest_dir.join(format!(".preparing-{}", std::process::id()));
        let tmp_pkg = tmp.join("model.mlpackage");
        if let Err(e) = copy_resolving_symlinks(&package, &tmp_pkg) {
            let _ = fs::remove_dir_all(&tmp);
            return Err(BundleError::Io(e));
        }
        if let Err(e) = self.verify_materialized(&tmp_pkg) {
            let _ = fs::remove_dir_all(&tmp);
            return Err(e);
        }
        if !dest.is_dir() {
            // Loser of a same-content race keeps the winner's copy.
            match fs::rename(&tmp_pkg, &dest) {
                Ok(()) => {}
                Err(e) if dest.is_dir() => {
                    let _ = e;
                }
                Err(e) => {
                    let _ = fs::remove_dir_all(&tmp);
                    return Err(BundleError::Io(e));
                }
            }
        }
        let _ = fs::remove_dir_all(&tmp);
        self.verify_materialized(&dest)?;
        Ok(dest)
    }

    /// Where the compiled `model.mlmodelc` is cached: `cache_root/<key>/`,
    /// same key as materialization. Never inside the (read-only) bundle.
    pub fn compiled_model_dir(&self, cache_root: &Path) -> PathBuf {
        let key = self
            .config
            .package_sha256
            .clone()
            .unwrap_or_else(|| "unknown".to_string());
        cache_root.join(key).join("model.mlmodelc")
    }

    /// Verify the materialized package members against manifest digests
    /// (only the `model.mlpackage/…` entries).
    fn verify_materialized(&self, package: &Path) -> Result<(), BundleError> {
        for (name, entry) in &self.config.files {
            let Some(rel) = name.strip_prefix("model.mlpackage/") else {
                continue;
            };
            let path = package.join(rel);
            if !path.exists() {
                return Err(BundleError::Missing(name.clone()));
            }
            if sha256_file(&path)? != entry.sha256 {
                return Err(BundleError::Digest(name.clone()));
            }
        }
        Ok(())
    }
}

fn tree_has_symlink(dir: &Path) -> Result<bool, io::Error> {
    if dir.symlink_metadata()?.is_symlink() {
        return Ok(true);
    }
    if !dir.is_dir() {
        return Ok(false);
    }
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        if tree_has_symlink(&entry.path())? {
            return Ok(true);
        }
    }
    Ok(false)
}

fn copy_resolving_symlinks(src: &Path, dst: &Path) -> Result<(), io::Error> {
    let meta = fs::metadata(src)?; // follows links
    if meta.is_dir() {
        fs::create_dir_all(dst)?;
        for entry in fs::read_dir(src)? {
            let entry = entry?;
            copy_resolving_symlinks(&entry.path(), &dst.join(entry.file_name()))?;
        }
    } else {
        fs::copy(src, dst)?;
    }
    Ok(())
}
