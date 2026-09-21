//! Host-side math of the ANE split runtime — mirrors `laya_coreml.ane`
//! and `laya_coreml.result`: embedding/type-vector gather, additive fp16
//! attention masks, marker one-hot map, then logits masking + softmax +
//! entropy features + the fp32 `act_head` MLP + temperature-calibrated
//! answer decode. Pure Rust; unit-tested on every platform.

use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

use half::f16;
use memmap2::Mmap;
use serde::Deserialize;
use serde_json::{json, Map, Value};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum HostError {
    #[error("host weights io: {0}")]
    Io(#[from] std::io::Error),
    #[error("host weights: {0}")]
    Format(String),
    #[error("non-finite model output")]
    NonFinite,
}

#[derive(Debug, Deserialize)]
struct TensorMeta {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: (usize, usize),
}

/// `host_weights.safetensors`, memory-mapped: the fp16 embedding table
/// stays in the page cache; the small fp32 action head is copied out.
pub struct HostWeights {
    map: Mmap,
    base: usize,
    tensors: HashMap<String, TensorMeta>,
}

impl HostWeights {
    pub fn open(path: &Path) -> Result<Self, HostError> {
        let file = File::open(path)?;
        let map = unsafe { Mmap::map(&file)? };
        if map.len() < 8 {
            return Err(HostError::Format("truncated safetensors".into()));
        }
        let header_len = u64::from_le_bytes(map[..8].try_into().unwrap()) as usize;
        let header_end = 8 + header_len;
        if map.len() < header_end {
            return Err(HostError::Format("safetensors header overruns file".into()));
        }
        let header: HashMap<String, TensorMeta> = serde_json::from_slice(&map[8..header_end])
            .map_err(|e| HostError::Format(format!("safetensors header: {e}")))?;
        let mut w = Self {
            map,
            base: header_end,
            tensors: header,
        };
        w.tensors.remove("__metadata__");
        Ok(w)
    }

    fn meta(&self, name: &str) -> Result<&TensorMeta, HostError> {
        self.tensors
            .get(name)
            .ok_or_else(|| HostError::Format(format!("missing tensor {name}")))
    }

    /// Raw f16 bits of a 2-D tensor row (`[row*width ..]`). Copied:
    /// safetensors data offsets are not guaranteed u16-aligned in the map.
    pub fn row_f16(&self, name: &str, row: usize) -> Result<Vec<u16>, HostError> {
        let m = self.meta(name)?;
        if m.dtype != "F16" || m.shape.len() != 2 || row >= m.shape[0] {
            return Err(HostError::Format(format!("tensor {name} is not a 2-D F16")));
        }
        let width = m.shape[1];
        let start = self.base + m.data_offsets.0 + row * width * 2;
        let end = start + width * 2;
        if end > self.map.len() {
            return Err(HostError::Format(format!("tensor {name} overruns file")));
        }
        Ok(self.map[start..end]
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect())
    }

    /// Whole tensor as f32 (small tensors only: the action head).
    pub fn vec_f32(&self, name: &str) -> Result<(Vec<f32>, Vec<usize>), HostError> {
        let m = self.meta(name)?;
        let start = self.base + m.data_offsets.0;
        let end = self.base + m.data_offsets.1;
        if end > self.map.len() {
            return Err(HostError::Format(format!("tensor {name} overruns file")));
        }
        let bytes = &self.map[start..end];
        let out: Vec<f32> = match m.dtype.as_str() {
            "F16" => bytes
                .chunks_exact(2)
                .map(|c| f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
                .collect(),
            "F32" => bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect(),
            other => {
                return Err(HostError::Format(format!(
                    "tensor {name} dtype {other} unsupported"
                )))
            }
        };
        Ok((out, m.shape.clone()))
    }

    pub fn shape(&self, name: &str) -> Result<Vec<usize>, HostError> {
        Ok(self.meta(name)?.shape.clone())
    }
}

/// One CoreML-ready fp16 input set (C-order u16 buffers) plus the
/// scalar metadata decode needs (`k`, `qtype`, raw length).
pub struct AneInputs {
    pub embeddings: Vec<u16>,   // [1, W, 1, L]
    pub full_mask: Vec<u16>,    // [1, L, 1, L]
    pub local_mask: Vec<u16>,   // [1, L, 1, L]
    pub type_vectors: Vec<u16>, // [1, W, 1, 1]
    pub marker_map: Vec<u16>,   // [1, L, 1, K]
    pub k: usize,
    pub input_len: usize,
}

/// Fixed-shape validation mirrors `collate_items`: anything outside the
/// exported envelope is a shape fault, never a silent truncation.
pub fn validate_shape(
    ids_len: usize,
    markers: &[u32],
    qtype: u8,
    fixed_len: usize,
    max_options: usize,
    vocab: usize,
    ids: &[u32],
) -> Result<(), String> {
    if ids_len == 0 || ids_len > fixed_len {
        return Err(format!(
            "input has {ids_len} tokens, but this export supports at most {fixed_len}"
        ));
    }
    if markers.len() > max_options {
        return Err(format!(
            "question exceeds the exported max_options {max_options}"
        ));
    }
    if qtype > 2 {
        return Err(format!("question type must be 0, 1 or 2 (got {qtype})"));
    }
    for &m in markers {
        if m as usize >= ids_len {
            return Err(format!("marker position {m} outside the sequence"));
        }
    }
    for &id in ids {
        if id as usize >= vocab {
            return Err(format!("token id {id} outside checkpoint vocabulary"));
        }
    }
    Ok(())
}

/// Build the five fp16 model inputs for one prepared sequence.
///
/// Layout (CoreML BC1S): `full_mask[0, key, 0, query]` and
/// `local_mask[0, key, 0, query]` are additive pre-softmax masks
/// (0 allowed / -1e4 blocked); `embeddings[0, w, 0, i]` is the gathered
/// token embedding column-major across positions; `marker_map[0, pos, 0,
/// option]` one-hots each option's `[MASK]` position.
pub fn build_inputs(
    weights: &HostWeights,
    ids: &[u32],
    markers: &[u32],
    qtype: u8,
    pad_id: u32,
    fixed_len: usize,
    max_options: usize,
    hidden: usize,
    local_attention: usize,
) -> Result<AneInputs, HostError> {
    let vocab = weights.shape("encoder.embeddings.tok_embeddings.weight")?[0];
    validate_shape(
        ids.len(),
        markers,
        qtype,
        fixed_len,
        max_options,
        vocab,
        ids,
    )
    .map_err(HostError::Format)?;

    let l = fixed_len;
    let k = markers.len();
    let zero = f16::from_f32(0.0).to_bits();
    let neg = f16::from_f32(-1e4).to_bits();

    // attention_mask semantics: padded positions are invalid keys AND
    // invalid queries; a dummy key at position 0 keeps padded rows legal
    // (collate_items sets attention_mask[:, 0] = 1).
    let valid = |i: usize| i < ids.len() || i == 0;

    let mut embeddings = vec![zero; hidden * l];
    let mut padded_ids = vec![pad_id; l];
    padded_ids[..ids.len()].copy_from_slice(ids);
    for (i, &id) in padded_ids.iter().enumerate() {
        let row = weights.row_f16("encoder.embeddings.tok_embeddings.weight", id as usize)?;
        for (w, &v) in row.iter().enumerate() {
            embeddings[w * l + i] = v;
        }
    }

    let mut full_mask = vec![zero; l * l];
    let mut local_mask = vec![zero; l * l];
    let half_window = local_attention / 2;
    for j in 0..l {
        for i in 0..l {
            let allowed_full = valid(j);
            let allowed_local = (i.abs_diff(j) <= half_window || !valid(i)) && valid(j);
            full_mask[j * l + i] = if allowed_full { zero } else { neg };
            local_mask[j * l + i] = if allowed_local { zero } else { neg };
        }
    }

    let type_vectors: Vec<u16> = weights.row_f16("type_emb.weight", qtype as usize)?;

    let mut marker_map = vec![zero; l * max_options];
    let one = f16::from_f32(1.0).to_bits();
    for (slot, &pos) in markers.iter().enumerate() {
        marker_map[pos as usize * max_options + slot] = one;
    }

    Ok(AneInputs {
        embeddings,
        full_mask,
        local_mask,
        type_vectors,
        marker_map,
        k,
        input_len: ids.len(),
    })
}

/// Calibrated answer decode for one question (mirrors
/// `ResultMixin.system_one` and `ANEAgent.forward` post-processing).
/// `logits32` is the model's [1,1,1,K] marker logits; `pooled` the
/// [1,768,1,1] representation; `act_*` the fp32 host action head.
#[allow(clippy::too_many_arguments)]
pub fn decode_answer(
    logits32: &[f32],
    pooled: &[f32],
    k: usize,
    qtype: u8,
    qdef: &Value,
    question: &jevalaya_render::Question,
    temperature: &[f64; 3],
    temperature_by_options: &HashMap<String, f64>,
    act_head: &ActHead,
) -> Result<Value, HostError> {
    if !logits32.iter().all(|v| v.is_finite()) || !pooled.iter().all(|v| v.is_finite()) {
        return Err(HostError::NonFinite);
    }
    let kk = logits32.len();
    if kk < 2 || k == 0 || k > kk {
        return Err(HostError::Format(format!(
            "logits length {kk} cannot serve k={k} options"
        )));
    }
    // Masked softmax across all K slots (invalid markers → -1e4).
    let mut masked = logits32.to_vec();
    for (i, v) in masked.iter_mut().enumerate() {
        if i >= k {
            *v = -1e4;
        }
    }
    let p32 = softmax(&masked);
    let k2 = k.max(2) as f32;
    let entropy: f32 = -p32.iter().map(|&p| p * p.max(1e-9).ln()).sum::<f32>() / k2.ln();
    let mut sorted = p32.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let (top2, top1) = (sorted[kk - 2], sorted[kk - 1]);
    // `k` is clamped before both the entropy norm and the k/255 feature.
    let features = [top1, top1 - top2, entropy, k2 / 255.0];
    let mut x = pooled.to_vec();
    x.extend_from_slice(&features);
    let act = act_head.forward(&x);
    let actp = softmax(&act);
    if !actp.iter().all(|v| v.is_finite()) {
        return Err(HostError::NonFinite);
    }

    // Temperature-calibrated option probabilities over the k valid slots.
    let scale = temperature_by_options
        .get(&temp_bucket(qtype, k))
        .copied()
        .unwrap_or(temperature[qtype as usize])
        .max(1e-3);
    let z: Vec<f32> = logits32[..k].iter().map(|v| *v / scale as f32).collect();
    let p = softmax(&z);
    let t = qtype_name(qtype);
    let confidence = if k < 2 {
        1.0
    } else {
        let ent: f64 = -p[..k]
            .iter()
            .map(|&pp| (pp as f64) * (pp as f64).clamp(1e-12, 1.0).ln())
            .sum::<f64>();
        (1.0 - ent / (k as f64).ln()).clamp(0.0, 1.0)
    };

    let mut answer = Map::new();
    answer.insert("type".into(), json!(t));
    answer.insert("confidence".into(), json!(round4(confidence)));
    match qtype {
        0 => {
            let labels = &question.choice_labels;
            let best = p[..k]
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or(0);
            answer.insert("choice".into(), json!(labels[best]));
            let probs: Map<String, Value> = labels
                .iter()
                .take(k)
                .zip(p.iter())
                .map(|(l, v)| (l.clone(), json!(round4(*v as f64))))
                .collect();
            answer.insert("probabilities".into(), Value::Object(probs));
        }
        1 => {
            let score: f64 = p[..k]
                .iter()
                .enumerate()
                .map(|(i, v)| i as f64 * *v as f64)
                .sum();
            answer.insert("score".into(), json!(round4(score)));
            let mut legend = Map::new();
            if let Some(Value::Array(crit)) = qdef.get("criteria") {
                for (i, c) in crit.iter().take(k).enumerate() {
                    legend.insert(i.to_string(), c.clone());
                }
            }
            answer.insert("legend".into(), Value::Object(legend));
            let probs: Map<String, Value> = (0..k)
                .map(|i| (i.to_string(), json!(round4(p[i] as f64))))
                .collect();
            answer.insert("probabilities".into(), Value::Object(probs));
        }
        _ => {
            let p1 = p.get(1).copied().unwrap_or(0.0) as f64;
            answer.insert("noul".into(), json!(round4(p1)));
            answer.insert("confidence".into(), json!(round4(p1.max(1.0 - p1))));
        }
    }
    let mut action = Map::new();
    action.insert("act_probability".into(), json!(round4(actp[0] as f64)));
    answer.insert("action".into(), Value::Object(action));
    Ok(Value::Object(answer))
}

/// The fp32 host action head: `772 → (linear+erf-GELU) → 256 → linear → 2`.
pub struct ActHead {
    w0: Vec<f32>, // [256, 772]
    b0: Vec<f32>,
    w2: Vec<f32>, // [2, 256]
    b2: Vec<f32>,
    dim_in: usize,
    dim_h: usize,
}

impl ActHead {
    pub fn load(weights: &HostWeights) -> Result<Self, HostError> {
        let (w0, s0) = weights.vec_f32("act_head.0.weight")?;
        let (b0, _) = weights.vec_f32("act_head.0.bias")?;
        let (w2, s2) = weights.vec_f32("act_head.2.weight")?;
        let (b2, _) = weights.vec_f32("act_head.2.bias")?;
        if s0.len() != 2
            || s2.len() != 2
            || s2[1] != s0[0]
            || b0.len() != s0[0]
            || b2.len() != s2[0]
        {
            return Err(HostError::Format(format!(
                "act head shape mismatch: w0={s0:?} w2={s2:?}"
            )));
        }
        Ok(Self {
            dim_h: s0[0],
            dim_in: s0[1],
            w0,
            b0,
            w2,
            b2,
        })
    }

    /// Exact-erf GELU MLP (the reference deliberately avoids tanh/sigmoid
    /// approximations for these 256 host elements).
    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        debug_assert_eq!(x.len(), self.dim_in);
        let mut h = vec![0f32; self.dim_h];
        for (r, row) in h.iter_mut().enumerate() {
            let w = &self.w0[r * self.dim_in..(r + 1) * self.dim_in];
            *row = w.iter().zip(x.iter()).map(|(a, b)| a * b).sum::<f32>() + self.b0[r];
            *row *= (1.0 + libm::erf((*row as f64) / std::f64::consts::SQRT_2) as f32) / 2.0;
        }
        let mut out = vec![0f32; self.b2.len()];
        for (r, v) in out.iter_mut().enumerate() {
            let w = &self.w2[r * self.dim_h..(r + 1) * self.dim_h];
            *v = w.iter().zip(h.iter()).map(|(a, b)| a * b).sum::<f32>() + self.b2[r];
        }
        out
    }
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut p: Vec<f32> = logits.iter().map(|v| (v - max).exp()).collect();
    let sum: f32 = p.iter().sum();
    for v in p.iter_mut() {
        *v /= sum;
    }
    p
}

/// `temp_bucket`: `"<name>:<size>"` — 2 / 3-5 / 6-10 / 11+.
fn temp_bucket(qtype: u8, k: usize) -> String {
    let size = if k <= 2 {
        "2"
    } else if k <= 5 {
        "3-5"
    } else if k <= 10 {
        "6-10"
    } else {
        "11+"
    };
    format!("{}:{}", qtype_name(qtype), size)
}

fn qtype_name(qtype: u8) -> &'static str {
    match qtype {
        0 => "choice",
        1 => "score",
        _ => "noul",
    }
}

/// Python `round(x, 4)` — half-to-even on a 1e4 grid.
fn round4(x: f64) -> f64 {
    (x * 1e4).round_ties_even() / 1e4
}

#[cfg(test)]
mod tests {
    use super::*;
    use jevalaya_render::to_internal;
    use serde_json::json;
    use std::io::Write;

    /// Build a tiny safetensors file: embedding [8,4], type_emb [3,4],
    /// act_head 8→3→2 — small enough to verify decode by hand.
    fn fixture_weights() -> (tempfile::TempDir, std::path::PathBuf) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("host_weights.safetensors");
        let mut tensors: Vec<(String, Vec<usize>, Vec<f32>)> = Vec::new();
        // embedding rows are just id/8 per channel so gathers are obvious.
        tensors.push((
            "encoder.embeddings.tok_embeddings.weight".into(),
            vec![8, 4],
            (0..32).map(|i| i as f32 / 8.0).collect(),
        ));
        tensors.push((
            "type_emb.weight".into(),
            vec![3, 4],
            (0..12).map(|i| i as f32).collect(),
        ));
        tensors.push(("act_head.0.weight".into(), vec![3, 8], vec![0.0; 24]));
        tensors.push(("act_head.0.bias".into(), vec![3], vec![0.0; 3]));
        tensors.push((
            "act_head.2.weight".into(),
            vec![2, 3],
            vec![0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        ));
        tensors.push(("act_head.2.bias".into(), vec![2], vec![0.5, -0.5]));

        let mut data = Vec::new();
        let mut header = Map::new();
        for (name, shape, vals) in &tensors {
            let start = data.len();
            for v in vals {
                data.extend_from_slice(&f16::from_f32(*v).to_bits().to_le_bytes());
            }
            header.insert(
                name.clone(),
                json!({
                    "dtype": "F16",
                    "shape": shape,
                    "data_offsets": [start, data.len()],
                }),
            );
        }
        let header_bytes = serde_json::to_vec(&Value::Object(header)).unwrap();
        let mut file = File::create(&path).unwrap();
        file.write_all(&(header_bytes.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(&header_bytes).unwrap();
        file.write_all(&data).unwrap();
        (dir, path)
    }

    fn load_fixture() -> (tempfile::TempDir, HostWeights) {
        let (dir, path) = fixture_weights();
        let w = HostWeights::open(&path).unwrap();
        (dir, w)
    }

    #[test]
    fn gathers_embedding_columns_and_marker_map() {
        let (_d, w) = load_fixture();
        // hidden=4, L=8, K=4, window 4 (half=2)
        let inputs = build_inputs(&w, &[2, 5, 7], &[1, 2], 0, 0, 8, 4, 4, 4).unwrap();
        // embeddings[w*L+i] = E[id_i][w] = (id*4+w)/8
        for (i, &id) in [2usize, 5, 7].iter().enumerate() {
            for c in 0..4 {
                let got = f16::from_bits(inputs.embeddings[c * 8 + i]).to_f32();
                let want = (id * 4 + c) as f32 / 8.0;
                assert_eq!(got, want, "embedding[{c}][{i}]");
            }
        }
        // padded positions hold pad row 0
        for i in 3..8 {
            let got = f16::from_bits(inputs.embeddings[1 * 8 + i]).to_f32();
            assert_eq!(got, 1.0 / 8.0);
        }
        // marker_map[pos*K + slot]
        assert_eq!(f16::from_bits(inputs.marker_map[1 * 4 + 0]).to_f32(), 1.0);
        assert_eq!(f16::from_bits(inputs.marker_map[2 * 4 + 1]).to_f32(), 1.0);
        assert!(inputs
            .marker_map
            .iter()
            .enumerate()
            .all(|(i, &v)| (i == 4 || i == 9) || f16::from_bits(v).to_f32() == 0.0));
        // type_vectors = row 0 of type_emb
        for c in 0..4 {
            assert_eq!(f16::from_bits(inputs.type_vectors[c]).to_f32(), c as f32);
        }
    }

    #[test]
    fn mask_layout_is_key_major_with_local_window() {
        let (_d, w) = load_fixture();
        // ids len 6 of an L=8 export; window half=2
        let inputs = build_inputs(&w, &[1, 2, 3, 4, 5, 6], &[1], 2, 0, 8, 4, 4, 4).unwrap();
        let at = |m: &Vec<u16>, j: usize, i: usize| f16::from_bits(m[j * 8 + i]).to_f32();
        // full mask: invalid keys (j>=6) block every query
        assert_eq!(at(&inputs.full_mask, 7, 0), -1e4);
        assert_eq!(at(&inputs.full_mask, 0, 7), 0.0);
        // local: |i-j|<=2 for valid queries; key must be valid
        assert_eq!(at(&inputs.local_mask, 0, 0), 0.0);
        assert_eq!(at(&inputs.local_mask, 5, 0), -1e4); // valid pair, |0-5|=5>2
        assert_eq!(at(&inputs.local_mask, 2, 0), 0.0); // |0-2|<=2
                                                       // invalid query (i=7) attends every VALID key (j=0)…
        assert_eq!(at(&inputs.local_mask, 0, 7), 0.0);
        // …but an invalid key (j=6) stays blocked even for invalid queries
        assert_eq!(at(&inputs.local_mask, 6, 7), -1e4);
        assert_eq!(at(&inputs.local_mask, 6, 0), -1e4);
    }

    #[test]
    fn shape_validation_boundaries() {
        let (_d, w) = load_fixture();
        // vocab fixture is 8
        assert!(build_inputs(&w, &[1, 2], &[1], 0, 0, 8, 4, 4, 4).is_ok());
        assert!(build_inputs(&w, &[1, 2], &[1], 0, 0, 2, 4, 4, 4).is_ok()); // == limit
        assert!(build_inputs(&w, &[1, 2, 3], &[1], 0, 0, 2, 4, 4, 4).is_err()); // over
        assert!(build_inputs(&w, &[1, 2], &[1, 2, 3, 4, 5], 0, 0, 8, 4, 4, 4).is_err());
        assert!(build_inputs(&w, &[1, 2], &[9], 0, 0, 8, 4, 4, 4).is_err()); // marker OOB
        assert!(build_inputs(&w, &[1, 2], &[1], 3, 0, 8, 4, 4, 4).is_err()); // qtype
        assert!(build_inputs(&w, &[1, 99], &[1], 0, 0, 8, 4, 4, 4).is_err()); // vocab
        assert!(build_inputs(&w, &[], &[], 0, 0, 8, 4, 4, 4).is_err()); // empty
    }

    #[test]
    fn decode_matches_reference_math() {
        let (_d, w) = load_fixture();
        let act = ActHead::load(&w).unwrap();
        let qdef =
            json!({"type": "choice", "instructions": "i", "criteria": {"a": null, "b": null}});
        let question = to_internal(&qdef).unwrap();
        // logits: option 1 clearly wins; k=2 of K=4
        let logits = vec![0.0f32, 3.0, 7.0, -9.0];
        let pooled = vec![0.1f32, 0.2, 0.3, 0.4];
        let ans = decode_answer(
            &logits,
            &pooled,
            2,
            0,
            &qdef,
            &question,
            &[1.0, 1.0, 1.0],
            &HashMap::new(),
            &act,
        )
        .unwrap();
        assert_eq!(ans["type"], "choice");
        assert_eq!(ans["choice"], "b");
        let p_b = ans["probabilities"]["b"].as_f64().unwrap();
        // softmax([0,3]) → e^3/(1+e^3) = 0.9526
        assert!((p_b - 0.9526).abs() < 1e-3);
        // act head: x = pooled ++ feats; w0=0 → h=0 → act=[0.5,-0.5] → p=[0.7311,0.2689]
        let act_p = ans["action"]["act_probability"].as_f64().unwrap();
        assert!((act_p - 0.7311).abs() < 1e-3);
        assert_eq!(ans["action"]["act_probability"], json!(0.7311));
    }

    #[test]
    fn decode_noul_and_score() {
        let (_d, w) = load_fixture();
        let act = ActHead::load(&w).unwrap();
        let noul_def = json!({"type": "noul", "instructions": "holds?"});
        let nq = to_internal(&noul_def).unwrap();
        let ans = decode_answer(
            &[0.0, 2.0, -3.0, -3.0],
            &[0.0; 4],
            2,
            2,
            &noul_def,
            &nq,
            &[1.0, 1.0, 1.0],
            &HashMap::new(),
            &act,
        )
        .unwrap();
        // softmax([0,2]) → 0.8808
        assert_eq!(ans["noul"], json!(0.8808));
        assert_eq!(ans["confidence"], json!(0.8808));

        let score_def =
            json!({"type": "score", "instructions": "rate", "criteria": ["lo", "mid", "hi"]});
        let sq = to_internal(&score_def).unwrap();
        let ans = decode_answer(
            &[0.0, 0.0, 2.0, -9.0],
            &[0.0; 4],
            3,
            1,
            &score_def,
            &sq,
            &[1.0, 1.0, 1.0],
            &HashMap::new(),
            &act,
        )
        .unwrap();
        // softmax([0,0,2]) → [0.1065,0.1065,0.787] → E[level] ≈ 1.6805
        let score = ans["score"].as_f64().unwrap();
        assert!((score - 1.6805).abs() < 1e-3);
        assert_eq!(ans["legend"]["2"], "hi");
    }

    #[test]
    fn nonfinite_output_is_rejected() {
        let (_d, w) = load_fixture();
        let act = ActHead::load(&w).unwrap();
        let qdef = json!({"type": "noul", "instructions": "i"});
        let q = to_internal(&qdef).unwrap();
        let err = decode_answer(
            &[f32::NAN, 0.0, 0.0, 0.0],
            &[0.0; 4],
            2,
            2,
            &qdef,
            &q,
            &[1.0, 1.0, 1.0],
            &HashMap::new(),
            &act,
        )
        .unwrap_err();
        assert!(matches!(err, HostError::NonFinite));
    }

    #[test]
    fn temperature_scales_option_probs() {
        let (_d, w) = load_fixture();
        let act = ActHead::load(&w).unwrap();
        let qdef = json!({"type": "choice", "instructions": "i", "criteria": ["a", "b"]});
        let q = to_internal(&qdef).unwrap();
        let cold = decode_answer(
            &[0.0, 2.0, -9.0, -9.0],
            &[0.0; 4],
            2,
            0,
            &qdef,
            &q,
            &[0.5, 1.0, 1.0],
            &HashMap::new(),
            &act,
        )
        .unwrap();
        // z = [0,4] → p_b = e^4/(1+e^4) = 0.982
        let p_b = cold["probabilities"]["b"].as_f64().unwrap();
        assert!((p_b - 0.982).abs() < 1e-3);
    }
}
