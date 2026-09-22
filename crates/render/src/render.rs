//! Canonical prompt layout: `[CLS] head [SEP] ([MASK] opt)* [SEP] state [SEP]`.
//!
//! Exact port of `build_prefix` / `build_sequence` (common.py) and the
//! per-question flow of `Agent.prepare` (agent.py) / `PrefixCache.prepare`
//! (prepared.py). The rendered id sequence is what the ANE gate counts, so
//! every truncation constant below must stay in lockstep with laya-mlx:
//!
//! * option text is encoded with a leading space and cut to 48 tokens,
//!   with a prepended `[MASK]` id that is NOT part of the 48;
//! * `opt_budget = head_max_len - total_option_len`; if `< 16`, each option
//!   is re-cut to `max(4, (head_max_len - 16) / n_options)`;
//! * head is cut to `max(8, opt_budget)`;
//! * state fills `room = max_len - prefix_len - 1`, then a final `[SEP]`,
//!   then hard-cut to `max_len`; markers `>= max_len` are dropped;
//! * `[MASK]` occurrences inside instructions/options/state are scrubbed to
//!   a space BEFORE encoding (mask injection must never create markers).

use serde_json::Value;
use thiserror::Error;

use crate::question::{to_internal, Question, QuestionError};
use crate::tokenizer::LayaTokenizer;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum RenderError {
    #[error(transparent)]
    Question(#[from] QuestionError),
    /// Deliberate option ceiling: at most `max_options` of this question's
    /// options fit the token budget, but `received` were sent. The router
    /// maps this to 400 `invalid_request` (never 500) — fix the question,
    /// not the service.
    #[error("too_many_options: at most {max_options} options fit the token budget, received {received}")]
    TooManyOptions { max_options: usize, received: usize },
    #[error("questions must be a dictionary keyed by question id")]
    QuestionsNotADict,
    #[error("option_order length {0} does not match option count {1}")]
    BadOptionOrder(usize, usize),
}

/// One rendered question: token ids, marker positions, numeric type id.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rendered {
    pub ids: Vec<u32>,
    pub markers: Vec<u32>,
    pub qtype: u8,
}

/// Question-only prefix: ids plus one marker per option (position of its
/// `[MASK]`). Mirrors `build_prefix`.
pub fn build_prefix(
    tok: &LayaTokenizer,
    question: &Question,
    head_max_len: usize,
    option_order: Option<&[usize]>,
) -> Result<(Vec<u32>, Vec<u32>), RenderError> {
    let n = question.options.len();
    let order: Vec<usize> = match option_order {
        Some(o) => {
            if o.len() != n {
                return Err(RenderError::BadOptionOrder(o.len(), n));
            }
            o.to_vec()
        }
        None => (0..n).collect(),
    };

    let ins = question.instructions.replace(tok.mask_token(), " ");
    let head_text = format!("{} question: {}", question.qtype.name(), ins);
    let mut head_ids = tok.encode(&head_text);

    let mut opt_ids: Vec<Vec<u32>> = Vec::with_capacity(n);
    for i in order {
        let text = format!(" {}", question.options[i].replace(tok.mask_token(), " "));
        let mut ids = vec![tok.mask_id()];
        ids.extend(tok.encode(&text).into_iter().take(48));
        opt_ids.push(ids);
    }

    let mut opt_budget =
        head_max_len as isize - opt_ids.iter().map(Vec::len).sum::<usize>() as isize;
    if opt_budget < 16 {
        let per = std::cmp::max(
            4,
            (head_max_len as isize - 16) / std::cmp::max(1, n as isize),
        );
        for o in opt_ids.iter_mut() {
            o.truncate(per as usize);
        }
        opt_budget = head_max_len as isize - opt_ids.iter().map(Vec::len).sum::<usize>() as isize;
    }
    head_ids.truncate(std::cmp::max(8, opt_budget) as usize);

    let mut ids =
        Vec::with_capacity(head_ids.len() + opt_ids.iter().map(Vec::len).sum::<usize>() + 3);
    ids.push(tok.cls_id());
    ids.extend(head_ids);
    ids.push(tok.sep_id());
    let mut markers = Vec::with_capacity(n);
    for o in &opt_ids {
        markers.push(ids.len() as u32);
        ids.extend(o.iter().copied());
    }
    ids.push(tok.sep_id());
    Ok((ids, markers))
}

/// Serialize + mask-scrub + encode the full state, with no budget applied.
/// Single copy of the state path shared by [`build_sequence`] and
/// [`gate_counts`] so the two can never drift apart.
fn encode_full_state(tok: &LayaTokenizer, state: &Value) -> Vec<u32> {
    let text = match state {
        Value::String(s) => s.clone(),
        other => crate::pyjson::py_dumps(other),
    }
    .replace(tok.mask_token(), " ");
    tok.encode(&text)
}

/// Markers surviving the `max_len` cut. Shared predicate so the ceiling,
/// the sequence builder, and the gate can never disagree on what fits.
fn surviving(markers: &[u32], max_len: usize) -> usize {
    markers.iter().filter(|m| (**m as usize) < max_len).count()
}

/// Deliberate option ceiling: how many of this question's options fit the
/// token budget. Markers are appended in option order, so the survivors are
/// always a prefix — the count that fits IS the max. Computed from the head
/// budget (`head_max_len` shapes the prefix; `max_len` cuts the markers).
pub fn option_ceiling(
    tok: &LayaTokenizer,
    question: &Question,
    head_max_len: usize,
    max_len: usize,
) -> Result<usize, RenderError> {
    let (_, markers) = build_prefix(tok, question, head_max_len, None)?;
    Ok(surviving(&markers, max_len))
}

/// Full sequence for one question: prefix + state + final `[SEP]`.
/// Mirrors `build_sequence`. `truncate_left` takes the state's tail instead
/// of its head (default right-side truncation matches `Agent.prepare`).
///
/// Fails closed with [`RenderError::TooManyOptions`] when markers fall past
/// `max_len` — every consumer (execution render, ANE dispatch, MLX report)
/// gets the same deliberate limit, never silently dropped options.
pub fn build_sequence(
    tok: &LayaTokenizer,
    state: &Value,
    question: &Question,
    max_len: usize,
    head_max_len: usize,
    option_order: Option<&[usize]>,
    truncate_left: bool,
) -> Result<(Vec<u32>, Vec<u32>), RenderError> {
    let (prefix_ids, markers) = build_prefix(tok, question, head_max_len, option_order)?;
    let received = markers.len();
    let kept: Vec<u32> = markers
        .into_iter()
        .filter(|m| (*m as usize) < max_len)
        .collect();
    if kept.len() != received {
        return Err(RenderError::TooManyOptions {
            max_options: kept.len(),
            received,
        });
    }
    let state_ids = encode_full_state(tok, state);
    let room = (max_len as isize - prefix_ids.len() as isize - 1).max(0) as usize;
    let state_ids = if truncate_left {
        let skip = state_ids.len().saturating_sub(room);
        &state_ids[skip..]
    } else {
        let take = room.min(state_ids.len());
        &state_ids[..take]
    };
    let mut ids = prefix_ids;
    ids.extend(state_ids.iter().copied());
    ids.push(tok.sep_id());
    ids.truncate(max_len);
    Ok((ids, kept))
}

/// Pre-truncation gate counts for one question (DESIGN.md §Canonical prompt).
///
/// * `raw_count` — full semantic length: prefix + COMPLETE state + final
///   `[SEP]`, before any `max_len` truncation. This is what the ANE gate
///   decides on; truncating state to duck under the limit is forbidden.
/// * `aligned_count` — `raw_count` rounded up to `alignment` (1 when the
///   model specifies none). The gate compares this against `max_ane_tokens`.
///
/// `build_sequence` ids are the model-bounded execution sequence
/// (`execution_token_count`); they must never feed the gate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GateCounts {
    pub raw_count: usize,
    pub aligned_count: usize,
}

pub fn gate_counts(
    tok: &LayaTokenizer,
    state: &Value,
    question: &Question,
    max_len: usize,
    head_max_len: usize,
    alignment: usize,
) -> Result<GateCounts, RenderError> {
    let (prefix_ids, markers) = build_prefix(tok, question, head_max_len, None)?;
    let received = markers.len();
    let max_options = surviving(&markers, max_len);
    if max_options != received {
        return Err(RenderError::TooManyOptions {
            max_options,
            received,
        });
    }
    let state_ids = encode_full_state(tok, state);
    let raw_count = prefix_ids.len() + state_ids.len() + 1;
    let align = alignment.max(1);
    let aligned_count = raw_count.div_ceil(align) * align;
    Ok(GateCounts {
        raw_count,
        aligned_count,
    })
}

/// Render every question in a `{qid: qdef}` map, in map order.
/// `build_sequence` fails closed on over-budget option sets, so the
/// marker/option invariant from `Agent.prepare` holds by construction.
pub fn prepare(
    tok: &LayaTokenizer,
    state: &Value,
    questions: &serde_json::Map<String, Value>,
    max_len: usize,
    head_max_len: usize,
) -> Result<Vec<(String, Rendered)>, RenderError> {
    let mut out = Vec::with_capacity(questions.len());
    for (qid, qdef) in questions {
        let question = to_internal(qdef)?;
        let (ids, markers) =
            build_sequence(tok, state, &question, max_len, head_max_len, None, false)?;
        debug_assert_eq!(markers.len(), question.options.len());
        out.push((
            qid.clone(),
            Rendered {
                ids,
                markers,
                qtype: question.qtype.id(),
            },
        ));
    }
    Ok(out)
}

/// Rendered-token count feeding the ANE eligibility gate.
pub fn rendered_len(rendered: &Rendered) -> usize {
    rendered.ids.len()
}

/// ANE serves short single-question multilingual prompts only:
/// `rendered_len <= max_ane_tokens` (default 96).
pub const DEFAULT_MAX_ANE_TOKENS: usize = 96;

pub fn ane_eligible(rendered_len: usize, max_ane_tokens: usize) -> bool {
    rendered_len <= max_ane_tokens
}
