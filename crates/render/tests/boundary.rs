//! T026: the option ceiling is deliberate and exact.
//!
//! For a fixed option shape, `option_ceiling` reports the largest count that
//! fits; `prepare` succeeds at exactly that count and fails closed at one
//! more, carrying `max_options == max` and `received == max + 1` so the
//! router can answer 400 with both numbers. A CLINC150-shaped 150-label
//! question pins the original finding at unit level.

use std::path::PathBuf;

use jevalaya_render::render::RenderError;
use jevalaya_render::{option_ceiling, to_internal, LayaTokenizer};
use serde_json::{json, Map, Value};

const MAX_LEN: usize = 512;
const HEAD_MAX_LEN: usize = 192;

fn tokenizer() -> LayaTokenizer {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/hermetic/tokenizer");
    LayaTokenizer::from_tokenizer_dir(&root).expect("hermetic tokenizer")
}

fn choice_map(labels: &[String]) -> Map<String, Value> {
    let qdef = json!({
        "type": "choice",
        "instructions": "Choose one",
        "criteria": labels,
    });
    let mut map = Map::new();
    map.insert("q".to_string(), qdef);
    map
}

fn labels(n: usize) -> Vec<String> {
    (0..n).map(|i| format!("opt{i}")).collect()
}

#[test]
fn ceiling_is_exact_at_max_and_max_plus_one() {
    let tok = tokenizer();
    let state = Value::String("hello".to_string());

    // Find the flip point by probing upward (each probe is milliseconds).
    let mut max = 0;
    for n in 1..=2000 {
        let q = to_internal(&json!({
            "type": "choice", "instructions": "Choose one", "criteria": labels(n),
        }))
        .unwrap();
        match option_ceiling(&tok, &q, HEAD_MAX_LEN, MAX_LEN).unwrap() {
            m if m >= n => max = n,
            _ => break,
        }
    }
    assert!(max > 100, "sanity: ceiling should be large for tiny options, got {max}");

    // Exactly max succeeds through the full prepare path.
    let ok = jevalaya_render::prepare(&tok, &state, &choice_map(&labels(max)), MAX_LEN, HEAD_MAX_LEN);
    assert!(ok.is_ok(), "prepare at ceiling ({max}) must succeed");

    // One more fails closed carrying both numbers for the 400 body.
    let err = jevalaya_render::prepare(
        &tok,
        &state,
        &choice_map(&labels(max + 1)),
        MAX_LEN,
        HEAD_MAX_LEN,
    )
    .unwrap_err();
    assert_eq!(
        err,
        RenderError::TooManyOptions {
            max_options: max,
            received: max + 1,
        }
    );

    // gate_counts agrees on the same boundary (no second formula).
    let q_max1 = to_internal(&json!({
        "type": "choice", "instructions": "Choose one", "criteria": labels(max + 1),
    }))
    .unwrap();
    let gc_err = jevalaya_render::gate_counts(&tok, &state, &q_max1, MAX_LEN, HEAD_MAX_LEN, 1)
        .unwrap_err();
    assert_eq!(gc_err, err);
}

#[test]
fn clinc150_shaped_150_labels_fail_with_counts() {
    let tok = tokenizer();
    let state = Value::String("transfer money please".to_string());
    let q = to_internal(&json!({
        "type": "choice",
        "instructions": "What does the user want?",
        "criteria": (0..150).map(|i| format!("intent number {i}")).collect::<Vec<_>>(),
    }))
    .unwrap();
    let ceiling = option_ceiling(&tok, &q, HEAD_MAX_LEN, MAX_LEN).unwrap();
    assert!(ceiling < 150, "150-label question must exceed the ceiling, got {ceiling}");
    let err =
        jevalaya_render::prepare(&tok, &state, &{
            let mut m = Map::new();
            m.insert("intent".to_string(), json!({
                "type": "choice",
                "instructions": "What does the user want?",
                "criteria": (0..150).map(|i| format!("intent number {i}")).collect::<Vec<_>>(),
            }));
            m
        }, MAX_LEN, HEAD_MAX_LEN)
        .unwrap_err();
    match err {
        RenderError::TooManyOptions { max_options, received } => {
            assert_eq!(received, 150);
            assert_eq!(max_options, ceiling);
        }
        other => panic!("expected TooManyOptions, got {other}"),
    }
}
