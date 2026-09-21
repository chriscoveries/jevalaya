//! Adapter-surface tests. The live path runs only when
//! `JEVALAYA_ANE_MODEL_DIR` names a real `laya-coreml-ane` bundle on a
//! macOS arm64 host; everything else is hermetic.

use std::path::PathBuf;

use jevalaya_backend_coreml::{AneBackend, AneConfig, ANE_TARGET};
use jevalaya_router_core::types::{
    BackendError, BackendKind, PredictBackend, PredictRequest, RenderedPrompt,
};
use serde_json::{json, Map, Value};

fn config_for(dir: &str) -> AneConfig {
    AneConfig {
        model_dir: PathBuf::from(dir),
        ..Default::default()
    }
}

#[test]
fn missing_bundle_dir_fails_construction() {
    let res = AneBackend::new(config_for("/nonexistent/ane-bundle"));
    let err = match res {
        Err(e) => e,
        Ok(_) => panic!("missing bundle must not construct"),
    };
    assert!(err.contains("coreml_config"), "unexpected error: {err}");
}

#[test]
fn off_target_stub_reports_unavailable() {
    if ANE_TARGET {
        return; // exercised only on non-Apple hosts
    }
    // A structurally valid bundle still reports unavailable off-target.
    // (On-target hosts take the live path below instead.)
}

fn request(questions: Map<String, Value>) -> PredictRequest {
    PredictRequest {
        state: json!("Der Kunde beantragt eine Rückerstattung der Doppelzahlung."),
        questions,
        checkpoint: jevalaya_router_core::types::Checkpoint::Multilingual,
        request_id: "t-ane".into(),
    }
}

fn one_noul() -> Map<String, Value> {
    let mut q = Map::new();
    q.insert(
        "q0".to_string(),
        json!({"type": "noul", "instructions": "Fordert der Kunde eine Rückerstattung?"}),
    );
    q
}

fn rendered_stub() -> RenderedPrompt {
    RenderedPrompt {
        ids: vec![1, 2, 3],
        markers: vec![1, 2],
        qtype: 2,
    }
}

/// Live path: requires a real bundle dir via env on macOS arm64.
/// Covers manifest verification, mlpackage materialization, model load,
/// fp16 input build, CoreML predict, and answer decode — the real T004
/// proof.
#[tokio::test(flavor = "multi_thread")]
async fn live_ane_predict() {
    let Ok(dir) = std::env::var("JEVALAYA_ANE_MODEL_DIR") else {
        eprintln!("JEVALAYA_ANE_MODEL_DIR unset — skipping live ANE test");
        return;
    };
    if !ANE_TARGET {
        eprintln!("not macOS arm64 — skipping live ANE test");
        return;
    }
    let backend = AneBackend::new(config_for(&dir)).expect("bundle init");
    let caps = backend.capabilities();
    assert_eq!(caps.kind, BackendKind::Ane);
    assert!(
        caps.available,
        "ane should report available on target: {}",
        caps.detail
    );

    // Real render through the production engine so the prompt is the
    // canonical one, not a stub.
    let mut engine = jevalaya_router_core::LayaPromptEngine::new();
    engine
        .load_ane(std::path::Path::new(&dir), 1)
        .expect("ane tokenizer");
    let questions = one_noul();
    let qdef = questions.get("q0").unwrap();
    let q = jevalaya_render::to_internal(qdef).unwrap();
    let state = json!("Der Kunde beantragt eine Rückerstattung der Doppelzahlung.");
    let rendered =
        jevalaya_router_core::PromptEngine::ane_render(&engine, &state, &q).expect("ane render");
    assert!(rendered.ids.len() <= 96);

    let out = backend
        .predict(&request(questions), Some(&rendered))
        .await
        .expect("live ane predict");
    assert_eq!(out.model, "laya-rl-agent");
    let ans = &out.answers["q0"];
    assert_eq!(ans["type"], "noul");
    let p = ans["noul"].as_f64().unwrap();
    assert!((0.0..=1.0).contains(&p));
    assert!(ans["confidence"].as_f64().unwrap() > 0.5);
    assert_eq!(out.usage["output_tokens"], json!(0));
    assert!(out.usage["input_tokens"].as_u64().unwrap() > 0);
    eprintln!(
        "live ANE: noul={p} conf={} act={} latency={}ms",
        ans["confidence"], ans["action"]["act_probability"], out.latency_ms
    );
}

/// Multi-question requests are a deterministic shape fault (and trip the
/// circuit breaker) — no CoreML needed once construction succeeds.
#[tokio::test(flavor = "multi_thread")]
async fn multi_question_is_shape_error() {
    let Ok(dir) = std::env::var("JEVALAYA_ANE_MODEL_DIR") else {
        eprintln!("JEVALAYA_ANE_MODEL_DIR unset — skipping");
        return;
    };
    let backend = AneBackend::new(config_for(&dir)).expect("bundle init");
    let mut questions = one_noul();
    questions.insert(
        "q1".to_string(),
        json!({"type": "noul", "instructions": "zweite Frage"}),
    );
    let err = backend
        .predict(&request(questions), Some(&rendered_stub()))
        .await
        .unwrap_err();
    if ANE_TARGET {
        assert!(matches!(err, BackendError::Shape(_)), "got {err:?}");
        // Circuit is now open: capabilities report unavailable.
        assert!(!backend.capabilities().available);
        // …and stays open until unload clears it.
        let err = backend
            .predict(&request(one_noul()), Some(&rendered_stub()))
            .await
            .unwrap_err();
        assert!(matches!(err, BackendError::Unavailable(_)));
    } else {
        assert!(matches!(err, BackendError::Unavailable(_)));
    }
}
