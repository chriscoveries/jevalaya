//! Jev TypeSafe HTTP backend.
//!
//! Speaks the shared laya predict schema (`{state, questions}` in,
//! `{model, answers, usage}` out) over `POST <base_url>/v1/systemone`.
//! Behavior mirrors `jev_ultrafast/model.py`:
//!
//! * retry ONLY 429/529/503, up to three total attempts, 0.5s/1.0s backoff;
//!   transport errors and other statuses fail immediately;
//! * `validate_choice` invariants on choice answers (selected label is a
//!   criterion, probability keys match criteria, finite [0,1] values summing
//!   to ~1, selected is the argmax);
//! * key from the environment named by config (`TYPESAFE_API_KEY`), never
//!   from the repo, never in errors or logs.

use std::time::{Duration, Instant};

use jevalaya_router_core::{
    BackendCapabilities, BackendError, BackendKind, BackendResult, PredictBackend, PredictRequest,
    PreloadRequest, RenderedPrompt, UnloadTarget,
};
use serde_json::{Map, Value};

/// Retryable provider statuses, mirroring `post_json`.
fn is_retryable_status(status: u16) -> bool {
    matches!(status, 429 | 529 | 503)
}

#[derive(Debug, Clone)]
pub struct JevConfig {
    pub base_url: String,
    pub predict_path: String,
    pub api_key_env: String,
    pub model: String,
    pub request_timeout: Duration,
    /// Total attempts including the first (design default 3).
    pub max_attempts: u32,
    /// Backoff before retry N (design: 0.5s, 1.0s).
    pub backoff: Vec<Duration>,
}

impl Default for JevConfig {
    fn default() -> Self {
        Self {
            base_url: "https://api.typesafe.ai".to_string(),
            predict_path: "/v1/systemone".to_string(),
            api_key_env: "TYPESAFE_API_KEY".to_string(),
            model: "jev-latest".to_string(),
            request_timeout: Duration::from_millis(25_000),
            max_attempts: 3,
            backoff: vec![Duration::from_millis(500), Duration::from_millis(1000)],
        }
    }
}

impl JevConfig {
    pub fn endpoint(&self) -> String {
        format!(
            "{}{}",
            self.base_url.trim_end_matches('/'),
            self.predict_path
        )
    }
}

pub struct JevBackend {
    config: JevConfig,
    api_key: String,
    client: reqwest::Client,
}

// Manual Debug: the key must never appear in logs or test output.
impl std::fmt::Debug for JevBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("JevBackend")
            .field("config", &self.config)
            .field("api_key", &"<redacted>")
            .finish_non_exhaustive()
    }
}

impl JevBackend {
    /// Build with an explicit key (tests, custom resolution).
    pub fn with_key(config: JevConfig, api_key: String) -> Result<Self, BackendError> {
        if api_key.is_empty() {
            return Err(BackendError::MissingCredential(config.api_key_env.clone()));
        }
        let client = reqwest::Client::builder()
            .timeout(config.request_timeout)
            .build()
            .map_err(|e| BackendError::Transport(format!("client build: {e}")))?;
        Ok(Self {
            config,
            api_key,
            client,
        })
    }

    /// Build with the key read from the configured environment variable.
    /// Presence (not value) is the only thing ever checked or logged.
    pub fn from_env(config: JevConfig) -> Result<Self, BackendError> {
        let key = std::env::var(&config.api_key_env).unwrap_or_default();
        Self::with_key(config, key)
    }

    pub fn config(&self) -> &JevConfig {
        &self.config
    }

    async fn post_once(&self, body: &Value) -> Result<Value, BackendError> {
        let response = self
            .client
            .post(self.config.endpoint())
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(body)
            .send()
            .await
            .map_err(|e| {
                if e.is_timeout() {
                    BackendError::Timeout(self.config.request_timeout.as_millis() as u64)
                } else {
                    BackendError::Transport("connection failed".to_string())
                }
            })?;
        let status = response.status().as_u16();
        if response.status().is_success() {
            return response.json::<Value>().await.map_err(|e| {
                BackendError::InvalidResponse(format!("unreadable provider body: {e}"))
            });
        }
        // Never include the body or auth context in the error.
        Err(BackendError::Upstream(
            Some(status),
            format!("provider returned HTTP {status}"),
        ))
    }

    async fn post_with_retry(&self, body: &Value) -> Result<Value, BackendError> {
        let mut attempt: u32 = 0;
        loop {
            attempt += 1;
            match self.post_once(body).await {
                Ok(value) => return Ok(value),
                Err(BackendError::Upstream(Some(status), _))
                    if is_retryable_status(status) && attempt < self.config.max_attempts =>
                {
                    let backoff = self
                        .config
                        .backoff
                        .get((attempt - 1) as usize)
                        .copied()
                        .unwrap_or(Duration::from_secs(1));
                    tokio::time::sleep(backoff).await;
                }
                Err(other) => return Err(other),
            }
        }
    }

    /// Normalized predict: same `{state, questions}` schema as layalocally,
    /// answers validated, latency measured. `rendered` is unused — Jev
    /// tokenizes server-side, so `token_count` stays null for explicit Jev.
    pub async fn predict_json(
        &self,
        state: &Value,
        questions: &Map<String, Value>,
    ) -> Result<BackendResult, BackendError> {
        let questions = jev_questions(questions);
        let body = serde_json::json!({
            "model": self.config.model,
            "state": state,
            "questions": questions,
        });
        let started = Instant::now();
        let response = self.post_with_retry(&body).await?;
        let latency_ms = started.elapsed().as_millis() as u64;
        let model = response
            .get("model")
            .and_then(Value::as_str)
            .unwrap_or(&self.config.model)
            .to_string();
        let answers = response.get("answers").cloned().unwrap_or(Value::Null);
        validate_answers(&questions, &answers)?;
        let usage = response
            .get("usage")
            .cloned()
            .unwrap_or_else(|| serde_json::json!({}));
        Ok(BackendResult {
            model,
            answers,
            usage,
            latency_ms,
        })
    }
}

/// TypeSafe requires `criteria` as a dict for choice questions; the shared
/// laya schema also accepts a bare label list (`dict.fromkeys` semantics).
/// Upgrade all-string lists to `{label: null}` at the adapter boundary so
/// any laya-valid choice request is Jev-compatible (verified: the provider
/// accepts null descriptors; a list body is a 422).
///
/// Score level-lists are NOT upgraded: the list order IS the ordinal scale,
/// and rewriting it as a dict is exactly what the provider 422s (T029).
fn jev_questions(questions: &Map<String, Value>) -> Map<String, Value> {
    let mut out = questions.clone();
    for qdef in out.values_mut() {
        let Some(obj) = qdef.as_object_mut() else {
            continue;
        };
        if obj.get("type").and_then(Value::as_str) != Some("choice") {
            continue;
        }
        let Some(Value::Array(labels)) = obj.get("criteria") else {
            continue;
        };
        if labels.is_empty() || !labels.iter().all(|l| l.is_string()) {
            continue;
        }
        let dict: Map<String, Value> = labels
            .iter()
            .filter_map(|l| l.as_str().map(|s| (s.to_string(), Value::Null)))
            .collect();
        obj.insert("criteria".to_string(), Value::Object(dict));
    }
    out
}

/// Choice-answer invariants, mirroring `validate_choice`: the selected label
/// must be a declared criterion, probability keys must equal the criteria
/// set, values finite in [0,1] summing to ~1, selected is the argmax.
pub fn validate_answers(
    questions: &Map<String, Value>,
    answers: &Value,
) -> Result<(), BackendError> {
    let answers = answers
        .as_object()
        .ok_or_else(|| BackendError::InvalidResponse("answers is not an object".to_string()))?;
    for (qid, qdef) in questions {
        let answer = answers.get(qid).ok_or_else(|| {
            BackendError::InvalidResponse(format!("missing answer for question {qid:?}"))
        })?;
        if qdef.get("type").and_then(Value::as_str) != Some("choice") {
            continue;
        }
        let criteria_ids: Vec<String> = match qdef.get("criteria") {
            Some(Value::Object(m)) => m.keys().cloned().collect(),
            Some(Value::Array(items)) => items
                .iter()
                .filter_map(Value::as_str)
                .map(str::to_string)
                .collect(),
            _ => continue,
        };
        validate_choice_answer(qid, answer, &criteria_ids)?;
    }
    Ok(())
}

fn validate_choice_answer(qid: &str, answer: &Value, ids: &[String]) -> Result<(), BackendError> {
    let invalid = |why: &str| BackendError::InvalidResponse(format!("question {qid:?}: {why}"));
    let selected = answer
        .get("choice")
        .and_then(Value::as_str)
        .ok_or_else(|| invalid("no selected choice"))?;
    let probs = answer
        .get("probabilities")
        .and_then(Value::as_object)
        .ok_or_else(|| invalid("no probabilities"))?;
    let confidence = answer
        .get("confidence")
        .and_then(Value::as_f64)
        .ok_or_else(|| invalid("no confidence"))?;
    if !ids.contains(&selected.to_string()) {
        return Err(invalid("selected choice is not a declared criterion"));
    }
    let mut keys: Vec<&String> = probs.keys().collect();
    let mut want: Vec<&String> = ids.iter().collect();
    keys.sort();
    want.sort();
    if keys != want {
        return Err(invalid("probability keys do not match criteria"));
    }
    let mut sum = 0.0;
    let mut top = f64::NEG_INFINITY;
    for v in probs.values() {
        let n = v
            .as_f64()
            .ok_or_else(|| invalid("non-numeric probability"))?;
        if !(n.is_finite() && (0.0..=1.0).contains(&n)) {
            return Err(invalid("probability out of range"));
        }
        sum += n;
        top = top.max(n);
    }
    if !(confidence.is_finite() && (0.0..=1.0).contains(&confidence)) {
        return Err(invalid("confidence out of range"));
    }
    if (sum - 1.0).abs() >= 0.02 {
        return Err(invalid("probabilities do not sum to 1"));
    }
    let selected_p = probs[selected].as_f64().unwrap_or(f64::NEG_INFINITY);
    if selected_p < top - 1e-6 {
        return Err(invalid("selected choice is not the argmax"));
    }
    Ok(())
}

#[async_trait::async_trait]
impl PredictBackend for JevBackend {
    fn kind(&self) -> BackendKind {
        BackendKind::Jev
    }

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            kind: BackendKind::Jev,
            available: !self.api_key.is_empty(),
            detail: format!("typesafe {}", self.config.endpoint()),
        }
    }

    async fn preload(&self, _request: PreloadRequest) -> Result<(), BackendError> {
        Ok(())
    }

    async fn unload(&self, _target: UnloadTarget) -> Result<(), BackendError> {
        Ok(())
    }

    async fn predict(
        &self,
        request: &PredictRequest,
        _rendered: Option<&RenderedPrompt>,
    ) -> Result<BackendResult, BackendError> {
        self.predict_json(&request.state, &request.questions).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    fn test_questions() -> Map<String, Value> {
        serde_json::json!({
            "topic": {"type": "choice", "instructions": "Choose", "criteria": ["a", "b"]},
        })
        .as_object()
        .unwrap()
        .clone()
    }

    fn good_body() -> String {
        serde_json::json!({
            "model": "jev-latest",
            "answers": {
                "topic": {
                    "type": "choice", "choice": "a",
                    "probabilities": {"a": 0.7, "b": 0.3},
                    "confidence": 0.65,
                }
            },
            "usage": {"input_tokens": 10, "output_tokens": 0},
        })
        .to_string()
    }

    /// Minimal scripted HTTP server: returns one (status, body) per
    /// connection, records request bodies + hit count.
    struct Mock {
        addr: std::net::SocketAddr,
        hits: Arc<AtomicUsize>,
        bodies: Arc<std::sync::Mutex<Vec<String>>>,
    }

    fn serve(script: Vec<(u16, String)>) -> Mock {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let hits = Arc::new(AtomicUsize::new(0));
        let bodies = Arc::new(std::sync::Mutex::new(Vec::new()));
        let (h, b) = (hits.clone(), bodies.clone());
        std::thread::spawn(move || {
            for (status, body) in script {
                let (mut stream, _) = listener.accept().unwrap();
                let mut buf = vec![0u8; 65536];
                let n = stream.read(&mut buf).unwrap();
                let raw = String::from_utf8_lossy(&buf[..n]).to_string();
                let payload = raw.split("\r\n\r\n").nth(1).unwrap_or("").to_string();
                b.lock().unwrap().push(payload);
                h.fetch_add(1, Ordering::SeqCst);
                let reason = match status {
                    200 => "OK",
                    400 => "Bad Request",
                    429 => "Too Many Requests",
                    503 => "Service Unavailable",
                    _ => "Error",
                };
                let head = format!(
                    "HTTP/1.1 {status} {reason}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    body.len()
                );
                stream.write_all(head.as_bytes()).unwrap();
                stream.write_all(body.as_bytes()).unwrap();
            }
        });
        Mock { addr, hits, bodies }
    }

    fn client_for(mock: &Mock) -> JevBackend {
        JevBackend::with_key(
            JevConfig {
                base_url: format!("http://{}", mock.addr),
                backoff: vec![Duration::from_millis(1), Duration::from_millis(1)],
                ..JevConfig::default()
            },
            "test-key".to_string(),
        )
        .unwrap()
    }

    #[tokio::test]
    async fn success_sends_shared_schema_and_normalizes() {
        let mock = serve(vec![(200, good_body()), (200, good_body())]);
        let client = client_for(&mock);
        let out = client
            .predict_json(&serde_json::json!("hello"), &test_questions())
            .await
            .unwrap();
        assert_eq!(out.model, "jev-latest");
        assert_eq!(out.answers["topic"]["choice"], serde_json::json!("a"));
        assert_eq!(mock.hits.load(Ordering::SeqCst), 1);
        let sent: Value = serde_json::from_str(&mock.bodies.lock().unwrap()[0]).unwrap();
        assert_eq!(sent["model"], serde_json::json!("jev-latest"));
        assert_eq!(sent["state"], serde_json::json!("hello"));
        assert!(sent["questions"]["topic"].is_object());
        // Same path through the shared backend trait.
        let via_trait = PredictBackend::predict(
            &client,
            &PredictRequest {
                state: serde_json::json!("hello"),
                questions: test_questions(),
                checkpoint: jevalaya_router_core::Checkpoint::English,
                request_id: "t1".to_string(),
            },
            None,
        )
        .await
        .unwrap();
        assert_eq!(via_trait.answers["topic"]["choice"], serde_json::json!("a"));
    }

    #[tokio::test]
    async fn retries_429_then_succeeds_without_retrying_400() {
        let mock = serve(vec![(429, "{}".into()), (200, good_body())]);
        let client = client_for(&mock);
        client
            .predict_json(&serde_json::json!("hi"), &test_questions())
            .await
            .unwrap();
        assert_eq!(mock.hits.load(Ordering::SeqCst), 2);

        let mock = serve(vec![(400, "{}".into())]);
        let client = client_for(&mock);
        let err = client
            .predict_json(&serde_json::json!("hi"), &test_questions())
            .await
            .unwrap_err();
        assert!(matches!(err, BackendError::Upstream(Some(400), _)));
        assert_eq!(mock.hits.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn gives_up_after_three_attempts() {
        let mock = serve(vec![
            (503, "{}".into()),
            (503, "{}".into()),
            (503, "{}".into()),
        ]);
        let client = client_for(&mock);
        let err = client
            .predict_json(&serde_json::json!("hi"), &test_questions())
            .await
            .unwrap_err();
        assert!(matches!(err, BackendError::Upstream(Some(503), _)));
        assert_eq!(mock.hits.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn invalid_choice_answer_rejected() {
        let bad = serde_json::json!({
            "model": "jev-latest",
            "answers": {"topic": {"type": "choice", "choice": "zzz",
                "probabilities": {"a": 0.5, "b": 0.5}, "confidence": 0.5}},
            "usage": {},
        })
        .to_string();
        let mock = serve(vec![(200, bad)]);
        let client = client_for(&mock);
        let err = client
            .predict_json(&serde_json::json!("hi"), &test_questions())
            .await
            .unwrap_err();
        assert!(matches!(err, BackendError::InvalidResponse(_)));
    }

    #[tokio::test]
    async fn list_criteria_upgrade_to_dict() {
        // TypeSafe 422s on list criteria; the adapter rewrites them to
        // {label: null} before posting.
        let mock = serve(vec![(200, good_body())]);
        let client = client_for(&mock);
        client
            .predict_json(&serde_json::json!("hi"), &test_questions())
            .await
            .unwrap();
        let sent: Value = serde_json::from_str(&mock.bodies.lock().unwrap()[0]).unwrap();
        let crit = &sent["questions"]["topic"]["criteria"];
        assert_eq!(crit, &serde_json::json!({"a": null, "b": null}));
    }

    #[test]
    fn score_level_lists_are_never_upgraded() {
        // T029: score criteria are ordinal lists — rewriting them as dicts
        // is what the provider 422s. Only choice lists upgrade.
        let questions: Map<String, Value> = serde_json::json!({
            "sev": {"type": "score", "instructions": "How severe?",
                    "criteria": ["trivial", "annoying", "unusable"]},
            "topic": {"type": "choice", "instructions": "Pick",
                      "criteria": ["a", "b"]},
            "ok": {"type": "noul", "instructions": "OK?"},
        })
        .as_object()
        .unwrap()
        .clone();
        let out = jev_questions(&questions);
        assert_eq!(
            out["sev"]["criteria"],
            serde_json::json!(["trivial", "annoying", "unusable"])
        );
        assert_eq!(out["topic"]["criteria"], serde_json::json!({"a": null, "b": null}));
        assert!(out["ok"].get("criteria").is_none());
    }

    #[test]
    fn missing_key_fails_closed() {
        let err = JevBackend::with_key(JevConfig::default(), String::new()).unwrap_err();
        assert!(matches!(err, BackendError::MissingCredential(_)));
        // The env var *name* is config (safe to show); a key value must
        // never appear. Use a sentinel key and check it is absent.
        let key = "sk-test-sentinel-42";
        let backend = JevBackend::with_key(JevConfig::default(), key.to_string()).unwrap();
        assert!(!format!("{backend:?}").contains(key));
    }
}
