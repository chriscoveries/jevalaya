//! Structured routing events (DESIGN.md §Observability): one append-only
//! record per routed request, to JSONL or a local socket. Never carries
//! state, questions, auth headers, keys, or raw provider payloads. The
//! sink is bounded and non-blocking — observability cannot stall
//! inference; failed writes are counted and do not alter answers.

use std::io::Write;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{sync_channel, SyncSender};
use std::sync::Arc;

use serde::Serialize;
use serde_json::Value;

use crate::types::{BackendKind, Checkpoint};

/// One routed request. Field names are contract-stable — a future native
/// Mac app consumes this stream.
#[derive(Debug, Clone, Serialize)]
pub struct RoutingEvent {
    /// RFC3339 UTC timestamp.
    pub ts: String,
    /// Opaque caller/supplied id.
    pub request_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub backend: Option<BackendKind>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub checkpoint: Option<Checkpoint>,
    pub reason: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub token_count: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub margin: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub latency_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub queue_ms: Option<f64>,
    pub fallback: bool,
    pub escalated: bool,
    pub degraded: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cost: Option<Value>,
    pub available_backends: Vec<BackendKind>,
    /// HTTP status the request produced (or would produce).
    pub status: u16,
}

/// Where events go. `emit` must be non-blocking.
pub trait EventSink: Send + Sync {
    fn emit(&self, event: RoutingEvent);
}

/// Discards events — tests and `--no-events` operation.
pub struct NullSink;

impl EventSink for NullSink {
    fn emit(&self, _event: RoutingEvent) {}
}

/// In-memory sink for tests.
#[derive(Default)]
pub struct VecSink {
    pub events: std::sync::Mutex<Vec<RoutingEvent>>,
}

impl EventSink for VecSink {
    fn emit(&self, event: RoutingEvent) {
        self.events.lock().unwrap().push(event);
    }
}

/// Append-only JSONL sink over a bounded channel + writer thread.
/// `try_send` keeps emit non-blocking; overflow increments `dropped`.
pub struct JsonlSink {
    tx: SyncSender<String>,
    dropped: Arc<AtomicU64>,
    writer: Option<std::thread::JoinHandle<()>>,
}

impl JsonlSink {
    /// Open (create/truncate-then-append) `path`; spawn the writer thread.
    /// `capacity` bounds queued lines; overflow is counted, never stalls.
    pub fn open(path: &std::path::Path, capacity: usize) -> std::io::Result<Self> {
        let file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)?;
        Self::from_writer(file, capacity)
    }

    pub fn from_writer<W: Write + Send + 'static>(
        mut w: W,
        capacity: usize,
    ) -> std::io::Result<Self> {
        let (tx, rx) = sync_channel::<String>(capacity.max(1));
        let dropped = Arc::new(AtomicU64::new(0));
        let writer = std::thread::Builder::new()
            .name("jevalaya-events".to_string())
            .spawn(move || {
                while let Ok(line) = rx.recv() {
                    if writeln!(w, "{line}").and_then(|_| w.flush()).is_err() {
                        // Writer-side failures are counted via a side channel
                        // is overkill; the line is lost either way.
                        break;
                    }
                }
            })?;
        Ok(Self {
            tx,
            dropped,
            writer: Some(writer),
        })
    }

    /// Events dropped because the queue was full.
    pub fn dropped(&self) -> u64 {
        self.dropped.load(Ordering::Relaxed)
    }
}

impl EventSink for JsonlSink {
    fn emit(&self, event: RoutingEvent) {
        if let Ok(line) = serde_json::to_string(&event) {
            if self.tx.try_send(line).is_err() {
                self.dropped.fetch_add(1, Ordering::Relaxed);
            }
        }
    }
}

impl Drop for JsonlSink {
    fn drop(&mut self) {
        // Close the channel so the writer drains and exits.
        let (tx, _) = sync_channel::<String>(1);
        self.tx = tx;
        if let Some(h) = self.writer.take() {
            let _ = h.join();
        }
    }
}

/// Minimal RFC3339 UTC timestamp without pulling in a datetime crate.
pub fn rfc3339_now() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let days = secs / 86400;
    let rem = secs % 86400;
    let (h, m, s) = (rem / 3600, (rem % 3600) / 60, rem % 60);
    // Howard Hinnant's civil-from-days algorithm.
    let z = days as i64 + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = (z - era * 146097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let mo = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if mo <= 2 { y + 1 } else { y };
    format!("{y:04}-{mo:02}-{d:02}T{h:02}:{m:02}:{s:02}Z")
}
