"use strict";
// Nine identical asks: four per local backend, one paid API call.
// No retries, staggering, or fabricated timings.
// Dispatch concurrency is not admission concurrency (browser/server limits still apply).
const Q = new URLSearchParams(location.search);
const CFG = Object.assign({ url: "", token: "" }, window.JEVALAYA || {});
if (Q.get("url")) CFG.url = Q.get("url");
if (Q.get("token")) CFG.token = Q.get("token");
const API = (CFG.url || "") + "/predict";
const ITEMS = (CFG.url || "") + "/items?limit=200&offset=0";
const cv = document.getElementById("race"), cx = cv.getContext("2d");
const W = 720, H = 720, BURST_SIZE = 4, HOLD_MS = 1000, REQUEST_TIMEOUT_MS = 30000;
const LABELS = ["World", "Sports", "Business", "Sci/Tech"];
const BACKENDS = ["ane", "mlx", "jev"];
const BATCH_SIZE = { ane: BURST_SIZE, mlx: BURST_SIZE, jev: 1 };
const BACKEND_LABEL = { ane: "ANE", mlx: "MLX", jev: "API" };
const COLOR = { ane: "#58a6ff", mlx: "#3fb950", jev: "#d29922" };
const UI = {
  bg: "#0d1117", panel: "#161b22", text: "#f0f6fc", muted: "#8b949e", line: "#30363d", red: "#f85149",
  sans: '-apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
  mono: 'ui-monospace, "SFMono-Regular", Menlo, Consolas, monospace',
};
const questions = { topic: { type: "choice", instructions: "Classify the news article by topic.",
  criteria: { World: null, Sports: null, Business: null, "Sci/Tech": null } } };
const reducedMotion = matchMedia("(prefers-reduced-motion: reduce)");
const autostart = Q.get("autostart") !== "0";
const controllers = new Set();
let generation = 0, queue = [], current = null, roundNumber = 0;
let paused = false, loading = true, loadError = "", lastFrame = 0;
let scores = { ane: 0, mlx: 0, jev: 0 }, samples = { ane: [], mlx: [], jev: [] };
let sessionMaxMs = 0;

function headlineOf(text) {
  const i = text.search(/[.!?]\s/);
  return (i > 20 ? text.slice(0, i + 1) : text.slice(0, 140));
}
function reset() {
  const version = ++generation;
  for (const controller of controllers) controller.abort();
  controllers.clear();
  queue = []; current = null; roundNumber = 0; paused = false; loading = true; loadError = "";
  scores = { ane: 0, mlx: 0, jev: 0 }; samples = { ane: [], mlx: [], jev: [] };
  sessionMaxMs = 0;
  const controller = new AbortController(); controllers.add(controller);
  const timeout = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
  fetch(ITEMS, { signal: controller.signal })
    .then(r => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json(); })
    .then(data => {
      if (generation !== version) return;
      if (!Array.isArray(data.items)) throw new Error("Invalid article list");
      queue = data.items.filter(row => typeof row.text === "string" && LABELS.includes(row.label));
      if (!queue.length) throw new Error("No usable articles");
      loading = false;
    })
    .catch(error => {
      if (generation !== version) return;
      loading = false; loadError = error.name === "AbortError" ? "Article load timed out" : error.message;
    })
    .finally(() => { clearTimeout(timeout); controllers.delete(controller); });
}
function startRound() {
  if (paused || !queue.length || (current && current.lanes.some(lane => lane.state === "pending"))) return;
  const row = queue.shift(); queue.push(row);
  const round = {
    generation, number: ++roundNumber, row, text: headlineOf(row.text),
    firstReply: null, holdRemaining: null, settledCount: 0, startedAt: performance.now(),
    lanes: BACKENDS.map(key => ({ key, state: "pending", motionMs: 0,
      calls: Array.from({ length: BATCH_SIZE[key] }, (_, index) => ({
        key, id: index + 1, state: "pending", receipt: null, error: "",
      })),
    })),
  };
  current = round;
  // Interleave lanes; alternate who is launched first each round. All nine
  // fetches start in this turn, but neither HTTP/1 nor the shared cap promises
  // nine admitted requests. Do not hide a 429 by silently retrying it.
  const order = round.number % 2 ? round.lanes : [...round.lanes].reverse();
  for (let index = 0; index < BURST_SIZE; index++) {
    order.forEach(lane => {
      if (lane.calls[index]) dispatchCall(round, lane, lane.calls[index]);
    });
  }
}
function isCurrent(round) { return generation === round.generation && current === round; }
function dispatchCall(round, lane, call) {
  const body = {
    state: round.text,
    questions,
    backend: lane.key,
    model: "multilingual",
    request_id: `race-${round.generation}-${round.number}-${lane.key}-${call.id}`,
  };
  const controller = new AbortController(); controllers.add(controller);
  let timedOut = false;
  const timeout = setTimeout(() => { timedOut = true; controller.abort(); }, REQUEST_TIMEOUT_MS);
  const t0 = performance.now(); call.startedAt = t0;
  fetch(API, { method: "POST",
    headers: { "Authorization": "Bearer " + CFG.token, "Content-Type": "application/json" },
    body: JSON.stringify(body), signal: controller.signal })
    .then(r => {
      if (!r.ok) throw Object.assign(new Error(`HTTP ${r.status}`), { status: r.status });
      return r.json();
    })
    .then(data => {
      if (!isCurrent(round)) return;
      const ans = data.answers?.topic, route = data.routing;
      if (!ans || !LABELS.includes(ans.choice) || !route || !Object.hasOwn(COLOR, route.backend)
          || !Number.isFinite(route.latency_ms) || route.latency_ms < 0) throw new Error("Invalid receipt");
      call.receipt = { backend: route.backend, ms: Math.round(route.latency_ms),
                       conf: ans.confidence, ckpt: route.checkpoint || "—" };
      call.serverMs = route.latency_ms;
      // Only validated, current-generation receipts can grow the shared axis.
      // Keep the exact maximum through subsequent rounds; R starts a session.
      sessionMaxMs = Math.max(sessionMaxMs, call.serverMs);
      call.choice = ans.choice;
      call.correct = ans.choice === round.row.label;
      call.escalated = !!route.escalated || (route.backend === "jev" && lane.key !== "jev");
      call.escalationError = !!route.escalation_error;
      call.fallback = !!route.fallback || (!call.escalated && route.backend !== lane.key);
      call.native = route.backend === lane.key && (lane.key === "jev" || route.checkpoint === "multilingual")
        && !call.fallback && !call.escalated;
      call.state = "done";
      if (!round.firstReply) round.firstReply = lane.key;
      // Rerouted requests remain visible, but never enter native stats. An
      // explicit API answer is not a local sample or a local escalation.
      if (call.native) {
        if (call.correct) scores[lane.key]++;
        samples[lane.key].push(call.serverMs);
      }
    })
    .catch(error => {
      if (!isCurrent(round)) return;
      call.state = error.status === 429 ? "busy" : "error";
      call.error = timedOut ? "Timed out" : error.name === "AbortError" ? "Cancelled" : error.message;
    })
    .finally(() => {
      clearTimeout(timeout); controllers.delete(controller);
      if (!isCurrent(round)) return;
      call.finishedAt = performance.now(); call.rtt = call.finishedAt - t0;
      call.arrivalOrder = ++round.settledCount;
      lane.state = lane.calls.every(other => other.state !== "pending") ? "done" : "pending";
      if (round.lanes.every(other => other.state !== "pending")) round.holdRemaining = HOLD_MS;
    });
}
function span(values) { return values.length < 2 ? null : Math.max(...values) - Math.min(...values); }
function laneStats(lane) {
  const settled = lane.calls.filter(call => call.state !== "pending");
  const measured = settled.filter(call => call.state === "done");
  return { settled: settled.length, measured: measured.length,
    busy: settled.filter(call => call.state === "busy").length,
    serverSpan: span(measured.map(call => call.serverMs)),
    arrivalSpan: span(settled.map(call => call.finishedAt)),
  };
}
function median(values) {
  if (!values.length) return null;
  const sorted = [...values].sort((a, b) => a - b), mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}
function frame(now) {
  requestAnimationFrame(frame);
  const dt = Math.min(100, now - (lastFrame || now)); lastFrame = now;
  if (!paused) {
    if (!current && !loading && !loadError && autostart) startRound();
    else if (current) {
      current.lanes.forEach(lane => { if (lane.state === "pending") lane.motionMs += dt; });
      if (current.holdRemaining !== null && autostart) {
        current.holdRemaining -= dt;
        if (current.holdRemaining <= 0) startRound();
      }
    }
  }
  draw(now);
}
function text(value, x, y, size = 14, color = UI.text, mono = false, align = "left", weight = 400) {
  cx.font = `${weight} ${size}px ${mono ? UI.mono : UI.sans}`;
  cx.fillStyle = color; cx.textAlign = align; cx.textBaseline = "alphabetic";
  cx.fillText(String(value), x, y);
}
function line(x1, y1, x2, y2, color = UI.line, width = 1.5) {
  cx.beginPath(); cx.moveTo(x1, y1); cx.lineTo(x2, y2);
  cx.strokeStyle = color; cx.lineWidth = width; cx.stroke();
}
function dot(x, y, color, radius = 5) {
  cx.beginPath(); cx.arc(x, y, radius, 0, Math.PI * 2); cx.fillStyle = color; cx.fill();
}
function fitText(value, width) {
  const chars = Array.from(String(value));
  if (cx.measureText(chars.join("")).width <= width) return chars.join("");
  while (chars.length && cx.measureText(chars.join("") + "…").width > width) chars.pop();
  return chars.join("") + "…";
}
function actualCalls(key, settledOnly = false) {
  if (!current) return [];
  return current.lanes.filter(lane => !settledOnly || lane.state === "done")
    .flatMap(lane => lane.calls).filter(call => call.state === "done" && call.receipt.backend === key)
    .sort((a, b) => a.arrivalOrder - b.arrivalOrder);
}
function redirected(call) {
  return !call.native;
}
function lanePresentation(lane) {
  // Rows and colors identify the request, never the destination. Hollow rings
  // and destination labels disclose reroutes without borrowing another hue.
  const calls = [...lane.calls].sort((a, b) => (a.arrivalOrder || Infinity) - (b.arrivalOrder || Infinity) || a.id - b.id);
  const received = calls.filter(call => call.state === "done");
  return { calls, received, serverSpan: span(received.map(call => call.serverMs)) };
}
function playfieldLayout() {
  return { laneY: [278, 404, 530], rowStep: 18, cardHeight: 110, font: 23, lineHeight: 24 };
}
function sharedScale() {
  return sessionMaxMs;
}
function headline(value, layout) {
  cx.beginPath(); cx.roundRect(32, 116, 656, layout.cardHeight, 12);
  cx.fillStyle = UI.panel; cx.fill(); cx.strokeStyle = UI.line; cx.lineWidth = 1.5; cx.stroke();
  cx.font = `500 ${layout.font}px ${UI.sans}`; cx.fillStyle = UI.text; cx.textAlign = "left";
  const lines = []; let row = "";
  for (const word of value.split(/\s+/)) {
    const next = row ? row + " " + word : word;
    if (row && cx.measureText(next).width > 608) { lines.push(row); row = word; }
    else row = next;
  }
  if (row) lines.push(row);
  const shown = lines.slice(0, 4), top = 125 + layout.cardHeight / 2 - (shown.length - 1) * layout.lineHeight / 2;
  shown.forEach((value, i) => cx.fillText(fitText(value + (i === 3 && lines.length > 4 ? " …" : ""), 608), 56, top + i * layout.lineHeight));
}
function callNote(call) {
  if (call.state === "pending") return "";
  if (call.state === "busy") return "busy · 429";
  if (call.state === "error") return call.error;
  const identity = call.receipt.backend !== call.key ? `→${BACKEND_LABEL[call.receipt.backend]}`
    : call.escalationError ? "Jev failed" : call.fallback ? "fallback"
    : !call.native ? call.receipt.ckpt : "";
  return [`${call.receipt.ms} ms`, identity, call.correct ? "" : "wrong"].filter(Boolean).join(" · ");
}
function callColor(call) {
  return call.state === "done" ? COLOR[call.key] : UI.muted;
}
function drawReceiptMarker(call, x, y) {
  // Hue always belongs to the requested row, even on redirects or mistakes.
  const color = COLOR[call.key];
  if (!redirected(call)) return dot(x, y, color, 4);
  // Transparent ring, not a filled fake-native dot. Coincident native dots
  // remain visible inside the ring when two real timings are exactly equal.
  cx.beginPath(); cx.arc(x, y, 7, 0, Math.PI * 2);
  cx.strokeStyle = color; cx.lineWidth = 2.5; cx.stroke();
}
function drawConnections(lane, y, index) {
  if (lane.state !== "done") return;
  const choices = new Set(lane.calls.filter(call => call.state === "done").map(call => call.choice));
  choices.forEach(choice => {
    const bin = LABELS.indexOf(choice), targetX = 109.5 + bin * 167 + (index - 1) * 6;
    const elbowX = [708, 700, 692][index];
    const elbowY = [611, 615, 619][index];
    cx.beginPath(); cx.moveTo(693, y); cx.lineTo(elbowX, y); cx.lineTo(elbowX, elbowY);
    cx.lineTo(targetX, elbowY); cx.lineTo(targetX, 622);
    cx.strokeStyle = COLOR[lane.key] + "88"; cx.lineWidth = 1.5; cx.stroke();
  });
}
function drawTimeAxis(key, y, scale) {
  const start = 112, end = 688;
  line(start, y, end, y, COLOR[key] + "66");
  for (let tick = 0; tick <= 4; tick++) {
    const x = start + (end - start) * tick / 4;
    line(x, y - 4, x, y + 4, UI.line);
  }
  // The right endpoint is the exact session maximum, not a rounded bucket.
  text(`max ${Math.round(scale)} ms`, end, y - 9, 16, UI.muted, true, "right");
}
function drawLane(lane, index, now, scale, layout) {
  const y = layout.laneY[index], start = 112, end = 688;
  const view = lanePresentation(lane), pending = lane.state === "pending";
  const wrong = view.received.filter(call => !call.correct);
  if (wrong.length && !reducedMotion.matches) {
    const pulse = Math.max(0, 1 - (now - Math.max(...wrong.map(call => call.finishedAt))) / 350);
    cx.fillStyle = COLOR[lane.key] + "14"; cx.globalAlpha = pulse;
    cx.fillRect(24, y - 50, 672, 125); cx.globalAlpha = 1;
  }
  text(BACKEND_LABEL[lane.key], 32, y - 22, 24, COLOR[lane.key], true, "left", 600);
  const elapsed = Math.floor(Math.max(0, now - current.startedAt));
  // Local spans and the API's single latency describe this requested row,
  // including redirects. Native-only p50/score remain separate above.
  const apiReceipt = lane.key === "jev" && view.received[0];
  const metric = lane.key === "jev" ? apiReceipt ? apiReceipt.serverMs : null : view.serverSpan;
  const server = metric === null ? "—" : Math.round(metric);
  text(`${pending ? elapsed : server} ms`, 360, y - 22, 34, UI.text, true, "center", 500);
  text(pending ? "elapsed" : lane.key === "jev" ? "server ms" : "server span", end, y - 33, 15, UI.muted, false, "right");
  drawTimeAxis(lane.key, y, scale);
  // A small pulse at the lane label means active, not simulated completion.
  if (pending) {
    cx.globalAlpha = paused || reducedMotion.matches ? 0.6 : 0.55 + 0.35 * Math.sin(lane.motionMs / 180);
    dot(46, y, COLOR[lane.key], 3); cx.globalAlpha = 1;
  }
  const labels = [];
  view.calls.forEach((call, row) => {
    const labelY = y + 20 + row * layout.rowStep;
    const color = callColor(call);
    const isRedirect = call.state === "done" && redirected(call);
    const size = isRedirect ? 17 : 16, weight = isRedirect ? 650 : 500;
    cx.font = `${weight} ${size}px ${UI.mono}`;
    const label = fitText(callNote(call), end - start - 8);
    const width = cx.measureText(label).width;
    if (call.state !== "done") {
      // Busy and errors have no measured latency and stay off the time axis.
      if (call.state !== "pending") dot(start - 18, labelY - 5, color, 3);
      labels.push({ label, x: start, y: labelY, width, color, size, weight });
      return;
    }
    // Empty/zero-only sessions have a zero-width time domain. Avoid 0/0,
    // without imposing an artificial floor once a positive receipt arrives.
    const markerX = start + (end - start) * call.serverMs / (scale || 1);
    const labelX = Math.max(start, Math.min(markerX + 10, end - width));
    const leaderX = labelX >= markerX ? labelX - 4 : labelX + width + 4;
    line(markerX, y + 8, markerX, labelY - 5, color + "55", 1);
    line(markerX, labelY - 5, leaderX, labelY - 5, color + "55", 1);
    drawReceiptMarker(call, markerX, y);
    labels.push({ label, x: labelX, y: labelY, width, color, size, weight });
  });
  // Draw labels after every leader so clustered response stems cannot strike
  // through an earlier label. Exact marker positions remain unchanged.
  labels.forEach(({ label, x, y, width, color, size, weight }) => {
    if (!label) return;
    cx.fillStyle = UI.bg; cx.fillRect(x - 2, y - 15, width + 4, 19);
    text(label, x, y, size, color, true, "left", weight);
  });
  drawConnections(lane, y, index);
}
function binHighlights(label) {
  return BACKENDS
    .map(key => ({ key, calls: actualCalls(key, true).filter(call => call.choice === label) }))
    .filter(lane => lane.calls.length);
}
function drawBin(label, index) {
  const x = 32 + index * 167, y = 622, width = 155, height = 70;
  const highlights = binHighlights(label), active = highlights.length > 0;
  // Bin colors identify actual answerers; connectors originate on request rows.
  if (active) {
    const laneWidth = width / highlights.length;
    highlights.forEach((lane, i) => {
      cx.save(); cx.shadowColor = COLOR[lane.calls[0].receipt.backend]; cx.shadowBlur = 22;
      cx.fillStyle = cx.shadowColor;
      cx.beginPath(); cx.roundRect(x + i * laneWidth, y, laneWidth, height, 12); cx.fill(); cx.restore();
    });
    cx.save(); cx.beginPath(); cx.roundRect(x, y, width, height, 12); cx.clip();
    highlights.forEach((lane, i) => lane.calls.forEach((call, j) => {
      cx.fillStyle = COLOR[call.receipt.backend];
      cx.fillRect(x + i * laneWidth + j * laneWidth / lane.calls.length, y, laneWidth / lane.calls.length + 0.5, height);
    }));
    cx.restore();
  } else {
    cx.fillStyle = UI.panel; cx.beginPath(); cx.roundRect(x, y, width, height, 12); cx.fill();
  }
  cx.beginPath(); cx.roundRect(x, y, width, height, 12);
  cx.strokeStyle = highlights.some(lane => lane.calls.some(call => !call.correct)) ? UI.red : active ? "#f0f6fc55" : UI.line;
  cx.lineWidth = active ? 2 : 1.5; cx.stroke();
  text(label, x + width / 2, y + 43, 22, active ? UI.bg : UI.muted, false, "center", active ? 750 : 500);
}
function draw(now = performance.now()) {
  cx.fillStyle = UI.bg; cx.fillRect(0, 0, W, H);
  text("jevalaya / burst race", 32, 42, 20, UI.muted, false, "left", 500);
  if (paused) text("Paused", 688, 42, 18, UI.text, false, "right");
  BACKENDS.forEach((key, i) => {
    const x = 32 + i * 224, p50 = median(samples[key]);
    text(`${BACKEND_LABEL[key]} ${scores[key]}`, x, 76, 24, COLOR[key], true, "left", 500);
    text(`p50 ${p50 === null ? "—" : Math.round(p50)} ms`, x, 99, 17, UI.muted, true);
  });
  const layout = playfieldLayout();
  headline(current ? current.text : loadError ? `Could not load articles: ${loadError}` : "Waiting for the first headline…", layout);
  const scale = sharedScale();
  if (current) current.lanes.forEach((lane, i) => drawLane(lane, i, now, scale, layout));
  else BACKENDS.forEach((key, i) => {
    const y = layout.laneY[i];
    text(BACKEND_LABEL[key], 32, y - 22, 24, COLOR[key], true, "left", 600);
    drawTimeAxis(key, y, scale);
  });
  LABELS.forEach(drawBin);
}
addEventListener("keydown", e => {
  if (e.repeat) return;
  if (e.code === "Space") { paused = !paused; e.preventDefault(); }
  if (e.code === "KeyR") reset();
});
reset();
requestAnimationFrame(frame);
