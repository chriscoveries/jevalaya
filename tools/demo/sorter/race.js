"use strict";
// Two identical classification requests, not a synthesized timing comparison.
// Marker crawl indicates activity, never an estimate of inference progress.
const Q = new URLSearchParams(location.search);
const CFG = Object.assign({ url: "", token: "" }, window.JEVALAYA || {});
if (Q.get("url")) CFG.url = Q.get("url");
if (Q.get("token")) CFG.token = Q.get("token");
const API = (CFG.url || "") + "/predict";
const ITEMS = (CFG.url || "") + "/items?limit=200&offset=0";
const cv = document.getElementById("race"), cx = cv.getContext("2d");
const W = 960, H = 720, HOLD_MS = 1000, REQUEST_TIMEOUT_MS = 30000;
const LABELS = ["World", "Sports", "Business", "Sci/Tech"];
const BACKENDS = ["ane", "mlx"];
const COLOR = { ane: "#58a6ff", mlx: "#3fb950", jev: "#d29922" };
const UI = {
  bg: "#0d1117", text: "#f0f6fc", muted: "#8b949e", line: "#30363d", red: "#f85149",
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
let scores = { ane: 0, mlx: 0 }, samples = { ane: [], mlx: [] };

function headlineOf(text) {
  const i = text.search(/[.!?]\s/);
  return (i > 20 ? text.slice(0, i + 1) : text.slice(0, 140));
}
function reset() {
  const version = ++generation;
  for (const controller of controllers) controller.abort();
  controllers.clear();
  queue = []; current = null; roundNumber = 0; paused = false; loading = true; loadError = "";
  scores = { ane: 0, mlx: 0 }; samples = { ane: [], mlx: [] };
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
    firstReply: null, holdRemaining: null,
    lanes: BACKENDS.map(key => ({ key, state: "pending", motionMs: 0, receipt: null, error: "" })),
  };
  current = round;
  // No await between dispatches: both POSTs are outstanding at the same time.
  round.lanes.forEach(lane => dispatchLane(round, lane));
}
function isCurrent(round) { return generation === round.generation && current === round; }
function dispatchLane(round, lane) {
  const body = {
    state: round.text,
    questions,
    backend: lane.key,
    model: "multilingual",
    request_id: `race-${round.generation}-${round.number}-${lane.key}`,
  };
  const controller = new AbortController(); controllers.add(controller);
  let timedOut = false;
  const timeout = setTimeout(() => { timedOut = true; controller.abort(); }, REQUEST_TIMEOUT_MS);
  const t0 = performance.now(); lane.startedAt = t0;
  fetch(API, { method: "POST",
    headers: { "Authorization": "Bearer " + CFG.token, "Content-Type": "application/json" },
    body: JSON.stringify(body), signal: controller.signal })
    .then(r => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json(); })
    .then(data => {
      if (!isCurrent(round)) return;
      const ans = data.answers?.topic, route = data.routing;
      if (!ans || !LABELS.includes(ans.choice) || !route || !Object.hasOwn(COLOR, route.backend)
          || !Number.isFinite(route.latency_ms) || route.latency_ms < 0) throw new Error("Invalid receipt");
      lane.receipt = { backend: route.backend, ms: Math.round(route.latency_ms),
                       conf: ans.confidence, ckpt: route.checkpoint || "—" };
      lane.serverMs = route.latency_ms;
      lane.choice = ans.choice;
      lane.correct = ans.choice === round.row.label;
      lane.escalated = !!route.escalated || route.backend === "jev";
      lane.escalationError = !!route.escalation_error;
      lane.fallback = !!route.fallback || (!lane.escalated && route.backend !== lane.key);
      lane.native = route.backend === lane.key && route.checkpoint === "multilingual"
        && !lane.fallback && !lane.escalated;
      lane.finishedAt = performance.now(); lane.rtt = lane.finishedAt - t0;
      lane.state = "done";
      if (!round.firstReply) round.firstReply = lane.key;
      // Rerouted requests remain visible, but are not ANE/MLX hardware samples.
      if (lane.native) {
        if (lane.correct) scores[lane.key]++;
        samples[lane.key].push(lane.serverMs);
      }
    })
    .catch(error => {
      if (!isCurrent(round)) return;
      lane.state = "error"; lane.finishedAt = performance.now(); lane.rtt = lane.finishedAt - t0;
      lane.error = timedOut ? "Timed out" : error.name === "AbortError" ? "Cancelled" : error.message;
    })
    .finally(() => {
      clearTimeout(timeout); controllers.delete(controller);
      if (isCurrent(round) && round.lanes.every(other => other.state !== "pending")) round.holdRemaining = HOLD_MS;
    });
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
function headline(value) {
  cx.font = `500 28px ${UI.sans}`; cx.fillStyle = UI.text; cx.textAlign = "left";
  const lines = []; let row = "";
  for (const word of value.split(/\s+/)) {
    const next = row ? row + " " + word : word;
    if (row && cx.measureText(next).width > 864) { lines.push(row); row = word; }
    else row = next;
  }
  if (row) lines.push(row);
  lines.slice(0, 3).forEach((value, i) => cx.fillText(fitText(value + (i === 2 && lines.length > 3 ? " …" : ""), 864), 48, 148 + i * 36));
}
function laneNote(lane) {
  if (lane.state === "pending") return "waiting";
  if (lane.state === "error") return lane.error;
  const actual = lane.receipt.backend.toUpperCase();
  const identity = lane.native ? "" : [actual,
    lane.fallback ? "fallback" : "",
    lane.escalated ? `escalated→jev${lane.escalationError ? " failed" : ""}` : "",
    !lane.fallback && !lane.escalated ? `checkpoint ${lane.receipt.ckpt}` : "",
  ].filter(Boolean).join(" · ");
  return [identity, lane.choice, lane.correct ? "correct" : "wrong"].filter(Boolean).join(" · ");
}
function drawLane(lane, index, now) {
  const y = 350 + index * 132, start = 152, end = 870, mid = (start + end) / 2;
  const completed = lane.state !== "pending";
  const wrong = lane.state === "done" && !lane.correct;
  const color = wrong || lane.state === "error" ? UI.red : lane.receipt ? COLOR[lane.receipt.backend] : COLOR[lane.key];
  if (wrong && !reducedMotion.matches) {
    const pulse = Math.max(0, 1 - (now - lane.finishedAt) / 350);
    cx.fillStyle = UI.red + "14"; cx.globalAlpha = pulse;
    cx.fillRect(40, y - 63, 840, 104); cx.globalAlpha = 1;
  }
  text(lane.key.toUpperCase(), 48, y + 8, 24, COLOR[lane.key], true, "left", 500);
  line(start, y, end, y, color + "66");
  const progress = completed ? 1 : reducedMotion.matches ? 0 : 0.88 * (1 - Math.exp(-lane.motionMs / 1800));
  dot(start + (end - start) * progress, y, color, 6);
  const ms = lane.state === "done" ? lane.receipt.ms : lane.state === "pending" ? Math.floor(Math.max(0, now - lane.startedAt)) : "—";
  text(ms, mid, y - 16, 38, lane.state === "error" ? UI.muted : UI.text, true, "center", 500);
  text(lane.state === "pending" ? "elapsed ms" : "server ms", mid, y + 27, 13, UI.muted, true, "center");
  if (completed) text(`rtt ${Math.round(lane.rtt)} ms`, end, y + 27, 12, UI.muted, true, "right");
  cx.font = `400 13px ${UI.sans}`;
  text(fitText(laneNote(lane), 550), start, y - 58, 13, wrong || lane.state === "error" ? UI.red : lane.native === false ? COLOR.jev : UI.muted);
  if (current.firstReply === lane.key) text("first reply", end, y - 58, 12, UI.muted, false, "right");
  if (lane.state === "done") {
    const bin = LABELS.indexOf(lane.choice), targetX = 150 + bin * 220 + (index ? 5 : -5);
    const elbowX = index ? 894 : 914, elbowY = index ? 577 : 557;
    cx.beginPath(); cx.moveTo(end, y); cx.lineTo(elbowX, y); cx.lineTo(elbowX, elbowY);
    cx.lineTo(targetX, elbowY); cx.lineTo(targetX, 602);
    cx.strokeStyle = color + "aa"; cx.lineWidth = 1.5; cx.stroke();
    line(targetX - 3, 598, targetX, 602, color); line(targetX + 3, 598, targetX, 602, color);
  }
}
function draw(now = performance.now()) {
  cx.fillStyle = UI.bg; cx.fillRect(0, 0, W, H);
  text("jevalaya / head-to-head", 48, 41, 15, UI.muted, false, "left", 500);
  text("correct answers · native server p50", 912, 41, 12, UI.muted, false, "right");
  BACKENDS.forEach((key, i) => {
    const x = 48 + i * 440, p50 = median(samples[key]);
    text(`${key.toUpperCase()} ${scores[key]}`, x, 83, 24, COLOR[key], true, "left", 500);
    text(`p50 ${p50 === null ? "—" : Math.round(p50)} ms`, x + 169, 81, 16, UI.muted, true);
  });
  line(48, 105, 912, 105);
  headline(current ? current.text : loadError ? "Could not load articles" : "Waiting for the first headline…");
  text(loadError || "Same headline · multilingual checkpoint · concurrent requests", 48, 254, 13, UI.muted);
  if (current) current.lanes.forEach((lane, i) => drawLane(lane, i, now));
  else BACKENDS.forEach((key, i) => {
    const y = 350 + i * 132;
    text(key.toUpperCase(), 48, y + 8, 24, COLOR[key], true, "left", 500);
    line(152, y, 870, y); dot(152, y, COLOR[key]);
  });
  LABELS.forEach((label, i) => {
    const x = 48 + i * 220;
    cx.beginPath(); cx.roundRect(x, 610, 204, 68, 6);
    cx.strokeStyle = UI.line; cx.lineWidth = 1.5; cx.stroke();
    text(label, x + 102, 652, 21, UI.text, false, "center", 500);
  });
  text(paused ? "Paused · in-flight calls continue" : "Space pause rounds   ·   R reset", 48, 707, 12, UI.muted);
  text("Fallbacks / escalations excluded from native stats", 912, 707, 11, UI.muted, false, "right");
}
addEventListener("keydown", e => {
  if (e.repeat) return;
  if (e.code === "Space") { paused = !paused; e.preventDefault(); }
  if (e.code === "KeyR") reset();
});
reset();
requestAnimationFrame(frame);
