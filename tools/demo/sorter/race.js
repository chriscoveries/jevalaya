"use strict";
// Eight identical asks, four per backend. No retries, staggering, or fabricated timings.
// Dispatch concurrency is not admission concurrency (browser/server limits still apply).
const Q = new URLSearchParams(location.search);
const CFG = Object.assign({ url: "", token: "" }, window.JEVALAYA || {});
if (Q.get("url")) CFG.url = Q.get("url");
if (Q.get("token")) CFG.token = Q.get("token");
const API = (CFG.url || "") + "/predict";
const ITEMS = (CFG.url || "") + "/items?limit=200&offset=0";
const cv = document.getElementById("race"), cx = cv.getContext("2d");
const W = 960, H = 720, BURST_SIZE = 4, HOLD_MS = 1000, REQUEST_TIMEOUT_MS = 30000;
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
    firstReply: null, holdRemaining: null, settledCount: 0, startedAt: performance.now(),
    lanes: BACKENDS.map(key => ({ key, state: "pending", motionMs: 0,
      calls: Array.from({ length: BURST_SIZE }, (_, index) => ({
        key, id: index + 1, state: "pending", receipt: null, error: "",
      })),
    })),
  };
  current = round;
  // Interleave lanes; alternate who is launched first each round. All eight
  // fetches start in this turn, but neither HTTP/1 nor the shared cap promises
  // eight admitted requests. Do not hide a 429 by silently retrying it.
  const order = round.number % 2 ? round.lanes : [...round.lanes].reverse();
  for (let index = 0; index < BURST_SIZE; index++) {
    order.forEach(lane => dispatchCall(round, lane, lane.calls[index]));
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
      call.choice = ans.choice;
      call.correct = ans.choice === round.row.label;
      call.escalated = !!route.escalated || route.backend === "jev";
      call.escalationError = !!route.escalation_error;
      call.fallback = !!route.fallback || (!call.escalated && route.backend !== lane.key);
      call.native = route.backend === lane.key && route.checkpoint === "multilingual"
        && !call.fallback && !call.escalated;
      call.state = "done";
      if (!round.firstReply) round.firstReply = lane.key;
      // Rerouted requests remain visible, but are not ANE/MLX hardware samples.
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
function serverScale(round) {
  const max = Math.max(300, ...round.lanes.flatMap(lane => lane.calls.map(call => call.serverMs || 0)));
  const step = 10 ** Math.floor(Math.log10(max));
  return Math.ceil(max / step) * step;
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
function callNote(call) {
  if (call.state === "pending") return `#${call.id} waiting`;
  if (call.state === "busy") return `#${call.id} busy · HTTP 429 · no server ms`;
  if (call.state === "error") return `#${call.id} ${call.error} · no server ms`;
  const actual = call.receipt.backend.toUpperCase();
  const identity = call.native ? "" : [actual,
    call.fallback ? "fallback" : "",
    call.escalated ? `escalated→jev${call.escalationError ? " failed" : ""}` : "",
    !call.fallback && !call.escalated ? `checkpoint ${call.receipt.ckpt}` : "",
  ].filter(Boolean).join(" · ");
  return [`#${call.id} ${call.receipt.ms} ms`, identity, call.choice, call.correct ? "correct" : "wrong"].filter(Boolean).join(" · ");
}
function callColor(call) {
  if (call.state === "pending" || call.state === "busy") return UI.muted;
  if (call.state === "error" || !call.correct) return UI.red;
  return COLOR[call.receipt.backend];
}
function drawLane(lane, index, now, scale) {
  const y = 324 + index * 156, start = 152, end = 912;
  const stats = laneStats(lane), pending = lane.state === "pending";
  const wrong = lane.calls.filter(call => call.state === "done" && !call.correct);
  if (wrong.length && !reducedMotion.matches) {
    const pulse = Math.max(0, 1 - (now - Math.max(...wrong.map(call => call.finishedAt))) / 350);
    cx.fillStyle = UI.red + "14"; cx.globalAlpha = pulse;
    cx.fillRect(40, y - 33, 880, 123); cx.globalAlpha = 1;
  }
  text(lane.key.toUpperCase(), 48, y - 17, 24, COLOR[lane.key], true, "left", 500);
  text(`${stats.settled}/4`, 111, y - 18, 12, UI.muted, true);
  const arrival = stats.arrivalSpan === null ? "—" : Math.round(stats.arrivalSpan);
  const elapsed = Math.floor(Math.max(0, now - current.startedAt));
  text(pending ? `${elapsed} elapsed ms` : `first→last ${arrival} ms rtt`, 190, y - 19, 14, UI.text, true);
  const server = stats.serverSpan === null ? "—" : Math.round(stats.serverSpan);
  text(`server span ${server} ms${stats.busy ? ` · ${stats.busy} busy` : ""}`, end, y - 19, 14, UI.muted, true, "right");
  line(start, y, end, y, COLOR[lane.key] + "66");
  for (let tick = 0; tick <= 4; tick++) {
    const x = start + (end - start) * tick / 4;
    line(x, y - 4, x, y + 4, UI.line);
  }
  // A small pulse at the lane label means active, not simulated completion.
  if (pending) {
    cx.globalAlpha = paused || reducedMotion.matches ? 0.6 : 0.55 + 0.35 * Math.sin(lane.motionMs / 180);
    dot(64, y, COLOR[lane.key], 3); cx.globalAlpha = 1;
  }
  const ordered = [...lane.calls].sort((a, b) => (a.arrivalOrder || Infinity) - (b.arrivalOrder || Infinity) || a.id - b.id);
  const labels = ordered.map((call, row) => {
    const labelY = y + 25 + row * 20, color = callColor(call);
    cx.font = `400 13px ${UI.mono}`;
    const label = fitText(callNote(call), end - start - 16), width = cx.measureText(label).width;
    if (call.state !== "done") {
      // Busy/errors have no server receipt: never plot them as zero ms.
      dot(start - 18, labelY - 4, color, 3);
      return { label, x: start, y: labelY, width, color };
    }
    const markerX = start + (end - start) * call.serverMs / scale;
    const labelX = Math.max(start, Math.min(markerX + 10, end - width));
    const leaderX = labelX >= markerX ? labelX - 4 : labelX + width + 4;
    line(markerX, y + 8, markerX, labelY - 5, color + "55", 1);
    line(markerX, labelY - 5, leaderX, labelY - 5, color + "55", 1);
    dot(markerX, y, color, 4);
    return { label, x: labelX, y: labelY, width, color };
  });
  // Draw labels after every leader so clustered response stems cannot strike
  // through an earlier label. Exact marker positions remain unchanged.
  labels.forEach(({ label, x, y, width, color }) => {
    cx.fillStyle = UI.bg; cx.fillRect(x - 2, y - 12, width + 4, 16);
    text(label, x, y, 13, color, true);
  });
  // Connect each returned category once; the bins retain the per-lane count.
  const choices = new Map();
  lane.calls.filter(call => call.state === "done").forEach(call => choices.set(call.choice, callColor(call)));
  choices.forEach((color, choice) => {
    const bin = LABELS.indexOf(choice), targetX = 150 + bin * 220 + (index ? 5 : -5);
    const elbowX = index ? 926 : 936, elbowY = index ? 596 : 587;
    cx.beginPath(); cx.moveTo(end + 5, y); cx.lineTo(elbowX, y); cx.lineTo(elbowX, elbowY);
    cx.lineTo(targetX, elbowY); cx.lineTo(targetX, 602);
    cx.strokeStyle = color + "66"; cx.lineWidth = 1.5; cx.stroke();
    line(targetX - 3, 598, targetX, 602, color); line(targetX + 3, 598, targetX, 602, color);
  });
}
function draw(now = performance.now()) {
  cx.fillStyle = UI.bg; cx.fillRect(0, 0, W, H);
  text("jevalaya / burst race", 48, 41, 15, UI.muted, false, "left", 500);
  text("correct answers · native server p50", 912, 41, 12, UI.muted, false, "right");
  BACKENDS.forEach((key, i) => {
    const x = 48 + i * 440, p50 = median(samples[key]);
    text(`${key.toUpperCase()} ${scores[key]}`, x, 83, 24, COLOR[key], true, "left", 500);
    text(`p50 ${p50 === null ? "—" : Math.round(p50)} ms`, x + 169, 81, 16, UI.muted, true);
  });
  line(48, 105, 912, 105);
  headline(current ? current.text : loadError ? "Could not load articles" : "Waiting for the first headline…");
  text(loadError || "4 calls per lane · same headline · multilingual · concurrent dispatch", 48, 246, 13, UI.muted);
  text("Shared admission cap applies · no retries · both adapters serialize model access", 48, 266, 12, UI.muted);
  const scale = current ? serverScale(current) : 300;
  if (current) current.lanes.forEach((lane, i) => drawLane(lane, i, now, scale));
  else BACKENDS.forEach((key, i) => {
    const y = 324 + i * 156;
    text(key.toUpperCase(), 48, y - 17, 24, COLOR[key], true, "left", 500);
    line(152, y, 912, y);
  });
  text(`Markers: server ms · shared scale 0–${scale} · rows: arrival order`, 912, 434, 11, UI.muted, false, "right");
  LABELS.forEach((label, i) => {
    const x = 48 + i * 220;
    cx.beginPath(); cx.roundRect(x, 610, 204, 68, 6);
    cx.strokeStyle = UI.line; cx.lineWidth = 1.5; cx.stroke();
    text(label, x + 102, 638, 20, UI.text, false, "center", 500);
    BACKENDS.forEach((key, j) => {
      const count = current ? current.lanes[j].calls.filter(call => call.state === "done" && call.choice === label).length : 0;
      text(`${key.toUpperCase()} ${count}`, x + 29 + j * 83, 660, 12, count ? COLOR[key] : UI.muted, true);
    });
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
