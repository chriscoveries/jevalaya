"use strict";
/* jevalaya sorter — a consumer proving live routing, honestly.
 * Headlines fall; each is one choice call to /predict. Short items pin the
 * multilingual checkpoint (labeled MULTI-pinned; the T023 fit gate routes it
 * to ANE when it fits the bundle's budget); full-text items run pure auto.
 * reports the receipt verbatim — backend, latency, checkpoint, confidence —
 * plus a lane strip of the last decisions. Lag-switch delays ACTING on the
 * answer (simulated network); the receipt still shows the true latency. */
const Q = new URLSearchParams(location.search);
const CFG = Object.assign({ url: "", token: "" }, window.JEVALAYA || {});
if (Q.get("url")) CFG.url = Q.get("url");
if (Q.get("token")) CFG.token = Q.get("token");
const API = (CFG.url || "") + "/predict";
const ITEMS = (CFG.url || "") + "/items?limit=200&offset=0";

const cv = document.getElementById("game"), cx = cv.getContext("2d");
const W = 1280, H = 720, BIN_Y = 600, BIN_H = 90;
const LABELS = ["World", "Sports", "Business", "Sci/Tech"];
const BIN_COLORS = ["#5b6ee1", "#26a69a", "#8e6fbf", "#d08a2d"];
const LANE = { ane: "#2f7bff", mlx: "#19b5a6", jev: "#e8a100" };
const FALL_PX_S = 230;

let queue, item, score, streak, decided, correct, jevCalls, lanes, latHist;
let lagMs, goldenArmed, paused, last, gapT;
function reset() {
  queue = []; item = null; score = 0; streak = 0; decided = 0; correct = 0;
  jevCalls = 0; lanes = []; latHist = [];
  lagMs = 0; goldenArmed = false; paused = false; gapT = 0;
  last = { backend: "—", ms: 0, conf: 0, ckpt: "—", mode: "—", status: "READY" };
  fetch(ITEMS).then(r => r.json()).then(d => { queue = d.items; });
}
function headlineOf(text) {
  const i = text.search(/[.!?]\s/);
  return (i > 20 ? text.slice(0, i + 1) : text.slice(0, 140));
}
function spawnNext() {
  if (!queue.length) return;
  const row = queue.shift(); queue.push(row);   // endless cycle
  const idx = decided % 2;
  const golden = goldenArmed; goldenArmed = false;
  const short = idx === 0;   // alternate: headline+pinned vs full-text+auto
  const text = short ? headlineOf(row.text) : row.text;
  item = { row, text, short, golden, y: -90, target: null, answered: false,
           result: null, id: row.id,
           tag: golden ? "GOLDEN (scripted jev)" : short ? "headline + MULTI pinned" : "full article + auto" };
  const body = {
    state: text,
    questions: { topic: { type: "choice", instructions: "Classify the news article by topic.",
      criteria: { World: null, Sports: null, Business: null, "Sci/Tech": null } } },
    backend: golden ? "jev" : "auto",
    request_id: "sorter-" + decided,
  };
  if (short && !golden) body.model = "multilingual";   // pinned lane, labeled on screen
  last.status = golden ? "PHONING A FRIEND…" : "ASKING…";
  last.mode = golden ? "GOLDEN (scripted jev)" : short ? "headline + MULTI pinned" : "full text + auto";
  const t0 = performance.now();
  fetch(API, { method: "POST",
    headers: { "Authorization": "Bearer " + CFG.token, "Content-Type": "application/json" },
    body: JSON.stringify(body) })
    .then(r => r.json())
    .then(data => {
      const ans = data.answers.topic, route = data.routing;
      const apply = () => {
        if (!item || item.id !== row.id) return;
        item.answered = true;
        item.target = ans.choice;
        item.receipt = { backend: route.backend, ms: Math.round(route.latency_ms),
                         conf: ans.confidence, ckpt: route.checkpoint || "—" };
        latHist.push(route.latency_ms); if (latHist.length > 20) latHist.shift();
        lanes.push(route.backend); if (lanes.length > 12) lanes.shift();
        if (golden) jevCalls++;
        last = { backend: route.backend, ms: Math.round(route.latency_ms),
                 conf: ans.confidence.toFixed(2), ckpt: route.checkpoint || "—",
                 mode: last.mode, status: "ANSWERED" };
      };
      if (lagMs > 0) setTimeout(apply, lagMs); else apply();
    })
    .catch(() => { last.status = "NO ANSWER"; });
}
function arrive() {
  // item reached the bins: score it (every settled item counts once)
  decided++;
  if (item.target === null || item.target === undefined) {
    item.result = "miss"; streak = 0;
    last.status = lagMs > 0 ? "MISSED (lag sim)" : "MISSED (no answer)";
  } else if (item.target === item.row.label) {
    item.result = "ok"; score++; streak++; correct++;
    last.status = "SORTED ✓";
  } else {
    item.result = "wrong"; streak = 0;
    last.status = "WRONG BIN ✗";
  }
  item.settle = 0;
}
let lastT = 0;
function frame(t) {
  requestAnimationFrame(frame);
  const dt = Math.min(50, t - (lastT || t)); lastT = t;
  if (paused) { draw(); return; }
  if (!item) { gapT += dt; if (gapT > 500 && queue.length) { gapT = 0; spawnNext(); } }
  else {
    item.y += FALL_PX_S * dt / 1000;
    if (item.y >= BIN_Y - 110 && !item.result) { arrive(); }
    if (item.result && (item.settle = (item.settle || 0) + dt) > 700) item = null;
  }
  draw();
}
function wrapText(text, x, y, maxW, lh) {
  const words = text.split(/\s+/); let line = "", yy = y;
  for (const w of words) {
    const t = line ? line + " " + w : w;
    if (cx.measureText(t).width > maxW && line) { cx.fillText(line, x, yy); line = w; yy += lh; }
    else line = t;
  }
  cx.fillText(line, x, yy);
}
function draw() {
  cx.fillStyle = "#f7f7f7"; cx.fillRect(0, 0, W, H);
  // bins
  const bw = W / 4;
  LABELS.forEach((lab, i) => {
    const glow = item && item.answered && item.target === lab;
    cx.fillStyle = glow ? BIN_COLORS[i] : "#e4e4e4";
    cx.fillRect(i * bw + 8, BIN_Y, bw - 16, BIN_H);
    cx.fillStyle = glow ? "#fff" : "#333";
    cx.font = "bold 26px monospace"; cx.textAlign = "center";
    cx.fillText(lab, i * bw + bw / 2, BIN_Y + 56);
    cx.textAlign = "left";
  });
  // falling card
  if (item) {
    const answered = item.answered;
    const border = !answered ? "#999"
      : item.result === "ok" ? "#2fae4e" : item.result === "wrong" ? "#d33" : backendColor();
    cx.fillStyle = "#fff"; cx.strokeStyle = border; cx.lineWidth = 5;
    const cw = 620, ch = 150, cx0 = W / 2 - cw / 2;
    cx.fillRect(cx0, item.y, cw, ch); cx.strokeRect(cx0, item.y, cw, ch);
    cx.fillStyle = "#222"; cx.font = "20px monospace";
    wrapText(item.text.slice(0, 170), cx0 + 16, item.y + 34, cw - 32, 26);
    cx.font = "15px monospace"; cx.fillStyle = "#777";
    cx.fillText(item.tag, cx0 + 16, item.y + ch - 12);
  }
  overlay();
}
function backendColor() {
  const b = item && item.receipt ? item.receipt.backend : null;
  return (LANE[b] || "#999");
}
function overlay() {
  const sorted = [...latHist].sort((a, b) => a - b);
  const p50 = sorted.length ? Math.round(sorted[Math.floor(sorted.length / 2)]) : 0;
  const acc = decided ? (correct / decided) : 0;
  cx.fillStyle = "rgba(17,17,17,0.86)"; cx.fillRect(0, 0, W, 132);
  cx.font = "24px monospace"; cx.fillStyle = "#fff";
  const lag = lagMs > 0 ? `  |  LAG SIM +${lagMs}ms` : "";
  cx.fillStyle = lagMs > 0 ? "#ff5555" : "#fff";
  cx.fillText(`backend=${last.backend}  ckpt=${last.ckpt}  answer=${last.ms}ms  conf=${last.conf}${lag}`, 18, 34);
  cx.fillStyle = "#fff";
  cx.fillText(`score=${score}  streak=${streak}  acc=${acc.toFixed(2)} (${correct}/${decided})  p50=${p50}ms  jev=${jevCalls}`, 18, 66);
  cx.fillStyle = "#bbb"; cx.font = "16px monospace";
  cx.fillText(last.mode, 18, 94);
  // lane strip: last decisions colored by backend
  cx.fillText("lanes:", 18, 120);
  lanes.forEach((b, i) => {
    cx.fillStyle = LANE[b] || "#666";
    cx.beginPath(); cx.arc(110 + i * 26, 114, 9, 0, 7); cx.fill();
  });
  cx.fillStyle = "#bbb";
  cx.fillText("●ane ●mlx ●jev", 110 + 12 * 26 + 10, 120);
  // status banner
  if (last.status !== "RUN" && last.status !== "READY" && last.status !== "ANSWERED") {
    cx.font = "bold 40px monospace";
    cx.fillStyle = last.status.includes("FRIEND") ? "#e8a100" : last.status.includes("WRONG") || last.status.includes("MISS") ? "#f44336" : lagMs > 0 ? "#ff5555" : "#4caf50";
    cx.fillText(last.status, 18, 175);
  }
}
addEventListener("keydown", e => {
  if (e.code === "Space") { paused = !paused; e.preventDefault(); }
  if (e.code === "KeyL") lagMs = lagMs > 0 ? 0 : 500;
  if (e.code === "KeyG") goldenArmed = true;
  if (e.code === "KeyR") reset();
});
reset();
if (Q.get("autostart") !== "0") requestAnimationFrame(frame);
