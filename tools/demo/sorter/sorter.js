"use strict";
/* jevalaya snake — classification with a scripted presentation, not spatial AI.
 * Each headline is one choice call to /predict. Short items pin the
 * multilingual checkpoint (labeled MULTI-pinned; the T023 fit gate routes it
 * to ANE when it fits the bundle's budget); full-text items run pure auto.
 * reports the receipt verbatim — backend, latency, checkpoint, confidence —
 * plus a lane strip of the last decisions. Lag-switch delays ACTING on the
 * answer (simulated network); the receipt still shows the true latency.
 * The legacy item.y/frame clock still owns deadlines and scoring. Snake
 * paths are rendering only: they cannot choose a bin or change an outcome. */
const Q = new URLSearchParams(location.search);
const CFG = Object.assign({ url: "", token: "" }, window.JEVALAYA || {});
if (Q.get("url")) CFG.url = Q.get("url");
if (Q.get("token")) CFG.token = Q.get("token");
const API = (CFG.url || "") + "/predict";
const ITEMS = (CFG.url || "") + "/items?limit=200&offset=0";

const cv = document.getElementById("game"), cx = cv.getContext("2d");
const W = 1280, H = 720, BIN_Y = 600, BIN_H = 90;
const LABELS = ["World", "Sports", "Business", "Sci/Tech"];
const LANE = { ane: "#58a6ff", mlx: "#3fb950", jev: "#d29922" };
const UI = {
  bg: "#0d1117", panel: "#161b22", raised: "#1c232d", line: "#30363d",
  text: "#f0f6fc", muted: "#8b949e", soft: "#b1bac4", red: "#f85149",
  sans: '-apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
  mono: 'ui-monospace, "SFMono-Regular", Menlo, Consolas, monospace',
};
const FALL_PX_S = 230;
const visualRounds = new WeakMap();
const motionPreference = matchMedia("(prefers-reduced-motion: reduce)");

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
// Presentation helpers only. Request, timing, scoring and key handling above/below
// are unchanged; simulation coordinates are only a clock for snake presentation.
function text(value, x, y, size = 14, color = UI.text, weight = 400, mono = false) {
  cx.font = `${weight} ${size}px ${mono ? UI.mono : UI.sans}`;
  cx.fillStyle = color; cx.textAlign = "left"; cx.textBaseline = "alphabetic";
  cx.fillText(String(value), x, y);
}
function panel(x, y, w, h, fill = UI.panel, stroke = UI.line, r = 10) {
  cx.beginPath(); cx.roundRect(x, y, w, h, r);
  cx.fillStyle = fill; cx.fill();
  if (stroke) { cx.strokeStyle = stroke; cx.lineWidth = 1.5; cx.stroke(); }
}
function line(x1, y1, x2, y2, color = UI.line, width = 1.5) {
  cx.beginPath(); cx.moveTo(x1, y1); cx.lineTo(x2, y2);
  cx.strokeStyle = color; cx.lineWidth = width; cx.stroke();
}
function dot(x, y, color, r = 3) {
  cx.beginPath(); cx.arc(x, y, r, 0, Math.PI * 2); cx.fillStyle = color; cx.fill();
}
function fitText(value, maxW) {
  const valueText = String(value);
  if (cx.measureText(valueText).width <= maxW) return valueText;
  const chars = Array.from(valueText);
  while (chars.length && cx.measureText(chars.join("") + "…").width > maxW) chars.pop();
  return chars.join("") + "…";
}
function wrapText(value, x, y, maxW, lh, maxLines = 3) {
  const words = value.split(/\s+/); const lines = []; let current = "";
  for (const word of words) {
    const next = current ? current + " " + word : word;
    if (cx.measureText(next).width > maxW && current) { lines.push(current); current = word; }
    else current = next;
  }
  if (current) lines.push(current);
  lines.slice(0, maxLines).forEach((row, i) => {
    const overflow = i === maxLines - 1 && lines.length > maxLines;
    cx.fillText(fitText(row + (overflow ? " …" : ""), maxW), x, y + i * lh);
  });
}
function badge(label, x, y, w, color = UI.muted) {
  panel(x, y, w, 26, color + "12", color + "55", 6);
  dot(x + 12, y + 13, color);
  text(label, x + 23, y + 18, 12, color, 500, true);
}
function brandMark(x, y) {
  cx.strokeStyle = LANE.ane; cx.lineWidth = 2; cx.lineCap = "round";
  for (const dy of [-10, 0, 10]) {
    cx.beginPath(); cx.moveTo(x, y); cx.lineTo(x + 12, y);
    cx.bezierCurveTo(x + 20, y, x + 23, y + dy, x + 33, y + dy); cx.stroke();
  }
}
function draw() {
  cx.clearRect(0, 0, W, H);
  cx.fillStyle = UI.bg; cx.fillRect(0, 0, W, H);
  // The article stays at the top; the arena below visualizes a returned choice.
  panel(24, 225, 1232, 349, UI.bg);
  line(24, 275, 1256, 275);
  for (let y = 442; y < 568; y += 24) {
    for (let x = 48; x < 1240; x += 24) dot(x, y, "#21262d", 0.7);
  }
  text("SCRIPTED MOTION / LIVE DECISION", 44, 423, 11, UI.muted, 500, true);
  const deadline = BIN_Y - 110;
  const remaining = item ? Math.max(0, (deadline - item.y) / FALL_PX_S) : 0;
  text(item ? item.result ? "ROUND CLOSED" : `FOOD WINDOW  ${remaining.toFixed(1)} s` : "AWAITING ARTICLE", 1060, 423, 11, UI.muted, 500, true);
  // The destination gets the backend's lane color, never a made-up backend.
  const binW = 299, binGap = 12;
  LABELS.forEach((lab, i) => {
    const x = 24 + i * (binW + binGap);
    const selected = item && item.answered && item.result !== "miss" && item.target === lab;
    const color = selected ? backendColor() : UI.line;
    line(x + binW / 2, 578, x + binW / 2, BIN_Y - 7, color);
    cx.beginPath(); cx.ellipse(x + binW / 2, 559, 28, 8, 0, 0, Math.PI * 2);
    cx.strokeStyle = color + "66"; cx.lineWidth = 1.5; cx.stroke();
    panel(x, BIN_Y, binW, BIN_H, selected ? color + "14" : UI.panel, color);
    text(`0${i + 1}`, x + 18, BIN_Y + 25, 12, selected ? color : UI.muted, 500, true);
    text(lab, x + 18, BIN_Y + 55, 24, UI.text, 600);
    text(selected ? "SELECTED DESTINATION" : "CATEGORY", x + 18, BIN_Y + 76, 10, selected ? color : UI.muted, 500, true);
    if (selected) {
      const mark = item.result === "ok" ? "✓" : item.result === "wrong" ? "×" : "↓";
      panel(x + binW - 48, BIN_Y + 29, 30, 30, color + "16", null, 7);
      text(mark, x + binW - 41, BIN_Y + 51, 21, item.result === "wrong" ? UI.red : color, 500);
    }
  });
  if (item) { drawCard(); drawSnakeRound(); }
  else {
    text("The next headline is on its way", 473, 334, 22, UI.soft, 500);
    text("A real topic choice. A scripted journey to the food.", 461, 361, 14, UI.muted);
    paintSnake(coilPoints(), UI.muted);
  }
  overlay();
}
function drawCard() {
  const x = 44, y = 286, w = 1192, h = 112;
  const color = item.golden ? LANE.jev : item.answered ? backendColor() : UI.muted;
  const border = item.result === "ok" ? LANE.mlx : item.result === "wrong" || item.result === "miss" ? UI.red : color;
  panel(x, y, w, h, UI.raised, border);
  line(x + 18, y + 12, x + 18, y + 26, color, 2);
  text(`ARTICLE ${String(item.id).slice(0, 28)}`, x + 29, y + 23, 11, UI.muted, 500, true);
  const target = item.target || "Awaiting answer";
  cx.font = `500 13px ${UI.sans}`;
  const targetW = cx.measureText(target).width;
  text(target, x + w - 24 - targetW, y + 23, 13, item.answered ? color : UI.muted, 500);
  cx.font = `500 21px ${UI.sans}`; cx.fillStyle = UI.text;
  wrapText(item.text.slice(0, 170), x + 24, y + 49, w - 48, 25, 2);
  text(item.tag, x + 24, y + 99, 11, item.golden ? LANE.jev : UI.muted, 400, true);
  if (item.receipt) {
    const receipt = `${item.receipt.backend.toUpperCase()} · ${item.receipt.ms} ms`;
    cx.font = `500 12px ${UI.mono}`;
    text(receipt, x + w - 24 - cx.measureText(receipt).width, y + 99, 12, color, 500, true);
  }
}
function clamp01(value) { return Math.max(0, Math.min(1, value)); }
function coilPoints() {
  return Array.from({ length: 61 }, (_, i) => {
    const t = i / 60, angle = (1 - t) * Math.PI * 2.1, radius = 11 + 23 * t;
    return { x: 606 + Math.cos(angle) * radius, y: 467 + Math.sin(angle) * radius * 0.75 };
  });
}
function waitingPoints(y, golden) {
  const points = coilPoints();
  // A neutral approach can freeze while pending; it reveals no destination.
  const lead = golden || motionPreference.matches ? 0 : clamp01((y + 90) / 35) * 24;
  for (let i = 1; i <= 12; i++) points.push({ x: 640, y: 467 + lead * i / 12 });
  return points;
}
function measurePath(points) {
  let length = 0;
  const distances = points.map((p, i) => {
    if (i) length += Math.hypot(p.x - points[i - 1].x, p.y - points[i - 1].y);
    return length;
  });
  return { points, distances, length };
}
function pointAlong(path, distance) {
  const d = Math.max(0, Math.min(path.length, distance));
  let i = 1;
  while (i < path.distances.length - 1 && path.distances[i] < d) i++;
  const a = path.points[i - 1], b = path.points[i];
  const t = (d - path.distances[i - 1]) / (path.distances[i] - path.distances[i - 1] || 1);
  return { x: a.x + (b.x - a.x) * t, y: a.y + (b.y - a.y) * t };
}
function routeVisual(round) {
  let visual = visualRounds.get(round);
  if (!visual) { visual = { path: null }; visualRounds.set(round, visual); }
  // A late receipt must not resurrect food after the real scorer records a miss.
  if (!visual.path && round.answered && round.result !== "miss" && LABELS.includes(round.target)) {
    visual.answerY = round.y;
    visual.durationY = Math.max(FALL_PX_S * 0.4, BIN_Y - 110 - round.y);
    const prefix = waitingPoints(round.y, round.golden);
    const from = prefix[prefix.length - 1];
    const goal = { x: 24 + LABELS.indexOf(round.target) * 311 + 149.5, y: 552 };
    const route = Array.from({ length: 91 }, (_, i) => {
      const t = i / 90, s = 1 - t;
      return {
        x: s ** 3 * from.x + 3 * s * s * t * from.x + 3 * s * t * t * goal.x + t ** 3 * goal.x,
        y: s ** 3 * from.y + 3 * s * s * t * 526 + 3 * s * t * t * 476 + t ** 3 * goal.y,
      };
    });
    visual.prefixLength = measurePath(prefix).length;
    visual.path = measurePath(prefix.concat(route.slice(1)));
    visual.route = route;
    visual.goal = goal;
  }
  return visual;
}
function strokePoints(points, color, width) {
  cx.beginPath(); points.forEach((p, i) => i ? cx.lineTo(p.x, p.y) : cx.moveTo(p.x, p.y));
  cx.strokeStyle = color; cx.lineWidth = width; cx.lineCap = "round"; cx.lineJoin = "round"; cx.stroke();
}
function paintSnake(points, color) {
  strokePoints(points, UI.bg, 22);
  strokePoints(points, color, 16);
  // Small scale marks and two restrained eyes distinguish the snake from a rail.
  points.forEach((p, i) => { if (i % 5 === 0 && i < points.length - 4) dot(p.x, p.y, UI.bg + "55", 1.5); });
  const head = points[points.length - 1], neck = points[points.length - 2];
  cx.save(); cx.translate(head.x, head.y); cx.rotate(Math.atan2(head.y - neck.y, head.x - neck.x));
  cx.beginPath(); cx.ellipse(0, 0, 12, 9, 0, 0, Math.PI * 2); cx.fillStyle = color; cx.fill();
  dot(4, -4, UI.bg, 2.1); dot(4, 4, UI.bg, 2.1);
  cx.restore();
}
function foodCard(x, y, color, decay = 0, eaten = 0, wrong = false) {
  cx.save(); cx.translate(x, y); cx.rotate(decay * 0.25);
  cx.globalAlpha = (1 - eaten) * (1 - decay * 0.45);
  panel(-13, -17, 26, 34, UI.panel, color, 4);
  line(-7, -7, 7, -7, color); line(-7, -1, 7, -1, color); line(-7, 5, 2, 5, color);
  if (wrong || decay > 0) {
    line(-9, 12, 9, -12, UI.red);
    if (wrong) { cx.beginPath(); cx.arc(-13, 0, 6, 0, Math.PI * 2); cx.fillStyle = UI.bg; cx.fill(); }
  }
  cx.restore();
}
function drawSnakeRound() {
  const visual = routeVisual(item);
  const miss = item.result === "miss";
  const color = item.golden ? LANE.jev : item.answered ? backendColor() : UI.muted;
  if (!visual.path || miss) {
    const points = waitingPoints(miss ? BIN_Y - 110 : item.y, item.golden);
    paintSnake(points, miss ? UI.muted : color);
    const decay = miss ? clamp01((item.settle || 0) / 450) : 0;
    foodCard(640, 552, miss ? UI.red : color, decay);
    text(miss ? "EXPIRED" : item.golden ? "JEV · SCRIPTED CALL" : lagMs ? "WAITING · LAG SIM" : "WAITING FOR ANSWER", 683, 485, 12, miss ? UI.red : color, 500, true);
    return;
  }
  const elapsed = item.y - visual.answerY;
  const u = motionPreference.matches ? item.result ? 1 : 0 : clamp01(elapsed / visual.durationY);
  const ease = (1 - Math.cos(Math.PI * u)) / 2;
  const distance = visual.prefixLength + (visual.path.length - visual.prefixLength) * ease;
  const body = Array.from({ length: 49 }, (_, i) => {
    const d = distance - (48 - i) * 2.5;
    const p = pointAlong(visual.path, d);
    const next = pointAlong(visual.path, d + 1);
    const angle = Math.atan2(next.y - p.y, next.x - p.x) + Math.PI / 2;
    const wave = motionPreference.matches || item.result ? 0
      : Math.sin(d / 15 - (item.y + 90) / 26) * 3.2 * Math.sin(Math.PI * i / 48) * Math.sin(Math.PI * u);
    return { x: p.x + Math.cos(angle) * wave, y: p.y + Math.sin(angle) * wave };
  });
  cx.save(); cx.setLineDash([3, 7]); strokePoints(visual.route, color + "55", 1.5); cx.restore();
  const atFood = u >= 1 && item.result;
  const eaten = atFood && item.result === "ok" ? motionPreference.matches ? 1
    : clamp01((elapsed - visual.durationY) / (FALL_PX_S * 0.18)) : 0;
  foodCard(visual.goal.x, visual.goal.y, item.result === "wrong" ? UI.red : color, 0, eaten, atFood && item.result === "wrong");
  paintSnake(body, atFood && item.result === "wrong" ? UI.red : color);
  if (atFood) {
    const label = item.result === "ok" ? "+1 · EATEN" : "WRONG FOOD";
    const labelColor = item.result === "ok" ? LANE.mlx : UI.red;
    text(label, visual.goal.x - 42, 464, 12, labelColor, 500, true);
    if (!motionPreference.matches) {
      const pulse = clamp01((elapsed - visual.durationY) / (FALL_PX_S * 0.3));
      cx.save(); cx.globalAlpha = 1 - pulse;
      cx.beginPath(); cx.arc(visual.goal.x, visual.goal.y, 16 + pulse * 16, 0, Math.PI * 2);
      cx.strokeStyle = labelColor; cx.lineWidth = 1.5; cx.stroke(); cx.restore();
    }
  }
}
function backendColor() {
  const b = item && item.receipt ? item.receipt.backend : null;
  return (LANE[b] || UI.muted);
}
function overlay() {
  const sorted = [...latHist].sort((a, b) => a - b);
  const p50 = sorted.length ? Math.round(sorted[Math.floor(sorted.length / 2)]) : 0;
  const acc = decided ? (correct / decided) : 0;
  brandMark(25, 35);
  text("jevalaya snake — live routing demo", 74, 43, 24, UI.text, 600);
  text("The model picks a topic. The snake follows a scripted path — not spatial steering.", 25, 67, 14, UI.muted);
  badge(paused ? "PAUSED" : "LIVE", 1150, 23, 106, paused ? UI.muted : LANE.mlx);

  text("LATEST ROUTING RECEIPT", 25, 93, 11, UI.muted, 500, true);
  text("SESSION", 801, 93, 11, UI.muted, 500, true);
  panel(24, 103, 752, 75);
  panel(800, 103, 456, 75);
  const metric = (label, value, x, color = UI.text, size = 25) => {
    text(label, x, 124, 11, UI.muted, 500, true);
    text(value, x, 156, size, color, 500, true);
  };
  const laneColor = LANE[last.backend] || UI.muted;
  metric("BACKEND", last.backend.toUpperCase(), 44, laneColor);
  metric("LATENCY / ms", last.ms, 202);
  metric("CONFIDENCE", last.conf, 364);
  metric("CHECKPOINT", last.ckpt, 541, UI.soft, 18);
  [181, 343, 520].forEach(x => line(x, 120, x, 160));
  metric("SCORE", score, 820);
  metric("STREAK", streak, 914);
  metric(`ACC ${correct}/${decided}`, `${(acc * 100).toFixed(0)}%`, 1017);
  metric("p50 / ms", p50, 1151);
  [895, 998, 1132].forEach(x => line(x, 120, x, 160));

  text("LAST 12", 25, 205, 11, UI.muted, 500, true);
  for (let i = 0; i < 12; i++) {
    const color = LANE[lanes[i]];
    panel(96 + i * 23, 190, 17, 17, color ? color + "33" : UI.panel, color || UI.line, 4);
  }
  [["ANE", LANE.ane], ["MLX", LANE.mlx], ["Jev", LANE.jev]].forEach(([name, color], i) => {
    dot(407 + i * 77, 199, color);
    text(name, 417 + i * 77, 204, 12, UI.soft, 500, true);
  });
  text(`Jev calls ${jevCalls}`, 650, 204, 12, UI.muted, 400, true);
  if (goldenArmed) badge("GOLDEN ARMED", 800, 186, 154, LANE.jev);
  if (lagMs > 0) badge(`LAG SIM +${lagMs} ms`, 1025, 186, 231, UI.red);
  else text("server latency · no simulated delay", 992, 204, 12, UI.muted);

  // Persistent status row: no labels land on top of the moving article.
  const outcome = item && item.result === "miss" ? "EXPIRED · NO ANSWER BEFORE DEADLINE" : last.status;
  const failure = /WRONG|MISS|NO ANSWER/.test(outcome);
  const statusColor = paused ? UI.muted : failure ? UI.red
    : last.status.includes("FRIEND") ? LANE.jev : last.status.includes("SORTED") ? LANE.mlx
    : last.status === "ANSWERED" ? laneColor : LANE.ane;
  dot(45, 250, statusColor, 4);
  const status = paused ? `PAUSED · ${outcome}` : outcome;
  text(status, 59, 256, 17, statusColor, 600);
  cx.font = `400 13px ${UI.mono}`;
  const mode = fitText(last.mode === "—" ? "awaiting first article" : last.mode, 450);
  text(mode, 1235 - cx.measureText(mode).width, 255, 13, UI.muted, 400, true);

  const key = (name, label, x, w = 22) => {
    panel(x, 699, w, 17, UI.panel, UI.line, 3);
    text(name, x + 5, 711, 10, UI.soft, 500, true);
    text(label, x + w + 8, 712, 12, UI.muted);
  };
  key("SPACE", paused ? "Resume" : "Pause", 25, 44);
  key("L", "Lag +500 ms", 171);
  key("G", "Golden · 1 scripted Jev call", 324);
  key("R", "Restart", 568);
  text("AG News  /  live POST /predict", 1035, 712, 12, UI.muted, 400, true);
}
addEventListener("keydown", e => {
  if (e.code === "Space") { paused = !paused; e.preventDefault(); }
  if (e.code === "KeyL") lagMs = lagMs > 0 ? 0 : 500;
  if (e.code === "KeyG") goldenArmed = true;
  if (e.code === "KeyR") reset();
});
reset();
draw();
if (Q.get("autostart") !== "0") requestAnimationFrame(frame);
