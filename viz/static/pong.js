/* Frontend for the Pong visualizer.
 *
 * All model state comes from the server (the tested `homeostasis` package);
 * this file only renders frames and sends commands. Per-step series arrive
 * batched per frame, so the raster and traces show every model step at any
 * playback speed.
 */

"use strict";

const C = {
  panel: "#10141a", grid: "#2a313c", text: "#d7dde6", dim: "#8b95a3",
  red: "#ff5d5d", blue: "#5da9ff", green: "#3fd68f", yellow: "#ffd23f",
  pink: "#ff9ecb", accent: "#58a6ff", white: "#e8edf4",
};
const PAPER_RATE = 0.582;
const STRIP_CAP = 1200;

const buf = {
  ang: [], dang: [], prop: [], ev: [],
  clear() { this.ang = []; this.dang = []; this.prop = []; this.ev = []; },
  push(e) {
    this.ang.push(e.ang); this.dang.push(e.dang);
    this.prop.push(e.prop); this.ev.push(e.ev);
    while (this.ang.length > STRIP_CAP) {
      this.ang.shift(); this.dang.shift(); this.prop.shift(); this.ev.shift();
    }
  },
};
const trail = { pts: [], cap: 220,
  clear() { this.pts = []; },
  push(x, y) { this.pts.push([x, y]); while (this.pts.length > this.cap) this.pts.shift(); },
};
const spark = { x: [], target: [], threshold: [], cap: 316,
  clear() { this.x = []; this.target = []; this.threshold = []; },
  push(n) {
    this.x.push(n.x); this.target.push(n.target); this.threshold.push(n.threshold);
    while (this.x.length > this.cap) { this.x.shift(); this.target.shift(); this.threshold.shift(); }
  },
};

let lastT = -1;
let lastNodeIdx = 0;
let nNodes = 500;
let latest = null;
let flash = null;          // {kind, until} for hit/miss feedback
let stepTimestamps = [];

// ---------- websocket -------------------------------------------------------
let ws = null;
const connEl = document.getElementById("conn");

function connect() {
  ws = new WebSocket(`ws://${location.host}/pong/ws`);
  ws.onopen = () => { connEl.textContent = "connected"; connEl.className = "conn ok"; };
  ws.onclose = () => {
    connEl.textContent = "disconnected — retrying…"; connEl.className = "conn bad";
    setTimeout(connect, 1000);
  };
  ws.onmessage = (ev) => handleFrame(JSON.parse(ev.data));
}
function send(o) { if (ws && ws.readyState === 1) ws.send(JSON.stringify(o)); }

// ---------- frames ----------------------------------------------------------
function handleFrame(msg) {
  if (msg.type !== "frame") return;
  if (msg.t < lastT || msg.config.n_nodes !== nNodes) {
    buf.clear(); trail.clear(); spark.clear(); clearRaster();
    nNodes = msg.config.n_nodes;
    flash = null;
  }
  lastT = msg.t;
  latest = msg;

  for (const e of msg.series) {
    buf.push(e);
    trail.push(e.bx, e.by);
    pushRasterColumn(e.spikes);
    if (e.ev !== 0) flash = { kind: e.ev === 1 ? "hit" : "miss", until: performance.now() + 450 };
  }
  if (msg.series.length) {
    const now = performance.now();
    stepTimestamps.push([now, msg.series.length]);
    while (stepTimestamps.length && now - stepTimestamps[0][0] > 2000) stepTimestamps.shift();
  }
  if (msg.node.index !== lastNodeIdx) { spark.clear(); lastNodeIdx = msg.node.index; }
  spark.push(msg.node);

  syncControls(msg);
  drawAll(msg);
}

// ---------- canvases --------------------------------------------------------
const cv = (id) => { const c = document.getElementById(id); return [c, c.getContext("2d")]; };
const [fieldC, fieldX] = cv("field");
const [sensC, sensX] = cv("sensors");
const [effC, effX] = cv("effectors");
const [rasterC, rasterX] = cv("raster");
const [propC, propX] = cv("propchart");
const [scoreC, scoreX] = cv("scorechart");
const [angC, angX] = cv("anglechart");
const [histC, histX] = cv("whist");
const [sparkC, sparkX] = cv("node-spark");

function bg(x, c) { x.fillStyle = C.panel; x.fillRect(0, 0, c.width, c.height); }

function fitWide() {
  for (const c of [fieldC, rasterC, propC]) {
    const w = Math.max(320, Math.round(c.clientWidth) || c.width);
    if (c.width !== w) c.width = w;
  }
  clearRaster();
}
let resizeTimer = null;
window.addEventListener("resize", () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => { fitWide(); if (latest) drawAll(latest); }, 150);
});

// ---------- field -----------------------------------------------------------
function drawField(m) {
  bg(fieldX, fieldC);
  const cfg = m.config, now = m.now;
  const pad = 10;
  const sx = (fieldC.width - 2 * pad) / cfg.width;
  const sy = (fieldC.height - 2 * pad) / cfg.height;
  const X = (x) => pad + x * sx;
  // Model y grows upward; canvas y grows downward.
  const Y = (y) => fieldC.height - pad - y * sy;

  fieldX.strokeStyle = C.grid;
  fieldX.strokeRect(X(0), Y(cfg.height), cfg.width * sx, cfg.height * sy);

  // Miss line at x = 0 and the paddle's column.
  fieldX.strokeStyle = "rgba(255,93,93,0.5)";
  fieldX.setLineDash([4, 4]);
  fieldX.beginPath(); fieldX.moveTo(X(0), Y(cfg.height)); fieldX.lineTo(X(0), Y(0)); fieldX.stroke();
  fieldX.setLineDash([]);

  // Sensor visualization.
  const vals = now.sensor_values, acts = now.sensors;
  if (m.encoding === "egocentric") {
    const rayLen = Math.min(fieldC.width - X(cfg.paddle_x) - pad, 300);
    for (let i = 0; i < vals.length; i++) {
      const a = (vals[i] * Math.PI) / 180;
      const x0 = X(cfg.paddle_x), y0 = Y(now.paddle_y);
      const on = acts[i] > 0;
      fieldX.beginPath();
      fieldX.moveTo(x0, y0);
      fieldX.lineTo(x0 + Math.cos(a) * (on ? rayLen : 26) * (on ? 1 : 1),
                    y0 - Math.sin(a) * (on ? rayLen : 26));
      fieldX.strokeStyle = on ? "rgba(255,210,63,0.85)" : "rgba(88,166,255,0.13)";
      fieldX.lineWidth = on ? 1.6 : 0.7;
      fieldX.stroke();
    }
  } else {
    for (let i = 0; i < vals.length; i++) {
      const on = acts[i] > 0;
      fieldX.fillStyle = on ? C.yellow : "rgba(88,166,255,0.18)";
      fieldX.fillRect(X(0) + 1, Y(vals[i]) - 1.5, on ? 26 : 10, 3);
    }
  }
  fieldX.lineWidth = 1;

  // Ball trail.
  for (let i = 0; i < trail.pts.length; i++) {
    const [bx, by] = trail.pts[i];
    fieldX.fillStyle = `rgba(63,214,143,${(i / trail.pts.length) * 0.5})`;
    fieldX.fillRect(X(bx) - 1, Y(by) - 1, 2, 2);
  }

  // Paddle.
  const half = cfg.paddle_half_height;
  fieldX.strokeStyle = C.pink;
  fieldX.lineWidth = 4;
  fieldX.beginPath();
  fieldX.moveTo(X(cfg.paddle_x), Y(now.paddle_y - half));
  fieldX.lineTo(X(cfg.paddle_x), Y(now.paddle_y + half));
  fieldX.stroke();
  fieldX.lineWidth = 1;

  // Ball and its velocity.
  const [bx, by] = now.ball;
  fieldX.beginPath();
  fieldX.arc(X(bx), Y(by), 5, 0, 7);
  fieldX.fillStyle = C.green;
  fieldX.fill();
  fieldX.beginPath();
  fieldX.moveTo(X(bx), Y(by));
  fieldX.lineTo(X(bx + now.vel[0] * 4), Y(by + now.vel[1] * 4));
  fieldX.strokeStyle = "rgba(232,237,244,0.55)";
  fieldX.stroke();

  // Hit / miss flash.
  if (flash && performance.now() < flash.until) {
    fieldX.fillStyle = flash.kind === "hit" ? "rgba(63,214,143,0.13)" : "rgba(255,93,93,0.15)";
    fieldX.fillRect(0, 0, fieldC.width, fieldC.height);
    fieldX.fillStyle = flash.kind === "hit" ? C.green : C.red;
    fieldX.font = "600 15px system-ui";
    fieldX.textAlign = "center";
    fieldX.fillText(flash.kind === "hit" ? "HIT" : "MISS", fieldC.width / 2, 22);
    fieldX.textAlign = "left";
  }

  document.getElementById("field-readout").textContent =
    `ball (${bx.toFixed(0)}, ${by.toFixed(0)})  v(${now.vel[0].toFixed(0)}, ${now.vel[1].toFixed(0)})` +
    `  ·  paddle y ${now.paddle_y.toFixed(1)}  Δ ${now.d_paddle.toFixed(1)}` +
    `  ·  θ ${now.angle.toFixed(1)}°  ·  active sensors ${now.sensors.filter((v) => v > 0).length}`;
}

// ---------- sensor / effector bars ------------------------------------------
function drawSensors(m) {
  bg(sensX, sensC);
  const acts = m.now.sensors, n = acts.length;
  const w = sensC.width / n;
  for (let i = 0; i < n; i++) {
    const h = acts[i] * (sensC.height - 16);
    sensX.fillStyle = acts[i] > 0 ? C.yellow : "#2b3340";
    sensX.fillRect(i * w + 0.5, sensC.height - Math.max(h, 2), Math.max(1, w - 1), Math.max(h, 2));
  }
  const vals = m.now.sensor_values;
  const unit = m.encoding === "egocentric" ? "°" : "px";
  sensX.fillStyle = C.dim;
  sensX.font = "10px monospace";
  sensX.fillText(`${vals[0]}${unit}`, 2, sensC.height - 3);
  sensX.textAlign = "right";
  sensX.fillText(`${vals[vals.length - 1]}${unit}`, sensC.width - 2, sensC.height - 3);
  sensX.textAlign = "left";
  document.getElementById("sensor-legend").textContent =
    m.encoding === "egocentric"
      ? `46 sensors, ball angle relative to paddle (no 0° sensor: grid is ±2, ±6, …)`
      : `50 sensors, ball height in the field (paddle motion cannot change this)`;
}

function drawEffectors(m) {
  bg(effX, effC);
  const [up, down] = m.now.outputs;
  const bw = Math.round(effC.width * 0.26), base = effC.height - 16, maxH = effC.height - 34;
  const ux = effC.width / 2 - bw - 16, dx = effC.width / 2 + 16;
  effX.fillStyle = "#39424f";
  effX.fillRect(ux, base - maxH, bw, maxH);
  effX.fillRect(dx, base - maxH, bw, maxH);
  effX.fillStyle = C.accent;
  effX.fillRect(ux, base - up * maxH, bw, up * maxH);
  effX.fillRect(dx, base - down * maxH, bw, down * maxH);
  effX.fillStyle = C.text;
  effX.font = "11px monospace";
  effX.textAlign = "center";
  effX.fillText(`up ${up.toFixed(2)}`, ux + bw / 2, effC.height - 3);
  effX.fillText(`down ${down.toFixed(2)}`, dx + bw / 2, effC.height - 3);
  effX.fillText(`Δy = ${m.config.gain}·(u−d) = ${m.now.d_paddle.toFixed(1)} px`, effC.width / 2, 11);
  effX.textAlign = "left";
}

// ---------- score -----------------------------------------------------------
function drawScore(m) {
  const s = m.score;
  const rateEl = document.getElementById("score-rate");
  rateEl.textContent = s.hit_rate === null ? "—" : `${(s.hit_rate * 100).toFixed(1)}%`;
  rateEl.style.color = s.hit_rate === null ? C.dim
    : (s.hit_rate > s.chance ? C.green : C.red);
  document.getElementById("score-detail").textContent =
    `${s.hits}/${s.opportunities} opportunities · chance ${(s.chance * 100).toFixed(0)}%`;

  bg(scoreX, scoreC);
  const W = scoreC.width, H = scoreC.height;
  const yFor = (v) => H - 6 - v * (H - 16);
  for (const [v, color, dash] of [[s.chance, C.dim, [3, 3]], [PAPER_RATE, C.pink, [5, 3]]]) {
    scoreX.strokeStyle = color;
    scoreX.setLineDash(dash);
    scoreX.beginPath(); scoreX.moveTo(0, yFor(v)); scoreX.lineTo(W, yFor(v)); scoreX.stroke();
    scoreX.setLineDash([]);
  }
  const curve = s.curve;
  if (curve.length > 1) {
    scoreX.strokeStyle = C.green;
    scoreX.lineWidth = 1.5;
    scoreX.beginPath();
    curve.forEach((v, i) => {
      const x = (i / (curve.length - 1)) * W;
      i === 0 ? scoreX.moveTo(x, yFor(v)) : scoreX.lineTo(x, yFor(v));
    });
    scoreX.stroke();
    scoreX.lineWidth = 1;
  }
  scoreX.fillStyle = C.dim;
  scoreX.font = "10px monospace";
  scoreX.fillText("1.0", 2, yFor(1.0) + 8);
  scoreX.fillText("0", 2, yFor(0) - 1);

  const recentEl = document.getElementById("recent");
  recentEl.innerHTML = s.recent
    .map((v) => `<i class="${v > 0 ? "hit" : "miss"}"></i>`).join("");
}

// ---------- angle chart -----------------------------------------------------
function drawAngle() {
  bg(angX, angC);
  const W = angC.width, H = angC.height, n = buf.ang.length;
  const mid = H * 0.55;
  angX.strokeStyle = C.grid;
  angX.beginPath(); angX.moveTo(0, mid); angX.lineTo(W, mid); angX.stroke();
  angX.fillStyle = C.dim; angX.font = "10px monospace";
  angX.fillText("θ +180", 2, 9); angX.fillText("0", 2, mid - 2); angX.fillText("−180", 2, mid + 10);
  if (!n) return;
  const xFor = (i) => W - (n - i) * (W / STRIP_CAP);
  // theta
  angX.strokeStyle = C.accent;
  angX.beginPath();
  let started = false;
  for (let i = 0; i < n; i++) {
    const x = xFor(i);
    if (x < 34) continue;
    const y = mid - (buf.ang[i] / 180) * (mid - 8);
    started ? angX.lineTo(x, y) : (angX.moveTo(x, y), (started = true));
  }
  angX.stroke();
  // |dtheta| on its own baseline at the bottom
  const dMax = Math.max(2, ...buf.dang);
  angX.strokeStyle = C.yellow;
  angX.beginPath();
  started = false;
  for (let i = 0; i < n; i++) {
    const x = xFor(i);
    if (x < 34) continue;
    const y = H - 4 - (buf.dang[i] / dMax) * (H - mid - 12);
    started ? angX.lineTo(x, y) : (angX.moveTo(x, y), (started = true));
  }
  angX.stroke();
  // hit/miss markers
  for (let i = 0; i < n; i++) {
    if (!buf.ev[i]) continue;
    const x = xFor(i);
    if (x < 34) continue;
    angX.fillStyle = buf.ev[i] === 1 ? C.green : C.red;
    angX.fillRect(x - 1, 0, 2, H);
  }
  angX.fillStyle = C.yellow;
  angX.font = "10px monospace";
  angX.fillText(`|Δθ| max ${dMax.toFixed(1)}°`, 34, H - 3);
}

// ---------- raster ----------------------------------------------------------
function clearRaster() {
  rasterX.fillStyle = C.panel;
  rasterX.fillRect(0, 0, rasterC.width, rasterC.height);
}
clearRaster();

function pushRasterColumn(spikeIdx) {
  const W = rasterC.width, H = rasterC.height;
  rasterX.drawImage(rasterC, -1, 0);
  rasterX.fillStyle = C.panel;
  rasterX.fillRect(W - 1, 0, 1, H);
  rasterX.fillStyle = C.yellow;
  const rowH = H / nNodes;
  for (const i of spikeIdx) rasterX.fillRect(W - 1, i * rowH, 1, Math.max(1, rowH));
}

rasterC.addEventListener("click", (ev) => {
  const r = rasterC.getBoundingClientRect();
  const y = ((ev.clientY - r.top) / r.height) * rasterC.height;
  const idx = Math.min(nNodes - 1, Math.max(0, Math.floor((y / rasterC.height) * nNodes)));
  document.getElementById("node-idx").value = idx;
  send({ cmd: "select_node", index: idx });
});

function drawProp() {
  bg(propX, propC);
  const W = propC.width, H = propC.height, n = buf.prop.length;
  propX.fillStyle = C.dim; propX.font = "10px monospace";
  propX.fillText("prop. spiked 1.0 —", 2, 10);
  propX.fillText("0 —", 2, H - 3);
  propX.strokeStyle = C.white;
  propX.beginPath();
  let started = false;
  for (let i = 0; i < n; i++) {
    const x = W - (n - i) * (W / STRIP_CAP);
    if (x < 34) continue;
    const y = H - 4 - buf.prop[i] * (H - 14);
    started ? propX.lineTo(x, y) : (propX.moveTo(x, y), (started = true));
  }
  propX.stroke();
}

// ---------- histogram / node / stats ----------------------------------------
function drawHist(m) {
  bg(histX, histC);
  const { counts, edges } = m.hist;
  const W = histC.width, H = histC.height;
  const maxC = Math.max(...counts, 1);
  const bw = W / counts.length;
  histX.fillStyle = C.accent;
  for (let i = 0; i < counts.length; i++) {
    const h = (counts[i] / maxC) * (H - 20);
    histX.fillRect(i * bw + 0.5, H - 13 - h, Math.max(1, bw - 1), h);
  }
  const lo = edges[0], hi = edges[edges.length - 1];
  if (lo < 0 && hi > 0) {
    const zx = ((0 - lo) / (hi - lo)) * W;
    histX.strokeStyle = C.dim;
    histX.setLineDash([3, 3]);
    histX.beginPath(); histX.moveTo(zx, 0); histX.lineTo(zx, H - 13); histX.stroke();
    histX.setLineDash([]);
  }
  histX.fillStyle = C.dim; histX.font = "10px monospace";
  histX.fillText(lo.toFixed(2), 2, H - 2);
  histX.textAlign = "right"; histX.fillText(hi.toFixed(2), W - 2, H - 2); histX.textAlign = "left";
}

function drawNode(m) {
  const n = m.node;
  const f = (v) => (typeof v === "number" ? v.toFixed(4) : String(v));
  const rows = [
    ["activation x′", f(n.x)], ["target T", f(n.target)],
    ["threshold 2T", f(n.threshold)], ["error E", f(n.error)],
    ["spiked now", n.spiked ? "yes" : "no"],
    ["in / out degree", `${n.in_degree} / ${n.out_degree}`],
    ["input links", String(n.n_input_links)],
    ["Σ incoming w", f(n.in_weight_sum)],
    ["spiking in-nbrs", String(n.spiking_in_neighbors)],
  ];
  document.getElementById("node-fields").innerHTML =
    rows.map(([k, v]) => `<span class="k">${k}</span><span class="v">${v}</span>`).join("");

  bg(sparkX, sparkC);
  const W = sparkC.width, H = sparkC.height;
  const all = [...spark.x, ...spark.target, ...spark.threshold];
  if (!all.length) return;
  const lo = Math.min(...all), hi = Math.max(...all), span = hi - lo || 1;
  const yFor = (v) => H - 4 - ((v - lo) / span) * (H - 8);
  const series = (arr, color, dash) => {
    sparkX.strokeStyle = color;
    sparkX.setLineDash(dash || []);
    sparkX.beginPath();
    arr.forEach((v, i) => {
      const x = W - (arr.length - i);
      i === 0 ? sparkX.moveTo(x, yFor(v)) : sparkX.lineTo(x, yFor(v));
    });
    sparkX.stroke();
    sparkX.setLineDash([]);
  };
  series(spark.threshold, C.dim, [3, 3]);
  series(spark.target, C.yellow);
  series(spark.x, C.accent);
}

function drawStats(m) {
  const now = m.now;
  const measured = stepTimestamps.reduce((s, [, n]) => s + n, 0) /
    Math.max(0.001, (performance.now() - (stepTimestamps[0]?.[0] ?? performance.now())) / 1000);
  const rows = [
    ["mean activation", now.mean_x.toFixed(4)],
    ["mean |error|", now.mean_abs_error.toFixed(4)],
    ["mean target", now.mean_target.toFixed(4)],
    ["prop. spiked", now.prop.toFixed(3)],
    ["spiking now", `${now.spikes.length} / ${m.config.n_nodes}`],
    ["learning", m.learning ? "on" : "OFF"],
    ["measured speed", m.playing ? `${measured.toFixed(0)} steps/s` : "paused"],
  ];
  document.getElementById("net-stats").innerHTML =
    rows.map(([k, v]) => `<span class="k">${k}</span><span class="v">${v}</span>`).join("");
  const fp = m.fingerprint;
  document.getElementById("fingerprint").textContent =
    `fingerprint @ t=${m.t}: Σx=${fp.sum_x} ΣT=${fp.sum_targets} ΣW=${fp.sum_weights} ` +
    `opps=${fp.n_opportunities}` +
    (m.custom_params || m.encoding !== "egocentric"
      ? " — non-default configuration; the fingerprint script replays published defaults only"
      : ` — verify with scripts/pong_fingerprint.py --seed ${m.seed} --steps ${m.t}` +
        " (valid for an untouched run: no learning toggles mid-run)");
}

// ---------- controls --------------------------------------------------------
const PARAM_ORDER = [
  "n_nodes", "p_link", "input_weight", "weight_init_mean", "weight_init_sd",
  "inhibitory_fraction", "inhibitory_weight_mean", "leak", "target_lr",
  "threshold_ratio", "gain", "paddle_half_height", "ball_speed_x", "ball_speed_y",
];
function buildParams(config) {
  const box = document.getElementById("params");
  if (!box.childElementCount) {
    for (const name of PARAM_ORDER) {
      const label = document.createElement("label");
      label.textContent = name;
      const input = document.createElement("input");
      input.id = `param-${name}`;
      input.type = "number";
      input.step = "any";
      box.appendChild(label);
      box.appendChild(input);
    }
  }
  // Mirror the live config so Reset re-applies what is actually running
  // (only skip a field while the user is editing it).
  for (const name of PARAM_ORDER) {
    const input = document.getElementById(`param-${name}`);
    if (input && document.activeElement !== input) input.value = config[name];
  }
}
function readParams() {
  const params = {};
  for (const name of PARAM_ORDER) {
    const el = document.getElementById(`param-${name}`);
    if (el && el.value !== "") params[name] = Number(el.value);
  }
  return params;
}

function buildLoadouts(loadouts) {
  const sel = document.getElementById("loadout");
  if (!loadouts || sel.options.length) return;
  sel.appendChild(new Option("custom / edited", ""));
  for (const l of loadouts) sel.appendChild(new Option(l.label, l.id));
  sel.onchange = () => {
    if (sel.value)
      send({ cmd: "loadout", id: sel.value,
             seed: Number(document.getElementById("seed").value) || 0 });
  };
}

function syncControls(m) {
  buildParams(m.config);
  buildLoadouts(m.loadouts);
  const play = document.getElementById("btn-play");
  play.textContent = m.playing ? "⏸ Pause" : "▶ Play";
  play.classList.toggle("primary", m.playing);
  document.getElementById("stat-t").textContent = m.t;
  document.getElementById("stat-seed").textContent = m.seed;
  document.getElementById("stat-enc").textContent = m.encoding;
  if (document.activeElement?.id !== "chk-learning")
    document.getElementById("chk-learning").checked = m.learning;
  if (document.activeElement?.id !== "chk-allo")
    document.getElementById("chk-allo").checked = m.encoding === "allocentric";
}

function doReset() {
  send({
    cmd: "reset",
    seed: Number(document.getElementById("seed").value) || 0,
    params: readParams(),
    encoding: document.getElementById("chk-allo").checked ? "allocentric" : "egocentric",
  });
}

document.getElementById("btn-play").onclick = () =>
  send({ cmd: latest && latest.playing ? "pause" : "play" });
document.getElementById("btn-step1").onclick = () => send({ cmd: "step", n: 1 });
document.getElementById("btn-step10").onclick = () => send({ cmd: "step", n: 10 });
document.getElementById("btn-step100").onclick = () => send({ cmd: "step", n: 100 });
document.getElementById("btn-reset").onclick = doReset;
document.getElementById("chk-learning").onchange = (e) =>
  send({ cmd: "learning", enabled: e.target.checked });
document.getElementById("chk-allo").onchange = doReset;  // encoding change needs a rebuild
document.getElementById("node-idx").onchange = (e) =>
  send({ cmd: "select_node", index: Number(e.target.value) || 0 });

const speedEl = document.getElementById("speed");
const sliderToSps = (v) => Math.round(Math.pow(10, (v / 100) * 3.6));
speedEl.oninput = () => {
  const sps = sliderToSps(Number(speedEl.value));
  document.getElementById("speed-val").textContent = `${sps}/s`;
  send({ cmd: "speed", sps });
};
speedEl.oninput();

document.addEventListener("keydown", (ev) => {
  if (ev.target.tagName === "INPUT") return;
  if (ev.code === "Space") { ev.preventDefault(); document.getElementById("btn-play").click(); }
  if (ev.key === "s") document.getElementById("btn-step1").click();
  if (ev.key === "S") document.getElementById("btn-step10").click();
});

// ---------- render ----------------------------------------------------------
function drawAll(m) {
  drawField(m);
  drawSensors(m);
  drawEffectors(m);
  drawScore(m);
  drawAngle();
  drawProp();
  drawHist(m);
  drawNode(m);
  drawStats(m);
}

fitWide();
connect();
