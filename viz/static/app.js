/* Frontend for the homeostatic reservoir visualizer.
 *
 * All model state comes from the server (the tested `homeostasis` package);
 * this file only renders frames and sends commands. Per-step series arrive
 * batched per frame so the charts are exact (one column per model step, no
 * subsampling), regardless of playback speed.
 */

"use strict";

// ---------- palette (matches style.css) ------------------------------------
const C = {
  bg: "#14171c", panel: "#10141a", grid: "#2a313c",
  text: "#d7dde6", dim: "#8b95a3",
  red: "#ff5d5d", blue: "#5da9ff", green: "#3fd68f",
  yellow: "#ffd23f", pink: "#ff9ecb", accent: "#58a6ff", white: "#e8edf4",
};

// ---------- ring-ish buffers for the strip charts ---------------------------
const STRIP_CAP = 1920; // steps kept for heading/error/prop charts (0.5 px/step)
const buf = {
  heading: [], stim: [], err: [], speed: [], targetSpeed: [], prop: [], flip: [],
  clear() {
    this.heading = []; this.stim = []; this.err = []; this.speed = [];
    this.targetSpeed = []; this.prop = []; this.flip = [];
  },
  push(entry, prevDir) {
    this.heading.push(entry.heading);
    this.stim.push(entry.stim);
    this.err.push(entry.err);
    this.speed.push(entry.speed);
    this.targetSpeed.push(entry.target_speed);
    this.prop.push(entry.prop);
    this.flip.push(prevDir !== null && entry.dir !== prevDir);
    while (this.heading.length > STRIP_CAP) {
      this.heading.shift(); this.stim.shift(); this.err.shift(); this.speed.shift();
      this.targetSpeed.shift();
      this.prop.shift(); this.flip.shift();
    }
  },
};
const spark = { x: [], target: [], threshold: [], cap: 380,
  clear() { this.x = []; this.target = []; this.threshold = []; },
  push(n) {
    this.x.push(n.x); this.target.push(n.target); this.threshold.push(n.threshold);
    while (this.x.length > this.cap) { this.x.shift(); this.target.shift(); this.threshold.shift(); }
  },
};

let lastT = -1;
let lastDir = null;
let lastNodeIdx = 0;
let nNodes = 200;
let latest = null;           // last frame received
let stepTimestamps = [];     // for measured steps/sec
let displayedConfigSignature = null;

// ---------- websocket -------------------------------------------------------
let ws = null;
const connEl = document.getElementById("conn");

function connect() {
  ws = new WebSocket(`ws://${location.host}/ws`);
  ws.onopen = () => { connEl.textContent = "connected"; connEl.className = "conn ok"; };
  ws.onclose = () => {
    connEl.textContent = "disconnected — retrying…"; connEl.className = "conn bad";
    setTimeout(connect, 1000);
  };
  ws.onmessage = (ev) => handleFrame(JSON.parse(ev.data));
}
function send(obj) { if (ws && ws.readyState === 1) ws.send(JSON.stringify(obj)); }

// ---------- frame handling --------------------------------------------------
function handleFrame(msg) {
  if (msg.type !== "frame") return;

  const cfgN = msg.config.n_nodes;
  if (msg.t < lastT || cfgN !== nNodes) {   // reset happened
    buf.clear(); spark.clear();
    clearRaster();
    lastDir = null;
    nNodes = cfgN;
  }
  lastT = msg.t;
  latest = msg;

  for (const e of msg.series) {
    buf.push(e, lastDir);
    lastDir = e.dir;
    pushRasterColumn(e.spikes);
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
const cv = (id) => {
  const c = document.getElementById(id);
  return [c, c.getContext("2d")];
};
const [arenaC, arenaX] = cv("arena");
const [sensC, sensX] = cv("sensors");
const [effC, effX] = cv("effectors");
const [stripC, stripX] = cv("strip");
const [errC, errX] = cv("errstrip");
const [motionSpeedC, motionSpeedX] = cv("speedchart");
const [rasterC, rasterX] = cv("raster");
const [propC, propX] = cv("propchart");
const [histC, histX] = cv("whist");
const [sparkC, sparkX] = cv("node-spark");
spark.cap = sparkC.width;

function bg(x, c) { x.fillStyle = C.panel; x.fillRect(0, 0, c.width, c.height); }

// Wide charts stretch to their column; match the backing store to the CSS
// size so pixels stay 1:1 (setting .width clears the canvas, so the raster
// history restarts on resize — the strip charts redraw from their buffers).
function fitWideCanvases() {
  for (const c of [stripC, errC, motionSpeedC, rasterC, propC]) {
    const w = Math.max(300, Math.round(c.clientWidth) || c.width);
    if (c.width !== w) c.width = w;
  }
  clearRaster();
}
let resizeTimer = null;
window.addEventListener("resize", () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => { fitWideCanvases(); if (latest) drawAll(latest); }, 150);
});

// ---------- arena -----------------------------------------------------------
const rad = (d) => (d * Math.PI) / 180;
function arenaXY(angleDeg, r) {
  // math convention: 0 deg = east, counter-clockwise positive; canvas y down.
  const cx = arenaC.width / 2, cy = arenaC.height / 2;
  return [cx + r * Math.cos(rad(angleDeg)), cy - r * Math.sin(rad(angleDeg))];
}

function drawArena(m) {
  bg(arenaX, arenaC);
  const cx = arenaC.width / 2, cy = arenaC.height / 2;
  const R_STIM = Math.min(cx, cy) - 22, R_AGENT = Math.round(R_STIM * 0.48);
  const now = m.now;

  // field of view shading: heading ± 90
  arenaX.beginPath();
  arenaX.moveTo(cx, cy);
  arenaX.arc(cx, cy, R_STIM + 16, -rad(now.heading + 90), -rad(now.heading - 90));
  arenaX.closePath();
  arenaX.fillStyle = "rgba(88,166,255,0.06)";
  arenaX.fill();

  // stimulus orbit
  arenaX.beginPath();
  arenaX.arc(cx, cy, R_STIM, 0, Math.PI * 2);
  arenaX.strokeStyle = C.grid;
  arenaX.stroke();

  // agent body
  arenaX.beginPath();
  arenaX.arc(cx, cy, R_AGENT, 0, Math.PI * 2);
  arenaX.fillStyle = "rgba(255,158,203,0.25)";
  arenaX.fill();
  arenaX.strokeStyle = C.pink;
  arenaX.stroke();

  // sensor ticks colored by activation
  const offsets = now.sensor_offsets, acts = now.sensors;
  for (let i = 0; i < offsets.length; i++) {
    const a = now.heading + offsets[i];      // absolute direction (server adds heading itself for sensing)
    const v = acts[i];
    const [x1, y1] = arenaXY(a, R_AGENT + 5);
    const [x2, y2] = arenaXY(a, R_AGENT + 5 + 3 + 13 * v);
    arenaX.beginPath();
    arenaX.moveTo(x1, y1); arenaX.lineTo(x2, y2);
    const eye = i < offsets.length / 2 ? C.red : C.blue;
    arenaX.strokeStyle = v > 0.02 ? eye : "#3a4350";
    arenaX.lineWidth = v > 0.02 ? 2 : 1;
    arenaX.stroke();
  }
  arenaX.lineWidth = 1;

  // heading arrow
  const [hx, hy] = arenaXY(now.heading, R_AGENT);
  arenaX.beginPath(); arenaX.moveTo(cx, cy); arenaX.lineTo(hx, hy);
  arenaX.strokeStyle = C.white; arenaX.lineWidth = 2; arenaX.stroke();
  arenaX.lineWidth = 1;

  // eyes at heading ± 30 on the body
  const [lx, ly] = arenaXY(now.heading + 30, R_AGENT);
  const [rx, ry] = arenaXY(now.heading - 30, R_AGENT);
  arenaX.beginPath(); arenaX.arc(lx, ly, 5, 0, 7); arenaX.fillStyle = C.red; arenaX.fill();
  arenaX.beginPath(); arenaX.arc(rx, ry, 5, 0, 7); arenaX.fillStyle = C.blue; arenaX.fill();

  // stimulus
  const [sx, sy] = arenaXY(now.stim, R_STIM);
  arenaX.beginPath(); arenaX.arc(sx, sy, 8, 0, 7);
  arenaX.fillStyle = C.green; arenaX.fill();

  // direction hint arrow on orbit
  const dirOffset = now.dir * 8;
  const [ax1, ay1] = arenaXY(now.stim + dirOffset, R_STIM + 14);
  arenaX.fillStyle = C.dim;
  arenaX.beginPath(); arenaX.arc(ax1, ay1, 2.5, 0, 7); arenaX.fill();

  document.getElementById("arena-readout").textContent =
    `heading ${now.heading.toFixed(1)}°  ·  stimulus ${now.stim.toFixed(1)}° ` +
    `(${now.dir === 1 ? "CCW" : "CW"}, ${now.speed.toFixed(2)}°/step)  ·  ` +
    `error ${now.err.toFixed(1)}°  ·  ΔH ${now.dh.toFixed(2)}°`;
}

// arena drag -> manual stimulus
let dragging = false;
function arenaAngle(ev) {
  const r = arenaC.getBoundingClientRect();
  const x = ((ev.clientX - r.left) / r.width) * arenaC.width - arenaC.width / 2;
  const y = ((ev.clientY - r.top) / r.height) * arenaC.height - arenaC.height / 2;
  return ((Math.atan2(-y, x) * 180) / Math.PI + 360) % 360;
}
arenaC.addEventListener("pointerdown", (ev) => {
  dragging = true;
  arenaC.setPointerCapture(ev.pointerId);
  if (!document.getElementById("chk-manual").checked) {
    document.getElementById("chk-manual").checked = true;
    send({ cmd: "stim_mode", manual: true });
  }
  send({ cmd: "stim_set", angle: arenaAngle(ev) });
});
arenaC.addEventListener("pointermove", (ev) => {
  if (dragging) send({ cmd: "stim_set", angle: arenaAngle(ev) });
});
arenaC.addEventListener("pointerup", () => { dragging = false; });

// ---------- sensor bars -----------------------------------------------------
function drawSensors(m) {
  bg(sensX, sensC);
  const acts = m.now.sensors, n = acts.length;
  const w = sensC.width / n;
  for (let i = 0; i < n; i++) {
    const h = acts[i] * (sensC.height - 14);
    sensX.fillStyle = i < n / 2 ? C.red : C.blue;
    sensX.fillRect(i * w + 0.5, sensC.height - h, Math.max(1, w - 1), h);
  }
  sensX.strokeStyle = C.grid;
  sensX.beginPath();
  sensX.moveTo(sensC.width / 2, 0); sensX.lineTo(sensC.width / 2, sensC.height);
  sensX.stroke();
  sensX.fillStyle = C.dim; sensX.font = "10px monospace";
  sensX.fillText("1.0 —", 2, 10);
}

// ---------- effectors -------------------------------------------------------
function drawEffectors(m) {
  bg(effX, effC);
  const [l, r] = m.now.outputs;
  const bw = Math.round(effC.width * 0.27), base = effC.height - 18, maxH = effC.height - 34;
  const lx = effC.width / 2 - bw - 20, rx = effC.width / 2 + 20;
  effX.fillStyle = "#39424f";
  effX.fillRect(lx, base - maxH, bw, maxH);
  effX.fillRect(rx, base - maxH, bw, maxH);
  effX.fillStyle = C.accent;
  effX.fillRect(lx, base - l * maxH, bw, l * maxH);
  effX.fillRect(rx, base - r * maxH, bw, r * maxH);
  effX.fillStyle = C.text; effX.font = "12px monospace"; effX.textAlign = "center";
  effX.fillText(`left ${l.toFixed(2)}`, lx + bw / 2, effC.height - 4);
  effX.fillText(`right ${r.toFixed(2)}`, rx + bw / 2, effC.height - 4);
  effX.fillText(`ΔH = 10·(L−R) = ${m.now.dh.toFixed(2)}°`, effC.width / 2, 12);
  effX.textAlign = "left";
}

// ---------- heading / error strips -----------------------------------------
function stripXpos(i, len, width) { return width - (len - i) * (width / STRIP_CAP); }

function drawStrip() {
  bg(stripX, stripC);
  const W = stripC.width, H = stripC.height, len = buf.stim.length;
  stripX.fillStyle = C.dim; stripX.font = "10px monospace";
  for (const y of [0, 90, 180, 270, 360]) {
    const py = H - 12 - (y / 360) * (H - 24);
    stripX.fillText(String(y), 2, py + 3);
    stripX.strokeStyle = C.grid; stripX.globalAlpha = 0.5;
    stripX.beginPath(); stripX.moveTo(24, py); stripX.lineTo(W, py); stripX.stroke();
    stripX.globalAlpha = 1;
  }
  const yFor = (deg) => H - 12 - (deg / 360) * (H - 24);
  for (let i = 0; i < len; i++) {
    const x = stripXpos(i, len, W);
    if (x < 24) continue;
    if (buf.flip[i]) {
      stripX.fillStyle = "rgba(139,149,163,0.35)";
      stripX.fillRect(x, 0, 1, H);
    }
    stripX.fillStyle = C.white;
    stripX.fillRect(x, yFor(buf.stim[i]), 1.4, 1.8);
    stripX.fillStyle = C.red;
    stripX.fillRect(x, yFor(buf.heading[i]), 1.6, 2.2);
  }
}

function drawErrStrip() {
  bg(errX, errC);
  const W = errC.width, H = errC.height, len = buf.err.length;
  const yFor = (deg) => H / 2 - (deg / 180) * (H / 2 - 6);
  errX.fillStyle = "rgba(63,214,143,0.10)";
  errX.fillRect(0, yFor(45), W, yFor(-45) - yFor(45));
  errX.strokeStyle = C.grid;
  errX.beginPath(); errX.moveTo(0, H / 2); errX.lineTo(W, H / 2); errX.stroke();
  errX.fillStyle = C.dim; errX.font = "10px monospace";
  errX.fillText("+180", 2, 10); errX.fillText("0", 2, H / 2 + 3); errX.fillText("−180", 2, H - 3);
  errX.fillStyle = C.accent;
  for (let i = 0; i < len; i++) {
    const x = stripXpos(i, len, W);
    if (x < 24) continue;
    errX.fillRect(x, yFor(buf.err[i]), 1.4, 1.8);
  }
}

function drawSpeedChart(m) {
  bg(motionSpeedX, motionSpeedC);
  const W = motionSpeedC.width, H = motionSpeedC.height, len = buf.speed.length;
  const configuredMax = Math.max(m.config.stimulus_speed_max, m.config.stimulus_speed, 0.1);
  const observedMax = Math.max(...buf.speed, ...buf.targetSpeed, 0.1);
  const top = Math.max(configuredMax, observedMax) * 1.05;
  const yFor = (v) => H - 6 - (v / top) * (H - 16);
  motionSpeedX.fillStyle = C.dim; motionSpeedX.font = "10px monospace";
  motionSpeedX.fillText(`${top.toFixed(2)}°/step`, 2, 10);
  motionSpeedX.fillText("0", 2, H - 3);
  const drawSeries = (values, color, dash) => {
    motionSpeedX.strokeStyle = color;
    motionSpeedX.setLineDash(dash || []);
    motionSpeedX.beginPath();
    let started = false;
    for (let i = 0; i < values.length; i++) {
      const x = stripXpos(i, values.length, W);
      if (x < 36) continue;
      const y = yFor(values[i]);
      if (!started) { motionSpeedX.moveTo(x, y); started = true; }
      else motionSpeedX.lineTo(x, y);
    }
    motionSpeedX.stroke();
    motionSpeedX.setLineDash([]);
  };
  drawSeries(buf.targetSpeed, C.dim, [4, 3]);
  drawSeries(buf.speed, C.green);
}

// ---------- spike raster (incremental blit) ---------------------------------
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
  for (const i of spikeIdx) {
    rasterX.fillRect(W - 1, i * rowH, 1, Math.max(1, rowH));
  }
}

rasterC.addEventListener("click", (ev) => {
  const r = rasterC.getBoundingClientRect();
  const y = ((ev.clientY - r.top) / r.height) * rasterC.height;
  const idx = Math.min(nNodes - 1, Math.max(0, Math.floor((y / rasterC.height) * nNodes)));
  document.getElementById("node-idx").value = idx;
  send({ cmd: "select_node", index: idx });
});

function drawPropChart() {
  bg(propX, propC);
  const W = propC.width, H = propC.height, len = buf.prop.length;
  propX.fillStyle = C.dim; propX.font = "10px monospace";
  propX.fillText("prop. spiked  1.0 —", 2, 10);
  propX.fillText("0 —", 2, H - 3);
  propX.strokeStyle = C.white;
  propX.beginPath();
  let started = false;
  for (let i = 0; i < len; i++) {
    const x = stripXpos(i, len, W);
    if (x < 24) continue;
    const y = H - 4 - buf.prop[i] * (H - 14);
    if (!started) { propX.moveTo(x, y); started = true; } else propX.lineTo(x, y);
  }
  propX.stroke();
}

// ---------- weights histogram ----------------------------------------------
function drawHist(m) {
  bg(histX, histC);
  const { counts, edges } = m.hist;
  const W = histC.width, H = histC.height;
  const maxC = Math.max(...counts, 1);
  const bw = W / counts.length;
  histX.fillStyle = C.accent;
  for (let i = 0; i < counts.length; i++) {
    const h = (counts[i] / maxC) * (H - 22);
    histX.fillRect(i * bw + 0.5, H - 14 - h, Math.max(1, bw - 1), h);
  }
  // zero line
  const lo = edges[0], hi = edges[edges.length - 1];
  if (lo < 0 && hi > 0) {
    const zx = ((0 - lo) / (hi - lo)) * W;
    histX.strokeStyle = C.dim;
    histX.setLineDash([3, 3]);
    histX.beginPath(); histX.moveTo(zx, 0); histX.lineTo(zx, H - 14); histX.stroke();
    histX.setLineDash([]);
  }
  histX.fillStyle = C.dim; histX.font = "10px monospace";
  histX.textAlign = "left"; histX.fillText(lo.toFixed(2), 2, H - 3);
  histX.textAlign = "right"; histX.fillText(hi.toFixed(2), W - 2, H - 3);
  histX.textAlign = "left";
}

// ---------- node inspector --------------------------------------------------
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
  const lo = Math.min(...all), hi = Math.max(...all);
  const span = hi - lo || 1;
  const yFor = (v) => H - 4 - ((v - lo) / span) * (H - 8);
  const drawSeries = (arr, color, dash) => {
    sparkX.strokeStyle = color;
    sparkX.setLineDash(dash || []);
    sparkX.beginPath();
    arr.forEach((v, i) => {
      const x = W - (arr.length - i);
      if (i === 0) sparkX.moveTo(x, yFor(v)); else sparkX.lineTo(x, yFor(v));
    });
    sparkX.stroke();
    sparkX.setLineDash([]);
  };
  drawSeries(spark.threshold, C.dim, [3, 3]);
  drawSeries(spark.target, C.yellow);
  drawSeries(spark.x, C.accent);
  sparkX.fillStyle = C.dim; sparkX.font = "10px monospace";
  sparkX.fillText(`x′ (blue) · T (yellow) · 2T (dashed) — per frame`, 4, 10);
}

// ---------- stats -----------------------------------------------------------
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
    ["stimulus motion", m.motion_mode],
    ["stimulus speed", `${now.speed.toFixed(3)} → ${now.target_speed.toFixed(3)}°/step`],
    ["next speed target", now.steps_until_speed_change === null ? "fixed" : `${now.steps_until_speed_change} steps`],
    ["next reversal", `${now.steps_until_direction_change} steps`],
    ["measured speed", m.playing ? `${measured.toFixed(0)} steps/s` : "paused"],
  ];
  document.getElementById("net-stats").innerHTML =
    rows.map(([k, v]) => `<span class="k">${k}</span><span class="v">${v}</span>`).join("");
  const fp = m.fingerprint;
  document.getElementById("fingerprint").textContent =
    `fingerprint @ t=${m.t}: Σx=${fp.sum_x} ΣT=${fp.sum_targets} ΣW=${fp.sum_weights}` +
    (m.custom_params
      ? " — custom parameters active; the fingerprint script replays defaults only"
      : ` — verify with scripts/fingerprint.py --seed ${m.seed} --steps ${m.t}` +
        " (only valid for an untouched run: no manual stimulus, learning toggles, or flips)");
}

// ---------- controls --------------------------------------------------------
const PARAM_ORDER = [
  "n_nodes", "p_link", "input_weight", "weight_init_mean", "weight_init_sd",
  "leak", "target_lr", "threshold_ratio", "gain", "stimulus_speed", "reverse_every",
  "stimulus_speed_min", "stimulus_speed_max", "speed_smoothing",
  "speed_change_min_steps", "speed_change_max_steps",
  "reverse_min_steps", "reverse_max_steps",
];
function buildParamsPanel(config) {
  const box = document.getElementById("params");
  if (box.childElementCount) return;
  for (const name of PARAM_ORDER) {
    const label = document.createElement("label");
    label.textContent = name;
    const input = document.createElement("input");
    input.id = `param-${name}`;
    input.type = "number";
    input.step = "any";
    input.value = config[name];
    input.addEventListener("input", () => {
      document.getElementById("loadout").value = "";
      document.getElementById("loadout-metrics").textContent = "edited parameters";
    });
    box.appendChild(label);
    box.appendChild(input);
  }
}
function syncParamsPanel(config) {
  buildParamsPanel(config);
  const signature = JSON.stringify(PARAM_ORDER.map((name) => config[name]));
  if (signature === displayedConfigSignature) return false;
  for (const name of PARAM_ORDER) {
    const input = document.getElementById(`param-${name}`);
    if (input) input.value = config[name];
  }
  displayedConfigSignature = signature;
  return true;
}
function readParams() {
  const params = {};
  for (const name of PARAM_ORDER) {
    const el = document.getElementById(`param-${name}`);
    if (el && el.value !== "") params[name] = Number(el.value);
  }
  return params;
}

function formatLoadoutMetrics(loadout) {
  if (!loadout) return "";
  const m = loadout.metrics;
  return `score ${m.score.toFixed(3)} · dir-agree ${m.dir_agree.toFixed(2)} · prop spiked ${m.prop_spiked.toFixed(2)}`;
}
function buildLoadoutSelect(loadouts) {
  const select = document.getElementById("loadout");
  if (select.options.length) return;
  select.appendChild(new Option("custom / edited", ""));
  for (const loadout of loadouts) {
    select.appendChild(new Option(loadout.label, loadout.id));
  }
  select.onchange = () => {
    const loadout = loadouts.find((item) => item.id === select.value);
    document.getElementById("loadout-metrics").textContent = formatLoadoutMetrics(loadout);
  };
}

function syncControls(m) {
  const configChanged = syncParamsPanel(m.config);
  buildLoadoutSelect(m.loadouts);
  if (configChanged) {
    const select = document.getElementById("loadout");
    select.value = m.active_loadout || "";
    const loadout = m.loadouts.find((item) => item.id === select.value);
    document.getElementById("loadout-metrics").textContent = formatLoadoutMetrics(loadout);
  }
  document.getElementById("btn-play").textContent = m.playing ? "⏸ Pause" : "▶ Play";
  document.getElementById("btn-play").classList.toggle("primary", m.playing);
  document.getElementById("stat-t").textContent = m.t;
  document.getElementById("stat-seed").textContent = m.seed;
  document.getElementById("stat-dir").textContent = m.now.dir === 1 ? "↺ CCW" : "↻ CW";
  document.getElementById("stat-motion").textContent = m.motion_mode;
  document.getElementById("stat-stim-speed").textContent = `${m.now.speed.toFixed(2)}°/step`;
  if (document.activeElement?.id !== "chk-learning")
    document.getElementById("chk-learning").checked = m.learning;
  if (document.activeElement?.id !== "chk-manual")
    document.getElementById("chk-manual").checked = m.manual_stimulus;
  if (document.activeElement?.id !== "chk-variable")
    document.getElementById("chk-variable").checked = m.motion_mode === "variable";
}

document.getElementById("btn-play").onclick = () =>
  send({ cmd: latest && latest.playing ? "pause" : "play" });
document.getElementById("btn-step1").onclick = () => send({ cmd: "step", n: 1 });
document.getElementById("btn-step10").onclick = () => send({ cmd: "step", n: 10 });
document.getElementById("btn-step100").onclick = () => send({ cmd: "step", n: 100 });
document.getElementById("btn-flip").onclick = () => send({ cmd: "stim_flip" });
document.getElementById("btn-reset").onclick = () => {
  send({
    cmd: "reset",
    seed: Number(document.getElementById("seed").value) || 0,
    params: readParams(),
    motion_mode: document.getElementById("chk-variable").checked ? "variable" : "constant",
  });
};
document.getElementById("btn-loadout").onclick = () => {
  const id = document.getElementById("loadout").value;
  if (!id) return;
  send({
    cmd: "loadout",
    id,
    seed: Number(document.getElementById("seed").value) || 0,
    motion_mode: document.getElementById("chk-variable").checked ? "variable" : "constant",
  });
};
document.getElementById("chk-learning").onchange = (ev) =>
  send({ cmd: "learning", enabled: ev.target.checked });
document.getElementById("chk-variable").onchange = (ev) =>
  send({
    cmd: "motion_mode",
    mode: ev.target.checked ? "variable" : "constant",
    params: readParams(),
  });
document.getElementById("chk-manual").onchange = (ev) =>
  send({ cmd: "stim_mode", manual: ev.target.checked });
document.getElementById("node-idx").onchange = (ev) =>
  send({ cmd: "select_node", index: Number(ev.target.value) || 0 });

const speedEl = document.getElementById("speed");
function sliderToSps(v) { return Math.round(Math.pow(10, (v / 100) * 3.3)); } // 1..2000
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
  drawArena(m);
  drawSensors(m);
  drawEffectors(m);
  drawStrip();
  drawErrStrip();
  drawSpeedChart(m);
  drawPropChart();
  drawHist(m);
  drawNode(m);
  drawStats(m);
}

fitWideCanvases();
connect();
