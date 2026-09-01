/* Frontend for the single-node lab.
 *
 * All numbers come from the server (`/lab/api/single_node`, which runs the
 * tested `homeostasis` package); this file only maps sliders to query params
 * and renders the returned traces. No model logic lives here.
 */

"use strict";

const C = {
  panel: "#10141a", grid: "#2a313c", text: "#d7dde6", dim: "#8b95a3",
  red: "#ff5d5d", blue: "#5da9ff", green: "#3fd68f", yellow: "#ffd23f",
  pink: "#ff9ecb", accent: "#58a6ff", white: "#e8edf4", orange: "#ffa04d",
};

// Exemplar parameter sets, one per end state plus a bistable cell. Each was
// verified against /lab/api/single_node (steps 3000 and 6000, target_lr 0.01):
//   dead-floor   -> dead-floor   (f=0,     T at floor, mean E = -0.90)
//   silent-comf  -> silent-comf  (f=0,     T -> 1.20 = mu/leak, E -> 0)
//   spiking      -> spiking      (f=0.334  vs duty law 0.3364, T=1.084)
//   frozen-cycle -> frozen-cycle (f=0.090, mean E = -0.38, T stuck at floor)
//   bistable     -> cold: spiking (f=0.200 vs duty law 0.2017)
//                   hot:  silent-comf (T -> mu/leak ~ 9.4)
// (verified with the slider-quantized values the page actually sends)
const PRESETS = {
  dead:     { mu: 0.05, leak: 0.50, rho: 2.0, lr: 0.01, hot: false },
  silent:   { mu: 0.30, leak: 0.25, rho: 2.0, lr: 0.01, hot: false },
  spiking:  { mu: 1.00, leak: 0.25, rho: 2.0, lr: 0.01, hot: false },
  frozen:   { mu: 0.14, leak: 0.05, rho: 1.2, lr: 0.01, hot: false },
  bistable: { mu: 0.47, leak: 0.05, rho: 2.0, lr: 0.01, hot: false },
};

// ---------- sliders ---------------------------------------------------------
// Each slider runs 0..1000 and maps to its parameter range (log or linear).
const SLIDERS = {
  mu:   { el: "sl-mu",   out: "v-mu",   lo: 0.05,   hi: 20.0, log: true,  fmt: (v) => v.toFixed(v < 1 ? 3 : 2) },
  leak: { el: "sl-leak", out: "v-leak", lo: 0.01,   hi: 0.95, log: false, fmt: (v) => v.toFixed(2) },
  rho:  { el: "sl-rho",  out: "v-rho",  lo: 1.05,   hi: 4.0,  log: false, fmt: (v) => v.toFixed(2) },
  lr:   { el: "sl-lr",   out: "v-lr",   lo: 0.0001, hi: 0.2,  log: true,  fmt: (v) => v.toPrecision(2) },
};

function sliderValue(s) {
  const u = document.getElementById(s.el).value / 1000;
  return s.log ? s.lo * Math.exp(u * Math.log(s.hi / s.lo)) : s.lo + u * (s.hi - s.lo);
}
function setSlider(s, v) {
  const clamped = Math.min(Math.max(v, s.lo), s.hi);
  const u = s.log
    ? Math.log(clamped / s.lo) / Math.log(s.hi / s.lo)
    : (clamped - s.lo) / (s.hi - s.lo);
  document.getElementById(s.el).value = Math.round(u * 1000);
}
function syncLabels() {
  for (const s of Object.values(SLIDERS)) {
    document.getElementById(s.out).textContent = s.fmt(sliderValue(s));
  }
}

function params() {
  return {
    mu: sliderValue(SLIDERS.mu),
    leak: sliderValue(SLIDERS.leak),
    rho: sliderValue(SLIDERS.rho),
    target_lr: sliderValue(SLIDERS.lr),
    steps: Math.min(Math.max(parseInt(document.getElementById("in-steps").value) || 3000, 10), 20000),
    ic: document.getElementById("chk-hot").checked ? "hot" : "cold",
  };
}

// ---------- fetch -----------------------------------------------------------
const connEl = document.getElementById("conn");
let latest = null;
let runSeq = 0;

async function run() {
  const p = params();
  const seq = ++runSeq;
  connEl.textContent = "running…";
  connEl.className = "conn";
  const t0 = performance.now();
  try {
    const qs = new URLSearchParams({
      mu: p.mu.toFixed(5), leak: p.leak.toFixed(4), rho: p.rho.toFixed(4),
      target_lr: p.target_lr.toPrecision(4), steps: p.steps, ic: p.ic,
    });
    const res = await fetch(`/lab/api/single_node?${qs}`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    if (seq !== runSeq) return; // a newer run superseded this one
    latest = data;
    connEl.textContent = `ok · ${(performance.now() - t0).toFixed(0)} ms`;
    connEl.className = "conn ok";
    render(data);
  } catch (err) {
    if (seq !== runSeq) return;
    connEl.textContent = `error: ${err.message}`;
    connEl.className = "conn bad";
  }
}

// ---------- summary strip ---------------------------------------------------
function render(data) {
  const s = data.summary, p = data.params;
  const chip = document.getElementById("state-chip");
  chip.textContent = s.state;
  chip.className = `chip ${s.state}`;
  document.getElementById("tail-n").textContent = s.tail;
  document.getElementById("f-late").textContent = s.f_late.toFixed(4);
  document.getElementById("duty-pred").textContent =
    s.duty_pred === null ? "— (no spikes)" : s.duty_pred.toFixed(4);
  document.getElementById("t-late").textContent = s.T_late.toFixed(4);
  document.getElementById("mean-e").textContent = s.meanE_late.toFixed(4);
  document.getElementById("abs-e").textContent = s.absE_late.toFixed(4);
  document.getElementById("boundaries").textContent =
    `μ=${p.mu.toFixed(4)} · comfort split leak·T_floor=${s.mu_comfort.toFixed(4)}` +
    ` · cold-start crossing ρ·leak·T₀=${s.mu_spike_cold.toFixed(4)}` +
    ` · x*=μ/leak=${(p.mu / p.leak).toFixed(3)}`;
  document.getElementById("stat").textContent =
    `steps=${p.steps} · ic=${p.ic} · lr=${p.target_lr}`;
  drawTrace(data);
  drawError(data);
}

// ---------- plotting --------------------------------------------------------
const cv = (id) => { const c = document.getElementById(id); return [c, c.getContext("2d")]; };
const [traceC, traceX] = cv("trace");
const [errC, errX] = cv("etrace");
const PAD = { l: 52, r: 10, t: 10, b: 18 };

function bg(x, c) { x.fillStyle = C.panel; x.fillRect(0, 0, c.width, c.height); }

function fitWide() {
  for (const c of [traceC, errC]) {
    const w = Math.max(360, Math.round(c.clientWidth) || c.width);
    if (c.width !== w) c.width = w;
  }
}
let resizeTimer = null;
window.addEventListener("resize", () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => { fitWide(); if (latest) render(latest); }, 150);
});

// Per-column min/max of a series, seeded with the previous column's closing
// value so the envelope stays connected. Returns null when a plain polyline
// is fine (fewer than ~2 points per pixel).
function envelope(arr, cols) {
  const n = arr.length;
  if (n <= cols * 2) return null;
  const lo = new Float64Array(cols), hi = new Float64Array(cols), mean = new Float64Array(cols);
  let prev = arr[0];
  for (let cx = 0; cx < cols; cx++) {
    const i0 = Math.floor((cx / cols) * n);
    const i1 = Math.max(i0 + 1, Math.floor(((cx + 1) / cols) * n));
    let l = prev, h = prev, sum = 0;
    for (let i = i0; i < i1; i++) {
      const v = arr[i];
      if (v < l) l = v;
      if (v > h) h = v;
      sum += v;
    }
    prev = arr[i1 - 1];
    lo[cx] = l; hi[cx] = h; mean[cx] = sum / (i1 - i0);
  }
  return { lo, hi, mean };
}

function polyline(x, pts, xFor, yFor) {
  x.beginPath();
  for (let i = 0; i < pts.length; i++) {
    const px = xFor(i), py = yFor(pts[i]);
    i ? x.lineTo(px, py) : x.moveTo(px, py);
  }
  x.stroke();
}

// Stroke a series honestly at any zoom: a plain polyline when there are few
// points per pixel; otherwise a per-column min/max envelope. With band set
// (for fast-oscillating series) the envelope is a translucent band plus a
// solid line along the per-column peaks ("hi", for x′ against its threshold)
// or per-column means ("mean", for E), so slower series stay visible on top.
function strokeSeries(x, arr, xFor, yFor, color, lw = 1.4, dash = null, band = null) {
  const cols = Math.max(1, Math.round(xFor(arr.length - 1) - xFor(0)));
  const env = envelope(arr, cols);
  x.strokeStyle = color;
  x.lineWidth = lw;
  if (dash) x.setLineDash(dash);
  if (env === null) {
    polyline(x, arr, xFor, yFor);
  } else if (band) {
    const x0 = xFor(0);
    x.globalAlpha = 0.28;
    x.fillStyle = color;
    for (let cx = 0; cx < cols; cx++) {
      const yh = yFor(env.hi[cx]);
      x.fillRect(x0 + cx, yh, 1, Math.max(1, yFor(env.lo[cx]) - yh));
    }
    x.globalAlpha = 1;
    x.lineWidth = 1;
    polyline(x, env[band], (cx) => x0 + cx + 0.5, yFor);
  } else {
    x.beginPath();
    const x0 = xFor(0);
    for (let cx = 0; cx < cols; cx++) {
      x.moveTo(x0 + cx + 0.5, yFor(env.hi[cx]));
      x.lineTo(x0 + cx + 0.5, yFor(env.lo[cx]) + 0.5);
    }
    x.stroke();
  }
  if (dash) x.setLineDash([]);
  x.lineWidth = 1;
}

function frame(x, c, n, tail, yFor, ymin, ymax) {
  const W = c.width, H = c.height;
  const xFor = (i) => PAD.l + (i / Math.max(n - 1, 1)) * (W - PAD.l - PAD.r);
  // classification-tail shading
  x.fillStyle = "rgba(255,255,255,0.045)";
  x.fillRect(xFor(n - tail), PAD.t, xFor(n - 1) - xFor(n - tail), H - PAD.t - PAD.b);
  // zero line + frame
  x.strokeStyle = C.grid;
  x.strokeRect(PAD.l + 0.5, PAD.t + 0.5, W - PAD.l - PAD.r - 1, H - PAD.t - PAD.b - 1);
  if (ymin < 0 && ymax > 0) {
    x.beginPath(); x.moveTo(PAD.l, yFor(0)); x.lineTo(W - PAD.r, yFor(0)); x.stroke();
  }
  // axis labels
  x.fillStyle = C.dim;
  x.font = "10px monospace";
  x.textAlign = "right";
  x.fillText(fmtTick(ymax), PAD.l - 4, PAD.t + 8);
  x.fillText(fmtTick(ymin), PAD.l - 4, H - PAD.b);
  if (ymin < 0 && ymax > 0) x.fillText("0", PAD.l - 4, yFor(0) + 3);
  x.textAlign = "left";
  x.fillText("t=0", PAD.l, H - 5);
  x.textAlign = "right";
  x.fillText(`t=${n}`, W - PAD.r, H - 5);
  x.textAlign = "left";
  return xFor;
}
const fmtTick = (v) => Math.abs(v) >= 100 ? v.toFixed(0) : Math.abs(v) >= 1 ? v.toFixed(1) : v.toFixed(2);

function drawTrace(data) {
  const { x_post, T, threshold, spiked } = data.trace;
  const n = x_post.length;
  bg(traceX, traceC);
  let ymin = 0, ymax = -Infinity;
  for (const arr of [x_post, T, threshold]) {
    for (const v of arr) { if (v < ymin) ymin = v; if (v > ymax) ymax = v; }
  }
  const padY = (ymax - ymin) * 0.06 || 1;
  ymax += padY; ymin -= padY * 0.5;
  const H = traceC.height;
  const yFor = (v) => PAD.t + ((ymax - v) / (ymax - ymin)) * (H - PAD.t - PAD.b);
  const xFor = frame(traceX, traceC, n, data.summary.tail, yFor, ymin, ymax);

  strokeSeries(traceX, x_post, xFor, yFor, C.accent, 1.4, null, "hi");
  strokeSeries(traceX, threshold, xFor, yFor, C.dim, 1.1, [3, 3]);
  strokeSeries(traceX, T, xFor, yFor, C.yellow, 1.4);

  // spike ticks along the top edge (one per column that contains a spike)
  traceX.fillStyle = C.green;
  const cols = Math.max(1, Math.round(xFor(n - 1) - xFor(0)));
  if (n <= cols) {
    for (let i = 0; i < n; i++) if (spiked[i]) traceX.fillRect(xFor(i), PAD.t + 1, 1.2, 7);
  } else {
    for (let cx = 0; cx < cols; cx++) {
      const i0 = Math.floor((cx / cols) * n);
      const i1 = Math.max(i0 + 1, Math.floor(((cx + 1) / cols) * n));
      let any = false;
      for (let i = i0; i < i1 && !any; i++) any = spiked[i] === 1;
      if (any) traceX.fillRect(xFor(0) + cx, PAD.t + 1, 1, 7);
    }
  }
}

function drawError(data) {
  const E = data.trace.E;
  const n = E.length;
  bg(errX, errC);
  let m = 0.05; // keep the ±0.02 comfort band visible even for tiny errors
  for (const v of E) m = Math.max(m, Math.abs(v));
  const ymax = m * 1.08, ymin = -ymax;
  const H = errC.height;
  const yFor = (v) => PAD.t + ((ymax - v) / (ymax - ymin)) * (H - PAD.t - PAD.b);
  // comfort band first, so the frame and series draw over it
  errX.fillStyle = "rgba(93,169,255,0.10)";
  errX.fillRect(PAD.l, yFor(0.02), errC.width - PAD.l - PAD.r, yFor(-0.02) - yFor(0.02));
  const xFor = frame(errX, errC, n, data.summary.tail, yFor, ymin, ymax);
  strokeSeries(errX, E, xFor, yFor, C.red, 1.3, null, "mean");
}

// ---------- wiring ----------------------------------------------------------
for (const s of Object.values(SLIDERS)) {
  const el = document.getElementById(s.el);
  el.addEventListener("input", syncLabels);
  el.addEventListener("change", run); // auto-run on release
}
document.getElementById("in-steps").addEventListener("change", run);
document.getElementById("chk-hot").addEventListener("change", run);
document.getElementById("btn-run").addEventListener("click", run);
for (const btn of document.querySelectorAll("[data-preset]")) {
  btn.addEventListener("click", () => {
    const p = PRESETS[btn.dataset.preset];
    setSlider(SLIDERS.mu, p.mu);
    setSlider(SLIDERS.leak, p.leak);
    setSlider(SLIDERS.rho, p.rho);
    setSlider(SLIDERS.lr, p.lr);
    document.getElementById("chk-hot").checked = p.hot;
    syncLabels();
    run();
  });
}

fitWide();
setSlider(SLIDERS.mu, PRESETS.spiking.mu);
setSlider(SLIDERS.leak, PRESETS.spiking.leak);
setSlider(SLIDERS.rho, PRESETS.spiking.rho);
setSlider(SLIDERS.lr, PRESETS.spiking.lr);
syncLabels();
run();
