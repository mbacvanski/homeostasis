/* Frontend for the trajectory viewer.
 *
 * All simulation happens server-side (/lab/api/traj runs the tested
 * `homeostasis` package with the lab campaign's exact step order); this file
 * only maps controls to query params and renders the returned traces.
 */

"use strict";

const C = {
  panel: "#10141a", grid: "#2a313c", text: "#d7dde6", dim: "#8b95a3",
  red: "#ff5d5d", blue: "#5da9ff", green: "#3fd68f", yellow: "#ffd23f",
  pink: "#ff9ecb", accent: "#58a6ff",
};

// Batch status goes to its own element now that #conn shows the LIVE
// connection state (the live section is served by lab_traj_live.js).
const connEl = document.getElementById("batch-status") || document.getElementById("conn");
const batchBox = document.getElementById("batch-box");
let latest = null;
let runSeq = 0;

// ---------- fetch -----------------------------------------------------------
function params() {
  const swapRaw = document.getElementById("swap").value;
  return {
    variant: document.getElementById("variant").value,
    seed: Math.max(parseInt(document.getElementById("seed").value) || 0, 0),
    steps: Math.min(Math.max(parseInt(document.getElementById("steps").value) || 7200, 720), 14400),
    swap_at: swapRaw === "" ? -1 : Math.max(parseInt(swapRaw) || 0, 0),
    noise: parseFloat(document.getElementById("noise").value) || 0,
  };
}

function showNoise() {
  document.getElementById("v-noise").textContent =
    (parseFloat(document.getElementById("noise").value) || 0).toFixed(2);
}

async function run() {
  if (batchBox && !batchBox.open) return;  // batch is lazy: render only when shown
  const p = params();
  if (p.variant === "stack") {  // live-only loadout (needs common.py's pin_output_p)
    connEl.textContent = "stack is live-only — use the live section above";
    connEl.className = "conn";
    return;
  }
  const seq = ++runSeq;
  connEl.textContent = "running…";
  connEl.className = "conn";
  const t0 = performance.now();
  try {
    const res = await fetch(`/lab/api/traj?${new URLSearchParams(p)}`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    if (data.error) throw new Error(data.error);
    if (seq !== runSeq) return;
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

// ---------- plotting helpers (as /lab, honest at any density) ---------------
const cv = (id) => { const c = document.getElementById(id); return [c, c.getContext("2d")]; };
const [folC, folX] = cv("follow");
const [errC, errX] = cv("errc");
const [segC, segX] = cv("segc");
const PAD = { l: 56, r: 10, t: 10, b: 18 };

function bg(x, c) { x.fillStyle = C.panel; x.fillRect(0, 0, c.width, c.height); }

function fitWide() {
  for (const c of [folC, errC, segC]) {
    const w = Math.max(360, Math.round(c.clientWidth) || c.width);
    if (c.width !== w) c.width = w;
  }
}
let resizeTimer = null;
window.addEventListener("resize", () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => { fitWide(); if (latest) render(latest); }, 150);
});

function envelope(arr, cols) {
  const n = arr.length;
  if (n <= cols * 2) return null;
  const lo = new Float64Array(cols), hi = new Float64Array(cols);
  let prev = arr[0];
  for (let cx = 0; cx < cols; cx++) {
    const i0 = Math.floor((cx / cols) * n);
    const i1 = Math.max(i0 + 1, Math.floor(((cx + 1) / cols) * n));
    let l = prev, h = prev;
    for (let i = i0; i < i1; i++) {
      const v = arr[i];
      if (v < l) l = v;
      if (v > h) h = v;
    }
    prev = arr[i1 - 1];
    lo[cx] = l; hi[cx] = h;
  }
  return { lo, hi };
}

function strokeSeries(x, arr, xFor, yFor, color, lw = 1.4) {
  const cols = Math.max(1, Math.round(xFor(arr.length - 1) - xFor(0)));
  const env = envelope(arr, cols);
  x.strokeStyle = color;
  x.lineWidth = lw;
  x.beginPath();
  if (env === null) {
    for (let i = 0; i < arr.length; i++) {
      const px = xFor(i), py = yFor(arr[i]);
      i ? x.lineTo(px, py) : x.moveTo(px, py);
    }
  } else {
    const x0 = xFor(0);
    for (let cx = 0; cx < cols; cx++) {
      x.moveTo(x0 + cx + 0.5, yFor(env.hi[cx]));
      x.lineTo(x0 + cx + 0.5, yFor(env.lo[cx]) + 0.5);
    }
  }
  x.stroke();
  x.lineWidth = 1;
}

const fmtTick = (v) => Math.abs(v) >= 100 ? v.toFixed(0) : Math.abs(v) >= 1 ? v.toFixed(1) : v.toFixed(2);

function frame(x, c, tMax, yFor, ymin, ymax) {
  const W = c.width, H = c.height;
  x.strokeStyle = C.grid;
  x.strokeRect(PAD.l + 0.5, PAD.t + 0.5, W - PAD.l - PAD.r - 1, H - PAD.t - PAD.b - 1);
  if (ymin < 0 && ymax > 0) {
    x.beginPath(); x.moveTo(PAD.l, yFor(0)); x.lineTo(W - PAD.r, yFor(0)); x.stroke();
  }
  x.fillStyle = C.dim;
  x.font = "10px monospace";
  x.textAlign = "right";
  x.fillText(fmtTick(ymax), PAD.l - 4, PAD.t + 8);
  x.fillText(fmtTick(ymin), PAD.l - 4, H - PAD.b);
  if (ymin < 0 && ymax > 0) x.fillText("0", PAD.l - 4, yFor(0) + 3);
  x.textAlign = "left";
  x.fillText("t=0", PAD.l, H - 5);
  x.textAlign = "right";
  x.fillText(`t=${tMax}`, W - PAD.r, H - 5);
  x.textAlign = "left";
}

function xScale(c, tArr) {
  const t0 = tArr[0], t1 = tArr[tArr.length - 1] || 1;
  return (i) => PAD.l + ((tArr[i] - t0) / (t1 - t0)) * (c.width - PAD.l - PAD.r);
}

function markSwap(x, c, tArr, swapAt) {
  if (swapAt === null || swapAt === undefined || swapAt < 0) return;
  const t1 = tArr[tArr.length - 1] || 1;
  const px = PAD.l + (Math.min(swapAt, t1) / t1) * (c.width - PAD.l - PAD.r);
  x.strokeStyle = C.pink;
  x.setLineDash([4, 4]);
  x.beginPath(); x.moveTo(px, PAD.t); x.lineTo(px, c.height - PAD.b); x.stroke();
  x.setLineDash([]);
}

// ---------- panels ----------------------------------------------------------
function drawFollow(d) {
  const { heading, stimulus, t } = d.trace;
  bg(folX, folC);
  let ymin = Infinity, ymax = -Infinity;
  for (const arr of [heading, stimulus]) for (const v of arr) { if (v < ymin) ymin = v; if (v > ymax) ymax = v; }
  const pad = (ymax - ymin) * 0.06 || 10;
  ymin -= pad; ymax += pad;
  const H = folC.height;
  const yFor = (v) => PAD.t + ((ymax - v) / (ymax - ymin)) * (H - PAD.t - PAD.b);
  frame(folX, folC, d.params.steps, yFor, ymin, ymax);
  const xFor = xScale(folC, t);
  strokeSeries(folX, stimulus, xFor, yFor, C.yellow, 1.4);
  strokeSeries(folX, heading, xFor, yFor, C.accent, 1.4);
  markSwap(folX, folC, t, d.params.swap_at);
}

function drawErr(d) {
  const { err, t } = d.trace;
  bg(errX, errC);
  const ymax = 185, ymin = -185;
  const H = errC.height;
  const yFor = (v) => PAD.t + ((ymax - v) / (ymax - ymin)) * (H - PAD.t - PAD.b);
  errX.fillStyle = "rgba(93,169,255,0.10)";
  errX.fillRect(PAD.l, yFor(45), errC.width - PAD.l - PAD.r, yFor(-45) - yFor(45));
  frame(errX, errC, d.params.steps, yFor, ymin, ymax);
  errX.fillStyle = C.dim;
  errX.font = "10px monospace";
  errX.textAlign = "right";
  errX.fillText("+45", PAD.l - 4, yFor(45) + 3);
  errX.fillText("−45", PAD.l - 4, yFor(-45) + 3);
  errX.textAlign = "left";
  strokeSeries(errX, err, xScale(errC, t), yFor, C.red, 1.2);
  markSwap(errX, errC, t, d.params.swap_at);
}

function drawSegs(d) {
  const segs = d.summary.seg_scores, segLen = d.params.seg_len;
  bg(segX, segC);
  const H = segC.height, W = segC.width;
  const yFor = (v) => PAD.t + (1 - v) * (H - PAD.t - PAD.b);
  frame(segX, segC, d.params.steps, yFor, 0, 1);
  const innerW = W - PAD.l - PAD.r;
  const bw = innerW / segs.length;
  segs.forEach((s, i) => {
    const x0 = PAD.l + i * bw;
    segX.fillStyle = s >= 0.35 ? C.green : "#5a3a44";
    segX.fillRect(x0 + 2, yFor(s), Math.max(2, bw - 4), yFor(0) - yFor(s));
    segX.fillStyle = s >= 0.35 ? C.green : C.dim;
    segX.font = "10px monospace";
    segX.textAlign = "center";
    segX.fillText(s.toFixed(2), x0 + bw / 2, yFor(s) - 3);
    segX.textAlign = "left";
  });
  segX.strokeStyle = C.yellow;
  segX.setLineDash([4, 4]);
  segX.beginPath(); segX.moveTo(PAD.l, yFor(0.35)); segX.lineTo(W - PAD.r, yFor(0.35)); segX.stroke();
  segX.setLineDash([]);
  if (d.params.swap_at !== null && d.params.swap_at >= 0) {
    const px = PAD.l + (Math.min(d.params.swap_at, d.params.steps) / d.params.steps) * innerW;
    segX.strokeStyle = C.pink;
    segX.setLineDash([4, 4]);
    segX.beginPath(); segX.moveTo(px, PAD.t); segX.lineTo(px, H - PAD.b); segX.stroke();
    segX.setLineDash([]);
  }
}

function render(d) {
  drawFollow(d);
  drawErr(d);
  drawSegs(d);
  const s = d.summary, cfg = d.config;
  const p3 = (v) => Number(v.toPrecision(3));
  document.getElementById("summary").textContent =
    `score ${s.score.toFixed(3)} · score_late ${s.score_late.toFixed(3)} · ` +
    `prop_spiked ${s.prop_spiked.toFixed(3)} · N=${cfg.n_nodes} leak=${p3(cfg.leak)} ` +
    `wlr=${p3(cfg.weight_lr)} tlr=${p3(cfg.target_lr)} gain=${cfg.gain.toFixed(1)}` +
    (d.params.swap_at !== null && d.params.swap_at >= 0 ? ` · swap at t=${d.params.swap_at}` : "") +
    (d.params.sensor_noise > 0 ? ` · noise σ=${d.params.sensor_noise.toFixed(2)}` : "");
  document.getElementById("stat").textContent =
    `${d.params.variant} · seed ${d.params.seed} · ${d.params.steps} steps (every ${d.params.subsample}th shown)`;
}

// ---------- wiring ----------------------------------------------------------
for (const id of ["variant", "seed", "steps", "swap", "noise"]) {
  document.getElementById(id).addEventListener("change", run);
}
document.getElementById("noise").addEventListener("input", showNoise);
document.getElementById("btn-run").addEventListener("click", run);
showNoise();
if (batchBox) {
  let batchRan = false;
  batchBox.addEventListener("toggle", () => {
    if (batchBox.open && !batchRan) { batchRan = true; fitWide(); run(); }
  });
} else {
  fitWide();
  run();
}
