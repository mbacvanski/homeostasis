/* Frontend for the self-repair exhibit.
 *
 * All simulation happens server-side (/lab/api/repair runs both H53 arms
 * through scripts/lab/common.py run_closed_loop — the tested harness on the
 * tested package); this file only renders the returned observables.
 */

"use strict";

const C = {
  panel: "#10141a", grid: "#2a313c", text: "#d7dde6", dim: "#8b95a3",
  red: "#ff5d5d", blue: "#5da9ff", green: "#3fd68f", yellow: "#ffd23f",
  pink: "#ff9ecb", accent: "#58a6ff", orange: "#ffa04d",
};
const ARM_COLOR = { learning: C.green, frozen: C.orange };

// Batch status goes to its own element now that #conn shows the LIVE
// connection state (the live section is served by lab_repair_live.js).
const connEl = document.getElementById("batch-status") || document.getElementById("conn");
const batchBox = document.getElementById("batch-box");
let latest = null;
let runSeq = 0;

// ---------- fetch -----------------------------------------------------------
function params() {
  return {
    kill: parseFloat(document.getElementById("kill").value) || 0.3,
    seed: Math.max(parseInt(document.getElementById("seed").value) || 0, 0),
    steps: Math.min(Math.max(parseInt(document.getElementById("steps").value) || 14400, 1440), 21600),
  };
}

async function run() {
  if (batchBox && !batchBox.open) return;  // batch is lazy: render only when shown
  const p = params();
  const seq = ++runSeq;
  connEl.textContent = "running… (two 14k-step arms)";
  connEl.className = "conn";
  const t0 = performance.now();
  try {
    const res = await fetch(`/lab/api/repair?${new URLSearchParams(p)}`);
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

// ---------- helpers (as the other lab pages) --------------------------------
const cv = (id) => { const c = document.getElementById(id); return [c, c.getContext("2d")]; };
const [fC, fX] = cv("fc");
const [wC, wX] = cv("wc");
const [segC, segX] = cv("segc");
const PAD = { l: 56, r: 10, t: 10, b: 18 };

function bg(x, c) { x.fillStyle = C.panel; x.fillRect(0, 0, c.width, c.height); }

function fitWide() {
  for (const c of [fC, wC, segC]) {
    const w = Math.max(360, Math.round(c.clientWidth) || c.width);
    if (c.width !== w) c.width = w;
  }
}
let resizeTimer = null;
window.addEventListener("resize", () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => { fitWide(); if (latest) render(latest); }, 150);
});

const fmtTick = (v) => Math.abs(v) >= 100 ? v.toFixed(0) : Math.abs(v) >= 1 ? v.toFixed(1) : v.toFixed(2);

function frame(x, c, tMax, yFor, ymin, ymax) {
  const W = c.width, H = c.height;
  x.strokeStyle = C.grid;
  x.strokeRect(PAD.l + 0.5, PAD.t + 0.5, W - PAD.l - PAD.r - 1, H - PAD.t - PAD.b - 1);
  x.fillStyle = C.dim;
  x.font = "10px monospace";
  x.textAlign = "right";
  x.fillText(fmtTick(ymax), PAD.l - 4, PAD.t + 8);
  x.fillText(fmtTick(ymin), PAD.l - 4, H - PAD.b);
  x.textAlign = "left";
  x.fillText("t=0", PAD.l, H - 5);
  x.textAlign = "right";
  x.fillText(`t=${tMax}`, W - PAD.r, H - 5);
  x.textAlign = "left";
}

function markKill(x, c, killAt, steps) {
  const px = PAD.l + (killAt / steps) * (c.width - PAD.l - PAD.r);
  x.strokeStyle = C.pink;
  x.setLineDash([4, 4]);
  x.beginPath(); x.moveTo(px, PAD.t); x.lineTo(px, c.height - PAD.b); x.stroke();
  x.setLineDash([]);
}

// snap trajectories are short (one point per 240 steps): plain polylines.
// The frozen arm is dashed so the identical pre-kill halves both stay visible.
function polyline(x, tArr, vals, xFor, yFor, color, dashed, lw = 1.6) {
  x.strokeStyle = color;
  x.lineWidth = lw;
  if (dashed) x.setLineDash([5, 3]);
  x.beginPath();
  for (let i = 0; i < vals.length; i++) {
    const px = xFor(tArr[i]), py = yFor(vals[i]);
    i ? x.lineTo(px, py) : x.moveTo(px, py);
  }
  x.stroke();
  x.setLineDash([]);
  x.lineWidth = 1;
}

// 5-window centered running mean (presentation only): single-seed f_win is
// bursty; the mean is what the H53 dip-and-recovery anchors describe.
function smooth5(vals) {
  const out = new Array(vals.length);
  for (let i = 0; i < vals.length; i++) {
    const a = Math.max(0, i - 2), b = Math.min(vals.length, i + 3);
    let s = 0;
    for (let j = a; j < b; j++) s += vals[j];
    out[i] = s / (b - a);
  }
  return out;
}

function drawSnapPanel(x, c, d, field, withMean) {
  bg(x, c);
  const steps = d.params.steps;
  // with the running-mean guide, scale to it and let raw bursts clip at top
  let hi = 0;
  for (const arm of Object.values(d.arms))
    for (const v of withMean ? smooth5(arm.snaps[field]) : arm.snaps[field])
      hi = Math.max(hi, v);
  hi = hi * (withMean ? 1.25 : 1.1) || 1;
  const H = c.height;
  const yFor = (v) =>
    Math.max(PAD.t + ((hi - v) / hi) * (H - PAD.t - PAD.b), PAD.t + 1);
  const xFor = (t) => PAD.l + (t / steps) * (c.width - PAD.l - PAD.r);
  frame(x, c, steps, yFor, 0, hi);
  markKill(x, c, d.params.kill_at, steps);
  for (const [name, arm] of Object.entries(d.arms)) {
    const dashed = name === "frozen";
    if (withMean) {
      x.globalAlpha = 0.4;
      polyline(x, arm.snaps.t, arm.snaps[field], xFor, yFor, ARM_COLOR[name], dashed, 1.0);
      x.globalAlpha = 1;
      polyline(x, arm.snaps.t, smooth5(arm.snaps[field]), xFor, yFor, ARM_COLOR[name], dashed, 2.4);
    } else {
      polyline(x, arm.snaps.t, arm.snaps[field], xFor, yFor, ARM_COLOR[name], dashed);
    }
  }
}

function drawSegs(d) {
  const segsL = d.arms.learning.seg_scores, segsF = d.arms.frozen.seg_scores;
  bg(segX, segC);
  const H = segC.height, W = segC.width;
  const yFor = (v) => PAD.t + (1 - v) * (H - PAD.t - PAD.b);
  frame(segX, segC, d.params.steps, yFor, 0, 1);
  const innerW = W - PAD.l - PAD.r;
  const bw = innerW / segsL.length;
  segsL.forEach((s, i) => {
    const x0 = PAD.l + i * bw;
    const half = Math.max(2, bw / 2 - 3);
    segX.fillStyle = ARM_COLOR.learning;
    segX.fillRect(x0 + 2, yFor(s), half, yFor(0) - yFor(s));
    segX.fillStyle = ARM_COLOR.frozen;
    segX.fillRect(x0 + 2 + half + 1, yFor(segsF[i]), half, yFor(0) - yFor(segsF[i]));
  });
  segX.strokeStyle = C.yellow;
  segX.setLineDash([4, 4]);
  segX.beginPath(); segX.moveTo(PAD.l, yFor(0.35)); segX.lineTo(W - PAD.r, yFor(0.35)); segX.stroke();
  segX.setLineDash([]);
  markKill(segX, segC, d.params.kill_at, d.params.steps);
}

// ---------- render ----------------------------------------------------------
function render(d) {
  drawSnapPanel(fX, fC, d, "f_win", true);
  drawSnapPanel(wX, wC, d, "w_mean", false);
  drawSegs(d);
  const arm = (name) => {
    const a = d.arms[name];
    return `${name} ${a.pre.toFixed(3)} → ${a.drop.toFixed(3)} → ${a.rec.toFixed(3)}`;
  };
  const wEnd = (name) => d.arms[name].snaps.w_mean.at(-1).toFixed(3);
  document.getElementById("summary").textContent =
    `score pre → post-kill → late: ${arm("learning")} · ${arm("frozen")} · ` +
    `final w̄: learning ${wEnd("learning")} vs frozen ${wEnd("frozen")}`;
  document.getElementById("stat").textContent =
    `kill ${(d.params.kill * 100).toFixed(0)}% (${d.config.n_killed} of ${d.config.n_nodes} nodes at t=${d.params.kill_at})` +
    ` · seed ${d.params.seed} · wlr=${d.config.weight_lr} tlr=${d.config.target_lr} leak=${d.config.leak}`;
}

// ---------- wiring ----------------------------------------------------------
for (const id of ["kill", "seed", "steps"]) {
  document.getElementById(id).addEventListener("change", run);
}
document.getElementById("btn-run").addEventListener("click", run);
if (batchBox) {
  let batchRan = false;
  batchBox.addEventListener("toggle", () => {
    if (batchBox.open && !batchRan) { batchRan = true; fitWide(); run(); }
  });
} else {
  fitWide();
  run();
}
