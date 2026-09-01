/* Frontend for the wall-avoidance viewer.
 *
 * All simulation happens server-side (/lab/api/wall runs the tested
 * `homeostasis` package: run_wall for the wall arms, run_tracking for the
 * evolved flank champion); this file only renders the returned traces.
 * The sensor rays are drawn from the returned sensor VALUES (length
 * (1 - value) x box diagonal), so after the perturbation they show what the
 * network was told, not the true geometry — which is the point.
 */

"use strict";

const C = {
  panel: "#10141a", grid: "#2a313c", text: "#d7dde6", dim: "#8b95a3",
  red: "#ff5d5d", blue: "#5da9ff", green: "#3fd68f", yellow: "#ffd23f",
  pink: "#ff9ecb", accent: "#58a6ff",
};

const connEl = document.getElementById("conn");
let latest = null;
let runSeq = 0;

// ---------- fetch -----------------------------------------------------------
function params() {
  return {
    variant: document.getElementById("variant").value,
    seed: Math.max(parseInt(document.getElementById("seed").value) || 0, 0),
    steps: Math.min(Math.max(parseInt(document.getElementById("steps").value) || 3600, 200), 14400),
    wlr: document.getElementById("wlr").value || "1.0",
    tlr: document.getElementById("tlr").value || "0.01",
  };
}

async function run() {
  const p = params();
  const seq = ++runSeq;
  connEl.textContent = "running…";
  connEl.className = "conn";
  const t0 = performance.now();
  try {
    const res = await fetch(`/lab/api/wall?${new URLSearchParams(p)}`);
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

// ---------- shared plotting helpers -----------------------------------------
const cv = (id) => { const c = document.getElementById(id); return [c, c.getContext("2d")]; };
const [arenaC, arenaX] = cv("arena");
const [sensC, sensX] = cv("sensors");
const [propC, propX] = cv("propc");
const [ringC, ringX] = cv("ring");
const [ferrC, ferrX] = cv("ferr");
const PAD = { l: 52, r: 10, t: 10, b: 18 };

function bg(x, c) { x.fillStyle = C.panel; x.fillRect(0, 0, c.width, c.height); }

function fitWide() {
  for (const c of [sensC, propC, ferrC]) {
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

function strokeSeries(x, arr, xFor, yFor, color, lw = 1.3) {
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

function markStep(x, c, tArr, atStep, color) {
  if (atStep === null || atStep === undefined) return;
  const t1 = tArr[tArr.length - 1] || 1;
  if (atStep > t1) return;
  const px = PAD.l + (atStep / t1) * (c.width - PAD.l - PAD.r);
  x.strokeStyle = color;
  x.setLineDash([4, 4]);
  x.beginPath(); x.moveTo(px, PAD.t); x.lineTo(px, c.height - PAD.b); x.stroke();
  x.setLineDash([]);
}

// ---------- wall arena ------------------------------------------------------
function drawArena(d) {
  const { x, y, heading, hit, s_left, s_right } = d.trace;
  const box = d.config.box_size, r = d.config.agent_radius;
  const n = x.length;
  bg(arenaX, arenaC);
  const side = arenaC.width, pad = 14;
  const s = (side - 2 * pad) / box;
  const X = (u) => pad + u * s;
  const Y = (v) => side - pad - v * s;   // world y grows upward

  arenaX.strokeStyle = C.grid;
  arenaX.strokeRect(X(0), Y(box), box * s, box * s);

  // fading trail
  for (let i = 1; i < n; i++) {
    arenaX.strokeStyle = C.accent;
    arenaX.globalAlpha = 0.05 + 0.5 * (i / n);
    arenaX.beginPath();
    arenaX.moveTo(X(x[i - 1]), Y(y[i - 1]));
    arenaX.lineTo(X(x[i]), Y(y[i]));
    arenaX.stroke();
  }
  arenaX.globalAlpha = 1;

  // hit marks
  arenaX.strokeStyle = C.red;
  arenaX.lineWidth = 1.2;
  for (let i = 0; i < n; i++) {
    if (!hit[i]) continue;
    const px = X(x[i]), py = Y(y[i]);
    arenaX.beginPath();
    arenaX.moveTo(px - 3.5, py - 3.5); arenaX.lineTo(px + 3.5, py + 3.5);
    arenaX.moveTo(px - 3.5, py + 3.5); arenaX.lineTo(px + 3.5, py - 3.5);
    arenaX.stroke();
  }
  arenaX.lineWidth = 1;

  // final agent + sensor rays (clip rays to the box)
  const i = n - 1;
  const hx = X(x[i]), hy = Y(y[i]), hr = heading[i];
  arenaX.save();
  arenaX.beginPath();
  arenaX.rect(X(0), Y(box), box * s, box * s);
  arenaX.clip();
  const diag = Math.SQRT2 * box;
  for (const [off, val, color] of [[45, s_left[i], C.red], [-45, s_right[i], C.blue]]) {
    const a = hr + (off * Math.PI) / 180;
    const ox = x[i] + r * Math.cos(a), oy = y[i] + r * Math.sin(a);
    const dWorld = Math.max(0, Math.min(1 - val, 1)) * diag;
    arenaX.strokeStyle = color;
    arenaX.globalAlpha = 0.35 + 0.65 * Math.min(Math.max(val, 0), 1);
    arenaX.lineWidth = 1.6;
    arenaX.beginPath();
    arenaX.moveTo(X(ox), Y(oy));
    arenaX.lineTo(X(ox + dWorld * Math.cos(a)), Y(oy + dWorld * Math.sin(a)));
    arenaX.stroke();
  }
  arenaX.restore();
  arenaX.globalAlpha = 1;
  arenaX.lineWidth = 1;

  arenaX.fillStyle = "#22303f";
  arenaX.strokeStyle = C.text;
  arenaX.beginPath();
  arenaX.arc(hx, hy, r * s, 0, 2 * Math.PI);
  arenaX.fill();
  arenaX.stroke();
  arenaX.strokeStyle = C.yellow;
  arenaX.lineWidth = 1.8;
  arenaX.beginPath();
  arenaX.moveTo(hx, hy);
  arenaX.lineTo(hx + 2.2 * r * s * Math.cos(hr), hy - 2.2 * r * s * Math.sin(hr));
  arenaX.stroke();
  arenaX.lineWidth = 1;
}

function drawWallStrips(d) {
  const { t, s_left, s_right, prop } = d.trace;
  // sensors
  bg(sensX, sensC);
  let hi = 1;
  for (const arr of [s_left, s_right]) for (const v of arr) hi = Math.max(hi, v);
  const yFor = (v) => PAD.t + ((hi * 1.05 - v) / (hi * 1.05)) * (sensC.height - PAD.t - PAD.b);
  frame(sensX, sensC, d.params.steps, yFor, 0, hi * 1.05);
  const xFor = xScale(sensC, t);
  strokeSeries(sensX, s_left, xFor, yFor, C.red);
  strokeSeries(sensX, s_right, xFor, yFor, C.blue);
  markStep(sensX, sensC, t, d.config.perturb_at, C.pink);
  // prop
  bg(propX, propC);
  const yP = (v) => PAD.t + (1 - v) * (propC.height - PAD.t - PAD.b);
  frame(propX, propC, d.params.steps, yP, 0, 1);
  strokeSeries(propX, prop, xScale(propC, t), yP, C.green);
  markStep(propX, propC, t, d.config.perturb_at, C.pink);
}

// ---------- flank (tracking) view -------------------------------------------
// Egocentric ring: heading up; positive error = stimulus CCW of heading (left).
function ringPt(cx, cy, R, aDeg) {
  const a = (aDeg * Math.PI) / 180;
  return [cx - R * Math.sin(a), cy - R * Math.cos(a)];
}

function drawRing(d) {
  const err = d.trace.err;
  const n = err.length;
  bg(ringX, ringC);
  const side = ringC.width, cx = side / 2, cy = side / 2, R = side / 2 - 34;

  // target band arcs |a| in [50, 90]
  ringX.strokeStyle = "rgba(63,214,143,0.16)";
  ringX.lineWidth = 20;
  for (const sgn of [1, -1]) {
    ringX.beginPath();
    for (let a = 50; a <= 90; a += 2) {
      const [px, py] = ringPt(cx, cy, R, sgn * a);
      a === 50 ? ringX.moveTo(px, py) : ringX.lineTo(px, py);
    }
    ringX.stroke();
  }
  ringX.lineWidth = 1;

  // ring + labels
  ringX.strokeStyle = C.grid;
  ringX.beginPath();
  ringX.arc(cx, cy, R, 0, 2 * Math.PI);
  ringX.stroke();
  ringX.fillStyle = C.dim;
  ringX.font = "10px monospace";
  ringX.textAlign = "center";
  ringX.fillText("0° (heading)", cx, cy - R - 8);
  ringX.fillText("180°", cx, cy + R + 14);
  ringX.textAlign = "left";
  ringX.fillText("−90°", cx + R + 6, cy + 3);
  ringX.textAlign = "right";
  ringX.fillText("+90°", cx - R - 6, cy + 3);
  ringX.textAlign = "left";

  // stimulus dots over time, fading with age
  for (let i = 0; i < n; i++) {
    const [px, py] = ringPt(cx, cy, R, err[i]);
    ringX.fillStyle = C.text;
    ringX.globalAlpha = 0.03 + 0.3 * (i / n);
    ringX.fillRect(px - 1, py - 1, 2, 2);
  }
  ringX.globalAlpha = 1;
  const [fx, fy] = ringPt(cx, cy, R, err[n - 1]);
  ringX.fillStyle = C.green;
  ringX.beginPath();
  ringX.arc(fx, fy, 5, 0, 2 * Math.PI);
  ringX.fill();

  // agent arrow at center, pointing up
  ringX.strokeStyle = C.accent;
  ringX.lineWidth = 2;
  ringX.beginPath();
  ringX.moveTo(cx, cy + 14); ringX.lineTo(cx, cy - 16);
  ringX.moveTo(cx - 5, cy - 9); ringX.lineTo(cx, cy - 16); ringX.lineTo(cx + 5, cy - 9);
  ringX.stroke();
  ringX.lineWidth = 1;
}

function drawFlankErr(d) {
  const { err, t } = d.trace;
  bg(ferrX, ferrC);
  const ymax = 185, ymin = -185;
  const H = ferrC.height;
  const yFor = (v) => PAD.t + ((ymax - v) / (ymax - ymin)) * (H - PAD.t - PAD.b);
  ferrX.fillStyle = "rgba(63,214,143,0.13)";
  ferrX.fillRect(PAD.l, yFor(90), ferrC.width - PAD.l - PAD.r, yFor(50) - yFor(90));
  ferrX.fillRect(PAD.l, yFor(-50), ferrC.width - PAD.l - PAD.r, yFor(-90) - yFor(-50));
  frame(ferrX, ferrC, d.params.steps, yFor, ymin, ymax);
  ferrX.fillStyle = C.dim;
  ferrX.font = "10px monospace";
  ferrX.textAlign = "right";
  for (const v of [90, 50, -50, -90]) ferrX.fillText(String(v), PAD.l - 4, yFor(v) + 3);
  ferrX.textAlign = "left";
  strokeSeries(ferrX, err, xScale(ferrC, t), yFor, C.yellow, 1.2);
}

// ---------- render ----------------------------------------------------------
function render(d) {
  const isWall = d.kind === "wall";
  document.getElementById("view-wall").style.display = isWall ? "" : "none";
  document.getElementById("view-flank").style.display = isWall ? "none" : "";
  if (isWall) {
    drawArena(d);
    drawWallStrips(d);
    document.getElementById("w-hits").textContent = d.summary.hits_total;
    document.getElementById("w-late-hits").textContent = d.summary.hits_last_1000;
    document.getElementById("w-late-dh").textContent = d.summary.late_mean_abs_dh.toFixed(4);
    document.getElementById("stat").textContent =
      `${d.params.variant} · seed ${d.params.seed} · ${d.params.steps} steps · ` +
      `wlr ${d.params.wlr} tlr ${d.params.tlr} · learning ${d.config.learning ? "on" : "off"}`;
  } else {
    drawRing(d);
    drawFlankErr(d);
    document.getElementById("f-band").textContent = `${(d.summary.band_frac * 100).toFixed(1)}%`;
    document.getElementById("f-band-late").textContent = `${(d.summary.band_frac_late * 100).toFixed(1)}%`;
    document.getElementById("f-w45").textContent = `${(d.summary.within45 * 100).toFixed(1)}%`;
    document.getElementById("f-abs").textContent = `${d.summary.mean_abs_err.toFixed(1)}°`;
    document.getElementById("stat").textContent =
      `flank-champion (tracking) · seed ${d.params.seed} · ${d.params.steps} steps · ` +
      `N=${d.config.n_nodes} gain=${d.config.gain}`;
  }
}

// ---------- wiring ----------------------------------------------------------
document.getElementById("variant").addEventListener("change", () => {
  const flank = document.getElementById("variant").value === "flank-champion";
  if (flank) {
    document.getElementById("seed").value = 5;      // verified band-holder
    document.getElementById("steps").value = 7200;
  } else {
    document.getElementById("steps").value = 3600;
  }
  document.getElementById("wlr").disabled = flank;  // champion fixes its own rates
  document.getElementById("tlr").disabled = flank;
  run();
});
for (const id of ["seed", "steps", "wlr", "tlr"]) {
  document.getElementById(id).addEventListener("change", run);
}
document.getElementById("btn-run").addEventListener("click", run);
fitWide();
run();
