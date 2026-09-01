/* Frontend for the pursuit viewer.
 *
 * All simulation happens server-side (/lab/api/pursuit runs the tested
 * `homeostasis` package with the exact scripts/lab/h34b_verify.py
 * construction); this file only renders the returned traces.
 */

"use strict";

const C = {
  panel: "#10141a", grid: "#2a313c", text: "#d7dde6", dim: "#8b95a3",
  red: "#ff5d5d", blue: "#5da9ff", green: "#3fd68f", yellow: "#ffd23f",
  pink: "#ff9ecb", accent: "#58a6ff",
};

const CAPTIONS = {
  "perfect-pursuer":
    "the perfect pursuer does not chase — it orbits phase-locked inside the target's orbit (velocity entrainment in 2D)",
  "pursuit-fails":
    "the same agent on unpredictable motion: a resonator, not a follower",
};

const connEl = document.getElementById("conn");
let latest = null;
let runSeq = 0;

// ---------- fetch -----------------------------------------------------------
function params() {
  return {
    preset: document.getElementById("preset").value,
    seed: Math.max(parseInt(document.getElementById("seed").value) || 0, 0),
    steps: Math.min(Math.max(parseInt(document.getElementById("steps").value) || 3600, 200), 14400),
  };
}

async function run() {
  const p = params();
  const seq = ++runSeq;
  connEl.textContent = "running…";
  connEl.className = "conn";
  const t0 = performance.now();
  try {
    const res = await fetch(`/lab/api/pursuit?${new URLSearchParams(p)}`);
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
const [arenaC, arenaX] = cv("arena");
const [distC, distX] = cv("distc");
const PAD = { l: 52, r: 10, t: 10, b: 18 };

function bg(x, c) { x.fillStyle = C.panel; x.fillRect(0, 0, c.width, c.height); }

function fitWide() {
  const w = Math.max(360, Math.round(distC.clientWidth) || distC.width);
  if (distC.width !== w) distC.width = w;
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

// ---------- arena -----------------------------------------------------------
function drawArena(d) {
  const { x, y, sx, sy, heading, hit } = d.trace;
  const box = d.config.box_size, r = d.config.agent_radius;
  const n = x.length;
  bg(arenaX, arenaC);
  const side = arenaC.width, pad = 14;
  const s = (side - 2 * pad) / box;
  const X = (u) => pad + u * s;
  const Y = (v) => side - pad - v * s;

  arenaX.strokeStyle = C.grid;
  arenaX.strokeRect(X(0), Y(box), box * s, box * s);

  // ideal orbit path (orbit preset): faint red circle around the center
  if (d.config.motion === "orbit") {
    arenaX.strokeStyle = C.red;
    arenaX.globalAlpha = 0.25;
    arenaX.setLineDash([4, 4]);
    arenaX.beginPath();
    arenaX.arc(X(box / 2), Y(box / 2), d.config.orbit_radius * s, 0, 2 * Math.PI);
    arenaX.stroke();
    arenaX.setLineDash([]);
    arenaX.globalAlpha = 1;
  }

  // stimulus trail (its actual path), faint red
  arenaX.strokeStyle = C.red;
  arenaX.globalAlpha = 0.28;
  arenaX.beginPath();
  for (let i = 0; i < n; i++) {
    const px = X(sx[i]), py = Y(sy[i]);
    i ? arenaX.lineTo(px, py) : arenaX.moveTo(px, py);
  }
  arenaX.stroke();
  arenaX.globalAlpha = 1;

  // agent trail, fading blue
  for (let i = 1; i < n; i++) {
    arenaX.strokeStyle = C.accent;
    arenaX.globalAlpha = 0.05 + 0.5 * (i / n);
    arenaX.beginPath();
    arenaX.moveTo(X(x[i - 1]), Y(y[i - 1]));
    arenaX.lineTo(X(x[i]), Y(y[i]));
    arenaX.stroke();
  }
  arenaX.globalAlpha = 1;

  // wall-hit marks
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

  // stimulus dot (final)
  arenaX.fillStyle = C.red;
  arenaX.beginPath();
  arenaX.arc(X(sx[n - 1]), Y(sy[n - 1]), 5, 0, 2 * Math.PI);
  arenaX.fill();

  // agent (final) + heading tick
  const hx = X(x[n - 1]), hy = Y(y[n - 1]), hr = heading[n - 1];
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

// ---------- distance strip --------------------------------------------------
function drawDist(d) {
  const { dist, hit, t } = d.trace;
  bg(distX, distC);
  let hi = 4;
  for (const v of dist) hi = Math.max(hi, v);
  hi *= 1.06;
  const H = distC.height, W = distC.width;
  const yFor = (v) => PAD.t + ((hi - v) / hi) * (H - PAD.t - PAD.b);
  const xFor = (i) => PAD.l + (t[i] / (t[t.length - 1] || 1)) * (W - PAD.l - PAD.r);

  // near-3 band and the scored (late) half
  distX.fillStyle = "rgba(63,214,143,0.12)";
  distX.fillRect(PAD.l, yFor(3), W - PAD.l - PAD.r, yFor(0) - yFor(3));
  const lateX = xFor(Math.floor(t.length / 2));
  distX.fillStyle = "rgba(255,255,255,0.045)";
  distX.fillRect(lateX, PAD.t, W - PAD.r - lateX, H - PAD.t - PAD.b);

  distX.strokeStyle = C.grid;
  distX.strokeRect(PAD.l + 0.5, PAD.t + 0.5, W - PAD.l - PAD.r - 1, H - PAD.t - PAD.b - 1);
  distX.fillStyle = C.dim;
  distX.font = "10px monospace";
  distX.textAlign = "right";
  distX.fillText(hi.toFixed(1), PAD.l - 4, PAD.t + 8);
  distX.fillText("3", PAD.l - 4, yFor(3) + 3);
  distX.fillText("0", PAD.l - 4, yFor(0));
  distX.textAlign = "left";
  distX.fillText("t=0", PAD.l, H - 5);
  distX.textAlign = "right";
  distX.fillText(`t=${d.params.steps}`, W - PAD.r, H - 5);
  distX.textAlign = "left";

  strokeSeries(distX, dist, xFor, yFor, C.accent, 1.3);

  // wall hits as red ticks along the top
  distX.fillStyle = C.red;
  for (let i = 0; i < hit.length; i++) {
    if (hit[i]) distX.fillRect(xFor(i), PAD.t + 1, 1.2, 7);
  }
}

// ---------- render ----------------------------------------------------------
function render(d) {
  drawArena(d);
  drawDist(d);
  document.getElementById("caption").textContent = CAPTIONS[d.params.preset] || "";
  document.getElementById("p-dist").textContent = d.summary.dist_late.toFixed(2);
  document.getElementById("p-near").textContent = `${(d.summary.near3_late * 100).toFixed(1)}%`;
  document.getElementById("p-hits").textContent = d.summary.hits_total;
  const isChamp = d.params.seed === d.config.champ_seed;
  document.getElementById("stat").textContent =
    `${d.config.motion} · seed ${d.params.seed}${isChamp ? " (champion pair)" : " (fresh wiring)"}` +
    ` · ${d.params.steps} steps · N=${d.config.n_nodes} wheel_base=${d.config.wheel_base}` +
    ` intensity_scale=${d.config.intensity_scale}`;
}

// ---------- wiring ----------------------------------------------------------
for (const id of ["preset", "seed", "steps"]) {
  document.getElementById(id).addEventListener("change", run);
}
document.getElementById("btn-run").addEventListener("click", run);
fitWide();
run();
