/* Frontend for the shared-visibility ecology viewer (three agents, four
 * attention regimes — the H81→H85b progression).
 *
 * All simulation happens server-side (/lab/api/ecology3 mirrors the exact
 * co-simulation loop of scripts/lab/h85_shared.py run(), with the sticky
 * attention rule imported from that script); this file only renders the
 * returned traces and replays them.
 */

"use strict";

const C = {
  panel: "#10141a", grid: "#2a313c", text: "#d7dde6", dim: "#8b95a3",
  red: "#ff5d5d", blue: "#5da9ff", green: "#3fd68f", yellow: "#ffd23f",
  pink: "#ff9ecb", accent: "#58a6ff", gray: "#39424f",
};
// agent colors (A pacemaker, B follower, C follower) — also the color of an
// attention line/band, keyed by the *selected* agent
const AGENT = { a: C.red, b: C.accent, c: C.green };

const connEl = document.getElementById("conn");
let latest = null;
let runSeq = 0;

// playback state (sample index into the subsampled trace)
const play = { idx: 0, playing: false, speed: 1, raf: null, last: 0 };
const BASE_STEPS_PER_SEC = 900; // sim steps per second at 1×
const TRAIL = 240;              // vivid trail length, in samples

// ---------- fetch -----------------------------------------------------------
function params() {
  return {
    mode: document.getElementById("mode").value,
    steps: Math.min(Math.max(parseInt(document.getElementById("steps").value) || 10800, 600), 21600),
  };
}

async function run() {
  const p = params();
  const seq = ++runSeq;
  connEl.textContent = "running…";
  connEl.className = "conn";
  const t0 = performance.now();
  try {
    const res = await fetch(`/lab/api/ecology3?${new URLSearchParams(p)}`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    if (data.error) throw new Error(data.error);
    if (seq !== runSeq) return;
    latest = data;
    connEl.textContent = `ok · ${(performance.now() - t0).toFixed(0)} ms`;
    connEl.className = "conn ok";
    render(data);
    startPlayback();
  } catch (err) {
    if (seq !== runSeq) return;
    connEl.textContent = `error: ${err.message}`;
    connEl.className = "conn bad";
  }
}

// ---------- helpers (as the other lab pages) --------------------------------
const cv = (id) => { const c = document.getElementById(id); return [c, c.getContext("2d")]; };
const [arenaC, arenaX] = cv("arena");
const [attC, attX] = cv("attc");
const [distC, distX] = cv("distc");
const [lockC, lockX] = cv("lockc");
const WIDE = [attC, distC, lockC];
const PAD = { l: 64, r: 10, t: 10, b: 18 };

function bg(x, c) { x.fillStyle = C.panel; x.fillRect(0, 0, c.width, c.height); }

function fitWide() {
  const w = Math.max(360, Math.round(distC.clientWidth) || distC.width);
  for (const c of WIDE) if (c.width !== w) c.width = w;
}
let resizeTimer = null;
window.addEventListener("resize", () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => { fitWide(); if (latest) { render(latest); drawFrame(); } }, 150);
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

// x position of a full-resolution step time on the wide strips
function xForTime(d, W, time) {
  const tMax = d.params.steps || 1;
  return PAD.l + (time / tMax) * (W - PAD.l - PAD.r);
}

function switchMarks(x, d, W, H) {
  const s = d.summary;
  x.setLineDash([3, 3]);
  x.font = "9px monospace";
  x.textAlign = "left";
  for (const [times, color, lab] of [[s.b_switch_times, AGENT.b, "B"], [s.c_switch_times, AGENT.c, "C"]]) {
    x.strokeStyle = color;
    x.fillStyle = color;
    for (const t of times) {
      const px = Math.round(xForTime(d, W, t)) + 0.5;
      x.beginPath();
      x.moveTo(px, PAD.t);
      x.lineTo(px, H - PAD.b);
      x.stroke();
      x.fillText(`${lab} switch`, px + 3, PAD.t + (lab === "B" ? 8 : 17));
    }
  }
  x.setLineDash([]);
}

// ---------- static strips (drawn once per run, blitted during playback) -----
const off = {};  // name -> offscreen canvas
function offscreen(name, w, h, draw) {
  let c = off[name];
  if (!c) { c = off[name] = document.createElement("canvas"); }
  c.width = w; c.height = h;
  draw(c.getContext("2d"), w, h);
  return c;
}

function drawAttStatic(x, W, H, d) {
  const { sel_b, sel_c } = d.trace;
  const n = sel_b.length;
  bg(x, { width: W, height: H });
  const rows = [
    { sel: sel_b, y0: PAD.t, colors: { "-1": C.gray, 0: AGENT.a, 1: AGENT.c }, lab: "B attends" },
    { sel: sel_c, y0: PAD.t + 34, colors: { "-1": C.gray, 0: AGENT.a, 1: AGENT.b }, lab: "C attends" },
  ];
  const span = W - PAD.l - PAD.r;
  for (const r of rows) {
    x.fillStyle = C.dim;
    x.font = "10px monospace";
    x.textAlign = "right";
    x.fillText(r.lab, PAD.l - 6, r.y0 + 16);
    for (let i = 0; i < n; i++) {
      const x0 = PAD.l + (i / n) * span;
      const x1 = PAD.l + ((i + 1) / n) * span;
      x.fillStyle = r.colors[r.sel[i]];
      x.fillRect(x0, r.y0, x1 - x0 + 0.6, 24);
    }
    x.strokeStyle = C.grid;
    x.strokeRect(PAD.l + 0.5, r.y0 + 0.5, span - 1, 24);
  }
  switchMarks(x, d, W, H);
}

function drawDistStatic(x, W, H, d) {
  const { d_ba, d_ca, d_cb, t } = d.trace;
  bg(x, { width: W, height: H });
  let hi = 6;
  for (const arr of [d_ba, d_ca, d_cb]) for (const v of arr) hi = Math.max(hi, v);
  hi *= 1.06;
  const yFor = (v) => PAD.t + ((hi - v) / hi) * (H - PAD.t - PAD.b);
  const xFor = (i) => PAD.l + (t[i] / (t[t.length - 1] || 1)) * (W - PAD.l - PAD.r);

  x.fillStyle = "rgba(63,214,143,0.10)";
  x.fillRect(PAD.l, yFor(4), W - PAD.l - PAD.r, yFor(0) - yFor(4));
  const lateX = xFor(Math.floor(t.length / 2));
  x.fillStyle = "rgba(255,255,255,0.045)";
  x.fillRect(lateX, PAD.t, W - PAD.r - lateX, H - PAD.t - PAD.b);

  x.strokeStyle = C.grid;
  x.strokeRect(PAD.l + 0.5, PAD.t + 0.5, W - PAD.l - PAD.r - 1, H - PAD.t - PAD.b - 1);
  x.fillStyle = C.dim;
  x.font = "10px monospace";
  x.textAlign = "right";
  x.fillText(hi.toFixed(1), PAD.l - 4, PAD.t + 8);
  x.fillText("4", PAD.l - 4, yFor(4) + 3);
  x.fillText("0", PAD.l - 4, yFor(0));
  x.textAlign = "left";
  x.fillText("t=0", PAD.l, H - 5);
  x.textAlign = "right";
  x.fillText(`t=${d.params.steps}`, W - PAD.r, H - 5);
  x.textAlign = "left";

  strokeSeries(x, d_ca, xFor, yFor, C.pink, 1.0);
  strokeSeries(x, d_cb, xFor, yFor, AGENT.c, 1.2);
  strokeSeries(x, d_ba, xFor, yFor, AGENT.b, 1.2);
  x.fillStyle = AGENT.b; x.fillText("B–A", PAD.l + 4, PAD.t + 9);
  x.fillStyle = AGENT.c; x.fillText("C–B", PAD.l + 34, PAD.t + 9);
  x.fillStyle = C.pink; x.fillText("C–A", PAD.l + 64, PAD.t + 9);
  switchMarks(x, d, W, H);
}

function drawLockStatic(x, W, H, d) {
  const { lock_ba, lock_ca, lock_cb, t } = d.trace;
  const s = d.summary;
  bg(x, { width: W, height: H });
  const yFor = (v) => PAD.t + (1 - v) * (H - PAD.t - PAD.b);
  const xFor = (i) => PAD.l + (t[i] / (t[t.length - 1] || 1)) * (W - PAD.l - PAD.r);

  x.strokeStyle = C.grid;
  x.strokeRect(PAD.l + 0.5, PAD.t + 0.5, W - PAD.l - PAD.r - 1, H - PAD.t - PAD.b - 1);
  x.setLineDash([2, 4]);
  x.beginPath();
  x.moveTo(PAD.l, yFor(0.5)); x.lineTo(W - PAD.r, yFor(0.5));
  x.stroke();
  x.setLineDash([]);
  x.fillStyle = C.dim;
  x.font = "10px monospace";
  x.textAlign = "right";
  x.fillText("1", PAD.l - 4, yFor(1) + 4);
  x.fillText(".5", PAD.l - 4, yFor(0.5) + 3);
  x.fillText("0", PAD.l - 4, yFor(0));
  x.textAlign = "left";

  strokeSeries(x, lock_ca, xFor, yFor, C.pink, 1.0);
  strokeSeries(x, lock_cb, xFor, yFor, AGENT.c, 1.3);
  strokeSeries(x, lock_ba, xFor, yFor, AGENT.b, 1.3);
  // final scored locks (full-resolution summary values), tagged at the right
  const tags = [
    [`B→A ${s.B_A.toFixed(3)}`, s.B_A, AGENT.b],
    [`C→B ${s.C_B.toFixed(3)}`, s.C_B, AGENT.c],
    [`C→A ${s.C_A.toFixed(3)}`, s.C_A, C.pink],
  ];
  tags.sort((p, q) => q[1] - p[1]);
  let lastY = -99;
  for (const [txt, v, color] of tags) {
    let py = yFor(v) + 3;
    if (py - lastY < 11) py = lastY + 11;
    lastY = py;
    x.fillStyle = color;
    x.textAlign = "right";
    x.fillText(txt, W - PAD.r - 3, py);
  }
  x.textAlign = "left";
}

// ---------- arena (per-frame) -----------------------------------------------
function drawArena(d, idx) {
  const { ax: pax, ay: pay, bx, by, cx, cy, sel_b, sel_c } = d.trace;
  const box = d.config.box_size, r = d.config.agent_radius;
  bg(arenaX, arenaC);
  const side = arenaC.width, pad = 14;
  const s = (side - 2 * pad) / box;
  const X = (u) => pad + u * s;
  const Y = (v) => side - pad - v * s;

  arenaX.strokeStyle = C.grid;
  arenaX.strokeRect(X(0), Y(box), box * s, box * s);

  const agents = [
    { xs: pax, ys: pay, color: AGENT.a, lab: "A" },
    { xs: bx, ys: by, color: AGENT.b, lab: "B" },
    { xs: cx, ys: cy, color: AGENT.c, lab: "C" },
  ];

  // full path, faint
  arenaX.globalAlpha = 0.12;
  for (const a of agents) {
    arenaX.strokeStyle = a.color;
    arenaX.beginPath();
    for (let i = 0; i <= idx; i++) {
      i ? arenaX.lineTo(X(a.xs[i]), Y(a.ys[i])) : arenaX.moveTo(X(a.xs[i]), Y(a.ys[i]));
    }
    arenaX.stroke();
  }
  arenaX.globalAlpha = 1;

  // vivid trail, fading with age
  const i0 = Math.max(1, idx - TRAIL);
  for (const a of agents) {
    arenaX.strokeStyle = a.color;
    arenaX.lineWidth = 1.4;
    for (let i = i0; i <= idx; i++) {
      arenaX.globalAlpha = 0.08 + 0.72 * ((i - i0) / Math.max(idx - i0, 1));
      arenaX.beginPath();
      arenaX.moveTo(X(a.xs[i - 1]), Y(a.ys[i - 1]));
      arenaX.lineTo(X(a.xs[i]), Y(a.ys[i]));
      arenaX.stroke();
    }
  }
  arenaX.globalAlpha = 1;
  arenaX.lineWidth = 1;

  // attention: follower -> selected source, colored by the selected agent;
  // in summed-retina mode (sel = -1) both sources are sensed — dashed dim links
  const links = [
    { from: 1, sel: sel_b[idx], targets: [0, 2] },  // B: 0 = A, 1 = C
    { from: 2, sel: sel_c[idx], targets: [0, 1] },  // C: 0 = A, 1 = B
  ];
  for (const L of links) {
    const f = agents[L.from];
    const draw = (tj, color, dashed) => {
      const g = agents[tj];
      arenaX.strokeStyle = color;
      arenaX.globalAlpha = dashed ? 0.5 : 0.85;
      arenaX.lineWidth = dashed ? 1 : 1.6;
      if (dashed) arenaX.setLineDash([3, 4]);
      arenaX.beginPath();
      arenaX.moveTo(X(f.xs[idx]), Y(f.ys[idx]));
      arenaX.lineTo(X(g.xs[idx]), Y(g.ys[idx]));
      arenaX.stroke();
      arenaX.setLineDash([]);
      if (!dashed) {  // ring the attended source
        arenaX.beginPath();
        arenaX.arc(X(g.xs[idx]), Y(g.ys[idx]), Math.max(r * s, 4) + 3.5, 0, 2 * Math.PI);
        arenaX.stroke();
      }
    };
    if (L.sel < 0) {
      for (const tj of L.targets) draw(tj, C.dim, true);
    } else {
      const tj = L.targets[L.sel];
      draw(tj, agents[tj].color, false);
    }
  }
  arenaX.globalAlpha = 1;
  arenaX.lineWidth = 1;

  // agents at the playhead
  arenaX.font = "11px monospace";
  for (const [j, a] of agents.entries()) {
    const px = X(a.xs[idx]), py = Y(a.ys[idx]);
    arenaX.beginPath();
    arenaX.arc(px, py, Math.max(r * s, 4), 0, 2 * Math.PI);
    if (j === 0) {
      arenaX.fillStyle = a.color;
      arenaX.fill();
    } else {
      arenaX.fillStyle = "#22303f";
      arenaX.strokeStyle = a.color;
      arenaX.fill();
      arenaX.stroke();
    }
    arenaX.fillStyle = a.color;
    arenaX.fillText(a.lab, px + 7, py - 7);
  }
}

// ---------- playback --------------------------------------------------------
const scrubEl = document.getElementById("scrub");
const playBtn = document.getElementById("btn-play");
const tlabEl = document.getElementById("tlab");

function nSamples() { return latest ? latest.trace.t.length : 0; }

function drawFrame() {
  if (!latest) return;
  const d = latest;
  const idx = Math.min(Math.floor(play.idx), nSamples() - 1);
  drawArena(d, idx);
  // blit static strips + playhead cursor
  for (const [c, x, name] of [[attC, attX, "att"], [distC, distX, "dist"], [lockC, lockX, "lock"]]) {
    x.drawImage(off[name], 0, 0);
    const px = Math.round(xForTime(d, c.width, d.trace.t[idx])) + 0.5;
    x.strokeStyle = "rgba(255,255,255,0.65)";
    x.beginPath();
    x.moveTo(px, PAD.t);
    x.lineTo(px, c.height - PAD.b);
    x.stroke();
  }
  scrubEl.value = idx;
  tlabEl.textContent = `t = ${d.trace.t[idx]} / ${d.params.steps}`;
}

function tick(now) {
  if (!play.playing) return;
  const dt = Math.min((now - play.last) / 1000, 0.25);
  play.last = now;
  const perSample = latest.params.subsample;
  play.idx += (BASE_STEPS_PER_SEC * play.speed * dt) / perSample;
  if (play.idx >= nSamples() - 1) {
    play.idx = nSamples() - 1;
    setPlaying(false);
  }
  drawFrame();
  if (play.playing) play.raf = requestAnimationFrame(tick);
}

function setPlaying(on) {
  play.playing = on;
  playBtn.textContent = on ? "⏸ pause" : "▶ play";
  if (on) {
    play.last = performance.now();
    cancelAnimationFrame(play.raf);
    play.raf = requestAnimationFrame(tick);
  }
}

function startPlayback() {
  scrubEl.max = nSamples() - 1;
  play.idx = 0;
  setPlaying(true);
}

playBtn.addEventListener("click", () => {
  if (!latest) return;
  if (!play.playing && play.idx >= nSamples() - 1) play.idx = 0;  // replay
  setPlaying(!play.playing);
});
scrubEl.addEventListener("input", () => {
  if (!latest) return;
  setPlaying(false);
  play.idx = parseInt(scrubEl.value) || 0;
  drawFrame();
});
document.getElementById("speed").addEventListener("change", (e) => {
  play.speed = parseFloat(e.target.value) || 1;
});

// ---------- captions & render -----------------------------------------------
function caption(d) {
  const s = d.summary, m = d.params.mode;
  const f = (v) => v.toFixed(3);
  if (m === "off") {
    return `no attention — the two retinal bumps are summed; superposed flows leave no co-moving ` +
      `frame and nothing locks (B→A ${f(s.B_A)}, C→A ${f(s.C_A)}, C→B ${f(s.C_B)}): collapse.`;
  }
  if (m === "wta") {
    const cap = s.b_switch_times.length ? ` B abandons the pacemaker for C at t=${s.b_switch_times[0]};` : "";
    return `memoryless WTA — selection follows raw salience every step:${cap} no chain to A forms ` +
      `(B→A ${f(s.B_A)}, C→A ${f(s.C_A)}): collapse.`;
  }
  if (m === "sticky") {
    const cap = s.b_switch_times.length
      ? `then C orbits close to B, becomes B's brightest stimulus and captures its attention at t=${s.b_switch_times[0]} — the follower seduces the leader, and the pair goes adrift`
      : `and B keeps the pacemaker this run`;
    return `sticky 2×/100 — attention with persistence: a chain forms (C→B ${f(s.C_B)}), ${cap} (B→A ${f(s.B_A)}).`;
  }
  return `sticky 5×/300 — capture-resistant: the first stable all-visible homeostatic chain, ` +
    `B holds A ${f(s.B_A)} and C holds B ${f(s.C_B)} with ${s.b_switches + s.c_switches} switches.`;
}

function render(d) {
  fitWide();
  offscreen("att", attC.width, attC.height, (x, w, h) => drawAttStatic(x, w, h, d));
  offscreen("dist", distC.width, distC.height, (x, w, h) => drawDistStatic(x, w, h, d));
  offscreen("lock", lockC.width, lockC.height, (x, w, h) => drawLockStatic(x, w, h, d));
  scrubEl.max = Math.max(nSamples() - 1, 1);
  if (play.idx > nSamples() - 1) play.idx = nSamples() - 1;
  drawFrame();
  const s = d.summary;
  document.getElementById("e-ba").textContent = s.B_A.toFixed(3);
  document.getElementById("e-cb").textContent = s.C_B.toFixed(3);
  document.getElementById("e-ca").textContent = s.C_A.toFixed(3);
  const swB = s.b_switches ? `B ${s.b_switches} (t=${s.b_switch_times.join(",")})` : "B 0";
  const swC = s.c_switches ? `C ${s.c_switches} (t=${s.c_switch_times.join(",")})` : "C 0";
  document.getElementById("e-sw").textContent =
    d.config.sticky ? `${swB} · ${swC}` : "— (no selection)";
  document.getElementById("caption").textContent = caption(d);
  document.getElementById("stat").textContent =
    `${d.config.label} · ${d.params.steps} steps` +
    ` · followers N=${d.config.n_nodes}, champion wiring ${d.config.champ_seed} (twins)` +
    ` · B from (${d.config.b_start}), C from (${d.config.c_start})` +
    ` · pacemaker seed ${d.config.pace_seed}`;
}

// ---------- wiring ----------------------------------------------------------
for (const id of ["mode", "steps"]) {
  document.getElementById(id).addEventListener("change", run);
}
document.getElementById("btn-run").addEventListener("click", run);
fitWide();
run();
