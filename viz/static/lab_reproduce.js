/* Frontend for the reproduction-ladder viewer (H97's four inheritance rungs:
 * position only / + wiring heredity / pure clone / full-state budding).
 *
 * All simulation happens server-side (/lab/api/reproduce mirrors the exact
 * co-simulation loop of scripts/lab/h97_reproduce.py run(reproduce=True),
 * with the Agent class, champion genome and mutation operator imported from
 * that script); this file only renders the returned traces and replays them.
 */

"use strict";

const C = {
  panel: "#10141a", grid: "#2a313c", text: "#d7dde6", dim: "#8b95a3",
  red: "#ff5d5d", blue: "#5da9ff", green: "#3fd68f", yellow: "#ffd23f",
  pink: "#ff9ecb", accent: "#58a6ff", gray: "#39424f",
};
const PACE = C.red;  // the blind wall-circling pacemaker
// agent slot colors: A0 the founder, A1..A5 its (grand)children
const AGENT_COLORS = ["#5da9ff", "#3fd68f", "#ffd23f", "#ff9ecb", "#ffa04d", "#b48cff"];

const connEl = document.getElementById("conn");
let latest = null;
let runSeq = 0;

// playback state (sample index into the subsampled trace)
const play = { idx: 0, playing: false, speed: 1, raf: null, last: 0 };
const BASE_STEPS_PER_SEC = 900; // sim steps per second at 1×
const TRAIL = 240;              // vivid trail length, in samples
const PULSE = 90;               // birth-ring lifetime, in samples

// ---------- fetch -----------------------------------------------------------
function params() {
  return {
    mode: document.getElementById("mode").value,
    steps: Math.min(Math.max(parseInt(document.getElementById("steps").value) || 21600, 3600), 21600),
  };
}

async function run() {
  const p = params();
  const seq = ++runSeq;
  connEl.textContent = "running…";
  connEl.className = "conn";
  const t0 = performance.now();
  try {
    const res = await fetch(`/lab/api/reproduce?${new URLSearchParams(p)}`);
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
const [popC, popX] = cv("popc");
const [distC, distX] = cv("distc");
const WIDE = [popC, distC];
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

// first sample at which an agent exists (its arrays are null before birth)
function firstFinite(arr) {
  for (let i = 0; i < arr.length; i++) if (arr[i] != null) return i;
  return -1;
}

// x position of a full-resolution step time on the wide strips
function xForTime(d, W, time) {
  const tMax = d.params.steps || 1;
  return PAD.l + (time / tMax) * (W - PAD.l - PAD.r);
}

function spawnMarks(x, d, W, H) {
  x.setLineDash([3, 3]);
  x.font = "9px monospace";
  x.textAlign = "left";
  for (const [k, sp] of d.spawns.entries()) {
    const color = AGENT_COLORS[sp.child % AGENT_COLORS.length];
    x.strokeStyle = color;
    x.fillStyle = color;
    const px = Math.round(xForTime(d, W, sp.t)) + 0.5;
    x.beginPath();
    x.moveTo(px, PAD.t);
    x.lineTo(px, H - PAD.b);
    x.stroke();
    x.fillText(`A${sp.parent}→A${sp.child}`, px + 3, PAD.t + 8 + (k % 3) * 9);
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

function drawPopStatic(x, W, H, d) {
  const { pop, t } = d.trace;
  const cap = d.config.cap;
  bg(x, { width: W, height: H });
  const yFor = (v) => PAD.t + ((cap + 0.5 - v) / (cap + 0.5)) * (H - PAD.t - PAD.b);
  const xFor = (i) => PAD.l + (t[i] / (t[t.length - 1] || 1)) * (W - PAD.l - PAD.r);

  x.strokeStyle = C.grid;
  x.strokeRect(PAD.l + 0.5, PAD.t + 0.5, W - PAD.l - PAD.r - 1, H - PAD.t - PAD.b - 1);
  x.setLineDash([2, 4]);
  x.beginPath();
  x.moveTo(PAD.l, yFor(cap)); x.lineTo(W - PAD.r, yFor(cap));
  x.stroke();
  x.setLineDash([]);
  x.fillStyle = C.dim;
  x.font = "10px monospace";
  x.textAlign = "right";
  x.fillText(`cap ${cap}`, PAD.l - 4, yFor(cap) + 3);
  x.fillText("1", PAD.l - 4, yFor(1) + 3);
  x.fillText("0", PAD.l - 4, yFor(0));
  x.textAlign = "left";
  x.fillText("t=0", PAD.l, H - 5);
  x.textAlign = "right";
  x.fillText(`t=${d.params.steps}`, W - PAD.r, H - 5);
  x.textAlign = "left";

  // staircase: population only ever steps up, at spawns
  x.strokeStyle = C.accent;
  x.lineWidth = 1.6;
  x.beginPath();
  x.moveTo(xFor(0), yFor(pop[0]));
  for (let i = 1; i < pop.length; i++) {
    if (pop[i] !== pop[i - 1]) {
      x.lineTo(xFor(i), yFor(pop[i - 1]));
      x.lineTo(xFor(i), yFor(pop[i]));
    }
  }
  x.lineTo(xFor(pop.length - 1), yFor(pop[pop.length - 1]));
  x.stroke();
  x.lineWidth = 1;
  spawnMarks(x, d, W, H);
}

function drawDistStatic(x, W, H, d) {
  const { t } = d.trace;
  const lockD = d.config.lock_d;
  bg(x, { width: W, height: H });
  let hi = lockD * 2;
  for (const ag of d.trace.agents) {
    for (const v of ag.dist) if (v != null && v > hi) hi = v;
  }
  hi *= 1.06;
  const yFor = (v) => PAD.t + ((hi - v) / hi) * (H - PAD.t - PAD.b);
  const xFor = (i) => PAD.l + (t[i] / (t[t.length - 1] || 1)) * (W - PAD.l - PAD.r);

  // lock band and the scored late window
  x.fillStyle = "rgba(63,214,143,0.10)";
  x.fillRect(PAD.l, yFor(lockD), W - PAD.l - PAD.r, yFor(0) - yFor(lockD));
  const lateX = xForTime(d, W, Math.max(d.params.steps - d.config.late_window, 0));
  x.fillStyle = "rgba(255,255,255,0.045)";
  x.fillRect(lateX, PAD.t, W - PAD.r - lateX, H - PAD.t - PAD.b);

  x.strokeStyle = C.grid;
  x.strokeRect(PAD.l + 0.5, PAD.t + 0.5, W - PAD.l - PAD.r - 1, H - PAD.t - PAD.b - 1);
  x.fillStyle = C.dim;
  x.font = "10px monospace";
  x.textAlign = "right";
  x.fillText(hi.toFixed(1), PAD.l - 4, PAD.t + 8);
  x.fillText(lockD.toFixed(1), PAD.l - 4, yFor(lockD) + 3);
  x.fillText("0", PAD.l - 4, yFor(0));
  x.textAlign = "left";
  x.fillText("t=0", PAD.l, H - 5);
  x.textAlign = "right";
  x.fillText(`t=${d.params.steps}`, W - PAD.r, H - 5);
  x.textAlign = "left";

  for (const [j, ag] of d.trace.agents.entries()) {
    const i0 = firstFinite(ag.dist);
    if (i0 < 0) continue;
    strokeSeries(x, ag.dist.slice(i0), (i) => xFor(i + i0), yFor,
                 AGENT_COLORS[j % AGENT_COLORS.length], j === 0 ? 1.3 : 1.1);
  }
  spawnMarks(x, d, W, H);

  // final lock tags (the campaign's late lock fraction), stacked at the right
  const tags = d.agents.map((a, j) => {
    const dist = d.trace.agents[j].dist;
    let last = 0;
    for (let i = dist.length - 1; i >= 0; i--) if (dist[i] != null) { last = dist[i]; break; }
    return [`A${j} ${a.locked ? "✓" : "✗"} ${a.lock_late.toFixed(3)}`,
            yFor(last), AGENT_COLORS[j % AGENT_COLORS.length]];
  });
  tags.sort((p, q) => p[1] - q[1]);
  let lastY = -99;
  const placed = [];
  for (const [txt, py0, color] of tags) {
    let py = py0 + 3;
    if (py - lastY < 11) py = lastY + 11;
    lastY = py;
    placed.push([txt, py, color]);
  }
  // keep the stack inside the plot (six coincident locks would run off it)
  const over = lastY - (H - PAD.b - 2);
  if (over > 0) for (const p of placed) p[1] -= over;
  x.textAlign = "right";
  for (const [txt, py, color] of placed) {
    x.fillStyle = color;
    x.fillText(txt, W - PAD.r - 3, py);
  }
  x.textAlign = "left";
}

// ---------- arena (per-frame) -----------------------------------------------
function drawArena(d, idx) {
  const { ax: pax, ay: pay } = d.trace;
  const box = d.config.box_size, r = d.config.agent_radius;
  const sub = d.params.subsample;
  const tFull = d.trace.t[idx];
  bg(arenaX, arenaC);
  const side = arenaC.width, pad = 14;
  const s = (side - 2 * pad) / box;
  const X = (u) => pad + u * s;
  const Y = (v) => side - pad - v * s;
  const rpx = Math.max(r * s, 4);

  arenaX.strokeStyle = C.grid;
  arenaX.strokeRect(X(0), Y(box), box * s, box * s);

  const bodies = [{ xs: pax, ys: pay, color: PACE, lab: "P", i0: 0 }];
  for (const [j, ag] of d.trace.agents.entries()) {
    bodies.push({ xs: ag.x, ys: ag.y, color: AGENT_COLORS[j % AGENT_COLORS.length],
                  lab: `A${j}`, i0: firstFinite(ag.x), j });
  }

  // full path, faint
  arenaX.globalAlpha = 0.12;
  for (const b of bodies) {
    if (b.i0 < 0 || b.i0 > idx) continue;
    arenaX.strokeStyle = b.color;
    arenaX.beginPath();
    for (let i = b.i0; i <= idx; i++) {
      i === b.i0 ? arenaX.moveTo(X(b.xs[i]), Y(b.ys[i])) : arenaX.lineTo(X(b.xs[i]), Y(b.ys[i]));
    }
    arenaX.stroke();
  }
  arenaX.globalAlpha = 1;

  // vivid trail, fading with age
  for (const b of bodies) {
    if (b.i0 < 0 || b.i0 > idx) continue;
    const i0 = Math.max(b.i0 + 1, idx - TRAIL);
    arenaX.strokeStyle = b.color;
    arenaX.lineWidth = 1.4;
    for (let i = i0; i <= idx; i++) {
      arenaX.globalAlpha = 0.08 + 0.72 * ((i - i0) / Math.max(idx - i0, 1));
      arenaX.beginPath();
      arenaX.moveTo(X(b.xs[i - 1]), Y(b.ys[i - 1]));
      arenaX.lineTo(X(b.xs[i]), Y(b.ys[i]));
      arenaX.stroke();
    }
  }
  arenaX.globalAlpha = 1;
  arenaX.lineWidth = 1;

  // lineage: a faint dashed line parent -> child while both are on the board
  arenaX.setLineDash([2, 3]);
  for (const b of bodies.slice(1)) {
    const a = d.agents[b.j];
    if (a.parent < 0 || b.i0 < 0 || b.i0 > idx) continue;
    const p = bodies[1 + a.parent];
    arenaX.strokeStyle = b.color;
    arenaX.globalAlpha = 0.35;
    arenaX.beginPath();
    arenaX.moveTo(X(p.xs[idx]), Y(p.ys[idx]));
    arenaX.lineTo(X(b.xs[idx]), Y(b.ys[idx]));
    arenaX.stroke();
  }
  arenaX.setLineDash([]);
  arenaX.globalAlpha = 1;

  // birth pulses: an expanding ring at each spawn position
  for (const sp of d.spawns) {
    const st = Math.floor(sp.t / sub);
    if (idx < st || idx - st >= PULSE) continue;
    const f = (idx - st) / PULSE;
    arenaX.strokeStyle = AGENT_COLORS[sp.child % AGENT_COLORS.length];
    arenaX.globalAlpha = 0.9 * (1 - f);
    arenaX.lineWidth = 2;
    arenaX.beginPath();
    arenaX.arc(X(sp.x), Y(sp.y), rpx + 2 + f * 26, 0, 2 * Math.PI);
    arenaX.stroke();
    if (f < 0.5) {
      arenaX.font = "11px monospace";
      arenaX.fillStyle = AGENT_COLORS[sp.child % AGENT_COLORS.length];
      arenaX.fillText(`A${sp.child} born`, X(sp.x) + rpx + 8, Y(sp.y) + rpx + 12);
    }
  }
  arenaX.globalAlpha = 1;
  arenaX.lineWidth = 1;

  // agents at the playhead; arc = lock streak charging toward a spawn.
  // Budded copies are exact, so a lineage can move in perfect lockstep:
  // agent radii grow with index (concentric rings when superimposed) and
  // labels of co-located agents stack upward instead of overprinting.
  arenaX.font = "11px monospace";
  const seen = new Map();
  for (const b of bodies) {
    if (b.i0 < 0 || b.i0 > idx) continue;
    const px = X(b.xs[idx]), py = Y(b.ys[idx]);
    const rr = b.lab === "P" ? rpx : rpx + 1.5 * b.j;
    arenaX.beginPath();
    arenaX.arc(px, py, rr, 0, 2 * Math.PI);
    if (b.lab === "P") {
      arenaX.fillStyle = b.color;
      arenaX.fill();
    } else {
      arenaX.strokeStyle = b.color;
      if (b.j === 0) {
        arenaX.fillStyle = "#22303f";
        arenaX.fill();
      }
      arenaX.stroke();
      const streak = d.trace.agents[b.j].streak[idx];
      const hasSpawned = d.spawns.some((sp) => sp.parent === b.j && sp.t <= tFull);
      if (!hasSpawned && streak > 0) {
        arenaX.strokeStyle = b.color;
        arenaX.lineWidth = 2;
        arenaX.beginPath();
        arenaX.arc(px, py, rr + 3.5, -Math.PI / 2,
                   -Math.PI / 2 + 2 * Math.PI * Math.min(streak / d.config.spawn_after, 1));
        arenaX.stroke();
        arenaX.lineWidth = 1;
      }
    }
    const key = `${Math.round(px / 8)},${Math.round(py / 8)}`;
    const stack = seen.get(key) || 0;
    seen.set(key, stack + 1);
    arenaX.fillStyle = b.color;
    arenaX.fillText(b.lab, px + rr + 4, py - 7 - stack * 10);
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
  for (const [c, x, name] of [[popC, popX, "pop"], [distC, distX, "dist"]]) {
    x.drawImage(off[name], 0, 0);
    const px = Math.round(xForTime(d, c.width, d.trace.t[idx])) + 0.5;
    x.strokeStyle = "rgba(255,255,255,0.65)";
    x.beginPath();
    x.moveTo(px, PAD.t);
    x.lineTo(px, c.height - PAD.b);
    x.stroke();
  }
  scrubEl.value = idx;
  tlabEl.textContent = `t = ${d.trace.t[idx]} / ${d.params.steps} · pop ${d.trace.pop[idx]}`;
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

// ---------- captions, legend, verdict ---------------------------------------
function caption(d) {
  const s = d.summary, m = d.params.mode;
  const lockTxt = `${s.locked}/${s.n_agents} locked`;
  const t0 = d.spawns.length ? d.spawns[0].t : null;
  if (m === "position") {
    return `rung 1, position only — A0 holds lock for ${d.config.spawn_after} steps and buds A1 at ` +
      `its own in-basin position${t0 !== null ? ` (t=${t0})` : ""}, with fresh wiring and a mutated ` +
      `genome; the child loses the competence lottery and never locks: ${lockTxt}.`;
  }
  if (m === "wiring") {
    return `rung 2, + wiring heredity — the child also inherits the parent's exact wiring seed, but ` +
      `its internal state starts cold; it loses the acquisition lottery: ${lockTxt}. Structure is not enough.`;
  }
  if (m === "clone") {
    return `rung 3, pure clone — genome and wiring bit-identical to a proven parent (mutation ` +
      `exonerated), yet born cold the child still fails: ${lockTxt}. The barrier is internal state.`;
  }
  return `rung 4, budding — each child copies its parent's entire dynamical state (x, T, W, spike ` +
    `buffers, heading) and is born mid-lock; ${s.n_spawns} spawns cascade to the cap and ` +
    `${lockTxt} (the copies are exact, so the lineage moves in perfect lockstep — superimposed ` +
    `agents draw as concentric rings). The heritable unit is the entire dynamical state — ` +
    `you can't inherit a lock.`;
}

function legendHtml(d) {
  const bits = [
    `<span><span class="sw" style="background:${PACE}"></span>P — pacemaker, blind wall-avoider</span>`,
    `<span><span class="sw" style="background:${AGENT_COLORS[0]}"></span>A0 — founder, h48e champion</span>`,
  ];
  for (const a of d.agents.slice(1)) {
    bits.push(`<span><span class="sw" style="background:${AGENT_COLORS[a.id % AGENT_COLORS.length]}"></span>` +
              `A${a.id} — child of A${a.parent}</span>`);
  }
  return bits.join(" ");
}

function verdictHtml(d) {
  const tick = (ok) => ok
    ? `<span style="color:var(--green)">✓ locked</span>`
    : `<span style="color:var(--dim)">✗ not locked</span>`;
  return d.agents.map((a) => {
    const col = AGENT_COLORS[a.id % AGENT_COLORS.length];
    const who = a.parent < 0
      ? `founder · wiring ${a.seed}`
      : `← A${a.parent} @ t=${a.birth} · ` +
        (a.seed === d.config.champ_seed ? "wiring inherited" : `fresh wiring ${a.seed}`) +
        (a.mutant ? " · mutant genome" : " · genome clone");
    return `<span><span class="sw" style="background:${col}"></span>` +
      `<span style="color:${col}">A${a.id}</span> · ${who} · lock ${a.lock_late.toFixed(3)} ${tick(a.locked)}</span>`;
  }).join(" ");
}

function render(d) {
  fitWide();
  offscreen("pop", popC.width, popC.height, (x, w, h) => drawPopStatic(x, w, h, d));
  offscreen("dist", distC.width, distC.height, (x, w, h) => drawDistStatic(x, w, h, d));
  scrubEl.max = Math.max(nSamples() - 1, 1);
  if (play.idx > nSamples() - 1) play.idx = nSamples() - 1;
  drawFrame();
  const s = d.summary;
  document.getElementById("e-pop").textContent = `${s.n_agents}`;
  document.getElementById("e-locked").textContent = `${s.locked}/${s.n_agents}`;
  document.getElementById("e-spawns").textContent = `${s.n_spawns}`;
  document.getElementById("e-mut").textContent = `${s.mutants_locked}`;
  document.getElementById("legend").innerHTML = legendHtml(d);
  document.getElementById("verdict").innerHTML = verdictHtml(d);
  document.getElementById("caption").textContent = caption(d);
  document.getElementById("stat").textContent =
    `${d.config.label} · ${d.params.steps} steps` +
    ` · lock dist<${d.config.lock_d} · spawn after ${d.config.spawn_after} locked steps` +
    ` · cap ${d.config.cap} · founder N=${d.config.n_nodes} wiring ${d.config.champ_seed}` +
    ` · pacemaker seed ${d.config.pace_seed} · lineage seed ${d.config.lineage_seed}`;
}

// ---------- wiring ----------------------------------------------------------
for (const id of ["mode", "steps"]) {
  document.getElementById(id).addEventListener("change", run);
}
document.getElementById("btn-run").addEventListener("click", run);
fitWide();
run();
