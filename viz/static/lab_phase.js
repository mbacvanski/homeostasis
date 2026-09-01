/* Frontend for the phase-map browser.
 *
 * Pure display of recorded sweep rows served by /lab/api/phase (which slims
 * the archived JSON server-side). Nothing is simulated here.
 */

"use strict";

const C = {
  panel: "#10141a", grid: "#2a313c", text: "#d7dde6", dim: "#8b95a3",
  red: "#ff5d5d", blue: "#5da9ff", green: "#3fd68f", yellow: "#ffd23f",
  pink: "#ff9ecb", accent: "#58a6ff", darktext: "#0b1520",
};

// Axes per map: x/y accessors and display names
const MAPS = {
  A1: { x: (r) => r.wlr, y: (r) => r.tlr, xName: "weight_lr", yName: "target_lr" },
  A3: { x: (r) => r.leak, y: (r) => r.wlr, xName: "leak", yName: "weight_lr" },
};
const FRAC_THRESHOLD = 0.35;

let DATA = null;
let selected = null; // {tag, x, y}

const connEl = document.getElementById("conn");
const metricEl = document.getElementById("metric");

// ---------- color ramp ------------------------------------------------------
const STOPS = [
  [0.00, [20, 25, 32]],
  [0.35, [49, 84, 127]],
  [0.55, [75, 143, 214]],
  [0.75, [63, 214, 143]],
  [1.00, [255, 210, 63]],
];
function rampColor(u) {
  u = Math.min(Math.max(u, 0), 1);
  for (let i = 1; i < STOPS.length; i++) {
    if (u <= STOPS[i][0]) {
      const [u0, c0] = STOPS[i - 1], [u1, c1] = STOPS[i];
      const f = (u - u0) / (u1 - u0);
      const c = c0.map((v, k) => Math.round(v + f * (c1[k] - v)));
      return { css: `rgb(${c[0]},${c[1]},${c[2]})`, lum: (0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]) / 255 };
    }
  }
  return { css: "rgb(255,210,63)", lum: 0.85 };
}

// diverged runs arrive with null g_final / mean_abs_E — render those as "—"
const num = (v) => (v === null || v === undefined ? null : v);
const show = (v, digits) => (num(v) === null ? "—" : v.toFixed(digits));

function metricDomain(metric) {
  if (metric !== "input_flow") return [0, 1];
  let hi = 0;
  for (const r of DATA.rows) if (num(r.input_flow) !== null) hi = Math.max(hi, r.input_flow);
  return [0, Math.max(hi, 1e-9)];
}

function drawColorbar(metric) {
  const c = document.getElementById("colorbar"), x = c.getContext("2d");
  x.clearRect(0, 0, c.width, c.height);
  const [lo, hi] = metricDomain(metric);
  const W = c.width, barH = 12, y0 = 4;
  for (let px = 0; px < W; px++) x.fillStyle = rampColor(px / (W - 1)).css, x.fillRect(px, y0, 1, barH);
  x.strokeStyle = C.grid;
  x.strokeRect(0.5, y0 + 0.5, W - 1, barH - 1);
  x.fillStyle = C.dim;
  x.font = "10px monospace";
  x.fillText(lo.toFixed(metric === "input_flow" ? 2 : 1), 0, y0 + barH + 11);
  x.textAlign = "center";
  x.fillText(metric, W / 2, y0 + barH + 11);
  x.textAlign = "right";
  x.fillText(hi.toFixed(metric === "input_flow" ? 2 : 1), W, y0 + barH + 11);
  x.textAlign = "left";
}

// ---------- grid building ---------------------------------------------------
function cellsFor(tag) {
  const spec = MAPS[tag];
  const rows = DATA.rows.filter((r) => r.tag === tag);
  const xs = [...new Set(rows.map(spec.x))].sort((a, b) => a - b);
  const ys = [...new Set(rows.map(spec.y))].sort((a, b) => a - b);
  const cells = new Map();
  for (const r of rows) {
    const key = `${spec.x(r)}|${spec.y(r)}`;
    if (!cells.has(key)) cells.set(key, []);
    cells.get(key).push(r);
  }
  return { spec, xs, ys, cells };
}

function cellLink(tag, x, y) {
  const hit = (DATA.links || []).find((l) => l.tag === tag && l.x === x && l.y === y);
  return hit ? hit.loadout : null;
}

function fmt(v) {
  return Math.abs(v) >= 100 ? v.toFixed(0) : Number(v.toPrecision(3)).toString();
}

function buildMap(tag) {
  const box = document.getElementById(`map-${tag}`);
  const { spec, xs, ys, cells } = cellsFor(tag);
  const metric = metricEl.value;
  const [lo, hi] = metricDomain(metric);
  box.innerHTML = "";
  box.style.gridTemplateColumns = `auto repeat(${xs.length}, minmax(56px, 1fr))`;

  for (let yi = ys.length - 1; yi >= 0; yi--) {  // largest y on top
    const yl = document.createElement("div");
    yl.className = "hlab mono";
    yl.textContent = yi === ys.length - 1 ? `${spec.yName} ${fmt(ys[yi])}` : fmt(ys[yi]);
    box.appendChild(yl);
    for (const xv of xs) {
      const rows = (cells.get(`${xv}|${ys[yi]}`) || []).filter((r) => num(r[metric]) !== null);
      const cell = document.createElement("div");
      cell.className = "hcell mono";
      if (!rows.length) { cell.classList.add("empty"); box.appendChild(cell); continue; }
      const mean = rows.reduce((s, r) => s + r[metric], 0) / rows.length;
      const nOk = rows.filter((r) => r.score_late >= FRAC_THRESHOLD).length;
      const col = rampColor((mean - lo) / (hi - lo));
      cell.style.background = col.css;
      cell.style.color = col.lum > 0.55 ? C.darktext : C.text;
      cell.innerHTML =
        `<b>${mean.toFixed(2)}</b><i>≥.35: ${nOk}/${rows.length}</i>` +
        (cellLink(tag, xv, ys[yi]) ? `<u title="matches a registered loadout">◆</u>` : "");
      if (selected && selected.tag === tag && selected.x === xv && selected.y === ys[yi])
        cell.classList.add("selected");
      cell.onclick = () => { selected = { tag, x: xv, y: ys[yi] }; render(); };
      box.appendChild(cell);
    }
  }
  // x-axis label row
  const corner = document.createElement("div");
  corner.className = "hlab";
  box.appendChild(corner);
  xs.forEach((xv, i) => {
    const xl = document.createElement("div");
    xl.className = "hlab mono center";
    xl.textContent = i === 0 ? `${spec.xName} ${fmt(xv)}` : fmt(xv);
    box.appendChild(xl);
  });
}

// ---------- side panel ------------------------------------------------------
function renderSide() {
  const box = document.getElementById("cell-info");
  if (!selected) { box.className = "hint"; box.textContent = "no cell selected"; return; }
  const { tag, x, y } = selected;
  const { spec, cells } = cellsFor(tag);
  const rows = (cells.get(`${x}|${y}`) || []).slice().sort((a, b) => a.seed - b.seed);
  box.className = "";
  const mean = (k) => {
    const vals = rows.map((r) => num(r[k])).filter((v) => v !== null);
    return vals.length ? vals.reduce((s, v) => s + v, 0) / vals.length : null;
  };
  const link = cellLink(tag, x, y);
  box.innerHTML =
    `<div class="mono cellhead">${tag} · ${spec.xName}=${fmt(x)} · ${spec.yName}=${fmt(y)}</div>` +
    `<table class="seedtab mono"><tr><th>seed</th><th>score_late</th><th>prop_spk</th><th>g_final</th><th>|E|</th></tr>` +
    rows.map((r) =>
      `<tr class="${r.score_late >= FRAC_THRESHOLD ? "ok" : ""}"><td>${r.seed}</td>` +
      `<td>${show(r.score_late, 3)}</td><td>${show(r.prop_spiked, 3)}</td>` +
      `<td>${show(r.g_final, 2)}</td><td>${show(r.mean_abs_E, 3)}</td></tr>`).join("") +
    `<tr class="meanrow"><td>mean</td><td>${show(mean("score_late"), 3)}</td>` +
    `<td>${show(mean("prop_spiked"), 3)}</td><td>${show(mean("g_final"), 2)}</td>` +
    `<td>${show(mean("mean_abs_E"), 3)}</td></tr></table>` +
    (link
      ? `<a class="btn" href="/?loadout=${encodeURIComponent(link)}">open “${link}” in the tracking visualizer →</a>`
      : `<div class="hint">no registered loadout matches this cell</div>`);
}

function render() {
  buildMap("A1");
  buildMap("A3");
  drawColorbar(metricEl.value);
  renderSide();
}

// ---------- load ------------------------------------------------------------
async function load() {
  try {
    const res = await fetch("/lab/api/phase");
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    DATA = await res.json();
    if (DATA.error) throw new Error(DATA.error);
    connEl.textContent = "ok";
    connEl.className = "conn ok";
    document.getElementById("stat").textContent =
      `source ${DATA.source} · ${DATA.rows.length} runs`;
    render();
  } catch (err) {
    connEl.textContent = `error: ${err.message}`;
    connEl.className = "conn bad";
  }
}

metricEl.addEventListener("change", render);
load();
