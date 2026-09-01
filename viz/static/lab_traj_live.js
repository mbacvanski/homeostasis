/* Live frontend for the trajectory playground (/lab/ws/traj).
 *
 * All simulation happens server-side (viz.lab_server TrajLive replicates
 * scripts/lab/common.py run_closed_loop verbatim, with the campaign arms
 * generalized to any-time surgery commands); this file renders frames and
 * maps the surgery buttons / live noise slider onto those commands.
 */

"use strict";

(() => {
  const C = LabLive.colors;
  const [arenaC, arenaX] = LabLive.cv("live-arena");
  const [sensC, sensX] = LabLive.cv("live-sens");
  const [segC, segX] = LabLive.cv("live-seg");

  const headStrip = LabLive.strip("live-strip", {
    yMin: 0, yMax: 360,
    lanes: [
      { key: "stim", color: C.white, lw: 1.4, dots: true },
      { key: "heading", color: C.red, lw: 1.6, dots: true },
    ],
    label: "heading (red) vs stimulus (white), degrees",
  });
  const errStrip = LabLive.strip("live-err", {
    yMin: -185, yMax: 185,
    lanes: [{ key: "err", color: C.accent, lw: 1.4, dots: true }],
    bands: [{ lo: -45, hi: 45, color: "rgba(63,214,143,0.10)" }],
    hlines: [{ y: 0, color: "rgba(42,49,60,0.9)", dash: [] }],
    label: "heading error",
  });
  const propStrip = LabLive.strip("live-prop", {
    yMin: 0, yMax: 1,
    lanes: [{ key: "prop", color: C.green, lw: 1.3 }],
    label: "prop. spiked",
  });
  const strips = [headStrip, errStrip, propStrip];

  function clearAll() { for (const s of strips) s.clear(); }

  const rad = (d) => (d * Math.PI) / 180;
  function ringXY(deg, r) {
    const cx = arenaC.width / 2, cy = arenaC.height / 2;
    return [cx + r * Math.cos(rad(deg)), cy - r * Math.sin(rad(deg))];
  }

  function drawArena(m) {
    LabLive.bg(arenaX, arenaC);
    const now = m.now;
    const cx = arenaC.width / 2, cy = arenaC.height / 2;
    const R = Math.min(cx, cy) - 18, RA = Math.round(R * 0.5);

    // FOV wedge: heading ± 92 (the retina spans ±90; 2° of margin)
    arenaX.beginPath();
    arenaX.moveTo(cx, cy);
    arenaX.arc(cx, cy, R + 12, -rad(now.heading + 92), -rad(now.heading - 92));
    arenaX.closePath();
    arenaX.fillStyle = "rgba(88,166,255,0.07)";
    arenaX.fill();

    // stimulus orbit ring
    arenaX.beginPath();
    arenaX.arc(cx, cy, R, 0, 2 * Math.PI);
    arenaX.strokeStyle = C.grid;
    arenaX.stroke();

    // agent body + heading arrow
    arenaX.beginPath();
    arenaX.arc(cx, cy, RA, 0, 2 * Math.PI);
    arenaX.fillStyle = "rgba(255,158,203,0.18)";
    arenaX.fill();
    arenaX.strokeStyle = C.pink;
    arenaX.stroke();
    const [hx, hy] = ringXY(now.heading, RA);
    arenaX.beginPath();
    arenaX.moveTo(cx, cy);
    arenaX.lineTo(hx, hy);
    arenaX.strokeStyle = C.white;
    arenaX.lineWidth = 2;
    arenaX.stroke();
    arenaX.lineWidth = 1;

    // stimulus dot
    const [sx, sy] = ringXY(now.stim, R);
    arenaX.beginPath();
    arenaX.arc(sx, sy, 7, 0, 2 * Math.PI);
    arenaX.fillStyle = C.green;
    arenaX.fill();
  }

  function drawSensors(m) {
    LabLive.bg(sensX, sensC);
    const acts = m.now.sensors, n = acts.length;
    const w = sensC.width / n;
    for (let i = 0; i < n; i++) {
      const h = Math.min(Math.max(acts[i], 0), 1) * (sensC.height - 12);
      sensX.fillStyle = i < n / 2 ? C.red : C.blue;
      sensX.fillRect(i * w + 0.5, sensC.height - h, Math.max(1, w - 1), h);
    }
    sensX.strokeStyle = C.grid;
    sensX.beginPath();
    sensX.moveTo(sensC.width / 2, 0);
    sensX.lineTo(sensC.width / 2, sensC.height);
    sensX.stroke();
    sensX.fillStyle = C.dim;
    sensX.font = "10px monospace";
    sensX.fillText("1.0 —", 2, 9);
  }

  function drawSegs(m) {
    LabLive.bg(segX, segC);
    const seg = m.now.seg;
    const W = segC.width, H = segC.height;
    const scores = seg.scores;
    const slots = 12;
    const bw = W / slots;
    const yFor = (v) => H - 12 - v * (H - 22);
    for (let i = 0; i < scores.length; i++) {
      const x0 = W - (scores.length - i + 1) * bw;  // last slot = current segment
      segX.fillStyle = C.green;
      segX.fillRect(x0 + 2, yFor(scores[i]), bw - 4, yFor(0) - yFor(scores[i]));
    }
    // current (partial) segment: hollow bar
    segX.strokeStyle = C.green;
    segX.strokeRect(W - bw + 2, yFor(seg.cur), bw - 4, Math.max(yFor(0) - yFor(seg.cur), 1));
    segX.strokeStyle = C.yellow;
    segX.setLineDash([4, 4]);
    segX.beginPath();
    segX.moveTo(0, yFor(0.35));
    segX.lineTo(W, yFor(0.35));
    segX.stroke();
    segX.setLineDash([]);
    segX.fillStyle = C.dim;
    segX.font = "10px monospace";
    segX.fillText("1.0", 2, 10);
    segX.fillText("0", 2, H - 3);
    document.getElementById("live-seg-readout").textContent =
      `current segment: ${(seg.cur * 100).toFixed(0)}% within ±45° after ${seg.n}/${m.config.seg_len} steps` +
      (scores.length ? ` · last full: ${(scores[scores.length - 1] * 100).toFixed(0)}%` : "");
  }

  function syncSurgery(m) {
    const s = m.now.surgery;
    document.getElementById("sg-swap").classList.toggle("armed", s.swapped);
    document.getElementById("sg-freeze-w").classList.toggle("armed", s.w_frozen);
    document.getElementById("sg-freeze-t").classList.toggle("armed", s.t_frozen);
    document.getElementById("sg-freeze-both").classList.toggle("armed", !s.learning);
    const bits = [];
    if (s.swapped) bits.push("effectors SWAPPED");
    if (s.w_frozen) bits.push("W pinned");
    if (s.t_frozen) bits.push("T pinned");
    if (!s.learning) bits.push("learning OFF");
    if (s.killed_n) bits.push(`${s.killed_n}/${m.config.n_nodes} nodes dead`);
    if (s.noise > 0) bits.push(`noise σ=${s.noise.toFixed(2)}`);
    document.getElementById("sg-state").innerHTML =
      bits.length ? `→ <b>${bits.join(" · ")}</b>` : "→ untouched run";
  }

  function onFrame(m, wasReset) {
    if (wasReset) clearAll();
    for (const e of m.series) {
      headStrip.push({ t: e.t, heading: e.heading, stim: e.stim });
      errStrip.push({ t: e.t, err: e.err });
      propStrip.push({ t: e.t, prop: e.prop });
    }
    for (const s of strips) s.setEvents(m.now.events);
    drawArena(m);
    drawSensors(m);
    drawSegs(m);
    for (const s of strips) s.draw();
    syncSurgery(m);
    const now = m.now;
    document.getElementById("live-readout").textContent =
      `heading ${now.heading.toFixed(1)}° · stim ${now.stim.toFixed(1)}° · err ${now.err.toFixed(1)}° · ` +
      `ΔH ${now.dh.toFixed(2)}° · eff [${now.outputs[0].toFixed(2)}, ${now.outputs[1].toFixed(2)}] · f ${now.prop.toFixed(3)}`;
    document.getElementById("live-note").textContent =
      `${m.config.variant} · seed ${m.seed} · N=${m.config.n_nodes} wlr=${m.config.weight_lr} tlr=${m.config.target_lr}`;
  }

  const live = LabLive.connect({
    path: "/lab/ws/traj",
    onFrame,
    resetParams: () => ({
      variant: document.getElementById("variant").value,
      noise: parseFloat(document.getElementById("noise").value) || 0,
    }),
  });

  document.getElementById("variant").addEventListener("change", () => live.reset());
  document.getElementById("noise").addEventListener("input", (ev) => {
    live.send({ cmd: "noise", sigma: parseFloat(ev.target.value) || 0 });
  });

  const ops = {
    "sg-swap": { op: "swap" },
    "sg-freeze-w": { op: "freeze_w" },
    "sg-freeze-t": { op: "freeze_t" },
    "sg-freeze-both": { op: "freeze_both" },
    "sg-unfreeze": { op: "unfreeze" },
    "sg-kill10": { op: "kill", frac: 0.1 },
    "sg-kill30": { op: "kill", frac: 0.3 },
    "sg-kill50": { op: "kill", frac: 0.5 },
  };
  for (const [id, args] of Object.entries(ops)) {
    document.getElementById(id).onclick = () => live.send({ cmd: "surgery", ...args });
  }

  function fitAll() { for (const s of strips) s.fit(); }
  let rT = null;
  window.addEventListener("resize", () => {
    clearTimeout(rT);
    rT = setTimeout(() => { fitAll(); if (live.latest) onFrame(live.latest, false); }, 150);
  });
  fitAll();
})();
