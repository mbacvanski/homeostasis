/* Live frontend for the self-repair exhibit (/lab/ws/repair).
 *
 * All simulation happens server-side (viz.lab_server RepairLive steps two
 * same-seed tracking twins in lockstep with common.py's step order; the kill
 * command wounds both identically and freezes the frozen twin); this file
 * renders the twin strips and the paired segment bars.
 */

"use strict";

(() => {
  const C = LabLive.colors;
  const LEARN = C.green, FROZEN = C.orange;
  const [segC, segX] = LabLive.cv("live-seg");

  const fStrip = LabLive.strip("live-f", {
    yMin: 0, yMax: 1,
    lanes: [
      { key: "fa", color: LEARN, lw: 1.3 },
      { key: "fb", color: FROZEN, lw: 1.3, dash: [5, 3] },
    ],
    label: "spike rate f · learning green · frozen orange",
  });
  const wStrip = LabLive.strip("live-w", {
    yMin: 0, yMax: "auto",
    lanes: [
      { key: "wa", color: LEARN, lw: 1.3 },
      { key: "wb", color: FROZEN, lw: 1.3, dash: [5, 3] },
    ],
    label: "w̄ over surviving links",
  });
  const strips = [fStrip, wStrip];

  function clearAll() { for (const s of strips) s.clear(); }

  function drawSegs(m) {
    LabLive.bg(segX, segC);
    const a = m.now.a, b = m.now.b;
    const W = segC.width, H = segC.height;
    const slots = Math.max(a.seg.length + 1, 10);
    const bw = W / slots;
    const yFor = (v) => H - 12 - v * (H - 22);
    for (let i = 0; i < a.seg.length; i++) {
      const x0 = W - (a.seg.length - i + 1) * bw;
      const half = Math.max(2, bw / 2 - 3);
      segX.fillStyle = LEARN;
      segX.fillRect(x0 + 2, yFor(a.seg[i]), half, yFor(0) - yFor(a.seg[i]));
      segX.fillStyle = FROZEN;
      segX.fillRect(x0 + 2 + half + 1, yFor(b.seg[i]), half, yFor(0) - yFor(b.seg[i]));
    }
    // current (partial) segment: hollow pair
    const x0 = W - bw, half = Math.max(2, bw / 2 - 3);
    segX.strokeStyle = LEARN;
    segX.strokeRect(x0 + 2, yFor(a.cur), half, Math.max(yFor(0) - yFor(a.cur), 1));
    segX.strokeStyle = FROZEN;
    segX.strokeRect(x0 + 2 + half + 1, yFor(b.cur), half, Math.max(yFor(0) - yFor(b.cur), 1));
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
      `current segment (${m.now.seg_n}/${m.config.seg_len} steps): ` +
      `learning ${(a.cur * 100).toFixed(0)}% · frozen ${(b.cur * 100).toFixed(0)}%` +
      (a.seg.length
        ? ` · last full: ${(a.seg[a.seg.length - 1] * 100).toFixed(0)}% vs ${(b.seg[b.seg.length - 1] * 100).toFixed(0)}%`
        : "");
  }

  function onFrame(m, wasReset) {
    if (wasReset) clearAll();
    for (const e of m.series) {
      fStrip.push({ t: e.t, fa: e.fa, fb: e.fb });
      wStrip.push({ t: e.t, wa: e.wa, wb: e.wb });
    }
    const evts = m.now.kills.map((k) => ({ t: k.t, op: `kill ${Math.round(k.frac * 100)}%` }));
    for (const s of strips) { s.setEvents(evts); s.draw(); }
    drawSegs(m);
    const killed = m.now.killed_n;
    document.getElementById("sg-state").innerHTML = killed
      ? `→ <b>${killed}/${m.config.n_nodes} nodes dead · frozen twin's learning OFF since t=${m.now.kills[0].t}</b>`
      : "→ twins identical (no wound yet)";
    document.getElementById("live-note").textContent =
      `seed ${m.seed} · N=${m.config.n_nodes} wlr=${m.config.weight_lr} tlr=${m.config.target_lr} · ` +
      `f ${m.now.a.prop.toFixed(3)} vs ${m.now.b.prop.toFixed(3)} · w̄ ${m.now.a.w.toFixed(3)} vs ${m.now.b.w.toFixed(3)}`;
  }

  const live = LabLive.connect({ path: "/lab/ws/repair", onFrame });

  document.getElementById("sg-kill").onclick = () =>
    live.send({
      cmd: "kill",
      frac: parseFloat(document.getElementById("kill").value) || 0.3,
    });

  function fitAll() { for (const s of strips) s.fit(); }
  let rT = null;
  window.addEventListener("resize", () => {
    clearTimeout(rT);
    rT = setTimeout(() => { fitAll(); if (live.latest) onFrame(live.latest, false); }, 150);
  });
  fitAll();
})();
