/* Live frontend for the ecology viewer (/lab/ws/ecology).
 *
 * All simulation happens server-side (viz.lab_server EcologyLive: pacemaker
 * + saved H50-chain followers under sequential same-step coupling, exclusive
 * or all-visible sticky sensing); this file renders the arena with trails
 * and gaze lines, the link-distance strip, and the attention timeline.
 */

"use strict";

(() => {
  const C = LabLive.colors;
  // agent colors by index: 0 = pacemaker A, then followers B, C, D
  const AGENT = [C.red, C.accent, C.green, C.yellow];
  const NAMES = ["A", "B", "C", "D"];
  const [arenaC, arenaX] = LabLive.cv("live-arena");
  const [attC, attX] = LabLive.cv("live-att");

  const trails = [0, 1, 2, 3].map(() => LabLive.trail(900));
  const attRows = [];      // rolling [t, sel1, sel2, sel3]
  const ATT_CAP = 1920;

  const distStrip = LabLive.strip("live-dist", {
    yMin: 0, yMax: "auto",
    lanes: [
      { key: "d1", color: AGENT[1], lw: 1.3 },
      { key: "d2", color: AGENT[2], lw: 1.3 },
      { key: "d3", color: AGENT[3], lw: 1.3 },
    ],
    hlines: [{ y: 4, color: "rgba(139,149,163,0.7)" }],
    label: "link distance to predecessor · B blue · C green · D yellow",
  });

  function clearAll() {
    for (const tr of trails) tr.clear();
    distStrip.clear();
    attRows.length = 0;
  }

  function drawArena(m) {
    const now = m.now, box = m.config.box_size, r = m.config.agent_radius;
    LabLive.bg(arenaX, arenaC);
    const map = LabLive.boxMap(arenaC, box);
    arenaX.strokeStyle = C.grid;
    arenaX.strokeRect(map.X(0), map.Y(box), box * map.s, box * map.s);

    const agents = [[now.ax, now.ay, now.ah]].concat(
      now.followers.map((f) => [f.x, f.y, f.h]));
    for (let i = 0; i < agents.length; i++) {
      LabLive.drawTrail(arenaX, trails[i].pts, map, AGENT[i], 0.45);
    }

    // gaze lines: follower -> the agent its attention is latched on
    for (let j = 0; j < now.followers.length; j++) {
      const f = now.followers[j];
      const target = agents[f.sel];
      if (!target) continue;
      arenaX.strokeStyle = AGENT[f.sel];
      arenaX.globalAlpha = m.config.mode === "sticky" ? 0.55 : 0.25;
      arenaX.setLineDash([5, 4]);
      arenaX.beginPath();
      arenaX.moveTo(map.X(f.x), map.Y(f.y));
      arenaX.lineTo(map.X(target[0]), map.Y(target[1]));
      arenaX.stroke();
      arenaX.setLineDash([]);
      arenaX.globalAlpha = 1;
    }

    // agent bodies + heading ticks
    for (let i = 0; i < agents.length; i++) {
      const [x, y, h] = agents[i];
      const px = map.X(x), py = map.Y(y);
      arenaX.fillStyle = AGENT[i];
      arenaX.globalAlpha = 0.85;
      arenaX.beginPath();
      arenaX.arc(px, py, Math.max(r * map.s, 4), 0, 2 * Math.PI);
      arenaX.fill();
      arenaX.globalAlpha = 1;
      arenaX.strokeStyle = C.white;
      arenaX.lineWidth = 1.4;
      arenaX.beginPath();
      arenaX.moveTo(px, py);
      arenaX.lineTo(px + 2.4 * Math.max(r * map.s, 4) * Math.cos(h),
                    py - 2.4 * Math.max(r * map.s, 4) * Math.sin(h));
      arenaX.stroke();
      arenaX.lineWidth = 1;
      arenaX.fillStyle = C.text;
      arenaX.font = "10px monospace";
      arenaX.fillText(NAMES[i], px + 7, py - 7);
    }
  }

  function drawAtt(m) {
    LabLive.bg(attX, attC);
    const n = m.now.followers.length;
    const W = attC.width, H = attC.height;
    const rowH = (H - 12) / Math.max(n, 1);
    for (let i = 0; i < attRows.length; i++) {
      const px = W - (attRows.length - i) * (W / ATT_CAP);
      if (px < 14) continue;
      for (let j = 0; j < n; j++) {
        const sel = attRows[i][1 + j];
        if (sel === undefined || sel === null) continue;
        attX.fillStyle = AGENT[sel];
        attX.fillRect(px, 2 + j * rowH, Math.max(W / ATT_CAP, 1), rowH - 3);
      }
    }
    attX.fillStyle = C.dim;
    attX.font = "10px monospace";
    for (let j = 0; j < n; j++) attX.fillText(NAMES[j + 1], 2, 2 + j * rowH + rowH / 2 + 3);
  }

  function onFrame(m, wasReset) {
    if (wasReset) clearAll();
    for (const e of m.series) {
      // trails need per-frame positions only for the pacemaker (not in the
      // series); followers' exact path comes from now-positions per frame too
      distStrip.push({ t: e.t, d1: e.d1, d2: e.d2, d3: e.d3 });
      attRows.push([e.t, e.sel1, e.sel2, e.sel3]);
      if (attRows.length > ATT_CAP) attRows.shift();
    }
    if (m.series.length || wasReset) {
      const now = m.now;
      trails[0].push(now.ax, now.ay);
      now.followers.forEach((f, j) => trails[j + 1].push(f.x, f.y));
    }
    drawArena(m);
    distStrip.draw();
    drawAtt(m);
    const now = m.now;
    document.getElementById("live-readout").textContent =
      `A (${now.ax.toFixed(1)}, ${now.ay.toFixed(1)})` +
      now.followers.map((f, j) =>
        ` · ${NAMES[j + 1]} (${f.x.toFixed(1)}, ${f.y.toFixed(1)})`).join("");
    document.getElementById("live-links").innerHTML = now.followers.map((f, j) => {
      const name = NAMES[j + 1];
      return `<div class="sumstat"><div class="k">${name} → attends <b style="color:${AGENT[f.sel]}">${NAMES[f.sel]}</b>` +
        ` · switches ${f.switches}</div>` +
        `<div class="v mono small">d ${f.d.toFixed(2)} · hits ${f.hits}</div></div>`;
    }).join("");
    document.getElementById("live-note").textContent =
      `${m.config.mode}${m.config.mode === "sticky" ? ` ${m.config.ratio}×/${m.config.patience}` : ""}` +
      ` · ${m.config.links} link${m.config.links > 1 ? "s" : ""} · B seed ${m.seed}`;
  }

  const live = LabLive.connect({
    path: "/lab/ws/ecology",
    onFrame,
    resetParams: () => ({
      links: parseInt(document.getElementById("links").value) || 1,
      mode: document.getElementById("mode").value,
      ratio: parseFloat(document.getElementById("ratio").value) || 5,
      patience: parseInt(document.getElementById("patience").value) || 300,
    }),
  });

  function syncAttInputs() {
    const sticky = document.getElementById("mode").value === "sticky";
    document.getElementById("ratio").disabled = !sticky;
    document.getElementById("patience").disabled = !sticky;
  }
  for (const id of ["links", "mode", "ratio", "patience"]) {
    document.getElementById(id).addEventListener("change", () => {
      syncAttInputs();
      live.reset();
    });
  }
  syncAttInputs();

  function fitAll() {
    distStrip.fit();
    const w = Math.max(300, Math.round(attC.clientWidth) || attC.width);
    if (attC.width !== w) attC.width = w;
  }
  let rT = null;
  window.addEventListener("resize", () => {
    clearTimeout(rT);
    rT = setTimeout(() => { fitAll(); if (live.latest) onFrame(live.latest, false); }, 150);
  });
  fitAll();
})();
