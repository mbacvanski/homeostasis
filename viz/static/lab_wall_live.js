/* Live frontend for the wall-avoidance viewer (/lab/ws/wall).
 *
 * All simulation happens server-side (viz.lab_server WallLive owns one
 * package WallSimulation per connection); this file renders frames, keeps a
 * rolling trail from the exact per-step series, and maps clicks to the
 * teleport command (the H28c displacement probe by hand).
 */

"use strict";

(() => {
  const C = LabLive.colors;
  const [arenaC, arenaX] = LabLive.cv("live-arena");
  const trail = LabLive.trail(1400);
  let hitFlash = 0;      // frames left of the red hit flash
  const hitQueue = [];   // rolling per-step hit flags for the 720-window count
  let hitWindow = 0;

  const sensStrip = LabLive.strip("live-sens", {
    yMin: 0, yMax: "auto",
    lanes: [
      { key: "sl", color: C.red, lw: 1.3 },
      { key: "sr", color: C.blue, lw: 1.3 },
    ],
    label: "sensors (as fed to the network)",
  });
  const propStrip = LabLive.strip("live-prop", {
    yMin: 0, yMax: 1,
    lanes: [{ key: "prop", color: C.green, lw: 1.3 }],
    markKey: "hit", markColor: "rgba(255,93,93,0.6)",
    label: "prop. spiked",
  });
  const hwStrip = LabLive.strip("live-hw", {
    yMin: 0, yMax: "auto",
    lanes: [{ key: "hw", color: C.yellow, lw: 1.5 }],
    label: "hits / last 720 steps",
  });
  const strips = [sensStrip, propStrip, hwStrip];

  function clearAll() {
    trail.clear();
    for (const s of strips) s.clear();
    hitQueue.length = 0;
    hitWindow = 0;
    hitFlash = 0;
  }

  function drawArena(m) {
    const now = m.now, box = m.config.box_size, r = m.config.agent_radius;
    LabLive.bg(arenaX, arenaC);
    const map = LabLive.boxMap(arenaC, box);
    arenaX.strokeStyle = C.grid;
    arenaX.strokeRect(map.X(0), map.Y(box), box * map.s, box * map.s);

    LabLive.drawTrail(arenaX, trail.pts, map, C.accent);

    // sensor rays from the values the network saw (post-perturbation):
    // ray length = (1 - value) x box diagonal, clipped to the box
    const hx = map.X(now.x), hy = map.Y(now.y), hr = now.heading;
    arenaX.save();
    arenaX.beginPath();
    arenaX.rect(map.X(0), map.Y(box), box * map.s, box * map.s);
    arenaX.clip();
    const diag = Math.SQRT2 * box;
    const rays = [[45, now.sensors[0], C.red], [-45, now.sensors[1], C.blue]];
    for (const [off, val, color] of rays) {
      const a = hr + (off * Math.PI) / 180;
      const ox = now.x + r * Math.cos(a), oy = now.y + r * Math.sin(a);
      const d = Math.max(0, Math.min(1 - val, 1)) * diag;
      arenaX.strokeStyle = color;
      arenaX.globalAlpha = 0.35 + 0.65 * Math.min(Math.max(val, 0), 1);
      arenaX.lineWidth = 1.6;
      arenaX.beginPath();
      arenaX.moveTo(map.X(ox), map.Y(oy));
      arenaX.lineTo(map.X(ox + d * Math.cos(a)), map.Y(oy + d * Math.sin(a)));
      arenaX.stroke();
    }
    arenaX.restore();
    arenaX.globalAlpha = 1;
    arenaX.lineWidth = 1;

    // agent body + heading tick (+ red flash on hit)
    if (hitFlash > 0) {
      arenaX.beginPath();
      arenaX.arc(hx, hy, r * map.s + 6, 0, 2 * Math.PI);
      arenaX.strokeStyle = C.red;
      arenaX.lineWidth = 3;
      arenaX.globalAlpha = Math.min(1, hitFlash / 6);
      arenaX.stroke();
      arenaX.globalAlpha = 1;
      arenaX.lineWidth = 1;
      hitFlash--;
    }
    arenaX.fillStyle = "#22303f";
    arenaX.strokeStyle = C.text;
    arenaX.beginPath();
    arenaX.arc(hx, hy, r * map.s, 0, 2 * Math.PI);
    arenaX.fill();
    arenaX.stroke();
    arenaX.strokeStyle = C.yellow;
    arenaX.lineWidth = 1.8;
    arenaX.beginPath();
    arenaX.moveTo(hx, hy);
    arenaX.lineTo(hx + 2.2 * r * map.s * Math.cos(hr), hy - 2.2 * r * map.s * Math.sin(hr));
    arenaX.stroke();
    arenaX.lineWidth = 1;
  }

  function onFrame(m, wasReset) {
    if (wasReset) clearAll();
    for (const e of m.series) {
      trail.push(e.x, e.y);
      if (e.hit) hitFlash = 10;
      hitQueue.push(e.hit);
      hitWindow += e.hit;
      if (hitQueue.length > 720) hitWindow -= hitQueue.shift();
      propStrip.push({ t: e.t, prop: e.prop, hit: e.hit });
      hwStrip.push({ t: e.t, hw: hitWindow });
      sensStrip.push({ t: e.t, sl: e.sl, sr: e.sr });
    }
    drawArena(m);
    for (const s of strips) s.draw();
    const now = m.now;
    document.getElementById("live-readout").textContent =
      `x ${now.x.toFixed(2)} · y ${now.y.toFixed(2)} · heading ${((((now.heading * 180) / Math.PI) % 360) + 360) % 360 | 0}° · ` +
      `sensors [${now.sensors[0].toFixed(3)}, ${now.sensors[1].toFixed(3)}] · ` +
      `effectors [${now.outputs[0].toFixed(2)}, ${now.outputs[1].toFixed(2)}]`;
    document.getElementById("lw-hits").textContent = now.hits;
    document.getElementById("lw-hw").textContent = hitWindow;
    document.getElementById("lw-prop").textContent = now.prop.toFixed(3);
    const learn = document.getElementById("live-learning");
    if (document.activeElement !== learn) learn.checked = m.config.learning;
    document.getElementById("live-note").textContent =
      `${m.config.variant} · seed ${m.seed} · wlr ${m.config.wlr} tlr ${m.config.tlr}` +
      `${m.config.learning ? "" : " · learning OFF"}`;
  }

  const wallVariant = () => {
    const v = document.getElementById("variant").value;
    return v === "flank-champion" ? null : v;
  };
  let lastWallVariant = "base";

  const live = LabLive.connect({
    path: "/lab/ws/wall",
    onFrame,
    resetParams: () => ({
      variant: wallVariant() || lastWallVariant,
      wlr: parseFloat(document.getElementById("wlr").value) || 1.0,
      tlr: parseFloat(document.getElementById("tlr").value) || 0.01,
    }),
  });

  // click / drag anywhere in the arena teleports the agent (H28c by hand)
  let dragging = false;
  function teleport(ev) {
    if (!live.latest) return;
    const box = live.latest.config.box_size;
    const rect = arenaC.getBoundingClientRect();
    const px = ((ev.clientX - rect.left) / rect.width) * arenaC.width;
    const py = ((ev.clientY - rect.top) / rect.height) * arenaC.height;
    const map = LabLive.boxMap(arenaC, box);
    live.send({
      cmd: "teleport",
      x: (px - map.pad) / map.s,
      y: (arenaC.height - map.pad - py) / map.s,
    });
  }
  arenaC.addEventListener("pointerdown", (ev) => {
    dragging = true;
    arenaC.setPointerCapture(ev.pointerId);
    teleport(ev);
  });
  arenaC.addEventListener("pointermove", (ev) => { if (dragging) teleport(ev); });
  arenaC.addEventListener("pointerup", () => { dragging = false; });

  document.getElementById("live-learning").onchange = (ev) =>
    live.send({ cmd: "learning", enabled: ev.target.checked });

  document.getElementById("variant").addEventListener("change", () => {
    const v = wallVariant();
    const note = document.getElementById("live-note");
    document.getElementById("btn-reset").disabled = v === null;
    if (v === null) {
      note.textContent = "flank-champion is a tracking run — open the batch section below";
      return;
    }
    lastWallVariant = v;
    live.reset();
  });

  function fitAll() { for (const s of strips) s.fit(); }
  let rT = null;
  window.addEventListener("resize", () => {
    clearTimeout(rT);
    rT = setTimeout(() => { fitAll(); if (live.latest) onFrame(live.latest, false); }, 150);
  });
  fitAll();
})();
