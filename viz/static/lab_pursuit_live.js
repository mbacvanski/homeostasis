/* Live frontend for the pursuit viewer (/lab/ws/pursuit).
 *
 * All simulation happens server-side (viz.lab_server PursuitLive owns one
 * package PursuitSimulation per connection, constructed exactly like the
 * batch endpoint); this file renders frames, keeps rolling trails from the
 * exact per-step series, and maps the stim-speed slider to the live
 * stim_speed command.
 */

"use strict";

(() => {
  const C = LabLive.colors;
  const TEAL = "#4dd0c9";
  const [arenaC, arenaX] = LabLive.cv("live-arena");
  const agentTrail = LabLive.trail(1200);
  const stimTrail = LabLive.trail(1200);   // entries [x, y, jumped]
  let hitFlash = 0;
  let catchGlow = 0;

  const distStrip = LabLive.strip("live-dist", {
    yMin: 0, yMax: "auto",
    lanes: [{ key: "dist", color: C.accent, lw: 1.3 }],
    bands: [{ lo: 0, hi: 1.5, color: "rgba(63,214,143,0.14)" }],
    hlines: [{ y: 4.0, color: "rgba(139,149,163,0.7)" }],
    markKey: "hit", markColor: "rgba(255,93,93,0.6)",
    label: "distance · catch band <1.5 · lock line 4",
  });
  const propStrip = LabLive.strip("live-prop", {
    yMin: 0, yMax: 1,
    lanes: [{ key: "prop", color: C.green, lw: 1.3 }],
    label: "prop. spiked",
  });
  const strips = [distStrip, propStrip];

  function clearAll() {
    agentTrail.clear();
    stimTrail.clear();
    for (const s of strips) s.clear();
    hitFlash = 0;
    catchGlow = 0;
  }

  function drawArena(m) {
    const now = m.now, box = m.config.box_size, r = m.config.agent_radius;
    LabLive.bg(arenaX, arenaC);
    const map = LabLive.boxMap(arenaC, box);
    arenaX.strokeStyle = C.grid;
    arenaX.strokeRect(map.X(0), map.Y(box), box * map.s, box * map.s);

    if (m.config.motion === "orbit" || m.config.motion === "ellipse") {
      // the target's ideal path, faint
      arenaX.strokeStyle = C.orange;
      arenaX.globalAlpha = 0.25;
      arenaX.setLineDash([4, 4]);
      arenaX.beginPath();
      if (m.config.motion === "orbit") {
        arenaX.arc(map.X(box / 2), map.Y(box / 2), m.config.orbit_radius * map.s, 0, 2 * Math.PI);
      } else {
        arenaX.ellipse(map.X(box / 2), map.Y(box / 2),
                       m.config.ellipse_a * map.s, m.config.ellipse_b * map.s,
                       0, 0, 2 * Math.PI);
      }
      arenaX.stroke();
      arenaX.setLineDash([]);
      arenaX.globalAlpha = 1;
    }

    // stimulus trail (orange), broken at ballistic respawn jumps
    const sp = stimTrail.pts;
    arenaX.strokeStyle = C.orange;
    arenaX.globalAlpha = 0.4;
    arenaX.beginPath();
    for (let i = 0; i < sp.length; i++) {
      const px = map.X(sp[i][0]), py = map.Y(sp[i][1]);
      (i && !sp[i][2]) ? arenaX.lineTo(px, py) : arenaX.moveTo(px, py);
    }
    arenaX.stroke();
    arenaX.globalAlpha = 1;

    LabLive.drawTrail(arenaX, agentTrail.pts, map, TEAL);

    // link line agent -> target, green when inside the catch radius
    const ax = map.X(now.x), ay = map.Y(now.y);
    const sx = map.X(now.sx), sy = map.Y(now.sy);
    const caught = now.dist < m.config.catch_r;
    if (caught) catchGlow = 8;
    arenaX.strokeStyle = catchGlow > 0 ? C.green : "rgba(139,149,163,0.5)";
    arenaX.lineWidth = catchGlow > 0 ? 2 : 1;
    arenaX.beginPath();
    arenaX.moveTo(ax, ay);
    arenaX.lineTo(sx, sy);
    arenaX.stroke();
    arenaX.lineWidth = 1;
    if (catchGlow > 0) catchGlow--;

    // target dot
    arenaX.fillStyle = C.orange;
    arenaX.beginPath();
    arenaX.arc(sx, sy, 5, 0, 2 * Math.PI);
    arenaX.fill();

    // agent + heading tick (+ red flash on wall hit)
    if (hitFlash > 0) {
      arenaX.beginPath();
      arenaX.arc(ax, ay, r * map.s + 6, 0, 2 * Math.PI);
      arenaX.strokeStyle = C.red;
      arenaX.lineWidth = 3;
      arenaX.globalAlpha = Math.min(1, hitFlash / 6);
      arenaX.stroke();
      arenaX.globalAlpha = 1;
      arenaX.lineWidth = 1;
      hitFlash--;
    }
    arenaX.fillStyle = "#123734";
    arenaX.strokeStyle = TEAL;
    arenaX.beginPath();
    arenaX.arc(ax, ay, r * map.s, 0, 2 * Math.PI);
    arenaX.fill();
    arenaX.stroke();
    arenaX.strokeStyle = C.yellow;
    arenaX.lineWidth = 1.8;
    arenaX.beginPath();
    arenaX.moveTo(ax, ay);
    arenaX.lineTo(ax + 2.2 * r * map.s * Math.cos(now.heading),
                  ay - 2.2 * r * map.s * Math.sin(now.heading));
    arenaX.stroke();
    arenaX.lineWidth = 1;
  }

  function onFrame(m, wasReset) {
    if (wasReset) clearAll();
    for (const e of m.series) {
      agentTrail.push(e.x, e.y);
      const prev = stimTrail.pts[stimTrail.pts.length - 1];
      const jumped = prev && Math.hypot(e.sx - prev[0], e.sy - prev[1]) > 1.0;
      stimTrail.pts.push([e.sx, e.sy, !!jumped]);
      if (stimTrail.pts.length > 1200) stimTrail.pts.shift();
      if (e.hit) hitFlash = 10;
      distStrip.push({ t: e.t, dist: e.dist, hit: e.hit });
      propStrip.push({ t: e.t, prop: e.prop });
    }
    drawArena(m);
    for (const s of strips) s.draw();
    const now = m.now;
    document.getElementById("lp-dist").textContent = now.dist.toFixed(2);
    document.getElementById("lp-catch").textContent =
      m.config.motion === "ballistic"
        ? `${now.catches} / ${now.n_crossings}`
        : "—";
    document.getElementById("lp-hits").textContent = now.hits;
    document.getElementById("lp-prop").textContent = now.prop.toFixed(3);
    document.getElementById("live-readout").textContent =
      `agent (${now.x.toFixed(2)}, ${now.y.toFixed(2)}) · target (${now.sx.toFixed(2)}, ${now.sy.toFixed(2)})` +
      ` · effectors [${now.outputs[0].toFixed(2)}, ${now.outputs[1].toFixed(2)}]`;
    const motionTag =
      m.config.motion === "ellipse"
        ? `ellipse ${m.config.ellipse_a.toFixed(1)}×${m.config.ellipse_b.toFixed(2)}`
        : m.config.motion === "wander"
          ? `wander σ=${m.config.wander_sigma.toFixed(3)}`
          : m.config.motion;
    document.getElementById("live-note").textContent =
      `${m.config.genome} · ${motionTag} @ ${m.config.speed.toFixed(3)}/step · seed ${m.seed} · N=${m.config.n_nodes}`;
    document.getElementById("live-caption").textContent =
      (typeof CAPTIONS !== "undefined" &&
       CAPTIONS[`${m.config.genome}|${m.config.motion}`]) || "";
  }

  const live = LabLive.connect({
    path: "/lab/ws/pursuit",
    onFrame,
    resetParams: () => ({
      genome: document.getElementById("genome").value,
      motion: document.getElementById("motion").value,
      speed: parseFloat(document.getElementById("stim-speed").value) || 0.15,
      ellipse_ratio: parseFloat(document.getElementById("ellipse-ratio").value) || 1.6,
      wander_sigma: parseFloat(document.getElementById("wander-sigma").value) || 0.05,
    }),
  });

  function syncMotionCtls() {
    const motion = document.getElementById("motion").value;
    document.getElementById("ellipse-ctl").style.display =
      motion === "ellipse" ? "" : "none";
    document.getElementById("wander-ctl").style.display =
      motion === "wander" ? "" : "none";
    document.getElementById("v-wander").textContent =
      (parseFloat(document.getElementById("wander-sigma").value) || 0.05).toFixed(3);
  }
  for (const id of ["genome", "motion", "ellipse-ratio"]) {
    document.getElementById(id).addEventListener("change", () => {
      syncMotionCtls();
      live.reset();
    });
  }
  document.getElementById("wander-sigma").addEventListener("change", () => live.reset());
  document.getElementById("wander-sigma").addEventListener("input", syncMotionCtls);
  syncMotionCtls();
  // stim speed is LIVE: dragging retargets the running stimulus
  document.getElementById("stim-speed").addEventListener("input", (ev) => {
    live.send({ cmd: "stim_speed", v: parseFloat(ev.target.value) || 0.15 });
  });

  function fitAll() { for (const s of strips) s.fit(); }
  let rT = null;
  window.addEventListener("resize", () => {
    clearTimeout(rT);
    rT = setTimeout(() => { fitAll(); if (live.latest) onFrame(live.latest, false); }, 150);
  });
  fitAll();
})();
