/* Shared transport + controls + chart helpers for the LIVE lab pages.
 *
 * Every live lab page speaks the same WebSocket protocol as the main
 * tracking viewer (play / pause / step{n} / speed{sps} / reset{seed,...}),
 * so this file wires the same transport controls everywhere: #btn-play,
 * #btn-step1/#btn-step10/#btn-step100, #speed + #speed-val, #seed and
 * #btn-reset (each optional — only wired if present), plus the space / s / S
 * keyboard shortcuts. World rendering stays in the per-page *_live.js files;
 * this file renders nothing by itself.
 */

"use strict";

const LabLive = (() => {
  const C = {
    panel: "#10141a", grid: "#2a313c", text: "#d7dde6", dim: "#8b95a3",
    red: "#ff5d5d", blue: "#5da9ff", green: "#3fd68f", yellow: "#ffd23f",
    pink: "#ff9ecb", accent: "#58a6ff", orange: "#ffa04d", white: "#e8edf4",
    gray: "#39424f",
  };

  const el = (id) => document.getElementById(id);
  const cv = (id) => { const c = el(id); return [c, c.getContext("2d")]; };
  const bg = (x, c) => { x.fillStyle = C.panel; x.fillRect(0, 0, c.width, c.height); };
  const fmtv = (v) => Math.abs(v) >= 100 ? v.toFixed(0) : Math.abs(v) >= 1 ? v.toFixed(1) : v.toFixed(2);

  // same log mapping as the main viewer's speed slider: 0..100 -> 1..2000
  const sliderToSps = (v) => Math.round(Math.pow(10, (v / 100) * 3.3));

  // ---------- transport ------------------------------------------------------

  function connect(opts) {
    // opts: { path, onFrame(msg, wasReset), resetParams() -> extra reset
    //         fields, onOpen(live) }
    const live = {
      latest: null,
      ws: null,
      send(obj) { if (live.ws && live.ws.readyState === 1) live.ws.send(JSON.stringify(obj)); },
      reset(extra) {
        const seedEl = el("seed");
        live.send({
          cmd: "reset",
          ...(seedEl ? { seed: Math.max(parseInt(seedEl.value) || 0, 0) } : {}),
          ...(opts.resetParams ? opts.resetParams() : {}),
          ...(extra || {}),
        });
      },
    };
    const connEl = el("conn");

    function open() {
      const ws = new WebSocket(`ws://${location.host}${opts.path}`);
      live.ws = ws;
      ws.onopen = () => {
        if (connEl) { connEl.textContent = "live"; connEl.className = "conn ok"; }
        pushSpeed();               // server speed follows the slider
        if (opts.onOpen) opts.onOpen(live);
      };
      ws.onclose = () => {
        if (connEl) { connEl.textContent = "live disconnected — retrying…"; connEl.className = "conn bad"; }
        setTimeout(open, 1000);
      };
      ws.onmessage = (ev) => {
        const msg = JSON.parse(ev.data);
        if (msg.type !== "frame") return;
        const wasReset = live.latest !== null && msg.t < live.latest.t;
        live.latest = msg;
        const play = el("btn-play");
        if (play) {
          play.textContent = msg.playing ? "⏸ Pause" : "▶ Play";
          play.classList.toggle("primary", msg.playing);
        }
        const tEl = el("live-t");
        if (tEl) tEl.textContent = msg.t;
        opts.onFrame(msg, wasReset);
      };
    }

    const on = (id, fn) => { const e = el(id); if (e) e.onclick = fn; };
    on("btn-play", () => live.send({ cmd: live.latest && live.latest.playing ? "pause" : "play" }));
    on("btn-step1", () => live.send({ cmd: "step", n: 1 }));
    on("btn-step10", () => live.send({ cmd: "step", n: 10 }));
    on("btn-step100", () => live.send({ cmd: "step", n: 100 }));
    on("btn-reset", () => live.reset());

    const speedEl = el("speed");
    function pushSpeed() {
      if (!speedEl) return;
      const sps = sliderToSps(Number(speedEl.value));
      const label = el("speed-val");
      if (label) label.textContent = `${sps}/s`;
      live.send({ cmd: "speed", sps });
    }
    if (speedEl) {
      speedEl.oninput = pushSpeed;
      const label = el("speed-val");
      if (label) label.textContent = `${sliderToSps(Number(speedEl.value))}/s`;
    }

    document.addEventListener("keydown", (ev) => {
      if (["INPUT", "SELECT", "TEXTAREA"].includes(ev.target.tagName)) return;
      if (ev.code === "Space") { ev.preventDefault(); const b = el("btn-play"); if (b) b.click(); }
      if (ev.key === "s") { const b = el("btn-step1"); if (b) b.click(); }
      if (ev.key === "S") { const b = el("btn-step10"); if (b) b.click(); }
    });

    open();
    return live;
  }

  // ---------- rolling strip chart --------------------------------------------
  // One entry per model step (exact, no subsampling), rolling window of `cap`
  // steps drawn right-aligned — the main viewer's strip-chart idiom.

  function strip(canvasId, cfg) {
    // cfg: { cap=1920, lanes: [{key, color, lw, dash, dots}], yMin, yMax
    //        (number or "auto"), bands: [{lo, hi, color}], hlines: [{y,
    //        color, dash}], markKey, markColor, label }
    const c = el(canvasId), x = c.getContext("2d");
    const cap = cfg.cap || 1920;
    let rows = [];
    let events = [];  // [{t, op}] vertical dashed marks by step index

    function push(e) { rows.push(e); if (rows.length > cap) rows.shift(); }
    function clear() { rows = []; events = []; }
    function setEvents(evts) { events = evts || []; }
    function fit() {
      const w = Math.max(300, Math.round(c.clientWidth) || c.width);
      if (c.width !== w) c.width = w;
    }

    function draw() {
      bg(x, c);
      const W = c.width, H = c.height, len = rows.length;
      let lo = cfg.yMin, hi = cfg.yMax;
      if (typeof lo !== "number" || typeof hi !== "number") {
        let l = Infinity, h = -Infinity;
        for (const r of rows) for (const ln of cfg.lanes) {
          const v = r[ln.key];
          if (v === undefined || v === null) continue;
          if (v < l) l = v;
          if (v > h) h = v;
        }
        if (!isFinite(l)) { l = 0; h = 1; }
        if (l === h) { l -= 0.5; h += 0.5; }
        const pad = (h - l) * 0.06;
        if (typeof lo !== "number") lo = l - pad;
        if (typeof hi !== "number") hi = h + pad;
      }
      const yFor = (v) => H - 12 - ((v - lo) / (hi - lo)) * (H - 24);
      const xFor = (i) => W - (len - i) * (W / cap);
      for (const b of cfg.bands || []) {
        x.fillStyle = b.color;
        x.fillRect(0, yFor(b.hi), W, yFor(b.lo) - yFor(b.hi));
      }
      for (const hl of cfg.hlines || []) {
        x.strokeStyle = hl.color || C.grid;
        x.setLineDash(hl.dash || [4, 3]);
        x.beginPath(); x.moveTo(0, yFor(hl.y)); x.lineTo(W, yFor(hl.y)); x.stroke();
        x.setLineDash([]);
      }
      if (cfg.markKey) {
        x.fillStyle = cfg.markColor || C.red;
        for (let i = 0; i < len; i++) {
          if (rows[i][cfg.markKey]) x.fillRect(xFor(i), 2, 1, H - 4);
        }
      }
      for (const ev of events) {  // map step t -> index (rows carry .t)
        if (!len || ev.t < rows[0].t || ev.t > rows[len - 1].t) continue;
        const i = ev.t - rows[0].t;  // one entry per step
        x.strokeStyle = C.pink;
        x.setLineDash([4, 4]);
        x.beginPath(); x.moveTo(xFor(i), 0); x.lineTo(xFor(i), H); x.stroke();
        x.setLineDash([]);
        if (ev.op) {
          x.fillStyle = C.pink; x.font = "9px monospace";
          x.fillText(ev.op, Math.min(xFor(i) + 3, W - 40), 9);
        }
      }
      for (const ln of cfg.lanes) {
        if (ln.dots) {
          x.fillStyle = ln.color;
          for (let i = 0; i < len; i++) {
            const v = rows[i][ln.key];
            if (v === undefined || v === null) continue;
            const px = xFor(i);
            if (px < 0) continue;
            x.fillRect(px, yFor(v), ln.lw || 1.5, (ln.lw || 1.5) + 0.5);
          }
          continue;
        }
        x.strokeStyle = ln.color;
        x.lineWidth = ln.lw || 1.3;
        if (ln.dash) x.setLineDash(ln.dash);
        x.beginPath();
        let started = false;
        for (let i = 0; i < len; i++) {
          const v = rows[i][ln.key];
          if (v === undefined || v === null) { started = false; continue; }
          const px = xFor(i);
          if (px < 0) continue;
          const py = yFor(v);
          if (!started) { x.moveTo(px, py); started = true; } else x.lineTo(px, py);
        }
        x.stroke();
        x.setLineDash([]);
        x.lineWidth = 1;
      }
      x.fillStyle = C.dim; x.font = "10px monospace";
      x.fillText(fmtv(hi), 2, 10);
      x.fillText(fmtv(lo), 2, H - 3);
      if (cfg.label) {
        x.textAlign = "right"; x.fillText(cfg.label, W - 4, 10); x.textAlign = "left";
      }
    }

    return { push, clear, draw, fit, setEvents, canvas: c,
             get length() { return rows.length; } };
  }

  // ---------- small world-drawing helpers ------------------------------------

  function boxMap(canvas, box, pad = 14) {
    const s = (Math.min(canvas.width, canvas.height) - 2 * pad) / box;
    return {
      X: (u) => pad + u * s,
      Y: (v) => canvas.height - pad - v * s,  // world y grows upward
      s, pad,
    };
  }

  function trail(cap) {
    const pts = [];
    return {
      pts,
      push(x, y) { pts.push([x, y]); if (pts.length > cap) pts.shift(); },
      clear() { pts.length = 0; },
    };
  }

  function drawTrail(x, pts, map, color, maxAlpha = 0.55) {
    for (let i = 1; i < pts.length; i++) {
      x.strokeStyle = color;
      x.globalAlpha = 0.04 + maxAlpha * (i / pts.length);
      x.beginPath();
      x.moveTo(map.X(pts[i - 1][0]), map.Y(pts[i - 1][1]));
      x.lineTo(map.X(pts[i][0]), map.Y(pts[i][1]));
      x.stroke();
    }
    x.globalAlpha = 1;
  }

  return { connect, strip, boxMap, trail, drawTrail, colors: C, cv, bg, fmtv, sliderToSps };
})();
