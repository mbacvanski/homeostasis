"""Render a quarter-slide gameplay loop: tracking on top, Pong below.

A 480x1080 portrait video (one quarter of a 1920-wide slide) meant to play
silently next to a conclusion slide: no charts, no numbers, no clock — just
the two behaviors. Top: the tracking agent (body, sensor ticks, eyes,
heading arrow) following the green stimulus. Bottom: the Pong agent's
paddle, sensor fan, ball and fading trail, with a brief HIT/MISS flash.

Both simulations are the tested `homeostasis` package (same loadouts as the
full demo videos: tracking `paper` seed 0, Pong `pongEvo1` seed 13); this
script only records and draws. Motion is rendered at 60 fps with sub-step
linear interpolation of positions, so the gameplay is smooth rather than
strobed. There is no on-screen timer, so the clip loops without a visible
jump cue.

Frames are rendered in parallel worker processes (update() is a pure
function of the frame index) and stream-copy concatenated.

Usage: python scripts/render_conclusion_loop.py [--seconds 60] [--still N]
"""

from __future__ import annotations

import argparse
import dataclasses
import multiprocessing
import os
import pathlib
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle, Wedge

from homeostasis import (PONG_RESERVOIR_CONFIG, PongConfig, PongSimulation,
                         ReservoirConfig, VariableTrackingConfig,
                         VariableTrackingSimulation)
from viz.pong_server import LOADOUT_BY_ID as PONG_LOADOUTS
from viz.pong_server import PONG_PARAMS, RESERVOIR_PARAMS as PONG_RES_PARAMS
from viz.server import BASE_TRACKING_PARAMS, LOADOUT_BY_ID, RESERVOIR_PARAMS

INK = "#1c2733"
MUTED = "#7a8494"
EDGE = "#d9dee6"
AGENT = "#d62839"        # left eye / paddle / miss
BLUE = "#2266cc"         # right eye
GREEN = "#0c9c62"        # stimulus / ball / hit
AMBER = "#c9820a"        # pong sensor fan
PINK = "#f2b8c6"         # agent body

TRAIL_STEPS = 450        # model steps of ball path drawn as a fading trail
TRAIL_PTS = 200
FLASH_STEPS = 90         # pong steps a HIT/MISS flash takes to fade

plt.rcParams.update({
    "font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"],
    "text.color": INK,
})


def lerp(arr, tau):
    """Linear interpolation of a per-step series at fractional step tau."""
    i0 = min(int(tau), len(arr) - 1)
    i1 = min(i0 + 1, len(arr) - 1)
    u = tau - i0
    return arr[i0] * (1.0 - u) + arr[i1] * u


# --------------------------------------------------------------------------
# simulate (per-step recordings; drawing interpolates between steps)
# --------------------------------------------------------------------------

def simulate_tracking(seed, n_steps, loadout, warmup=0):
    params = LOADOUT_BY_ID[loadout]["params"]
    r_cfg = ReservoirConfig(**{k: v for k, v in params.items() if k in RESERVOIR_PARAMS})
    t_cfg = VariableTrackingConfig(
        **{k: v for k, v in params.items() if k in BASE_TRACKING_PARAMS})
    sim = VariableTrackingSimulation(r_cfg, t_cfg, seed=seed)
    for _ in range(warmup):     # start the clip with a settled tracker
        sim.step()
    rec = {"heading": np.empty(n_steps), "stim": np.empty(n_steps),
           "sensors": np.empty((n_steps, 62), dtype=np.float32)}
    for t in range(n_steps):
        rec["stim"][t] = sim.env.stimulus_angle
        state, _ = sim.step()
        rec["heading"][t] = sim.env.heading
        rec["sensors"][t] = state.inputs
    # unwrap so interpolation never sweeps the long way around the circle
    for key in ("heading", "stim"):
        rec[key] = np.degrees(np.unwrap(np.radians(rec[key])))
    return rec


def simulate_pong(seed, n_steps, loadout, warmup=0):
    entry = PONG_LOADOUTS[loadout]
    params = entry["params"]
    r_cfg = dataclasses.replace(
        PONG_RESERVOIR_CONFIG,
        **{k: v for k, v in params.items() if k in PONG_RES_PARAMS})
    pong_cfg = PongConfig(**{k: v for k, v in params.items() if k in PONG_PARAMS})
    sim = PongSimulation(r_cfg, pong_cfg, seed=seed)
    for _ in range(warmup):
        sim.step()
    n_sens = pong_cfg.n_sensors
    rec = {"ball": np.empty((n_steps, 2), dtype=np.float32),
           "paddle": np.empty(n_steps, dtype=np.float32),
           "sensors": np.zeros((n_steps, n_sens), dtype=np.float32),
           "event": np.zeros(n_steps, dtype=np.int8)}
    for t in range(n_steps):
        state, event, _ = sim.step()
        rec["ball"][t] = (sim.env.ball_x, sim.env.ball_y)
        rec["paddle"][t] = sim.env.paddle_y
        rec["sensors"][t] = sim.env.sense()
        rec["event"][t] = 1 if event == "hit" else (-1 if event == "miss" else 0)
    hits = np.asarray(sim.env.hits, dtype=float)
    rec["sensor_values"] = np.asarray(pong_cfg.sensor_values)
    rec["field"] = (pong_cfg.width, pong_cfg.height, pong_cfg.paddle_x,
                    pong_cfg.paddle_half_height)
    print(f"  pong: {hits.size} opportunities, hit rate "
          f"{hits.mean():.3f}" if hits.size else "  pong: no opportunities",
          flush=True)
    return rec


# --------------------------------------------------------------------------
# figure
# --------------------------------------------------------------------------

def build_figure(rec_t, rec_p, cfg):
    if cfg.layout == "wide":
        # full slide width, half height: arena and field side by side,
        # centered with balanced margins — 1920x540 at the default dpi
        fig = plt.figure(figsize=(12.8, 3.6), dpi=cfg.dpi)
        rect_a = [0.1010, 0.0222, 0.2542, 0.9037]
        rect_p = [0.3969, 0.0222, 0.5026, 0.9037]
        labels = ((0.1010, 0.945, "tracking"), (0.3969, 0.945, "pong"))
        sz = dict(tick_lw=1.5, arrow_lw=2.0, eye=7, stim=11, fan_on=1.5,
                  fan_off=0.8, trail_lw=1.4, paddle_lw=5.0, ball=9,
                  flash=13, label=10)
    else:
        # quarter-slide portrait: a square arena over the 2:1 field —
        # 480x760 at the default dpi (height follows the content)
        fig = plt.figure(figsize=(3.2, 760 / 150), dpi=cfg.dpi)
        rect_a = [0.03, 0.3724, 0.94, 0.5937]
        rect_p = [0.03, 0.0132, 0.94, 0.3003]
        labels = ((0.045, 0.972, "tracking"), (0.045, 0.322, "pong"))
        sz = dict(tick_lw=1.3, arrow_lw=1.8, eye=5.5, stim=9, fan_on=1.2,
                  fan_off=0.6, trail_lw=1.1, paddle_lw=4.0, ball=7,
                  flash=11, label=9.5)
    fig.patch.set_facecolor("white")
    art = {"fan_lw": (sz["fan_on"], sz["fan_off"])}

    for x, y, txt in labels:
        fig.text(x, y, txt, fontsize=sz["label"], color=MUTED)

    # ---- tracking arena (square) ----
    ax = fig.add_axes(rect_a)
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_aspect("equal")
    ax.axis("off")
    art["fov"] = Wedge((0, 0), 1.18, 0, 180, color="#eef3fa", zorder=0)
    ax.add_patch(art["fov"])
    ax.add_patch(Circle((0, 0), 1.0, fill=False, color=EDGE, lw=1.0, zorder=1))
    ax.add_patch(Circle((0, 0), 0.48, color=PINK, alpha=0.5, zorder=2))
    ax.add_patch(Circle((0, 0), 0.48, fill=False, color="#e2748e", lw=1.1,
                        zorder=3))
    art["ticks"] = LineCollection([], linewidths=sz["tick_lw"], zorder=4)
    ax.add_collection(art["ticks"])
    (art["arrow"],) = ax.plot([], [], color=INK, lw=sz["arrow_lw"], zorder=5,
                              solid_capstyle="round")
    (art["eyeL"],) = ax.plot([], [], "o", ms=sz["eye"], color=AGENT, zorder=6)
    (art["eyeR"],) = ax.plot([], [], "o", ms=sz["eye"], color=BLUE, zorder=6)
    (art["stimdot"],) = ax.plot([], [], "o", ms=sz["stim"], color=GREEN,
                                zorder=7)

    # ---- pong field (2:1) ----
    W, H, PX, PH = rec_p["field"]
    art["pong_geom"] = (W, H, PX, PH)
    ax = fig.add_axes(rect_p)
    ax.set_xlim(-22, W + 22)
    ax.set_ylim(-14, H + 14)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.add_patch(plt.Rectangle((0, 0), W, H, fill=False, ec=EDGE, lw=1.0,
                               zorder=1))
    ax.plot([0, 0], [0, H], color=AGENT, lw=1.0, ls=(0, (4, 4)), alpha=0.55,
            zorder=2)
    art["fan"] = LineCollection([], linewidths=sz["fan_off"], zorder=3)
    ax.add_collection(art["fan"])
    art["trail"] = LineCollection([], linewidths=sz["trail_lw"], zorder=4)
    ax.add_collection(art["trail"])
    (art["paddle"],) = ax.plot([], [], color=AGENT, lw=sz["paddle_lw"],
                               zorder=6, solid_capstyle="butt")
    (art["ball"],) = ax.plot([], [], "o", ms=sz["ball"], color=GREEN,
                             zorder=7)
    art["flash"] = ax.text(W / 2, H - 20, "", fontsize=sz["flash"],
                           fontweight="bold", ha="center", va="top", zorder=8)
    art["trail_colors"] = [(0.05, 0.61, 0.38, al)
                           for al in np.linspace(0.04, 0.5, TRAIL_PTS - 1)]
    return fig, art


# --------------------------------------------------------------------------
# per-frame update (pure function of the frame index)
# --------------------------------------------------------------------------

def make_update(rec_t, rec_p, art, cfg):
    n_t = len(rec_t["heading"])
    n_p = len(rec_p["paddle"])
    W, H, PX, PH = art["pong_geom"]
    sensor_vals = rec_p["sensor_values"]
    a = np.radians(sensor_vals)
    cos_a, sin_a = np.cos(a), np.sin(a)
    fan_len = W - PX - 20
    offs_base = np.concatenate([30 + np.linspace(-60, 60, 31),
                                -30 + np.linspace(-60, 60, 31)])

    def update(f):
        # ---- tracking ----
        tau = min(f * cfg.track_spf, n_t - 1)
        heading = float(lerp(rec_t["heading"], tau)) % 360.0
        stim = float(lerp(rec_t["stim"], tau)) % 360.0
        acts = lerp(rec_t["sensors"], tau)
        art["fov"].set_theta1(heading - 90)
        art["fov"].set_theta2(heading + 90)
        ang = np.radians(offs_base + heading)
        r0, seg_base, seg_len = 0.52, 0.03, 0.30
        r1 = r0 + seg_base + seg_len * acts
        segs = np.stack([np.column_stack([r0 * np.cos(ang), r0 * np.sin(ang)]),
                         np.column_stack([r1 * np.cos(ang), r1 * np.sin(ang)])],
                        axis=1)
        art["ticks"].set_segments(list(segs))
        art["ticks"].set_colors(
            [(AGENT if i < 31 else BLUE) if acts[i] > 0.02 else EDGE
             for i in range(62)])
        h = np.radians(heading)
        art["arrow"].set_data([0, 0.46 * np.cos(h)], [0, 0.46 * np.sin(h)])
        for key, off in (("eyeL", 30), ("eyeR", -30)):
            e = np.radians(heading + off)
            art[key].set_data([0.48 * np.cos(e)], [0.48 * np.sin(e)])
        s = np.radians(stim)
        art["stimdot"].set_data([np.cos(s)], [np.sin(s)])

        # ---- pong ----
        tau = min(f * cfg.pong_spf, n_p - 1)
        i0 = int(tau)
        bx, by = rec_p["ball"][i0]
        nxt = rec_p["ball"][min(i0 + 1, n_p - 1)]
        if abs(float(nxt[0]) - float(bx)) < 100:   # don't interpolate resets
            bx, by = lerp(rec_p["ball"], tau)
        py = float(lerp(rec_p["paddle"], tau))
        acts = rec_p["sensors"][i0]
        on = acts > 0
        eps = 1e-9
        t_x = (W - PX) / np.maximum(cos_a, eps)
        t_y = np.where(sin_a > eps, (H - py) / np.maximum(sin_a, eps),
                       np.where(sin_a < -eps,
                                (0.0 - py) / np.minimum(sin_a, -eps), np.inf))
        r = np.where(on, np.minimum(fan_len, np.minimum(t_x, t_y)), 30.0)
        segs = np.stack(
            [np.column_stack([np.full(len(a), PX), np.full(len(a), py)]),
             np.column_stack([PX + r * cos_a, py + r * sin_a])], axis=1)
        art["fan"].set_segments(list(segs))
        art["fan"].set_colors([AMBER if o else "#e8edf3" for o in on])
        lw_on, lw_off = art["fan_lw"]
        art["fan"].set_linewidths(np.where(on, lw_on, lw_off))

        start = max(0, i0 - TRAIL_STEPS)
        path = rec_p["ball"][start:i0 + 1]
        if len(path) > 2:
            idx = np.linspace(0, len(path) - 1, TRAIL_PTS).astype(int)
            pts = path[idx]
            tsegs = np.stack([pts[:-1], pts[1:]], axis=1)
            keep = np.abs(tsegs[:, 1, 0] - tsegs[:, 0, 0]) < 400
            art["trail"].set_segments(list(tsegs[keep]))
            art["trail"].set_colors(
                [c for c, k in zip(art["trail_colors"], keep) if k])
        art["paddle"].set_data([PX, PX], [py - PH, py + PH])
        art["ball"].set_data([bx], [by])

        # latest hit/miss within the fade window, alpha by age
        w0 = max(0, i0 - FLASH_STEPS)
        recent = np.flatnonzero(rec_p["event"][w0:i0 + 1])
        if recent.size:
            j = w0 + recent[-1]
            ev = rec_p["event"][j]
            age = (i0 - j) / FLASH_STEPS
            art["flash"].set_text("HIT" if ev > 0 else "MISS")
            art["flash"].set_color(GREEN if ev > 0 else AGENT)
            art["flash"].set_alpha(1.0 - age)
        else:
            art["flash"].set_text("")
        return []

    return update


# --------------------------------------------------------------------------

def _render_chunk(payload):
    """Render frames [a, b) into an mp4 part — runs in a worker process."""
    rec_t, rec_p, cfg_d, a, b, part_path = payload
    cfg = argparse.Namespace(**cfg_d)
    fig, art = build_figure(rec_t, rec_p, cfg)
    update = make_update(rec_t, rec_p, art, cfg)
    writer = animation.FFMpegWriter(fps=cfg.fps, bitrate=4000,
                                    extra_args=["-pix_fmt", "yuv420p"])
    with writer.saving(fig, part_path, dpi=cfg.dpi):
        for f in range(a, b):
            update(f)
            writer.grab_frame()
    plt.close(fig)
    return part_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--fps", type=int, default=60)
    ap.add_argument("--track-seed", type=int, default=0)
    ap.add_argument("--track-loadout", type=str, default="paper",
                    choices=sorted(LOADOUT_BY_ID))
    ap.add_argument("--track-spf", type=float, default=3.0,
                    help="tracking model steps per video frame")
    ap.add_argument("--pong-seed", type=int, default=13)
    ap.add_argument("--pong-loadout", type=str, default="pongEvo1",
                    choices=sorted(PONG_LOADOUTS))
    ap.add_argument("--pong-spf", type=float, default=2.0,
                    help="pong model steps per video frame")
    ap.add_argument("--track-warmup", type=int, default=2000,
                    help="tracking steps run before recording starts")
    ap.add_argument("--pong-warmup", type=int, default=1000)
    ap.add_argument("--layout", type=str, default="tall",
                    choices=("tall", "wide"),
                    help="tall = quarter-slide portrait (480x760); "
                         "wide = full-width half-height banner (1920x540)")
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--out", type=str, default="scripts/out/conclusion_loop.mp4")
    ap.add_argument("--still", type=int, default=None)
    ap.add_argument("--jobs", type=int,
                    default=max(1, (os.cpu_count() or 4) - 2))
    cfg = ap.parse_args()
    n_frames = int(round(cfg.seconds * cfg.fps))

    n_t = int(n_frames * cfg.track_spf) + 2
    n_p = int(n_frames * cfg.pong_spf) + 2
    print(f"simulating tracking '{cfg.track_loadout}' seed {cfg.track_seed} "
          f"({n_t} steps) + pong '{cfg.pong_loadout}' seed {cfg.pong_seed} "
          f"({n_p} steps)...", flush=True)
    rec_t = simulate_tracking(cfg.track_seed, n_t, cfg.track_loadout,
                              cfg.track_warmup)
    rec_p = simulate_pong(cfg.pong_seed, n_p, cfg.pong_loadout,
                          cfg.pong_warmup)

    out = pathlib.Path(cfg.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if cfg.still is not None:
        fig, art = build_figure(rec_t, rec_p, cfg)
        update = make_update(rec_t, rec_p, art, cfg)
        update(cfg.still)
        png = out.with_suffix("").as_posix() + f"_frame{cfg.still}.png"
        fig.savefig(png, dpi=cfg.dpi, facecolor="white")
        print(f"saved {png}", flush=True)
        return

    t0 = time.time()
    bounds = np.linspace(0, n_frames, max(cfg.jobs, 1) + 1).astype(int)
    parts = [out.with_name(f"{out.stem}_part{k:02d}.mp4")
             for k in range(max(cfg.jobs, 1))]
    payloads = [(rec_t, rec_p, vars(cfg), int(a), int(b), p.as_posix())
                for a, b, p in zip(bounds, bounds[1:], parts) if b > a]
    print(f"rendering {n_frames} frames ({cfg.seconds:.0f}s at {cfg.fps} fps, "
          f"480x1080) on {len(payloads)} workers...", flush=True)
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=max(cfg.jobs, 1),
                             mp_context=ctx) as ex:
        for done, path in enumerate(ex.map(_render_chunk, payloads), 1):
            print(f"  part {done}/{len(payloads)} done", flush=True)
    concat = out.with_name(f"{out.stem}_parts.txt")
    concat.write_text("".join(f"file '{p.resolve()}'\n"
                              for p in parts if p.exists()))
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe",
                    "0", "-i", concat.as_posix(), "-c", "copy",
                    out.as_posix()], check=True)
    for p in parts:
        p.unlink(missing_ok=True)
    concat.unlink()
    print(f"saved {out} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
