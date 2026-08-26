"""Render a slide-ready video of the Pong experiment.

Light-themed portrait video with the field, score, sensor activations,
effector activations, ball angle, reservoir spikes, and weight distribution.
No controls or titles. The simulation is the tested `homeostasis` package;
this script only records and draws.

Playback runs a two-phase speed schedule: a slow legible intro, then a fast
phase so a full 700+ scoring opportunities fit in a two-minute clip. Because
the fast phase advances many model steps per video frame, the time-domain
panels aggregate within each frame (the raster shows per-frame firing rate,
the angle panel shows the per-frame extreme of |dtheta|), and the ball trail
is drawn at step resolution so the field reads as a long exposure rather
than a strobe.

Usage:
  python scripts/render_pong_video.py --loadout pongEvo1 --seed 13
  python scripts/render_pong_video.py --time-frames 40   (speed probe)
"""

from __future__ import annotations

import argparse
import dataclasses
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.collections import LineCollection

from homeostasis import PONG_RESERVOIR_CONFIG, PongConfig, PongSimulation
from viz.pong_server import LOADOUT_BY_ID, PONG_PARAMS, RESERVOIR_PARAMS

# ---- light theme (matches the tracking renderer) ---------------------------
INK = "#1c2733"
MUTED = "#7a8494"
EDGE = "#d9dee6"
AGENT = "#d62839"        # paddle / miss
BLUE = "#2266cc"         # angle trace / down effector
GREEN = "#0c9c62"        # ball / hit
AMBER = "#c9820a"        # active sensors, |dtheta|
HIST = "#5b8fd6"
PAPER_RATE = 0.582

RASTER_SECONDS = 20.0    # playback seconds of history in the raster
ANGLE_SECONDS = 4.0      # playback seconds of history in the angle panel
TRAIL_STEPS = 450        # model steps of ball path drawn as a trail

plt.rcParams.update({
    "font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"],
    "text.color": INK,
    "axes.edgecolor": EDGE,
    "axes.labelcolor": MUTED,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
})


def frame_plan(cfg):
    """Steps per video frame: slow intro, then fast phase."""
    intro_frames = int(round(cfg.intro_seconds * cfg.fps))
    intro_step = max(1, int(round(cfg.intro_sps / cfg.fps)))
    main_step = max(1, int(round(cfg.main_sps / cfg.fps)))
    plan = np.full(cfg.frames, main_step, dtype=int)
    plan[:intro_frames] = intro_step
    return plan


def simulate(cfg, plan):
    """Run Pong, recording per-frame aggregates plus a step-resolution path."""
    entry = LOADOUT_BY_ID[cfg.loadout]
    params = entry["params"]
    r_cfg = dataclasses.replace(
        PONG_RESERVOIR_CONFIG,
        **{k: v for k, v in params.items() if k in RESERVOIR_PARAMS})
    pong_cfg = PongConfig(**{k: v for k, v in params.items() if k in PONG_PARAMS})
    sim = PongSimulation(r_cfg, pong_cfg, seed=cfg.seed)

    n_frames = len(plan)
    n_nodes = r_cfg.n_nodes
    n_sens = pong_cfg.n_sensors
    total = int(plan.sum())

    hist_bins = np.linspace(-8.0, 8.0, 161)
    rec = {
        "rates": np.zeros((n_nodes, n_frames), dtype=np.float32),
        "sensors": np.zeros((n_frames, n_sens), dtype=np.float32),
        "outputs": np.zeros((n_frames, 2), dtype=np.float32),
        "paddle": np.zeros(n_frames), "ball": np.zeros((n_frames, 2)),
        "vel": np.zeros((n_frames, 2)),
        "angle": np.zeros(n_frames), "dangle": np.zeros(n_frames),
        "hits_done": np.zeros(n_frames, dtype=int),
        "hits_sum": np.zeros(n_frames, dtype=int),
        "event": np.zeros(n_frames, dtype=int),
        "hists": np.zeros((n_frames, len(hist_bins) - 1), dtype=np.int32),
        "path": np.zeros((total, 2), dtype=np.float32),
        "frame_end": np.cumsum(plan),
        "hit_curve": [],       # (opportunity index, running rate)
        "hit_frames": [],      # (frame, +1 hit / -1 miss)
    }

    step = 0
    t0 = time.perf_counter()
    for f, n_steps in enumerate(plan):
        acc = np.zeros(n_nodes)
        dmax = 0.0
        ev = 0
        for _ in range(int(n_steps)):
            before = sim.env.ball_angle()
            state, event, _ = sim.step()
            ang = sim.env.ball_angle()
            acc += state.spiked
            d = abs((ang - before + 180.0) % 360.0 - 180.0)
            dmax = max(dmax, d)
            rec["path"][step] = (sim.env.ball_x, sim.env.ball_y)
            step += 1
            if event == "hit":
                ev = 1
            elif event == "miss":
                ev = -1
        rec["rates"][:, f] = acc / n_steps
        rec["sensors"][f] = sim.env.sense()
        rec["outputs"][f] = state.outputs
        rec["paddle"][f] = sim.env.paddle_y
        rec["ball"][f] = (sim.env.ball_x, sim.env.ball_y)
        rec["vel"][f] = (sim.env.dx, sim.env.dy)
        rec["angle"][f] = sim.env.ball_angle()
        rec["dangle"][f] = dmax
        hits = np.asarray(sim.env.hits, dtype=float)
        rec["hits_done"][f] = hits.size
        rec["hits_sum"][f] = int(hits.sum())
        rec["event"][f] = ev
        if ev:
            rec["hit_frames"].append((f, ev))
        w = sim.network.weights[sim.network.adjacency]
        rec["hists"][f] = np.histogram(w, bins=hist_bins)[0]

    hits = np.asarray(sim.env.hits, dtype=float)
    rec["hit_curve"] = (np.cumsum(hits) / np.arange(1, hits.size + 1)) if hits.size else np.zeros(0)
    rec["rates_g"] = np.sqrt(rec["rates"])   # gamma applied once, not per frame
    rec["hist_bins"] = hist_bins
    rec["n_nodes"] = n_nodes
    rec["sensor_values"] = pong_cfg.sensor_values
    rec["chance"] = pong_cfg.chance_hit_rate
    rec["field"] = (pong_cfg.width, pong_cfg.height, pong_cfg.paddle_x,
                    pong_cfg.paddle_half_height)
    rec["gain"] = pong_cfg.gain
    rec["label"] = entry["label"]
    rec["total_steps"] = total
    rec["final_hits"] = int(hits.sum())
    rec["final_opps"] = int(hits.size)
    print(f"  simulated {total:,} steps in {time.perf_counter()-t0:.0f}s | "
          f"{hits.size} opportunities, hit rate {hits.mean():.3f}", flush=True)
    return rec


def build_figure(rec, cfg):
    fig = plt.figure(figsize=(9.6, 10.8), dpi=cfg.dpi)
    fig.patch.set_facecolor("white")
    W, H, PX, PH = rec["field"]

    def panel(rect, title=None):
        ax = fig.add_axes(rect)
        ax.set_facecolor("white")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        if title:
            fig.text(rect[0], rect[1] + rect[3] + 0.005, title, fontsize=10.5,
                     fontweight="bold", color=INK, va="bottom")
        return ax

    art = {}
    art["clock"] = fig.text(0.968, 0.963, "", fontsize=11, color=MUTED,
                            ha="right", family="monospace")

    # ---- field --------------------------------------------------------------
    ax = panel([0.06, 0.585, 0.88, 0.355], "Field   ·   paddle 100 px in a 500 px field")
    ax.set_xlim(-25, W + 25)
    ax.set_ylim(-15, H + 15)
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.add_patch(plt.Rectangle((0, 0), W, H, fill=False, ec=EDGE, lw=1.0, zorder=1))
    ax.plot([0, 0], [0, H], color=AGENT, lw=1.2, ls=(0, (4, 4)), alpha=0.55, zorder=2)
    art["fan"] = LineCollection([], linewidths=1.0, zorder=3)
    ax.add_collection(art["fan"])
    art["trail"] = LineCollection([], linewidths=1.3, zorder=4)
    ax.add_collection(art["trail"])
    (art["paddle"],) = ax.plot([], [], color=AGENT, lw=5.0, zorder=6,
                               solid_capstyle="butt")
    (art["ball"],) = ax.plot([], [], "o", ms=9, color=GREEN, zorder=7)
    art["flash"] = ax.text(W / 2, H - 18, "", fontsize=13, fontweight="bold",
                           ha="center", va="top", zorder=8)

    # ---- score --------------------------------------------------------------
    ax = panel([0.06, 0.455, 0.40, 0.085], "Score  ·  running hit rate vs. scoring opportunities")
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0, 0.5, 1])
    ax.axhline(rec["chance"], color=MUTED, ls=(0, (3, 3)), lw=1.0)
    ax.text(0.99, rec["chance"] + 0.04, "chance", transform=ax.get_yaxis_transform(),
            fontsize=7, color=MUTED, ha="right")
    (art["score_tr"],) = ax.plot([], [], color=GREEN, lw=1.8, zorder=3)
    art["score_ax"] = ax
    art["score_big"] = ax.text(0.02, 0.94, "", transform=ax.transAxes,
                               fontsize=13, fontweight="bold", color=GREEN,
                               ha="left", va="top", family="monospace")

    # ---- ball angle ---------------------------------------------------------
    ax = panel([0.545, 0.455, 0.395, 0.085],
               f"Ball angle θ  ·  |Δθ| per frame  (last {ANGLE_SECONDS:.0f} s)")
    ax.set_xlim(-ANGLE_SECONDS, 0)
    ax.set_xticks([])
    ax.set_ylim(-190, 190)
    ax.set_yticks([-180, 0, 180])
    art["ang_ev"] = LineCollection([], linewidths=1.6, zorder=4)
    ax.add_collection(art["ang_ev"])
    art["ang_tr"] = ax.scatter([], [], s=2.4, color=BLUE, zorder=3)
    ax_d = ax.twinx()
    ax_d.set_ylim(-0.25, 1.0)
    ax_d.set_yticks([])
    for side in ("top", "left", "right"):
        ax_d.spines[side].set_visible(False)
    (art["dang_tr"],) = ax_d.plot([], [], color=AMBER, lw=1.1, zorder=2)
    art["dang_ax"] = ax_d
    ax.text(0.01, 0.05, "θ", transform=ax.transAxes, fontsize=8.5, color=BLUE)
    ax.text(0.055, 0.05, "|Δθ|", transform=ax.transAxes, fontsize=8.5, color=AMBER)

    # ---- sensors ------------------------------------------------------------
    n_sens = len(rec["sensor_values"])
    ax = panel([0.06, 0.345, 0.40, 0.068], "Sensor activations  ·  binary")
    ax.set_xlim(-0.5, n_sens - 0.5)
    ax.set_ylim(0, 1.35)
    ax.set_xticks([])
    ax.set_yticks([0, 1])
    ax.text(0.0, 0.04, "−90°", transform=ax.transAxes, fontsize=7.5, color=MUTED)
    ax.text(1.0, 0.04, "+90°", transform=ax.transAxes, fontsize=7.5,
            color=MUTED, ha="right")
    art["sens"] = LineCollection([], linewidths=3.4, zorder=2)
    ax.add_collection(art["sens"])
    art["sens_n"] = ax.text(0.99, 0.95, "", transform=ax.transAxes, fontsize=7.5,
                            color=MUTED, ha="right", va="top")

    # ---- effectors ----------------------------------------------------------
    ax = panel([0.545, 0.345, 0.395, 0.068], "Effectors")
    out_max = max(float(rec["outputs"].max()), 0.02)
    ax.set_xlim(0, out_max * 1.06)
    ax.set_ylim(-0.65, 0.65)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axhline(0, color=EDGE, lw=1)
    art["eff_up"] = ax.barh(0.3, 0, height=0.36, color=GREEN, alpha=0.85)[0]
    art["eff_dn"] = ax.barh(-0.3, 0, height=0.36, color=BLUE, alpha=0.85)[0]
    ax.text(-0.015, 0.3, "up", fontsize=9, color=GREEN, ha="right", va="center",
            transform=ax.get_yaxis_transform())
    ax.text(-0.015, -0.3, "down", fontsize=9, color=BLUE, ha="right", va="center",
            transform=ax.get_yaxis_transform())
    ax.text(0.997, 0.04, f"full scale {out_max:.2f}", transform=ax.transAxes,
            fontsize=7.5, color=MUTED, ha="right", va="bottom")

    # ---- raster -------------------------------------------------------------
    raster_frames = int(RASTER_SECONDS * cfg.fps)
    ax = panel([0.06, 0.125, 0.88, 0.175],
               f"Reservoir spikes  ·  rows = {rec['n_nodes']} nodes  ·  "
               f"per-frame firing rate, √ scale  (last {RASTER_SECONDS:.0f} s)")
    ax.set_xticks([])
    ax.set_yticks([])
    art["raster"] = ax.imshow(np.zeros((rec["n_nodes"], raster_frames)),
                              aspect="auto", cmap="Greys", vmin=0, vmax=1.0,
                              extent=(-raster_frames, 0, 0, rec["n_nodes"]),
                              interpolation="nearest", zorder=2)
    art["raster_frames"] = raster_frames

    # ---- weights ------------------------------------------------------------
    ax = panel([0.06, 0.045, 0.88, 0.055], "Weights")
    bins = rec["hist_bins"]
    pooled = rec["hists"].sum(axis=0).astype(float)
    cdf = np.cumsum(pooled) / max(pooled.sum(), 1.0)
    w_lo = float(bins[int(np.searchsorted(cdf, 0.005))])
    w_hi = float(bins[min(int(np.searchsorted(cdf, 0.995)) + 1, len(bins) - 1)])
    pad = 0.06 * max(w_hi - w_lo, 1.0)
    ax.set_xlim(w_lo - pad, w_hi + pad)
    ax.set_yticks([])
    ax.set_xticks(sorted({round(w_lo, 1), 0.0, round(w_hi, 1)}))
    ax.axvline(0, color=EDGE, lw=1, ls=(0, (3, 3)))
    art["hist"] = ax.stairs(np.zeros(len(bins) - 1), bins, fill=True,
                            color=HIST, lw=0)
    art["hist_ax"] = ax
    return fig, art


def make_update(rec, art, cfg):
    W, H, PX, PH = rec["field"]
    sensor_vals = np.asarray(rec["sensor_values"])
    fan_len = W - PX - 20
    raster_frames = art["raster_frames"]
    ang_frames = int(ANGLE_SECONDS * cfg.fps)
    frame_end = rec["frame_end"]
    dang_scale = max(float(rec["dangle"].max()), 1.0)
    TRAIL_PTS = 220
    trail_colors = [(0.05, 0.61, 0.38, al)
                    for al in np.linspace(0.05, 0.55, TRAIL_PTS - 1)]

    def update(f):
        # ---- field ----
        py = rec["paddle"][f]
        bx, by = rec["ball"][f]
        acts = rec["sensors"][f]
        a = np.radians(sensor_vals)
        on = acts > 0
        # Clip each ray where it would leave the field, so the fan stays inside.
        eps = 1e-9
        cos_a, sin_a = np.cos(a), np.sin(a)
        t_x = (W - PX) / np.maximum(cos_a, eps)
        t_y = np.where(sin_a > eps, (H - py) / np.maximum(sin_a, eps),
                       np.where(sin_a < -eps, (0.0 - py) / np.minimum(sin_a, -eps),
                                np.inf))
        r = np.where(on, np.minimum(fan_len, np.minimum(t_x, t_y)), 34.0)
        segs = np.stack([np.column_stack([np.full(len(a), PX), np.full(len(a), py)]),
                         np.column_stack([PX + r * np.cos(a), py + r * np.sin(a)])],
                        axis=1)
        art["fan"].set_segments(list(segs))
        art["fan"].set_colors([AMBER if o else "#e8edf3" for o in on])
        art["fan"].set_linewidths(np.where(on, 1.5, 0.7))

        end = int(frame_end[f])
        start = max(0, end - TRAIL_STEPS)
        path = rec["path"][start:end]
        if len(path) > 2:
            # resample to a fixed point count so the fade palette is reusable
            idx = np.linspace(0, len(path) - 1, TRAIL_PTS).astype(int)
            pts = path[idx]
            tsegs = np.stack([pts[:-1], pts[1:]], axis=1)
            keep = np.abs(tsegs[:, 1, 0] - tsegs[:, 0, 0]) < 400
            art["trail"].set_segments(list(tsegs[keep]))
            art["trail"].set_colors([c for c, k in zip(trail_colors, keep) if k])
        art["paddle"].set_data([PX, PX], [py - PH, py + PH])
        art["ball"].set_data([bx], [by])
        ev = rec["event"][f]
        if ev:
            art["flash"].set_text("HIT" if ev > 0 else "MISS")
            art["flash"].set_color(GREEN if ev > 0 else AGENT)
        else:
            art["flash"].set_text("")

        # ---- score ----
        n_opp = rec["hits_done"][f]
        if n_opp:
            art["score_tr"].set_data(np.arange(1, n_opp + 1), rec["hit_curve"][:n_opp])
            art["score_ax"].set_xlim(0, max(n_opp, 8) * 1.04)
            rate = rec["hits_sum"][f] / n_opp
            art["score_big"].set_text(f"{rate * 100:4.1f}%   {rec['hits_sum'][f]}/{n_opp}")
        else:
            art["score_big"].set_text("—")

        # ---- ball angle ----
        lo = max(0, f - ang_frames + 1)
        xs = (np.arange(lo, f + 1) - f) / cfg.fps
        art["ang_tr"].set_offsets(np.c_[xs, rec["angle"][lo:f + 1]])
        art["dang_tr"].set_data(xs, rec["dangle"][lo:f + 1] / dang_scale)
        evs = [(fr - f) / cfg.fps for fr, _ in rec["hit_frames"] if lo <= fr <= f]
        cols = [GREEN if e > 0 else AGENT
                for fr, e in rec["hit_frames"] if lo <= fr <= f]
        art["ang_ev"].set_segments([[(x, -190), (x, -140)] for x in evs])
        art["ang_ev"].set_colors(cols)

        # ---- sensors ----
        idx = np.arange(len(acts))
        ssegs = np.stack([np.column_stack([idx, np.zeros_like(acts)]),
                          np.column_stack([idx, np.maximum(acts, 0.02)])], axis=1)
        art["sens"].set_segments(list(ssegs))
        art["sens"].set_colors([AMBER if v > 0 else "#e8edf3" for v in acts])
        art["sens_n"].set_text(f"{int(on.sum())} active")

        # ---- effectors ----
        up, dn = rec["outputs"][f]
        art["eff_up"].set_width(up)
        art["eff_dn"].set_width(dn)

        # ---- raster ----
        rlo = max(0, f - raster_frames + 1)
        win = np.zeros((rec["n_nodes"], raster_frames), dtype=np.float32)
        chunk = rec["rates_g"][:, rlo:f + 1]
        win[:, raster_frames - chunk.shape[1]:] = chunk
        art["raster"].set_data(win)

        # ---- weights ----
        counts = rec["hists"][f]
        art["hist"].set_data(counts)
        art["hist_ax"].set_ylim(0, max(counts.max(), 1) * 1.08)

        art["clock"].set_text(f"step {end:,}")
        return []

    return update


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loadout", type=str, default="pongEvo1",
                    choices=sorted(LOADOUT_BY_ID))
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--seconds", type=float, default=120.0)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--intro-seconds", type=float, default=10.0)
    ap.add_argument("--intro-sps", type=float, default=122.0)
    ap.add_argument("--main-sps", type=float, default=2050.0)
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--out", type=str, default="scripts/out/pong_evolved.mp4")
    ap.add_argument("--still", type=int, default=None)
    ap.add_argument("--time-frames", type=int, default=None,
                    help="render this many frames and report the rate, then exit")
    cfg = ap.parse_args()
    cfg.frames = int(round(cfg.seconds * cfg.fps))

    plan = frame_plan(cfg)
    print(f"'{LOADOUT_BY_ID[cfg.loadout]['label']}' seed {cfg.seed}: "
          f"{cfg.frames} frames, {plan.sum():,} steps "
          f"({cfg.intro_sps:.0f}/s for {cfg.intro_seconds:.0f}s, "
          f"then {cfg.main_sps:.0f}/s)", flush=True)
    rec = simulate(cfg, plan)
    fig, art = build_figure(rec, cfg)
    update = make_update(rec, art, cfg)

    out = pathlib.Path(cfg.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    if cfg.still is not None:
        update(cfg.still)
        png = out.with_suffix("").as_posix() + f"_frame{cfg.still}.png"
        fig.savefig(png, dpi=cfg.dpi, facecolor="white")
        print(f"saved {png}", flush=True)
        return

    if cfg.time_frames:
        writer = animation.FFMpegWriter(fps=cfg.fps, bitrate=cfg.bitrate
                                       if hasattr(cfg, "bitrate") else 6000)
        probe = out.with_suffix("").as_posix() + "_probe.mp4"
        t0 = time.perf_counter()
        with writer.saving(fig, probe, cfg.dpi):
            for f in range(cfg.time_frames):
                update(f)
                writer.grab_frame()
        dt = time.perf_counter() - t0
        print(f"{cfg.time_frames} frames in {dt:.1f}s = {dt / cfg.time_frames:.3f} s/frame"
              f"  ->  {cfg.frames} frames ≈ {dt / cfg.time_frames * cfg.frames / 60:.1f} min",
              flush=True)
        return

    print(f"rendering {cfg.frames} frames...", flush=True)
    anim = animation.FuncAnimation(fig, update, frames=cfg.frames, blit=False)
    writer = animation.FFMpegWriter(fps=cfg.fps, bitrate=6000,
                                    extra_args=["-pix_fmt", "yuv420p"])
    anim.save(out.as_posix(), writer=writer, dpi=cfg.dpi,
              progress_callback=lambda i, n: (i % 300 == 0) and print(
                  f"  frame {i}/{n}", flush=True))
    print(f"saved {out}  ({rec['final_hits']}/{rec['final_opps']} opportunities, "
          f"hit rate {rec['final_hits'] / max(rec['final_opps'], 1):.3f})", flush=True)


if __name__ == "__main__":
    main()
