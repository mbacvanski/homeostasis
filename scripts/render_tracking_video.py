"""Render a slide-ready video of the tracking experiment (irregular motion).

Reproduces the web visualizer's display panels — arena, sensor activations,
effectors, heading vs. stimulus, heading error, spike raster, weight
distribution — as a light-themed portrait video with no controls, titles, or
inspector panels. The simulation is the tested `homeostasis` package; this
script only records and draws.

Usage: python scripts/render_tracking_video.py [--seed 0] [--steps 7200]
       [--steps-per-frame 6] [--out scripts/out/tracking_irregular.mp4]
       [--still N]   (render only frame N to a PNG, for layout checks)
"""

from __future__ import annotations

import argparse
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle, Wedge

from homeostasis import VariableTrackingSimulation

# ---- light theme -----------------------------------------------------------
INK = "#1c2733"          # primary text / marks
MUTED = "#7a8494"        # secondary text
EDGE = "#d9dee6"         # panel hairlines
STIM = "#1c2733"         # stimulus trace (near-black, like Fig. 4A)
AGENT = "#d62839"        # agent trace / left eye
BLUE = "#2266cc"         # right eye / error line
GREEN = "#0c9c62"        # stimulus dot
PINK = "#f2b8c6"         # agent body
HIST = "#5b8fd6"         # weight histogram bars
BAND = "#0c9c62"         # +/-45 degree band shading

WINDOW = 1440            # steps shown in every time panel

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


def simulate(seed: int, n_steps: int, frame_stride: int):
    """Run the irregular-motion tracking model, recording what the panels need."""
    sim = VariableTrackingSimulation(seed=seed)
    n = sim.network.config.n_nodes
    rec = {
        "heading": np.empty(n_steps), "stim": np.empty(n_steps),
        "err": np.empty(n_steps), "dh": np.empty(n_steps),
        "sensors": np.empty((n_steps, 62), dtype=np.float32),
        "outputs": np.empty((n_steps, 2), dtype=np.float32),
        "spikes": np.zeros((n_steps, n), dtype=bool),
    }
    hist_bins = np.linspace(-2.0, 3.0, 51)
    hists = []
    flips = []
    prev_dir = sim.env.stimulus_direction
    for t in range(n_steps):
        rec["stim"][t] = sim.env.stimulus_angle
        rec["err"][t] = sim.env.heading_error()
        state, dh = sim.step()
        rec["heading"][t] = sim.env.heading
        rec["dh"][t] = dh
        rec["sensors"][t] = state.inputs
        rec["outputs"][t] = state.outputs
        rec["spikes"][t] = state.spiked
        if sim.env.stimulus_direction != prev_dir:
            flips.append(t)
            prev_dir = sim.env.stimulus_direction
        if t % frame_stride == 0:
            w = sim.network.weights[sim.network.adjacency]
            hists.append(np.histogram(w, bins=hist_bins)[0])
    rec["hists"] = np.array(hists)
    rec["hist_bins"] = hist_bins
    rec["flips"] = np.array(flips)
    rec["n_nodes"] = n
    return rec


def build_figure(rec, cfg):
    fig = plt.figure(figsize=(9.6, 10.8), dpi=cfg.dpi)
    fig.patch.set_facecolor("white")

    def panel(rect, title=None, title_pad=0.005):
        ax = fig.add_axes(rect)
        ax.set_facecolor("white")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        if title:
            fig.text(rect[0], rect[1] + rect[3] + title_pad, title,
                     fontsize=10.5, fontweight="bold", color=INK, va="bottom")
        return ax

    art = {}
    art["clock"] = fig.text(0.968, 0.964, "t = 0", fontsize=11, color=MUTED,
                            ha="right", family="monospace")

    # ---- arena (top left, square) ----
    ax = panel([0.035, 0.655, 0.33, 0.33 * (9.6 / 10.8)], "Agent & stimulus")
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_aspect("equal")
    ax.axis("off")
    art["fov"] = Wedge((0, 0), 1.18, 0, 180, color="#eef3fa", zorder=0)
    ax.add_patch(art["fov"])
    ax.add_patch(Circle((0, 0), 1.0, fill=False, color=EDGE, lw=1.0, zorder=1))
    ax.add_patch(Circle((0, 0), 0.48, color=PINK, alpha=0.5, zorder=2))
    ax.add_patch(Circle((0, 0), 0.48, fill=False, color="#e2748e", lw=1.2, zorder=3))
    art["ticks"] = LineCollection([], linewidths=1.7, zorder=4)
    ax.add_collection(art["ticks"])
    (art["arrow"],) = ax.plot([], [], color=INK, lw=2.1, zorder=5,
                              solid_capstyle="round")
    (art["eyeL"],) = ax.plot([], [], "o", ms=7.5, color=AGENT, zorder=6)
    (art["eyeR"],) = ax.plot([], [], "o", ms=7.5, color=BLUE, zorder=6)
    (art["stimdot"],) = ax.plot([], [], "o", ms=12, color=GREEN, zorder=7)

    # ---- sensors ----
    ax = panel([0.435, 0.845, 0.535, 0.09], "Sensor activations")
    ax.set_xlim(-0.5, 61.5)
    ax.set_ylim(0, 1.42)
    ax.set_xticks([])
    ax.set_yticks([0, 1])
    art["sens"] = ax.bar(np.arange(62), np.zeros(62), width=0.9,
                         color=[AGENT] * 31 + [BLUE] * 31)
    ax.text(0.01, 0.97, "left eye −30°…+90°", transform=ax.transAxes,
            fontsize=8, color=AGENT, va="top")
    ax.text(0.99, 0.97, "right eye −90°…+30°", transform=ax.transAxes,
            fontsize=8, color=BLUE, ha="right", va="top")

    # ---- effectors ----
    ax = panel([0.435, 0.745, 0.535, 0.062], "Effectors  ·  ΔH = 10·(L−R)")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.6, 0.6)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axhline(0, color=EDGE, lw=1)
    art["effL"] = ax.barh(0.3, 0, height=0.34, left=0, color=AGENT, alpha=0.85)[0]
    art["effR"] = ax.barh(-0.3, 0, height=0.34, left=0, color=BLUE, alpha=0.85)[0]
    ax.text(-0.015, 0.3, "L", fontsize=9.5, color=AGENT, ha="right", va="center",
            transform=ax.get_yaxis_transform())
    ax.text(-0.015, -0.3, "R", fontsize=9.5, color=BLUE, ha="right", va="center",
            transform=ax.get_yaxis_transform())

    # ---- weights ----
    ax = panel([0.435, 0.655, 0.535, 0.055], "Weights")
    bins = rec["hist_bins"]
    ax.set_xlim(bins[0], bins[-1])
    ax.set_yticks([])
    ax.set_xticks([-2, 0, 3])
    ax.axvline(0, color=EDGE, lw=1, ls=(0, (3, 3)))
    art["hist"] = ax.bar(0.5 * (bins[:-1] + bins[1:]), np.zeros(50),
                         width=(bins[1] - bins[0]) * 0.94, color=HIST)
    art["hist_ax"] = ax

    def time_axis(ax):
        ax.set_xlim(-WINDOW, 0)
        ax.set_xticks([])

    # ---- heading vs stimulus ----
    ax = panel([0.075, 0.395, 0.895, 0.20],
               "Heading vs. stimulus  (window: last 1,440 steps)")
    time_axis(ax)
    ax.set_ylim(-5, 365)
    ax.set_yticks([0, 90, 180, 270, 360])
    ax.set_ylabel("deg", fontsize=8.5)
    art["flips1"] = LineCollection([], colors=EDGE, linewidths=1.0, zorder=1)
    ax.add_collection(art["flips1"])
    art["stim_tr"] = ax.scatter([], [], s=2.6, color=STIM, zorder=2)
    art["head_tr"] = ax.scatter([], [], s=3.4, color=AGENT, zorder=3)
    ax.text(0.006, 0.915, "stimulus", transform=ax.transAxes, fontsize=8.5, color=STIM)
    ax.text(0.073, 0.915, "agent", transform=ax.transAxes, fontsize=8.5, color=AGENT)

    # ---- heading error ----
    ax = panel([0.075, 0.245, 0.895, 0.11], "Heading error  (±45° band shaded)")
    time_axis(ax)
    ax.set_ylim(-185, 185)
    ax.set_yticks([-180, 0, 180])
    ax.axhspan(-45, 45, color=BAND, alpha=0.10, zorder=0)
    art["flips2"] = LineCollection([], colors=EDGE, linewidths=1.0, zorder=1)
    ax.add_collection(art["flips2"])
    art["err_tr"] = ax.scatter([], [], s=2.6, color=BLUE, zorder=2)

    # ---- raster ----
    ax = panel([0.075, 0.035, 0.895, 0.17], "Reservoir spikes  (rows = 200 nodes)")
    time_axis(ax)
    ax.set_yticks([])
    art["raster"] = ax.imshow(np.zeros((rec["n_nodes"], WINDOW)), aspect="auto",
                              cmap="Greys", vmin=0, vmax=1.35,
                              extent=(-WINDOW, 0, 0, rec["n_nodes"]),
                              interpolation="nearest", zorder=2)

    return fig, art


def make_update(rec, art, cfg):
    n_steps = len(rec["heading"])

    def update(frame):
        t = min(frame * cfg.steps_per_frame + cfg.steps_per_frame - 1, n_steps - 1)
        lo = max(0, t - WINDOW + 1)
        xs = np.arange(lo, t + 1) - t

        heading = rec["heading"][t]
        stim = rec["stim"][t]

        # arena
        art["fov"].set_theta1(heading - 90)
        art["fov"].set_theta2(heading + 90)
        offs = np.concatenate([30 + np.linspace(-60, 60, 31),
                               -30 + np.linspace(-60, 60, 31)]) + heading
        acts = rec["sensors"][t]
        r0, seg_base, seg_len = 0.52, 0.03, 0.30
        a = np.radians(offs)
        segs, colors = [], []
        for i in range(62):
            r1 = r0 + seg_base + seg_len * acts[i]
            segs.append([(r0 * np.cos(a[i]), r0 * np.sin(a[i])),
                         (r1 * np.cos(a[i]), r1 * np.sin(a[i]))])
            on = acts[i] > 0.02
            colors.append((AGENT if i < 31 else BLUE) if on else EDGE)
        art["ticks"].set_segments(segs)
        art["ticks"].set_colors(colors)
        h = np.radians(heading)
        art["arrow"].set_data([0, 0.46 * np.cos(h)], [0, 0.46 * np.sin(h)])
        for key, off in (("eyeL", 30), ("eyeR", -30)):
            e = np.radians(heading + off)
            art[key].set_data([0.48 * np.cos(e)], [0.48 * np.sin(e)])
        s = np.radians(stim)
        art["stimdot"].set_data([np.cos(s)], [np.sin(s)])

        # sensors / effectors
        for bar, v in zip(art["sens"], acts):
            bar.set_height(v)
        L, R = rec["outputs"][t]
        art["effL"].set_width(L)
        art["effR"].set_width(R)

        # time panels
        art["stim_tr"].set_offsets(np.c_[xs, rec["stim"][lo:t + 1]])
        art["head_tr"].set_offsets(np.c_[xs, rec["heading"][lo:t + 1]])
        art["err_tr"].set_offsets(np.c_[xs, rec["err"][lo:t + 1]])
        vis = [f - t for f in rec["flips"] if lo <= f <= t]
        art["flips1"].set_segments([[(x, -5), (x, 365)] for x in vis])
        art["flips2"].set_segments([[(x, -185), (x, 185)] for x in vis])

        # raster window (right-aligned)
        win = np.zeros((rec["n_nodes"], WINDOW))
        win[:, WINDOW - (t + 1 - lo):] = rec["spikes"][lo:t + 1].T
        art["raster"].set_data(win)

        # weights histogram
        counts = rec["hists"][min(frame, len(rec["hists"]) - 1)]
        peak = counts.max() or 1
        art["hist_ax"].set_ylim(0, peak * 1.08)
        for bar, c in zip(art["hist"], counts):
            bar.set_height(c)

        art["clock"].set_text(f"t = {t + 1:,}")
        return []

    return update


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=7200)
    ap.add_argument("--steps-per-frame", type=int, default=6)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--out", type=str, default="scripts/out/tracking_irregular.mp4")
    ap.add_argument("--still", type=int, default=None,
                    help="render only this frame to PNG and exit")
    cfg = ap.parse_args()

    print(f"simulating seed {cfg.seed}, {cfg.steps} steps...", flush=True)
    rec = simulate(cfg.seed, cfg.steps, cfg.steps_per_frame)
    fig, art = build_figure(rec, cfg)
    update = make_update(rec, art, cfg)
    n_frames = cfg.steps // cfg.steps_per_frame

    out = pathlib.Path(cfg.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if cfg.still is not None:
        update(cfg.still)
        png = out.with_suffix("").as_posix() + f"_frame{cfg.still}.png"
        fig.savefig(png, dpi=cfg.dpi, facecolor="white")
        print(f"saved {png}", flush=True)
        return

    print(f"rendering {n_frames} frames at {cfg.fps} fps "
          f"({n_frames / cfg.fps:.0f} s of video)...", flush=True)
    anim = animation.FuncAnimation(fig, update, frames=n_frames, blit=False)
    writer = animation.FFMpegWriter(fps=cfg.fps, bitrate=6000,
                                    extra_args=["-pix_fmt", "yuv420p"])
    anim.save(out.as_posix(), writer=writer, dpi=cfg.dpi,
              progress_callback=lambda i, n: (i % 150 == 0) and print(
                  f"  frame {i}/{n}", flush=True))
    print(f"saved {out}", flush=True)


if __name__ == "__main__":
    main()
