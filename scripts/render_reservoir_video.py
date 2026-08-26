"""Render the reservoir explainer as the right-half companion of the tracking
demo video.

Runs the SAME simulation as scripts/render_tracking_video.py's original
output (paper configuration, irregular stimulus motion, seed 0) with the same
frame timing (6 steps per frame at 30 fps), so the two videos play in
lockstep on one 16:9 slide: tracking demo left, this network view right.

Top: the network as a ring. Sensors form an outer arc and the 200 neurons
are arranged by the stimulus direction they are wired to prefer, in the
agent's own frame with FRONT FACING UP (annotated) - so the lit region moves
around the ring as the stimulus moves across the retina, while every neuron
keeps its position. Node fill blends blue with activation (relative to
threshold) and red with the neuron's firing rate within the current frame.
Effectors sit at the center. Bottom: three real, connected neurons with
activation-vs-threshold traces.

Usage: python scripts/render_reservoir_video.py [--seed 0] [--steps 7200]
       [--steps-per-frame 6] [--still N]
"""

from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle, FancyArrowPatch

from homeostasis import VariableTrackingSimulation

INK = "#1c2733"
MUTED = "#7a8494"
EDGE = "#d9dee6"
AMBER = "#c9820a"        # sensory drive
SPIKE = "#d62839"        # firing
NODE = "#5b8fd6"         # activation fill
GREEN = "#0c9c62"        # stimulus / left effector
BLUE = "#2266cc"         # right effector
PINK = "#f2b8c6"

TRACE_WIN = 450          # steps shown in the microscope traces (2.5 s of video)

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


def simulate(seed: int, n_steps: int):
    sim = VariableTrackingSimulation(seed=seed)
    net = sim.network
    n = net.config.n_nodes
    rec = {
        "x": np.zeros((n_steps, n), dtype=np.float32),
        "thr": np.zeros((n_steps, n), dtype=np.float32),
        "spiked": np.zeros((n_steps, n), dtype=bool),
        "inputs": np.zeros((n_steps, 62), dtype=np.float32),
        "outputs": np.zeros((n_steps, 2), dtype=np.float32),
        "heading": np.zeros(n_steps), "stim": np.zeros(n_steps),
    }
    for t in range(n_steps):
        thr = net.thresholds.copy()
        state, _ = sim.step()
        rec["x"][t] = state.x
        rec["thr"][t] = thr
        rec["spiked"][t] = state.spiked
        rec["inputs"][t] = state.inputs
        rec["outputs"][t] = state.outputs
        rec["heading"][t] = sim.env.heading
        rec["stim"][t] = sim.env.stimulus_angle
    rec["adj"] = net.adjacency.copy()
    rec["in_adj"] = net.input_adjacency.copy()
    rec["out_adj"] = net.output_adjacency.copy()
    rec["sensor_offsets"] = sim.env.config.sensor_offsets.copy()
    rec["n_nodes"] = n
    return rec


def layout_ring(rec, rng):
    """Ring in the agent's frame, FRONT = UP: angle = 90 deg + 2 x offset."""
    offs = rec["sensor_offsets"]
    A = rec["in_adj"].astype(float)
    with np.errstate(invalid="ignore"):
        tuning = (A * offs[:, None]).sum(0) / np.maximum(A.sum(0), 1e-9)
    order = np.argsort(tuning)
    n = rec["n_nodes"]
    ang = np.zeros(n)
    # ranks spread over the full circle, rotated so tuning 0 sits at the top
    ang[order] = np.linspace(-np.pi, np.pi, n, endpoint=False) + np.pi / 2
    rad = 1.0 + rng.uniform(-0.055, 0.055, n)
    pos = np.column_stack([rad * np.cos(ang), rad * np.sin(ang)])
    s_ang = np.radians(90.0 + 2.0 * offs)
    s_pos = np.column_stack([1.30 * np.cos(s_ang), 1.30 * np.sin(s_ang)])
    return pos, s_pos


def pick_trio(rec):
    """Three moderately active nodes forming a real chain a -> b -> c."""
    act = rec["spiked"].mean(0)
    good = np.argsort(np.abs(act - 0.35))
    adj = rec["adj"]
    has_in = rec["in_adj"].sum(0) > 0
    for a in good[:60]:
        if not has_in[a]:
            continue
        for b in np.flatnonzero(adj[a])[:20]:
            if abs(act[b] - 0.35) > 0.3:
                continue
            for c in np.flatnonzero(adj[b])[:20]:
                if c != a and abs(act[c] - 0.35) <= 0.3:
                    return int(a), int(b), int(c)
    return int(good[0]), int(good[1]), int(good[2])


def build_figure(rec, cfg, rng):
    fig = plt.figure(figsize=(9.6, 10.8), dpi=cfg.dpi)
    fig.patch.set_facecolor("white")
    art = {}

    pos, s_pos = layout_ring(rec, rng)
    art["pos"], art["s_pos"] = pos, s_pos
    trio = pick_trio(rec)
    art["trio"] = trio

    def head(x, y, text):
        fig.text(x, y, text, fontsize=10.5, fontweight="bold", color=INK)

    # ================= ring panel =================
    ax = fig.add_axes([0.03, 0.305, 0.94, 0.63])
    ax.set_xlim(-1.55, 1.55)
    ax.set_ylim(-1.46, 1.50)
    ax.set_aspect("equal")
    ax.axis("off")
    head(0.05, 0.958, "Inside the reservoir")
    fig.text(0.05, 0.944,
             "the agent's frame, front facing up · neurons arranged by the "
             "stimulus direction they are wired to prefer · neurons stay put; "
             "the lit region moves with the stimulus", fontsize=8.5, color=MUTED)

    # legend: two stacked rows, top-left, clear of the ring
    legend = ((("●", "charging (activation)", NODE),
               ("●", "firing (rate this frame)", SPIKE)),
              (("—", "sensory input", AMBER),
               ("—", "spikes to neighbors", "#9aa3ad")))
    for r, row in enumerate(legend):
        lx = 0.05
        for swatch, label, col in row:
            fig.text(lx, 0.9245 - r * 0.0145, swatch, fontsize=9, color=col,
                     fontweight="bold")
            fig.text(lx + 0.013, 0.9245 - r * 0.0145, label, fontsize=8,
                     color=MUTED)
            lx += 0.030 + 0.0082 * len(label)

    art["in_edges"] = LineCollection([], colors=AMBER, linewidths=0.9, zorder=2)
    ax.add_collection(art["in_edges"])
    art["chords"] = LineCollection([], colors=INK, linewidths=0.3, alpha=0.05,
                                   zorder=1)
    ax.add_collection(art["chords"])
    art["eff_edges"] = LineCollection([], linewidths=0.8, zorder=3)
    ax.add_collection(art["eff_edges"])

    art["sens"] = ax.scatter(s_pos[:, 0], s_pos[:, 1], s=26, c="#eef1f5",
                             edgecolors=EDGE, linewidths=0.5, zorder=4)
    art["nodes"] = ax.scatter(pos[:, 0], pos[:, 1], s=52, c="#f2f5f9",
                              edgecolors=EDGE, linewidths=0.6, zorder=5)
    for i, node in enumerate(trio):
        ax.add_patch(Circle(pos[node], 0.075, fill=False, color=SPIKE,
                            lw=1.3, zorder=6))
        ax.annotate(f"{chr(97 + i)}", pos[node] + np.array([0.09, 0.09]),
                    fontsize=9, color=SPIKE, fontweight="bold", zorder=7)

    # orientation annotations: front = up
    ax.annotate("", xy=(0, 1.50), xytext=(0, 1.38),
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.4))
    ax.text(0.05, 1.44, "ahead", fontsize=8.5, color=INK, fontweight="bold",
            va="center")
    ax.text(-1.52, 0, "left", fontsize=8, color=MUTED, ha="left", va="center")
    ax.text(1.52, 0, "right", fontsize=8, color=MUTED, ha="right", va="center")

    for key, (x0, label, col) in {"effL": (-0.30, "turn left", GREEN),
                                  "effR": (0.30, "turn right", BLUE)}.items():
        art[key] = Circle((x0, 0), 0.14, facecolor="white", edgecolor=col,
                          lw=1.6, zorder=8)
        ax.add_patch(art[key])
        ax.text(x0, -0.225, label, fontsize=8, color=col, ha="center",
                va="top", zorder=9, fontweight="bold")

    ax.text(0, -1.44, "input → reservoir → output", fontsize=8.5,
            color=MUTED, ha="center")

    # world glyph (allocentric, for orientation against the left-half video)
    axg = fig.add_axes([0.825, 0.760, 0.145, 0.145 * (9.6 / 10.8)])
    axg.set_xlim(-1.25, 1.25)
    axg.set_ylim(-1.25, 1.25)
    axg.set_aspect("equal")
    axg.axis("off")
    axg.add_patch(Circle((0, 0), 1.0, fill=False, color=EDGE, lw=0.8))
    axg.add_patch(Circle((0, 0), 0.42, color=PINK, alpha=0.55))
    (art["g_head"],) = axg.plot([], [], color=INK, lw=1.6,
                                solid_capstyle="round")
    (art["g_stim"],) = axg.plot([], [], "o", ms=6, color=GREEN)
    axg.set_title("the world (bird's-eye)", fontsize=7.5, color=MUTED, pad=2)

    art["clock"] = fig.text(0.965, 0.958, "t = 0", fontsize=11, color=MUTED,
                            ha="right", family="monospace")

    # ================= microscope =================
    head(0.05, 0.276, "Three neurons up close")
    fig.text(0.05, 0.262,
             "each neuron leaks, sums weighted spikes from its neighbors, and "
             "fires when its activation crosses threshold — then subtracts it "
             "and starts again", fontsize=8.5, color=MUTED)

    axm = fig.add_axes([0.045, 0.032, 0.24, 0.21])
    axm.set_xlim(-1.15, 1.15)
    axm.set_ylim(-1.25, 1.30)
    axm.axis("off")
    tpos = {trio[0]: (-0.62, 0.72), trio[1]: (0.68, 0.18), trio[2]: (-0.45, -0.78)}
    art["tpos"] = tpos
    art["tcirc"] = {}
    art["tarrow"] = []
    adj = rec["adj"]
    for i, node in enumerate(trio):
        c = Circle(tpos[node], 0.30, facecolor="white", edgecolor=INK, lw=1.2,
                   zorder=5)
        axm.add_patch(c)
        art["tcirc"][node] = c
        axm.text(*(np.array(tpos[node]) + [0, 0.44]), f"{chr(97 + i)}",
                 fontsize=10, fontweight="bold", color=SPIKE, ha="center")
    for s in trio:
        for t in trio:
            if s != t and adj[s, t]:
                p0, p1 = np.array(tpos[s]), np.array(tpos[t])
                d = (p1 - p0) / np.linalg.norm(p1 - p0)
                ar = FancyArrowPatch(p0 + d * 0.33, p1 - d * 0.36,
                                     arrowstyle="-|>", mutation_scale=13,
                                     color=EDGE, lw=1.4, zorder=4,
                                     connectionstyle="arc3,rad=0.18")
                axm.add_patch(ar)
                art["tarrow"].append((s, t, ar))
    p0 = np.array(tpos[trio[0]])
    art["sens_arrow"] = FancyArrowPatch(p0 + [-0.55, 0.42], p0 + [-0.24, 0.20],
                                        arrowstyle="-|>", mutation_scale=13,
                                        color=EDGE, lw=1.6, zorder=4)
    axm.add_patch(art["sens_arrow"])
    axm.text(*(p0 + [-0.52, 0.60]), "sensors", fontsize=7.5, color=AMBER,
             ha="left")

    art["traces"] = []
    for i, node in enumerate(trio):
        axt = fig.add_axes([0.36, 0.187 - i * 0.076, 0.60, 0.060])
        axt.set_xlim(-TRACE_WIN, 0)
        axt.set_xticks([])
        axt.set_yticks([])
        for side in ("top", "right"):
            axt.spines[side].set_visible(False)
        (ln_x,) = axt.plot([], [], color=NODE, lw=0.9, zorder=3)
        (ln_t,) = axt.plot([], [], color=INK, lw=0.9, ls=(0, (3, 2)), zorder=2)
        sc = axt.scatter([], [], s=14, marker="|", color=SPIKE,
                         alpha=0.45, linewidths=0.7, zorder=4)
        axt.text(0.005, 0.80, f"{chr(97 + i)}", transform=axt.transAxes,
                 fontsize=9, fontweight="bold", color=SPIKE)
        if i == 0:
            axt.text(0.995, 0.10, "activation before reset · threshold (dashed) · spike",
                     transform=axt.transAxes, fontsize=7.5, color=MUTED,
                     ha="right", va="bottom")
        art["traces"].append((axt, ln_x, ln_t, sc))
    art["traces"][-1][0].set_xlabel(
        f"last {TRACE_WIN} steps (2.5 s of video)", fontsize=7.5)
    return fig, art


def make_update(rec, art, cfg):
    pos, s_pos = art["pos"], art["s_pos"]
    trio = art["trio"]
    adj = rec["adj"]
    in_adj = rec["in_adj"]
    out_adj = rec["out_adj"]
    spf = cfg.steps_per_frame
    n_steps = len(rec["x"])
    targets = [np.flatnonzero(adj[s]) for s in range(rec["n_nodes"])]
    in_targets = [np.flatnonzero(in_adj[s]) for s in range(62)]
    eff_sources = [np.flatnonzero(out_adj[:, k]) for k in range(2)]
    base_face = np.array(matplotlib.colors.to_rgba("#f2f5f9"))
    node_rgb = np.array(matplotlib.colors.to_rgba(NODE))
    spike_rgb = np.array(matplotlib.colors.to_rgba(SPIKE))
    edge_rgba = matplotlib.colors.to_rgba(EDGE)
    eff_cols = [GREEN, BLUE]

    def update(f):
        t = min(f * spf + spf - 1, n_steps - 1)
        w0 = max(0, t - spf + 1)
        window = rec["spiked"][w0:t + 1]           # (spf, N)
        rate = window.mean(0)[:, None]             # firing rate this frame
        x = rec["x"][t]
        thr = rec["thr"][t]
        acts = rec["inputs"][t]

        level = np.clip(x / np.maximum(thr, 1e-9), 0, 1)[:, None]
        faces = base_face + (node_rgb - base_face) * (0.15 + 0.85 * level)
        faces = faces + (spike_rgb - faces) * rate
        art["nodes"].set_facecolors(faces)
        art["nodes"].set_edgecolors(np.where(rate > 0.34, spike_rgb, edge_rgba))

        s_faces = [matplotlib.colors.to_rgba(AMBER, 0.15 + 0.85 * a) if a > 0.02
                   else matplotlib.colors.to_rgba("#eef1f5") for a in acts]
        art["sens"].set_facecolors(s_faces)

        segs, cols = [], []
        for s in np.flatnonzero(acts > 0.05):
            for nnn in in_targets[s]:
                segs.append([s_pos[s], pos[nnn]])
                cols.append(matplotlib.colors.to_rgba(AMBER, 0.10 + 0.35 * acts[s]))
        art["in_edges"].set_segments(segs)
        art["in_edges"].set_colors(cols)

        spikers = np.flatnonzero(window.any(0))
        art["chords"].set_segments(
            [[pos[s], pos[nnn]] for s in spikers for nnn in targets[s]])

        segs, cols = [], []
        spk_now = rec["spiked"][t]
        for k, (key, cx) in enumerate((("effL", -0.30), ("effR", 0.30))):
            src = eff_sources[k]
            for s in src[spk_now[src]]:
                segs.append([pos[s], (cx, 0)])
                cols.append(matplotlib.colors.to_rgba(eff_cols[k], 0.30))
            out = rec["outputs"][t, k]
            art[key].set_facecolor(matplotlib.colors.to_rgba(eff_cols[k],
                                                             0.08 + 0.7 * out))
        art["eff_edges"].set_segments(segs)
        art["eff_edges"].set_colors(cols)

        h = np.radians(rec["heading"][t])
        s_ = np.radians(rec["stim"][t])
        art["g_head"].set_data([0, 0.42 * np.cos(h)], [0, 0.42 * np.sin(h)])
        art["g_stim"].set_data([np.cos(s_)], [np.sin(s_)])

        win_any = window.any(0)
        for node, c in art["tcirc"].items():
            lv = float(np.clip(rec["x"][t, node] / max(rec["thr"][t, node], 1e-9), 0, 1))
            r = float(rate[node, 0])
            col = base_face + (node_rgb - base_face) * (0.1 + 0.9 * lv)
            col = col + (spike_rgb - col) * r
            c.set_facecolor(col)
            c.set_edgecolor(SPIKE if r > 0.34 else INK)
        for s, t_, ar in art["tarrow"]:
            on = win_any[s]
            ar.set_color(SPIKE if on else EDGE)
            ar.set_linewidth(2.2 if on else 1.2)
        drive = float(rec["inputs"][t] @ in_adj[:, trio[0]])
        art["sens_arrow"].set_color(AMBER if drive > 0.05 else EDGE)
        art["sens_arrow"].set_linewidth(2.2 if drive > 0.05 else 1.2)

        lo = max(0, t - TRACE_WIN + 1)
        xs = np.arange(lo, t + 1) - t
        for (axt, ln_x, ln_t, sc), node in zip(art["traces"], trio):
            # Show the PRE-reset activation: the model stores x' = x - 2T at
            # spike steps, so add the threshold back where a spike occurred -
            # the trace then visibly crosses the dashed line at every spike.
            sp_w = rec["spiked"][lo:t + 1, node]
            tv = rec["thr"][lo:t + 1, node]
            xv = rec["x"][lo:t + 1, node] + sp_w * tv
            ln_x.set_data(xs, xv)
            ln_t.set_data(xs, tv)
            sp = sp_w
            top = max(float(tv.max()) * 1.25, float(xv.max()) * 1.08, 0.5)
            sc.set_offsets(np.c_[xs[sp], np.full(sp.sum(), top * 0.96)]
                           if sp.any() else np.empty((0, 2)))
            axt.set_ylim(min(0.0, float(xv.min()) * 1.1), top)

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
    ap.add_argument("--out", type=str, default="scripts/out/reservoir_companion.mp4")
    ap.add_argument("--still", type=int, default=None)
    cfg = ap.parse_args()

    print(f"simulating seed {cfg.seed}, {cfg.steps} steps (irregular motion, "
          f"paper config — matches the tracking demo)...", flush=True)
    rec = simulate(cfg.seed, cfg.steps)
    rng = np.random.default_rng(0)
    fig, art = build_figure(rec, cfg, rng)
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

    print(f"rendering {n_frames} frames "
          f"({n_frames / cfg.fps:.0f} s at {cfg.fps} fps, synced to the "
          f"tracking demo)...", flush=True)
    anim = animation.FuncAnimation(fig, update, frames=n_frames, blit=False)
    writer = animation.FFMpegWriter(fps=cfg.fps, bitrate=6000,
                                    extra_args=["-pix_fmt", "yuv420p"])
    anim.save(out.as_posix(), writer=writer, dpi=cfg.dpi,
              progress_callback=lambda i, n: (i % 250 == 0) and print(
                  f"  frame {i}/{n}", flush=True))
    print(f"saved {out}", flush=True)


if __name__ == "__main__":
    main()
