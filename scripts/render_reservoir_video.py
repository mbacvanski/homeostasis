"""Render an explanatory video of the reservoir itself.

Top: the whole network as a ring. Sensors form an outer arc (lit by the
stimulus), the 200 reservoir neurons sit on the ring ordered by the stimulus
direction they are wired to prefer, and the two effectors sit at the center.
Input edges light as sensors drive their neurons, spikes flash and cascade as
chords across the ring, and effector edges light as spiking in-neighbors are
counted. A small arena glyph gives the world context.

Bottom: three real, connected neurons under the microscope - activation
traces against their (moving) thresholds show leak, integration, firing and
reset, while arrows between them flash as spikes propagate.

Driven by the tested `homeostasis` package (published tracking task).

Usage: python scripts/render_reservoir_video.py [--seed 6] [--steps 1500]
       [--still N]
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

from homeostasis import TrackingSimulation

INK = "#1c2733"
MUTED = "#7a8494"
EDGE = "#d9dee6"
AMBER = "#c9820a"        # sensory drive
SPIKE = "#d62839"        # spike flash
NODE = "#5b8fd6"         # activation fill
GREEN = "#0c9c62"        # stimulus / left effector
BLUE = "#2266cc"         # right effector
PINK = "#f2b8c6"

TRACE_WIN = 150          # steps shown in the microscope traces

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
    sim = TrackingSimulation(seed=seed)
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

    # spike glow with a short decay so flashes read at video rate
    glow = np.zeros((n_steps, n), dtype=np.float32)
    g = np.zeros(n, dtype=np.float32)
    for t in range(n_steps):
        g = np.maximum(rec["spiked"][t].astype(np.float32), g * 0.68)
        glow[t] = g
    rec["glow"] = glow

    rec["adj"] = net.adjacency.copy()
    rec["in_adj"] = net.input_adjacency.copy()
    rec["out_adj"] = net.output_adjacency.copy()
    rec["sensor_offsets"] = sim.env.config.sensor_offsets.copy()
    rec["n_nodes"] = n
    return rec


def layout_ring(rec, rng):
    """Node positions: ring ordered by preferred stimulus direction."""
    offs = rec["sensor_offsets"]
    A = rec["in_adj"].astype(float)                    # (62, N)
    with np.errstate(invalid="ignore"):
        tuning = (A * offs[:, None]).sum(0) / np.maximum(A.sum(0), 1e-9)
    order = np.argsort(tuning)
    n = rec["n_nodes"]
    ang = np.zeros(n)
    ang[order] = np.linspace(0, 2 * np.pi, n, endpoint=False) - np.pi
    rad = 1.0 + rng.uniform(-0.055, 0.055, n)
    pos = np.column_stack([rad * np.cos(ang), rad * np.sin(ang)])
    # sensors on the outer arc: offset (-90..90) -> full circle, same handedness
    s_ang = np.radians(offs * 2.0)
    s_pos = np.column_stack([1.30 * np.cos(s_ang), 1.30 * np.sin(s_ang)])
    return pos, s_pos, ang, tuning


def pick_trio(rec):
    """Three moderately active nodes forming a real chain a -> b -> c."""
    act = rec["spiked"].mean(0)
    good = np.argsort(np.abs(act - 0.35))              # near-median activity first
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

    pos, s_pos, ang, tuning = layout_ring(rec, rng)
    art["pos"], art["s_pos"] = pos, s_pos
    trio = pick_trio(rec)
    art["trio"] = trio

    def head(x, y, text):
        fig.text(x, y, text, fontsize=10.5, fontweight="bold", color=INK)

    # ================= ring panel =================
    ax = fig.add_axes([0.03, 0.315, 0.94, 0.635])
    ax.set_xlim(-1.52, 1.52)
    ax.set_ylim(-1.43, 1.43)
    ax.set_aspect("equal")
    ax.axis("off")
    art["ring_ax"] = ax
    head(0.05, 0.955, "The reservoir")
    fig.text(0.05, 0.941,
             "sensors (outer arc) drive neurons arranged by the stimulus direction "
             "they prefer · spikes cascade along the wiring · effectors count "
             "spiking inputs", fontsize=8.5, color=MUTED)

    # input edges (sensor -> node), rebuilt per frame
    art["in_edges"] = LineCollection([], colors=AMBER, linewidths=0.9, zorder=2)
    ax.add_collection(art["in_edges"])
    # recurrent chords from last step's spikers, rebuilt per frame
    art["chords"] = LineCollection([], colors=INK, linewidths=0.3, alpha=0.055,
                                   zorder=1)
    ax.add_collection(art["chords"])
    # effector edges
    art["eff_edges"] = LineCollection([], linewidths=0.8, zorder=3)
    ax.add_collection(art["eff_edges"])

    # sensor markers
    art["sens"] = ax.scatter(s_pos[:, 0], s_pos[:, 1], s=26, c="#eef1f5",
                             edgecolors=EDGE, linewidths=0.5, zorder=4)
    # reservoir nodes
    art["nodes"] = ax.scatter(pos[:, 0], pos[:, 1], s=52, c="#f2f5f9",
                              edgecolors=EDGE, linewidths=0.6, zorder=5)
    # highlight the microscope trio on the ring
    for i, node in enumerate(trio):
        ax.add_patch(Circle(pos[node], 0.075, fill=False, color=SPIKE,
                            lw=1.3, zorder=6))
        ax.annotate(f"{chr(97 + i)}", pos[node] + np.array([0.10, 0.10]),
                    fontsize=9, color=SPIKE, fontweight="bold", zorder=7)

    # effectors at the center
    for key, (x0, label, col) in {"effL": (-0.30, "turn left", GREEN),
                                  "effR": (0.30, "turn right", BLUE)}.items():
        art[key] = Circle((x0, 0), 0.14, facecolor="white", edgecolor=col,
                          lw=1.6, zorder=8)
        ax.add_patch(art[key])
        ax.text(x0, -0.225, label, fontsize=8, color=col, ha="center",
                va="top", zorder=9, fontweight="bold")

    ax.text(0, -1.40, "input → reservoir → output", fontsize=8.5,
            color=MUTED, ha="center")

    # arena glyph (top-right corner of the ring panel)
    axg = fig.add_axes([0.815, 0.755, 0.155, 0.155 * (9.6 / 10.8)])
    axg.set_xlim(-1.25, 1.25)
    axg.set_ylim(-1.25, 1.25)
    axg.set_aspect("equal")
    axg.axis("off")
    axg.add_patch(Circle((0, 0), 1.0, fill=False, color=EDGE, lw=0.8))
    axg.add_patch(Circle((0, 0), 0.42, color=PINK, alpha=0.55))
    (art["g_head"],) = axg.plot([], [], color=INK, lw=1.6,
                                solid_capstyle="round")
    (art["g_stim"],) = axg.plot([], [], "o", ms=6, color=GREEN)
    axg.set_title("the world", fontsize=7.5, color=MUTED, pad=2)

    art["clock"] = fig.text(0.965, 0.958, "", fontsize=10, color=MUTED,
                            ha="right", family="monospace")

    # ================= microscope =================
    head(0.05, 0.283, "Three neurons up close")
    fig.text(0.05, 0.269,
             "each neuron leaks, sums weighted spikes from its neighbors, and "
             "fires when its activation crosses threshold — then subtracts it "
             "and starts again", fontsize=8.5, color=MUTED)

    # trio diagram (left)
    axm = fig.add_axes([0.045, 0.035, 0.24, 0.215])
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
    # sensory drive arrow into node a
    p0 = np.array(tpos[trio[0]])
    art["sens_arrow"] = FancyArrowPatch(p0 + [-0.55, 0.42], p0 + [-0.24, 0.20],
                                        arrowstyle="-|>", mutation_scale=13,
                                        color=EDGE, lw=1.6, zorder=4)
    axm.add_patch(art["sens_arrow"])
    axm.text(*(p0 + [-0.52, 0.60]), "sensors", fontsize=7.5, color=AMBER,
             ha="left")

    # activation traces (right): one axis per trio node
    art["traces"] = []
    for i, node in enumerate(trio):
        axt = fig.add_axes([0.36, 0.192 - i * 0.078, 0.60, 0.062])
        axt.set_xlim(-TRACE_WIN, 0)
        axt.set_xticks([])
        axt.set_yticks([])
        for side in ("top", "right"):
            axt.spines[side].set_visible(False)
        (ln_x,) = axt.plot([], [], color=NODE, lw=1.1, zorder=3)
        (ln_t,) = axt.plot([], [], color=INK, lw=0.9, ls=(0, (3, 2)), zorder=2)
        sc = axt.scatter([], [], s=14, color=SPIKE, zorder=4)
        axt.text(0.005, 0.82, f"{chr(97 + i)}", transform=axt.transAxes,
                 fontsize=9, fontweight="bold", color=SPIKE)
        if i == 0:
            axt.text(0.995, 0.10, "activation · threshold (dashed) · spike",
                     transform=axt.transAxes, fontsize=7.5, color=MUTED,
                     ha="right", va="bottom")
        art["traces"].append((axt, ln_x, ln_t, sc))
    art["traces"][-1][0].set_xlabel(f"last {TRACE_WIN} steps", fontsize=7.5)

    return fig, art


def make_update(rec, art, cfg):
    pos, s_pos = art["pos"], art["s_pos"]
    trio = art["trio"]
    adj = rec["adj"]
    in_adj = rec["in_adj"]
    out_adj = rec["out_adj"]
    targets = [np.flatnonzero(adj[s]) for s in range(rec["n_nodes"])]
    in_targets = [np.flatnonzero(in_adj[s]) for s in range(62)]
    eff_sources = [np.flatnonzero(out_adj[:, k]) for k in range(2)]
    base_face = np.array(matplotlib.colors.to_rgba("#f2f5f9"))
    node_rgb = np.array(matplotlib.colors.to_rgba(NODE))
    spike_rgb = np.array(matplotlib.colors.to_rgba(SPIKE))
    eff_cols = [GREEN, BLUE]

    def update(f):
        t = min(f, len(rec["x"]) - 1)
        x = rec["x"][t]
        thr = rec["thr"][t]
        spk = rec["spiked"][t]
        glow = rec["glow"][t][:, None]
        acts = rec["inputs"][t]

        # node faces: pale -> blue with activation, flashing red on spikes
        level = np.clip(x / np.maximum(thr, 1e-9), 0, 1)[:, None]
        faces = base_face + (node_rgb - base_face) * (0.15 + 0.85 * level)
        faces = faces + (spike_rgb - faces) * glow
        art["nodes"].set_facecolors(faces)
        art["nodes"].set_edgecolors(np.where(glow > 0.4, spike_rgb, 
                                    matplotlib.colors.to_rgba(EDGE)))

        # sensor faces
        s_faces = [matplotlib.colors.to_rgba(AMBER, 0.15 + 0.85 * a) if a > 0.02
                   else matplotlib.colors.to_rgba("#eef1f5") for a in acts]
        art["sens"].set_facecolors(s_faces)

        # input edges from active sensors
        segs, cols = [], []
        for s in np.flatnonzero(acts > 0.05):
            for n in in_targets[s]:
                segs.append([s_pos[s], pos[n]])
                cols.append(matplotlib.colors.to_rgba(AMBER, 0.10 + 0.35 * acts[s]))
        art["in_edges"].set_segments(segs)
        art["in_edges"].set_colors(cols)

        # recurrent chords from the previous step's spikers (delivering now)
        if t > 0:
            spikers = np.flatnonzero(rec["spiked"][t - 1])
            segs = [ [pos[s], pos[n]] for s in spikers for n in targets[s] ]
            art["chords"].set_segments(segs)

        # effector edges + fills
        segs, cols = [], []
        for k, (key, cx) in enumerate((("effL", -0.30), ("effR", 0.30))):
            src = eff_sources[k]
            live = src[spk[src]]
            for s in live:
                segs.append([pos[s], (cx, 0)])
                cols.append(matplotlib.colors.to_rgba(eff_cols[k], 0.30))
            out = rec["outputs"][t, k]
            art[key].set_facecolor(matplotlib.colors.to_rgba(eff_cols[k],
                                                             0.08 + 0.7 * out))
        art["eff_edges"].set_segments(segs)
        art["eff_edges"].set_colors(cols)

        # arena glyph
        h = np.radians(rec["heading"][t])
        s = np.radians(rec["stim"][t])
        art["g_head"].set_data([0, 0.42 * np.cos(h)], [0, 0.42 * np.sin(h)])
        art["g_stim"].set_data([np.cos(s)], [np.sin(s)])

        # microscope
        prev_spk = rec["spiked"][t - 1] if t > 0 else np.zeros_like(spk)
        for node, c in art["tcirc"].items():
            lv = float(np.clip(rec["x"][t, node] / max(rec["thr"][t, node], 1e-9), 0, 1))
            g = float(rec["glow"][t, node])
            col = base_face + (node_rgb - base_face) * (0.1 + 0.9 * lv)
            col = col + (spike_rgb - col) * g
            c.set_facecolor(col)
            c.set_edgecolor(SPIKE if g > 0.4 else INK)
        for s_, t_, ar in art["tarrow"]:
            on = prev_spk[s_]
            ar.set_color(SPIKE if on else EDGE)
            ar.set_linewidth(2.2 if on else 1.2)
        drive = float(rec["inputs"][t] @ in_adj[:, trio[0]])
        art["sens_arrow"].set_color(AMBER if drive > 0.05 else EDGE)
        art["sens_arrow"].set_linewidth(2.2 if drive > 0.05 else 1.2)

        lo = max(0, t - TRACE_WIN + 1)
        xs = np.arange(lo, t + 1) - t
        for (axt, ln_x, ln_t, sc), node in zip(art["traces"], trio):
            xv = rec["x"][lo:t + 1, node]
            tv = rec["thr"][lo:t + 1, node]
            ln_x.set_data(xs, xv)
            ln_t.set_data(xs, tv)
            sp = rec["spiked"][lo:t + 1, node]
            sc.set_offsets(np.c_[xs[sp], tv[sp]] if sp.any() else
                           np.empty((0, 2)))
            top = max(float(tv.max()) * 1.25, float(xv.max()) * 1.1, 0.5)
            axt.set_ylim(min(0.0, float(xv.min()) * 1.1), top)

        art["clock"].set_text(f"step {t:,}")
        return []

    return update


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=6)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--out", type=str, default="scripts/out/reservoir_explainer.mp4")
    ap.add_argument("--still", type=int, default=None)
    cfg = ap.parse_args()

    print(f"simulating seed {cfg.seed}, {cfg.steps} steps...", flush=True)
    rec = simulate(cfg.seed, cfg.steps)
    rng = np.random.default_rng(0)
    fig, art = build_figure(rec, cfg, rng)
    update = make_update(rec, art, cfg)

    out = pathlib.Path(cfg.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if cfg.still is not None:
        update(cfg.still)
        png = out.with_suffix("").as_posix() + f"_frame{cfg.still}.png"
        fig.savefig(png, dpi=cfg.dpi, facecolor="white")
        print(f"saved {png}", flush=True)
        return

    print(f"rendering {cfg.steps} frames ({cfg.steps / cfg.fps:.0f} s at "
          f"{cfg.fps} fps = {cfg.fps} steps/s)...", flush=True)
    anim = animation.FuncAnimation(fig, update, frames=cfg.steps, blit=False)
    writer = animation.FFMpegWriter(fps=cfg.fps, bitrate=6000,
                                    extra_args=["-pix_fmt", "yuv420p"])
    anim.save(out.as_posix(), writer=writer, dpi=cfg.dpi,
              progress_callback=lambda i, n: (i % 250 == 0) and print(
                  f"  frame {i}/{n}", flush=True))
    print(f"saved {out}", flush=True)


if __name__ == "__main__":
    main()
