"""Right-half companion for pong_evolved.mp4: inside the Pong reservoir.

Runs the IDENTICAL simulation as render_pong_video.py (same loadout, seed,
and two-phase frame plan) and draws the reservoir itself, frame-for-frame in
lockstep, so the two 1440x1620 videos can sit side by side on one slide.

Layout mirrors the game's geometry: the sensor fan opens to the RIGHT
(toward the field, as in the left-half video), up is up, down is down, and
the region behind the paddle has no sensors. Unlike the tracking reservoir,
40% of this champion's synapses are inhibitory - spike deliveries are drawn
in two colors, and the microscope trio is chosen to include an inhibitory
link so a spike can be seen pushing a neighbor's activation DOWN.

Usage:
  python scripts/render_pong_reservoir_video.py            (matches final video)
  python scripts/render_pong_reservoir_video.py --still 4000
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
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle

from homeostasis import PONG_RESERVOIR_CONFIG, PongConfig, PongSimulation
from viz.pong_server import LOADOUT_BY_ID, PONG_PARAMS, RESERVOIR_PARAMS

INK = "#1c2733"
MUTED = "#7a8494"
EDGE = "#d9dee6"
AGENT = "#d62839"
BLUE = "#2266cc"
GREEN = "#0c9c62"
AMBER = "#c9820a"
NODE = "#5b8fd6"
SPIKE = "#d62839"
INH = "#c2432f"          # inhibitory spike deliveries
EXC = "#9aa3ad"          # excitatory spike deliveries

TRACE_WIN = 900          # steps shown in the microscope traces
MAX_CHORDS = 1400        # cap on drawn spike-delivery segments per frame
TOP_SPIKERS = 25         # busiest nodes whose deliveries are drawn per frame

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
    """Steps per video frame: slow intro, then fast phase (as the left half)."""
    intro_frames = int(round(cfg.intro_seconds * cfg.fps))
    intro_step = max(1, int(round(cfg.intro_sps / cfg.fps)))
    main_step = max(1, int(round(cfg.main_sps / cfg.fps)))
    plan = np.full(cfg.frames, main_step, dtype=int)
    plan[:intro_frames] = intro_step
    return plan


def make_sim(cfg):
    entry = LOADOUT_BY_ID[cfg.loadout]
    params = entry["params"]
    r_cfg = dataclasses.replace(
        PONG_RESERVOIR_CONFIG,
        **{k: v for k, v in params.items() if k in RESERVOIR_PARAMS})
    pong_cfg = PongConfig(**{k: v for k, v in params.items() if k in PONG_PARAMS})
    return PongSimulation(r_cfg, pong_cfg, seed=cfg.seed)


def simulate(cfg, plan):
    """Pass A: the full run, recording per-frame aggregates for every panel."""
    sim = make_sim(cfg)
    net = sim.network
    n = net.config.n_nodes
    n_frames = len(plan)
    n_sens = sim.env.config.n_sensors

    e_src, e_dst = np.nonzero(net.adjacency)
    n_edges = len(e_src)

    rec = {
        "rates": np.zeros((n, n_frames), dtype=np.float32),
        "x_end": np.zeros((n, n_frames), dtype=np.float32),
        "thr_end": np.zeros((n, n_frames), dtype=np.float32),
        "sensors": np.zeros((n_frames, n_sens), dtype=np.float32),
        "outputs": np.zeros((n_frames, 2), dtype=np.float32),
        "ball": np.zeros((n_frames, 2), dtype=np.float32),
        "paddle": np.zeros(n_frames, dtype=np.float32),
        "signs": np.zeros((n_frames, (n_edges + 7) // 8), dtype=np.uint8),
        "frame_end": np.cumsum(plan),
    }
    t0 = time.perf_counter()
    for f, n_steps in enumerate(plan):
        acc = np.zeros(n)
        for _ in range(int(n_steps)):
            state, _, _ = sim.step()
            acc += state.spiked
        rec["rates"][:, f] = acc / n_steps
        rec["x_end"][:, f] = state.x
        rec["thr_end"][:, f] = net.thresholds
        rec["sensors"][f] = sim.env.sense()
        rec["outputs"][f] = state.outputs
        rec["ball"][f] = (sim.env.ball_x, sim.env.ball_y)
        rec["paddle"][f] = sim.env.paddle_y
        rec["signs"][f] = np.packbits(net.weights[e_src, e_dst] > 0)
    print(f"  pass A: {rec['frame_end'][-1]:,} steps in "
          f"{time.perf_counter()-t0:.0f}s", flush=True)

    rec["adj"] = net.adjacency.copy()
    rec["in_adj"] = net.input_adjacency.copy()
    rec["out_adj"] = net.output_adjacency.copy()
    rec["e_src"], rec["e_dst"], rec["n_edges"] = e_src, e_dst, n_edges
    rec["n_nodes"] = n
    rec["sensor_values"] = sim.env.config.sensor_values.copy()
    pc = sim.env.config
    rec["field"] = (pc.width, pc.height, pc.paddle_x, pc.paddle_half_height)
    rec["rates_g"] = np.sqrt(rec["rates"])
    return rec


def record_trio(cfg, plan, trio):
    """Pass B: identical run (seed determinism), per-step traces for 3 nodes."""
    sim = make_sim(cfg)
    net = sim.network
    total = int(plan.sum())
    idx = np.asarray(trio)
    x3 = np.zeros((total, 3), dtype=np.float32)
    thr3 = np.zeros((total, 3), dtype=np.float32)
    sp3 = np.zeros((total, 3), dtype=bool)
    t0 = time.perf_counter()
    for t in range(total):
        thr3[t] = net.thresholds[idx]        # comparison threshold this step
        state, _, _ = sim.step()
        x3[t] = state.x[idx]
        sp3[t] = state.spiked[idx]
    print(f"  pass B: trio traces in {time.perf_counter()-t0:.0f}s", flush=True)
    return x3, thr3, sp3


def layout_ring(rec, rng):
    """Ring matching the game's frame: toward the field = RIGHT, up = up."""
    vals = rec["sensor_values"]              # degrees, -90 (down) .. +90 (up)
    A = rec["in_adj"].astype(float)
    has_in = A.sum(0) > 0
    with np.errstate(invalid="ignore"):
        tuning = (A * vals[:, None]).sum(0) / np.maximum(A.sum(0), 1e-9)
    # nodes with no sensor wiring inherit the mean tuning of their inputs
    adj = rec["adj"].astype(float)
    fill = ~has_in
    if fill.any():
        num = adj[:, fill].T @ np.where(has_in, tuning, 0.0)
        den = adj[:, fill].T @ has_in.astype(float)
        second = np.where(den > 0, num / np.maximum(den, 1e-9),
                          rng.uniform(-90, 90, int(fill.sum())))
        tuning[fill] = second + rng.uniform(-4, 4, int(fill.sum()))
    order = np.argsort(tuning)
    n = rec["n_nodes"]
    ang = np.zeros(n)
    # ranks over the full circle: median tuning (~0 deg, toward the field)
    # lands at the right; +/-90 sweep through north/south
    ang[order] = np.linspace(-np.pi, np.pi, n, endpoint=False)
    rad = 1.0 + rng.uniform(-0.075, 0.075, n)
    pos = np.column_stack([rad * np.cos(ang), rad * np.sin(ang)])
    s_ang = np.radians(vals)                 # 0 deg = east = toward the field
    s_pos = np.column_stack([1.30 * np.cos(s_ang), 1.30 * np.sin(s_ang)])
    return pos, s_pos


def pick_trio(rec):
    """A chain a -> b -> c of active nodes, a with sensor input, preferring a
    chain whose middle link is inhibitory so the trace shows suppression."""
    act = rec["rates"].mean(1)
    good = np.argsort(np.abs(act - 0.15))
    adj = rec["adj"]
    has_in = rec["in_adj"].sum(0) > 0
    w_pos = np.unpackbits(rec["signs"][0])[: rec["n_edges"]].astype(bool)
    sign = {}
    for k in range(rec["n_edges"]):
        sign[(rec["e_src"][k], rec["e_dst"][k])] = w_pos[k]
    best = None
    for a in good[:120]:
        if not has_in[a] or act[a] < 0.02:
            continue
        for b in np.flatnonzero(adj[a])[:25]:
            if act[b] < 0.02 or act[b] > 0.7:
                continue
            for c in np.flatnonzero(adj[b])[:25]:
                if c == a or act[c] < 0.02 or act[c] > 0.7:
                    continue
                trio = (int(a), int(b), int(c))
                if best is None:
                    best = trio
                if not sign[(int(a), int(b))] or not sign[(int(b), int(c))]:
                    return trio              # has an inhibitory link
    if best is not None:
        return best
    return tuple(int(v) for v in good[:3])


def build_figure(rec, cfg, rng):
    fig = plt.figure(figsize=(9.6, 10.8), dpi=cfg.dpi)
    fig.patch.set_facecolor("white")
    art = {}

    pos, s_pos = layout_ring(rec, rng)
    art["pos"], art["s_pos"] = pos, s_pos
    trio = art["trio"] = rec["trio"]
    inh_pct = 100.0 * (1.0 - np.unpackbits(rec["signs"][0])[: rec["n_edges"]].mean())

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
             "the paddle's frame, field to the right · neurons arranged by the "
             f"ball direction they are wired to prefer · {inh_pct:.0f}% of "
             "synapses are inhibitory", fontsize=8.5, color=MUTED)

    legend = ((("●", "charging (activation)", NODE),
               ("●", "firing (rate this frame)", SPIKE)),
              (("—", "sensory input", AMBER),
               ("—", "excitatory spikes", EXC),
               ("—", "inhibitory spikes", INH)))
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
    art["chords_exc"] = LineCollection([], colors=INK, linewidths=0.3,
                                       alpha=0.05, zorder=1)
    ax.add_collection(art["chords_exc"])
    art["chords_inh"] = LineCollection([], colors=INH, linewidths=0.35,
                                       alpha=0.11, zorder=1.5)
    ax.add_collection(art["chords_inh"])
    art["eff_edges_up"] = LineCollection([], colors=GREEN, linewidths=0.8,
                                         alpha=0.45, zorder=3)
    art["eff_edges_dn"] = LineCollection([], colors=BLUE, linewidths=0.8,
                                         alpha=0.45, zorder=3)
    ax.add_collection(art["eff_edges_up"])
    ax.add_collection(art["eff_edges_dn"])

    art["sens"] = ax.scatter(s_pos[:, 0], s_pos[:, 1], s=26, c="#eef1f5",
                             edgecolors=EDGE, linewidths=0.5, zorder=4)
    art["nodes"] = ax.scatter(pos[:, 0], pos[:, 1], s=9, c="#f2f5f9",
                              edgecolors="none", zorder=5)
    for i, node in enumerate(trio):
        ax.add_patch(Circle(pos[node], 0.075, fill=False, color=SPIKE,
                            lw=1.3, zorder=6))
        ax.annotate(f"{chr(97 + i)}", pos[node] + np.array([0.09, 0.09]),
                    fontsize=9, color=SPIKE, fontweight="bold", zorder=7)

    # orientation: the field (and the ball) are to the RIGHT, as on the left half
    ax.annotate("", xy=(1.55, 0), xytext=(1.43, 0),
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.4))
    ax.text(1.49, 0.055, "toward\nthe field", fontsize=8, color=INK,
            fontweight="bold", ha="center", va="bottom")
    ax.text(0, 1.485, "up", fontsize=8.5, color=MUTED, ha="center", va="top")
    ax.text(0, -1.44, "down", fontsize=8.5, color=MUTED, ha="center",
            va="bottom")
    ax.text(-1.52, 0, "behind the paddle\n(no sensors)", fontsize=7.5,
            color=MUTED, ha="left", va="center")

    for key, (y0, label, col) in {"effU": (0.35, "move up", GREEN),
                                  "effD": (-0.35, "move down", BLUE)}.items():
        art[key] = Circle((0, y0), 0.14, facecolor="white", edgecolor=col,
                          lw=1.6, zorder=8)
        ax.add_patch(art[key])
        ax.text(0.19, y0, label, fontsize=8, color=col, ha="left",
                va="center", zorder=9, fontweight="bold")

    ax.text(0, -1.53, "input → reservoir → output", fontsize=8.5,
            color=MUTED, ha="center", transform=ax.transData, clip_on=False)

    # mini field glyph, top-right, to anchor against the left-half video
    W, H, PX, PH = rec["field"]
    axg = fig.add_axes([0.79, 0.792, 0.175, 0.0778])
    axg.set_xlim(-35, W + 35)
    axg.set_ylim(-40, H + 40)
    axg.set_aspect("equal")
    axg.axis("off")
    axg.add_patch(Rectangle((0, 0), W, H, fill=False, color=EDGE, lw=0.8))
    (art["g_pad"],) = axg.plot([], [], color=AGENT, lw=2.4,
                               solid_capstyle="round")
    (art["g_ball"],) = axg.plot([], [], "o", ms=3.5, color=GREEN)
    axg.set_title("the field", fontsize=7.5, color=MUTED, pad=2)

    art["clock"] = fig.text(0.965, 0.958, "", fontsize=11, color=MUTED,
                            ha="right", family="monospace")

    # ================= microscope =================
    head(0.05, 0.276, "Three neurons up close")
    fig.text(0.05, 0.262,
             "each neuron leaks, sums weighted spikes from its neighbors "
             "(inhibitory ones subtract), and fires when its activation "
             "crosses threshold — then subtracts it and starts again",
             fontsize=8.5, color=MUTED)

    axm = fig.add_axes([0.045, 0.032, 0.24, 0.21])
    axm.set_xlim(-1.15, 1.15)
    axm.set_ylim(-1.25, 1.30)
    axm.axis("off")
    tpos = {trio[0]: (-0.62, 0.72), trio[1]: (0.68, 0.18),
            trio[2]: (-0.45, -0.78)}
    art["tpos"] = tpos
    art["tcirc"] = {}
    art["tarrow"] = []
    adj = rec["adj"]
    w_pos = np.unpackbits(rec["signs"][0])[: rec["n_edges"]].astype(bool)
    sign = {(int(s), int(d)): bool(p) for s, d, p
            in zip(rec["e_src"], rec["e_dst"], w_pos)}
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
                exc = sign.get((s, t), True)
                p0, p1 = np.array(tpos[s]), np.array(tpos[t])
                d = (p1 - p0) / np.linalg.norm(p1 - p0)
                ar = FancyArrowPatch(
                    p0 + d * 0.33, p1 - d * 0.36,
                    arrowstyle="-|>" if exc else "|-|,widthA=0,widthB=0.35",
                    mutation_scale=13, color=EDGE, lw=1.4, zorder=4,
                    connectionstyle="arc3,rad=0.18")
                axm.add_patch(ar)
                art["tarrow"].append((s, t, ar, exc))
    p0 = np.array(tpos[trio[0]])
    art["sens_arrow"] = FancyArrowPatch(p0 + [-0.55, 0.42], p0 + [-0.24, 0.20],
                                        arrowstyle="-|>", mutation_scale=13,
                                        color=EDGE, lw=1.6, zorder=4)
    axm.add_patch(art["sens_arrow"])
    axm.text(*(p0 + [-0.52, 0.60]), "sensors", fontsize=7.5, color=AMBER,
             ha="left")
    axm.text(0, -1.22, "▸ excitatory      ⊣ inhibitory", fontsize=7.5,
             color=MUTED, ha="center")

    art["traces"] = []
    for i, node in enumerate(trio):
        axt = fig.add_axes([0.36, 0.187 - i * 0.076, 0.60, 0.060])
        axt.set_xlim(-TRACE_WIN, 0)
        axt.set_xticks([])
        axt.set_yticks([])
        for side in ("top", "right"):
            axt.spines[side].set_visible(False)
        (ln_x,) = axt.plot([], [], color=NODE, lw=0.8, zorder=3)
        (ln_t,) = axt.plot([], [], color=INK, lw=0.9, ls=(0, (3, 2)), zorder=2)
        sc = axt.scatter([], [], s=14, marker="|", color=SPIKE,
                         alpha=0.45, linewidths=0.7, zorder=4)
        axt.text(0.005, 0.80, f"{chr(97 + i)}", transform=axt.transAxes,
                 fontsize=9, fontweight="bold", color=SPIKE)
        art["traces"].append((axt, ln_x, ln_t, sc))
    art["traces"][-1][0].set_xlabel(
        f"last {TRACE_WIN} steps   ·   activation before reset · "
        "threshold (dashed) · spike", fontsize=7.5)
    return fig, art


def make_update(rec, art, cfg):
    pos, s_pos = art["pos"], art["s_pos"]
    trio = art["trio"]
    in_adj = rec["in_adj"]
    out_adj = rec["out_adj"]
    e_src, e_dst = rec["e_src"], rec["e_dst"]
    n_edges = rec["n_edges"]
    seg_all = np.stack([pos[e_src], pos[e_dst]], axis=1)
    x3, thr3, sp3 = rec["x3"], rec["thr3"], rec["sp3"]
    frame_end = rec["frame_end"]

    base = np.array(matplotlib.colors.to_rgb("#f2f5f9"))
    blue = np.array(matplotlib.colors.to_rgb(NODE))
    red = np.array(matplotlib.colors.to_rgb(SPIKE))
    up_c = (0.0, 0.35)
    dn_c = (0.0, -0.35)
    up_nodes = np.flatnonzero(out_adj[:, 0])
    dn_nodes = np.flatnonzero(out_adj[:, 1])

    W, H, PX, PH = rec["field"]

    def update(f):
        end = int(frame_end[f])
        art["clock"].set_text(f"step {end:,}")

        # node faces: charge toward blue, firing toward red
        charge = np.clip(rec["x_end"][:, f] / np.maximum(rec["thr_end"][:, f],
                                                         1e-9), 0, 1)
        faces = base + (blue - base) * (0.15 + 0.85 * np.clip(charge * 1.5, 0, 1))[:, None]
        rate = rec["rates_g"][:, f][:, None]
        faces = faces + (red - faces) * np.clip(rate * 1.6, 0, 1)
        art["nodes"].set_facecolors(faces)

        # sensory input
        on = rec["sensors"][f] > 0
        segs, cols = [], []
        for si in np.flatnonzero(on):
            for node in np.flatnonzero(in_adj[si]):
                segs.append([s_pos[si], pos[node]])
        art["in_edges"].set_segments(segs)
        art["sens"].set_facecolors(np.where(on[:, None],
                                            matplotlib.colors.to_rgba(AMBER),
                                            matplotlib.colors.to_rgba("#eef1f5")))

        # spike deliveries: only the busiest spikers this frame, so the
        # ring reads as traffic rather than as the full wiring diagram
        spikers = rec["rates"][:, f] > 0
        r_now = rec["rates"][:, f]
        if spikers.sum() > TOP_SPIKERS:
            cut = np.partition(r_now, -TOP_SPIKERS)[-TOP_SPIKERS]
            top = r_now >= cut
        else:
            top = spikers
        active = top[e_src]
        w_pos = np.unpackbits(rec["signs"][f])[:n_edges].astype(bool)
        for key, mask in (("chords_exc", active & w_pos),
                          ("chords_inh", active & ~w_pos)):
            idx = np.flatnonzero(mask)
            if len(idx) > MAX_CHORDS:
                idx = idx[:: len(idx) // MAX_CHORDS + 1]
            art[key].set_segments(seg_all[idx])

        # effector edges from spiking contributor nodes
        for key, nodes, center in (("eff_edges_up", up_nodes, up_c),
                                   ("eff_edges_dn", dn_nodes, dn_c)):
            src = nodes[spikers[nodes]]
            art[key].set_segments([[pos[s], center] for s in src])
        o = rec["outputs"][f]
        art["effU"].set_facecolor((*matplotlib.colors.to_rgb(GREEN),
                                   float(np.clip(o[0], 0, 1)) * 0.85))
        art["effD"].set_facecolor((*matplotlib.colors.to_rgb(BLUE),
                                   float(np.clip(o[1], 0, 1)) * 0.85))

        # mini field
        bx, by = rec["ball"][f]
        py = rec["paddle"][f]
        art["g_pad"].set_data([PX, PX], [py - PH, py + PH])
        art["g_ball"].set_data([bx], [by])

        # microscope glyph
        for i, node in enumerate(trio):
            ch = float(charge[node])
            fc = base + (blue - base) * (0.15 + 0.85 * ch)
            if rec["rates"][node, f] > 0:
                fc = fc + (red - fc) * min(1.0, rec["rates_g"][node, f] * 1.6)
            art["tcirc"][node].set_facecolor(fc)
        sp_now = {node: rec["rates"][node, f] > 0 for node in trio}
        for s, t, ar, exc in art["tarrow"]:
            ar.set_color((EXC if exc else INH) if sp_now[s] else EDGE)
        a = trio[0]
        art["sens_arrow"].set_color(
            AMBER if (rec["sensors"][f] * in_adj[:, a][: len(rec["sensors"][f])]
                      ).any() else EDGE)

        # traces
        lo = max(0, end - TRACE_WIN)
        xs = np.arange(lo - end, 0)
        for i, (axt, ln_x, ln_t, sc) in enumerate(art["traces"]):
            tv = thr3[lo:end, i]
            sp = sp3[lo:end, i]
            xv = x3[lo:end, i] + sp * tv
            ln_x.set_data(xs, xv)
            ln_t.set_data(xs, tv)
            top = max(float(tv.max()) * 1.25, float(xv.max()) * 1.08, 0.5)
            bot = min(0.0, float(xv.min()) * 1.08)
            wh = np.flatnonzero(sp)
            sc.set_offsets(np.column_stack([xs[wh], np.full(len(wh),
                                                            top * 0.96)])
                           if len(wh) else np.empty((0, 2)))
            axt.set_ylim(bot, top)
        return []

    return update


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loadout", type=str, default="pongEvo1",
                    choices=sorted(LOADOUT_BY_ID))
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--seconds", type=float, default=320.0)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--intro-seconds", type=float, default=30.0)
    ap.add_argument("--intro-sps", type=float, default=122.0)
    ap.add_argument("--main-sps", type=float, default=775.0)
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--out", type=str,
                    default="scripts/out/pong_reservoir.mp4")
    ap.add_argument("--still", type=int, default=None)
    cfg = ap.parse_args()
    cfg.frames = int(round(cfg.seconds * cfg.fps))

    plan = frame_plan(cfg)
    print(f"loadout {cfg.loadout} ('{LOADOUT_BY_ID[cfg.loadout]['label']}') "
          f"seed {cfg.seed}: {cfg.frames} frames, {plan.sum():,} steps", flush=True)
    rec = simulate(cfg, plan)
    rec["trio"] = pick_trio(rec)
    print(f"  trio: {rec['trio']}", flush=True)
    rec["x3"], rec["thr3"], rec["sp3"] = record_trio(cfg, plan, rec["trio"])

    rng = np.random.default_rng(7)
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

    t0 = time.perf_counter()

    def frame_iter():
        for f in range(cfg.frames):
            if f and f % 500 == 0:
                el = time.perf_counter() - t0
                print(f"  frame {f}/{cfg.frames} ({el/f:.3f} s/frame, "
                      f"eta {(cfg.frames-f)*el/f/60:.0f} min)", flush=True)
            yield f

    anim = animation.FuncAnimation(fig, update, frames=frame_iter(),
                                   save_count=cfg.frames, blit=False)
    writer = animation.FFMpegWriter(fps=cfg.fps, bitrate=6000,
                                    extra_args=["-pix_fmt", "yuv420p"])
    anim.save(out.as_posix(), writer=writer)
    print(f"saved {out} ({cfg.frames} frames @ {cfg.fps} fps)", flush=True)


if __name__ == "__main__":
    main()
