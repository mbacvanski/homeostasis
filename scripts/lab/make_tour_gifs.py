"""Animated GIFs for docs/field_guide.md — every frame from the real model."""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Wedge

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "lab"))
OUT = ROOT / "docs" / "assets"
LAB = ROOT / "scripts" / "out" / "lab"

from common import make_configs  # noqa: E402
from homeostasis.reservoir import HomeostaticReservoir, ReservoirConfig  # noqa: E402
from homeostasis.tracking import TrackingEnv  # noqa: E402

INK = "#1C2025"; ACC = "#0E6E63"; WARM = "#C0562F"; GRAY = "#B9BFC6"


def gif(fig, update, frames, name, fps=18):
    anim = FuncAnimation(fig, update, frames=frames, blit=False)
    anim.save(OUT / name, writer=PillowWriter(fps=fps), dpi=72)
    plt.close(fig)
    print("saved", name)


# ── tracking triptych: statue / good / jittery, side by side ────────────
def run_tracking(wlr, seed, n):
    rcfg, tcfg = make_configs({"weight_lr": wlr, "target_lr": 0.01}, {})
    net = HomeostaticReservoir(rcfg, seed=seed)
    env = TrackingEnv(tcfg)
    H = np.zeros((n, 2))
    A = np.zeros((n, 62))
    for i in range(n):
        acts = env.sense()
        st = net.step(acts)
        env.apply_action(*st.outputs)
        env.advance_stimulus()
        H[i] = (env.heading, env.stimulus_angle)
        A[i] = acts
    return H, A


def tracking_gif():
    n, step, lo = 8200, 11, 5000
    runs = [("no learning at all:\nfrozen in place — the light\nsweeps past, ignored", run_tracking(0.0, 0, n)),
            ("learning speed just right:\nit follows the light", run_tracking(0.1, 0, n)),
            ("learns too fast:\nits own churn drowns the light", run_tracking(1.0, 2, n))]
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.6))
    artists = []
    for ax, (title, _) in zip(axes, runs):
        ax.set_xlim(-1.45, 1.45); ax.set_ylim(-1.45, 1.45)
        ax.set_aspect("equal"); ax.axis("off")
        th = np.linspace(0, 2 * np.pi, 100)
        ax.plot(np.cos(th), np.sin(th), color=GRAY, lw=1)
        wedge = Wedge((0, 0), 0.92, 0, 1, color=ACC, alpha=0.13)
        ax.add_patch(wedge)
        arrow, = ax.plot([], [], color=INK, lw=2.5)
        dot, = ax.plot([], [], "o", color=WARM, ms=11)
        ax.set_title(title, fontsize=9.5)
        artists.append((wedge, arrow, dot))
    fig.suptitle("Same brain design, three learning speeds  ·  arrow = which way the agent faces  ·  dot = the light", fontsize=10)

    def update(f):
        i = lo + f * step
        for (title, (H, _)), (wedge, arrow, dot) in zip(runs, artists):
            hd, st = H[i]
            wedge.set_theta1(hd - 92); wedge.set_theta2(hd + 92)
            r = np.deg2rad(hd)
            arrow.set_data([0, 0.55 * np.cos(r)], [0, 0.55 * np.sin(r)])
            s = np.deg2rad(st)
            dot.set_data([np.cos(s)], [np.sin(s)])
        return []
    gif(fig, update, (7200 - lo) // step, "tour_tracking.gif")


# ── the agent's view + arena (single, with retina strip) ────────────────
def tracking_view_gif():
    n, step, lo = 8200, 10, 5000
    H, A = run_tracking(0.1, 0, n)
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(4.6, 5.6),
                                  height_ratios=[3, 1])
    ax.set_xlim(-1.45, 1.45); ax.set_ylim(-1.45, 1.45)
    ax.set_aspect("equal"); ax.axis("off")
    th = np.linspace(0, 2 * np.pi, 100)
    ax.plot(np.cos(th), np.sin(th), color=GRAY, lw=1)
    wedge = Wedge((0, 0), 0.92, 0, 1, color=ACC, alpha=0.13)
    ax.add_patch(wedge)
    arrow, = ax.plot([], [], color=INK, lw=2.5)
    dot, = ax.plot([], [], "o", color=WARM, ms=11)
    ax.set_title("the arena (bird's-eye view)", fontsize=10)
    offs = np.sort(np.concatenate([e + (np.arange(31) - 15) * 4.0 for e in (30., -30.)]))
    order = np.argsort(np.concatenate([e + (np.arange(31) - 15) * 4.0 for e in (30., -30.)]))
    bars = ax2.bar(offs, np.zeros(62), width=3.4, color=ACC)
    ax2.set_ylim(0, 1.1); ax2.set_xlabel("what its sensors report", fontsize=9)
    ax2.set_yticks([])
    for s_ in ("top", "right", "left"):
        ax2.spines[s_].set_visible(False)

    def update(f):
        i = lo + f * step
        hd, st = H[i]
        wedge.set_theta1(hd - 92); wedge.set_theta2(hd + 92)
        r = np.deg2rad(hd)
        arrow.set_data([0, 0.55 * np.cos(r)], [0, 0.55 * np.sin(r)])
        s = np.deg2rad(st)
        dot.set_data([np.cos(s)], [np.sin(s)])
        acts = A[i][order]
        for b, v in zip(bars, acts):
            b.set_height(v)
        return []
    gif(fig, update, (6600 - lo) // step, "tour_tracking_view.gif")


# ── wall avoider ────────────────────────────────────────────────────────
def wall_gif():
    from homeostasis.simulation import run_wall
    h = run_wall(n_steps=3600, seed=1)
    step = 12
    fig, ax = plt.subplots(figsize=(4.4, 4.4))
    ax.set_xlim(-0.5, 15.5); ax.set_ylim(-0.5, 15.5)
    ax.set_aspect("equal"); ax.axis("off")
    ax.plot([0, 15, 15, 0, 0], [0, 0, 15, 15, 0], color=INK, lw=2)
    trail, = ax.plot([], [], color=ACC, lw=1, alpha=0.6)
    dot, = ax.plot([], [], "o", color=ACC, ms=9)
    txt = ax.set_title("", fontsize=10)

    def update(f):
        i = f * step
        lo = max(0, i - 500)
        trail.set_data(h.x[lo:i + 1], h.y[lo:i + 1])
        dot.set_data([h.x[i]], [h.y[i]])
        hits = int(h.hit[:i + 1].sum())
        txt.set_text(f"step {i}   wall bumps so far: {hits}")
        return []
    gif(fig, update, 3600 // step, "tour_wall.gif")


# ── the perfect pursuer ─────────────────────────────────────────────────
def pursuit_gif():
    from homeostasis.simulation import run_pursuit
    from homeostasis.pursuit import PursuitConfig
    champ = json.loads((LAB / "h34_joint.json").read_text())[-1]
    g, seed = champ["champion"], champ["champ_seed"]
    pc = PursuitConfig(eye_offsets=(0.0,), sensors_per_eye=91,
                       wheel_base=g["wheel_base"], intensity_scale=g["intensity_scale"])
    res_keys = ("n_nodes", "p_link", "input_weight", "weight_init_mean",
                "leak", "target_lr", "threshold_ratio", "weight_lr")
    res = ReservoirConfig(n_inputs=pc.n_sensors, **{k: g[k] for k in res_keys})
    h = run_pursuit(n_steps=2600, seed=seed, reservoir_config=res, pursuit_config=pc)
    step = 8
    fig, ax = plt.subplots(figsize=(4.4, 4.4))
    ax.set_xlim(-0.5, 15.5); ax.set_ylim(-0.5, 15.5)
    ax.set_aspect("equal"); ax.axis("off")
    ax.plot([0, 15, 15, 0, 0], [0, 0, 15, 15, 0], color=INK, lw=2)
    ttrail, = ax.plot([], [], color=WARM, lw=1, alpha=0.5)
    atrail, = ax.plot([], [], color=ACC, lw=1, alpha=0.5)
    tdot, = ax.plot([], [], "o", color=WARM, ms=9, label="the target")
    adot, = ax.plot([], [], "o", color=ACC, ms=9, label="the chaser")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title("it doesn't chase — it orbits alongside its target", fontsize=9.5)

    def update(f):
        i = f * step
        lo = max(0, i - 260)
        ttrail.set_data(h.sx[lo:i + 1], h.sy[lo:i + 1])
        atrail.set_data(h.x[lo:i + 1], h.y[lo:i + 1])
        tdot.set_data([h.sx[i]], [h.sy[i]])
        adot.set_data([h.x[i]], [h.y[i]])
        return []
    gif(fig, update, 2600 // step, "tour_pursuit.gif")


if __name__ == "__main__":
    tracking_gif()
    tracking_view_gif()
    wall_gif()
    pursuit_gif()


# ── the four-agent chain ────────────────────────────────────────────────
def chain_gif():
    from h50_depth import CHAIN_FILE, PACE_CFG, PACE_SEED, START_Y, make_follower
    from homeostasis.simulation import WallSimulation
    chain = [(g, s_) for g, s_ in json.loads(CHAIN_FILE.read_text())["chain"]]
    A = WallSimulation(wall_config=PACE_CFG, seed=PACE_SEED)
    links = [make_follower(g, s_, START_Y[i]) for i, (g, s_) in enumerate(chain)]
    n = 8000
    P = np.zeros((n, 4, 2))
    for i in range(n):
        A.step()
        tx, ty = A.env.x, A.env.y
        P[i, 0] = (tx, ty)
        for j, (net, env) in enumerate(links):
            env.sx, env.sy = tx, ty
            st = net.step(env.sense())
            env.apply_action(*map(float, st.outputs)); env.steps += 1
            tx, ty = env.x, env.y
            P[i, 1 + j] = (env.x, env.y)
    lo, step = 7200, 4
    fig, ax = plt.subplots(figsize=(4.8, 4.8))
    ax.set_xlim(-0.8, 30.8); ax.set_ylim(-0.8, 30.8)
    ax.set_aspect("equal"); ax.axis("off")
    ax.plot([0, 30, 30, 0, 0], [0, 0, 30, 30, 0], color=INK, lw=2)
    names = ["A (blind leader)", "B follows A", "C follows B", "D follows C"]
    cols = [WARM, "#2A6FB0", "#3E8E5A", "#7B5AA6"]
    trails = [ax.plot([], [], color=c, lw=1, alpha=0.5)[0] for c in cols]
    dots = [ax.plot([], [], "o", color=c, ms=8, label=nm)[0]
            for c, nm in zip(cols, names)]
    ax.legend(loc="lower left", fontsize=7.5)
    ax.set_title("a conga line of comfort-seekers, all circling one point", fontsize=9.5)

    def update(f):
        i = lo + f * step
        s0 = max(lo, i - 220)
        for j in range(4):
            trails[j].set_data(P[s0:i + 1, j, 0], P[s0:i + 1, j, 1])
            dots[j].set_data([P[i, j, 0]], [P[i, j, 1]])
        return []
    gif(fig, update, (n - lo) // step, "tour_chain.gif")


# ── budding cascade ─────────────────────────────────────────────────────
def budding_gif():
    import h97_reproduce as H
    from homeostasis.simulation import WallSimulation
    H.INHERIT_WIRING = True; H.CLONE_TEST = True; H.BUDDING = True
    rng = np.random.default_rng(971)
    A = WallSimulation(wall_config=H.PACE_CFG, seed=H.PACE_SEED)
    agents = [H.Agent(dict(H.CHAMP["champion"]), H.CHAMP["champ_seed"], (15.0, 10.0))]
    n = 12000
    pos = np.full((n, H.CAP, 2), np.nan)
    pa = np.zeros((n, 2))
    counts = np.zeros(n, dtype=int)
    births = []
    for i in range(n):
        A.step()
        target = (A.env.x, A.env.y)
        pa[i] = target
        for j, ag in enumerate(agents):
            ag.step(target)
            pos[i, j] = (ag.env.x, ag.env.y)
        counts[i] = len(agents)
        if len(agents) < H.CAP:
            for ag in list(agents):
                if ag.lock_streak >= H.SPAWN_AFTER and ag.spawned == 0 and len(agents) < H.CAP:
                    child = H.Agent(dict(ag.genome), H.CHAMP["champ_seed"],
                                    (ag.env.x, ag.env.y))
                    child.net.x = ag.net.x.copy()
                    child.net.targets = ag.net.targets.copy()
                    child.net.weights = ag.net.weights.copy()
                    child.net.spiked = ag.net.spiked.copy()
                    child.net._spiked_f = ag.net._spiked_f.copy()
                    child.env.heading = ag.env.heading
                    agents.append(child)
                    ag.spawned = 1; ag.lock_streak = 0
                    births.append(i)
    step = 40
    fig, ax = plt.subplots(figsize=(4.8, 4.8))
    ax.set_xlim(-0.8, 30.8); ax.set_ylim(-0.8, 30.8)
    ax.set_aspect("equal"); ax.axis("off")
    ax.plot([0, 30, 30, 0, 0], [0, 0, 30, 30, 0], color=INK, lw=2)
    ldot, = ax.plot([], [], "o", color=WARM, ms=8, label="the blind leader")
    ftrail, = ax.plot([], [], color=ACC, lw=1, alpha=0.5)
    rings = [ax.plot([], [], "o", mfc="none", mec=ACC, ms=8 + 4 * k, mew=1.6)[0]
             for k in range(6)]
    ax.legend(loc="lower left", fontsize=8)
    title = ax.set_title("", fontsize=10)

    def update(f):
        i = f * step
        ldot.set_data([pa[i, 0]], [pa[i, 1]])
        s0 = max(0, i - 300)
        ftrail.set_data(pos[s0:i + 1, 0, 0], pos[s0:i + 1, 0, 1])
        k = counts[i]
        for r_i, ring in enumerate(rings):
            if r_i < k:
                ring.set_data([pos[i, 0, 0]], [pos[i, 0, 1]])
            else:
                ring.set_data([], [])
        flash = any(abs(i - b) < 200 for b in births)
        title.set_text(f"family size: {k} of 6" + ("   ✳ a bud is born!" if flash else ""))
        return []
    gif(fig, update, n // step, "tour_budding.gif", fps=14)


# ── the seduction ───────────────────────────────────────────────────────
def seduction_gif():
    import h85_shared as HS
    from homeostasis.simulation import WallSimulation
    A = WallSimulation(wall_config=HS.PACE_CFG, seed=3)
    B = HS.StickyFollower(HS.CHAMP["champ_seed"], (15.0, 10.0), 2, True,
                          ratio=2.0, patience=100)
    C = HS.StickyFollower(HS.CHAMP["champ_seed"], (15.0, 8.0), 2, True,
                          ratio=2.0, patience=100)
    n = 6000
    P = np.zeros((n, 3, 2))
    selB = np.zeros(n, dtype=int)
    posB, posC = (15.0, 10.0), (15.0, 8.0)
    for i in range(n):
        A.step()
        pa = (A.env.x, A.env.y)
        newB = B.step([pa, posC], i)
        newC = C.step([pa, posB], i)
        posB, posC = newB, newC
        P[i] = [pa, posB, posC]
        selB[i] = B.sel
    step = 12
    fig, ax = plt.subplots(figsize=(4.8, 4.8))
    ax.set_xlim(-0.8, 30.8); ax.set_ylim(-0.8, 30.8)
    ax.set_aspect("equal"); ax.axis("off")
    ax.plot([0, 30, 30, 0, 0], [0, 0, 30, 30, 0], color=INK, lw=2)
    cols = [WARM, "#2A6FB0", "#3E8E5A"]
    names = ["A (blind leader)", "B (should follow A)", "C (follows B)"]
    trails = [ax.plot([], [], color=c, lw=1, alpha=0.5)[0] for c in cols]
    dots = [ax.plot([], [], "o", color=c, ms=8, label=nm)[0]
            for c, nm in zip(cols, names)]
    gaze, = ax.plot([], [], ls=":", color=INK, lw=1.4)
    ax.legend(loc="lower left", fontsize=7.5)
    title = ax.set_title("", fontsize=9.5)

    def update(f):
        i = f * step
        s0 = max(0, i - 220)
        for j in range(3):
            trails[j].set_data(P[s0:i + 1, j, 0], P[s0:i + 1, j, 1])
            dots[j].set_data([P[i, j, 0]], [P[i, j, 1]])
        tgt = P[i, 0] if selB[i] == 0 else P[i, 2]
        gaze.set_data([P[i, 1, 0], tgt[0]], [P[i, 1, 1], tgt[1]])
        title.set_text("B now watches C — seduced by its follower!" if selB[i] == 1
                       else "dotted line = whom B watches")
        return []
    gif(fig, update, n // step, "tour_seduction.gif")


# ── ellipse follower vs toll-booth ──────────────────────────────────────
def ellipse_gif():
    from homeostasis.simulation import run_pursuit
    from homeostasis.pursuit import PursuitConfig
    res_keys = ("n_nodes", "p_link", "input_weight", "weight_init_mean",
                "leak", "target_lr", "threshold_ratio", "weight_lr")
    follower = json.loads((LAB / "h38c_interp.json").read_text())[-1]
    booth = json.loads((LAB / "h38_manifold.json").read_text())["ellipse"][-1]
    runs = []
    for champ, a, b, label in (
            (follower, 4.5, 4.0, "gently squashed circle:\na true follower keeps pace"),
            (booth, 5.0, 2.5, "strongly squashed circle:\nbreeding gave up on following — it just\nloiters mid-court as the target swings past")):
        g, seed = champ["champion"], champ["champ_seed"]
        pc = PursuitConfig(eye_offsets=(0.0,), sensors_per_eye=91,
                           stimulus_motion="ellipse", ellipse_a=a, ellipse_b=b,
                           wheel_base=g["wheel_base"],
                           intensity_scale=g["intensity_scale"])
        res = ReservoirConfig(n_inputs=pc.n_sensors, **{k: g[k] for k in res_keys})
        h = run_pursuit(n_steps=2600, seed=seed, reservoir_config=res,
                        pursuit_config=pc)
        runs.append((label, h))
    step = 8
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.6))
    artists = []
    for ax, (label, h) in zip(axes, runs):
        ax.set_xlim(-0.5, 15.5); ax.set_ylim(-0.5, 15.5)
        ax.set_aspect("equal"); ax.axis("off")
        ax.plot([0, 15, 15, 0, 0], [0, 0, 15, 15, 0], color=INK, lw=2)
        tt, = ax.plot([], [], color=WARM, lw=1, alpha=0.5)
        at, = ax.plot([], [], color=ACC, lw=1, alpha=0.5)
        td, = ax.plot([], [], "o", color=WARM, ms=9)
        ad, = ax.plot([], [], "o", color=ACC, ms=9)
        ax.set_title(label, fontsize=9.5)
        artists.append((h, tt, at, td, ad))
    fig.suptitle("orange = the target   ·   teal = the bred chaser", fontsize=10)

    def update(f):
        i = f * step
        for h, tt, at, td, ad in artists:
            lo = max(0, i - 260)
            tt.set_data(h.sx[lo:i + 1], h.sy[lo:i + 1])
            at.set_data(h.x[lo:i + 1], h.y[lo:i + 1])
            td.set_data([h.sx[i]], [h.sy[i]])
            ad.set_data([h.x[i]], [h.y[i]])
        return []
    gif(fig, update, 2600 // step, "tour_ellipse.gif")
