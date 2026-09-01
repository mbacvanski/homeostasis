"""Illustrations for docs/field_guide.md — diagrams from real geometry/data."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle, Circle, Wedge, Polygon

ROOT = Path(__file__).resolve().parents[2]
LAB = ROOT / "scripts" / "out" / "lab"
OUT = ROOT / "docs" / "assets"
OUT.mkdir(exist_ok=True)

INK = "#1C2025"
ACC = "#0E6E63"
WARM = "#C0562F"
GRAY = "#8A9199"


def save(fig, name):
    fig.savefig(OUT / name, dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", name)


# ── 1. the unit as a leaky bucket ────────────────────────────────────────
def unit_diagram():
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 6.4); ax.axis("off")
    # bucket
    bx, by, bw, bh = 4.0, 1.2, 2.2, 3.4
    ax.add_patch(Polygon([[bx, by + bh], [bx + 0.25, by], [bx + bw - 0.25, by],
                          [bx + bw, by + bh]], closed=False, fill=False,
                         lw=2.5, edgecolor=INK))
    # water
    level = 1.9
    ax.add_patch(Polygon([[bx + 0.14, by + level], [bx + 0.21, by + 0.06],
                          [bx + bw - 0.21, by + 0.06], [bx + bw - 0.14, by + level]],
                         closed=True, color="#BFD9F2"))
    ax.text(bx + bw / 2, by + level - 0.45, "charge", ha="center", fontsize=11,
            color="#39627F", style="italic")
    # threshold line
    ax.plot([bx - 0.15, bx + bw + 0.15], [by + 2.7, by + 2.7], ls="--", c=WARM, lw=2)
    ax.plot([bx + bw + 0.15, 6.75], [by + 2.7, 2.75], c=WARM, lw=0.8, alpha=0.6)
    ax.text(6.55, 2.6, "the firing line: reach this\nand the unit FIRES —\nempties a gulp, pings\nits neighbours",
            va="top", ha="left", fontsize=9.5, color=WARM)
    # leak
    ax.add_patch(FancyArrowPatch((bx + bw / 2, by - 0.05), (bx + bw / 2, by - 0.9),
                                 arrowstyle="-|>", mutation_scale=16, color="#39627F"))
    ax.text(bx + bw / 2 + 0.15, by - 0.55, "the leak: charge constantly\ndrains away (some units leak\nfast, some slowly)",
            fontsize=9.5, va="center", color="#39627F")
    # inputs
    for i, (yy, lbl) in enumerate([(5.6, "ping from neighbour A"),
                                   (4.9, "ping from neighbour B"),
                                   (4.2, "signal from a light sensor")]):
        ax.add_patch(FancyArrowPatch((1.2, yy), (bx + 0.5 + 0.35 * i, by + bh + 0.05),
                                     arrowstyle="-|>", mutation_scale=13,
                                     color=ACC, lw=1.6,
                                     connectionstyle="arc3,rad=-0.15"))
        ax.text(1.1, yy, lbl, ha="right", fontsize=9.5, color=ACC)
    # comfort dial
    ax.add_patch(Circle((8.7, 5.45), 0.62, fill=False, lw=2, edgecolor=INK))
    ax.plot([8.7, 8.99], [5.45, 5.93], c=INK, lw=2)
    ax.text(8.7, 4.6, 'the comfort dial\n("how full do I like to be?")\nthe unit slowly adjusts this\ndial toward its usual charge',
            ha="center", va="top", fontsize=9.5)
    ax.set_title("One unit = a leaky bucket that wants to feel 'just right'",
                 fontsize=13, pad=12)
    save(fig, "tour_unit.png")


# ── 2. how the tracking agent sees (retina bump) ────────────────────────
def retina_diagram():
    offs = np.concatenate([eye + (np.arange(31) - 15) * 4.0 for eye in (30.0, -30.0)])
    offs = np.sort(offs)
    stim = 18.0
    d = np.abs((stim - offs + 180) % 360 - 180)
    act = np.exp(-(d ** 2) / 10.0)
    act[d <= 4.0] = 1.0
    fig, ax = plt.subplots(figsize=(7.6, 2.9))
    ax.bar(offs, act, width=3.4, color=ACC)
    ax.axvline(stim, color=WARM, lw=2)
    ax.text(stim + 3, 1.02, "the light is HERE\n(18° to the agent's left)", color=WARM, fontsize=9.5)
    ax.set_xlabel("direction, in degrees (0 = straight ahead; the 62 sensors cover ±92°)")
    ax.set_ylabel("how strongly each\nsensor responds")
    ax.set_ylim(0, 1.35)
    ax.set_title("What the agent 'sees': a hill of activity over its strip of light sensors", fontsize=12)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    save(fig, "tour_retina.png")


# ── 3. Pong court schematic ─────────────────────────────────────────────
def pong_diagram():
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    ax.set_xlim(-90, 1060); ax.set_ylim(-40, 560); ax.axis("off")
    ax.add_patch(Rectangle((0, 0), 1000, 500, fill=False, lw=2, edgecolor=INK))
    ax.add_patch(Rectangle((95, 200), 10, 100, color=ACC))
    ax.text(100, 175, "the paddle\n(moves up/down only)", ha="center", va="top",
            fontsize=9.5, color=ACC)
    bx, by = 560, 330
    ax.add_patch(Circle((bx, by), 9, color=WARM))
    ax.add_patch(FancyArrowPatch((bx, by), (bx - 90, by - 90), arrowstyle="-|>",
                                 mutation_scale=14, color=WARM))
    ax.text(bx + 14, by + 14, "the ball (bounces off the top,\nbottom and right walls)",
            fontsize=9.5, color=WARM)
    for ang in np.linspace(-80, 80, 9):
        a = np.deg2rad(ang)
        ax.plot([100, 100 + 130 * np.cos(a)], [250, 250 + 130 * np.sin(a)],
                color=GRAY, lw=0.8)
    ax.text(290, 60, "46 'angle sensors': each fires when the\nball sits at its particular angle from the\npaddle (the paddle is blind behind itself)",
            fontsize=9.5, color=GRAY)
    ax.plot([0, 0], [0, 500], color=WARM, lw=3)
    ax.text(-25, 250, "if the ball gets past → a miss;\nit is served again from the right",
            rotation=90, va="center", ha="center", fontsize=9, color=WARM)
    ax.set_title("World 2 — Pong: keep hitting the ball back", fontsize=13)
    save(fig, "tour_pong.png")


# ── 4. looming ramp vs loitering (real flow traces) ─────────────────────
def ramp_diagram():
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from homeostasis.pursuit import PursuitConfig, PursuitEnv
    rng = np.random.default_rng(7)
    # ballistic crossing, agent parked mid-box: brightness over one crossing
    env = PursuitEnv(PursuitConfig(stimulus_motion="ballistic", stimulus_speed=0.04),
                     rng=np.random.default_rng(3))
    env.x, env.y = 7.5, 7.5
    f_b = []
    for _ in range(900):
        env.advance_stimulus()
        f_b.append(env.sense().sum())
    env2 = PursuitEnv(PursuitConfig(stimulus_motion="waypoint", stimulus_speed=0.04),
                      rng=np.random.default_rng(3))
    env2.x, env2.y = 7.5, 7.5
    f_w = []
    for _ in range(900):
        env2.advance_stimulus()
        f_w.append(env2.sense().sum())
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 2.9), sharey=True)
    axes[0].plot(f_b[:260], color=ACC)
    axes[0].set_title('A target flying straight past:\none smooth swell — rises, peaks, fades.\nThat rise is a "looming ramp" it can climb.', fontsize=10)
    axes[1].plot(f_w, color=WARM)
    axes[1].set_title('A target milling about nearby:\nbrightness lurches and vanishes without\npattern — no steady swell to ride', fontsize=10)
    for ax in axes:
        ax.set_xlabel("time (steps)")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("total brightness\nthe agent sees")
    save(fig, "tour_ramp.png")


# ── 5. the followable-speed curve ───────────────────────────────────────
def band_diagram():
    band = json.load(open(LAB / "h75_band.json"))
    xs = sorted(float(k) for k in band)
    ys = [band[str(x)] for x in xs]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.plot(xs, ys, "o-", color=ACC, lw=2)
    ax.axvspan(0.10, 0.26, color=ACC, alpha=0.08)
    ax.text(0.18, 0.28, "the comfortable range:\nfollowed almost perfectly", ha="center",
            fontsize=9.5, color=ACC)
    ax.text(0.062, 0.66, "too slow →\nnothing to latch onto", fontsize=9, ha="left")
    ax.text(0.298, 0.55, "too fast →\ncan't keep up", fontsize=9, ha="right")
    ax.set_xlabel("how fast the leader moves (arena units per step)")
    ax.set_ylabel("how well the best possible\nfollower stays with it (1 = perfect)")
    ax.set_ylim(0, 1.1)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title("Followers only work for leaders moving at a comfortable speed", fontsize=11.5)
    save(fig, "tour_band.png")


# ── 6. the attention ladder ─────────────────────────────────────────────
def attention_diagram():
    fig, ax = plt.subplots(figsize=(7.4, 3.4))
    labels = ["no filter:\nsees both lights\nblended together",
              "flickery filter:\nattends whichever is\nbrighter each instant",
              "sticky filter:\npicks one light\nand stays on it"]
    vals = [0.034, 0.074, 1.000]
    colors = [GRAY, GRAY, ACC]
    bars = ax.bar(range(3), vals, color=colors, width=0.55)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.03, f"{v:.2f}", ha="center", fontsize=11)
    ax.set_xticks(range(3)); ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_ylabel("how well it follows its\nleader (1 = perfect)")
    ax.set_ylim(0, 1.15)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title("With two lights in view, only patient attention saves the follower", fontsize=12)
    save(fig, "tour_attention.png")


# ── 7. the ridge, plainly labeled ───────────────────────────────────────
def ridge_diagram():
    rows = json.load(open(LAB / "cluster1_results.json"))
    from collections import defaultdict
    cells = defaultdict(list)
    for r in rows:
        if set(r["res"].keys()) != {"leak", "weight_lr"}:
            continue
        cells[(r["res"]["leak"], r["res"]["weight_lr"])].append(r["score_late"])
    leaks = sorted({k[0] for k in cells})
    wlrs = sorted({k[1] for k in cells})
    M = np.array([[np.mean(cells[(lk, w)]) if (lk, w) in cells else np.nan
                   for w in wlrs] for lk in leaks])
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    im = ax.imshow(M, origin="lower", aspect="auto", cmap="viridis",
                   vmin=0.25, vmax=0.55)
    ax.set_xticks(range(len(wlrs))); ax.set_xticklabels(wlrs, fontsize=8)
    ax.set_yticks(range(len(leaks))); ax.set_yticklabels(leaks, fontsize=8)
    ax.set_xlabel("LEARNING speed (how hard each surprise tugs the wiring)")
    ax.set_ylabel("FORGETTING speed\n(how fast charge drains)")
    best = [int(np.nanargmax(M[i])) for i in range(len(leaks))]
    ax.plot(best, range(len(leaks)), "o-", color="white", lw=2, ms=5)
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("how well it follows the light\n(0.25 = pure chance)")
    ax.set_title("The ridge: for every forgetting speed there is a matching learning speed\n(white line = the crest; 4,800 runs)", fontsize=11)
    save(fig, "tour_ridge.png")


if __name__ == "__main__":
    unit_diagram()
    retina_diagram()
    pong_diagram()
    ramp_diagram()
    band_diagram()
    attention_diagram()
    ridge_diagram()
