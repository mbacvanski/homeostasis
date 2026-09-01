"""Render the depth-4 chain (A + B + C + D), four concentric rings."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from h50_depth import make_follower, START_Y, LAB
from h48c_live_chain import PACE_CFG, PACE_SEED
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from homeostasis.simulation import WallSimulation  # noqa: E402

def main():
    chain = [(g, s) for g, s in json.loads((LAB / "h50_chain.json").read_text())["chain"]]
    A = WallSimulation(wall_config=PACE_CFG, seed=PACE_SEED)
    links = [make_follower(g, s, START_Y[i]) for i, (g, s) in enumerate(chain)]
    n = 10800
    tr = [([], []) for _ in range(len(links) + 1)]
    for i in range(n):
        A.step()
        tx, ty = A.env.x, A.env.y
        if i % 3 == 0:
            tr[0][0].append(tx); tr[0][1].append(ty)
        for j, (net, env) in enumerate(links):
            env.sx, env.sy = tx, ty
            st = net.step(env.sense())
            env.apply_action(*map(float, st.outputs)); env.steps += 1
            tx, ty = env.x, env.y
            if i % 3 == 0:
                tr[j + 1][0].append(env.x); tr[j + 1][1].append(env.y)
    half = len(tr[0][0]) // 2
    fig, ax = plt.subplots(figsize=(7, 7))
    styles = [("tab:red", 2.2, "A pacemaker (blind wall-avoider)"),
              ("tab:blue", 1.2, "B follows A"),
              ("tab:green", 0.9, "C follows B"),
              ("tab:purple", 0.7, "D follows C")]
    for (xs, ys), (c, lw, lab) in zip(tr, styles):
        ax.plot(xs[half:], ys[half:], "-", lw=lw, color=c, label=lab)
    ax.set_xlim(0, 30); ax.set_ylim(0, 30); ax.set_aspect("equal")
    ax.legend(fontsize=8, loc="lower left")
    ax.set_title("The depth-4 entrainment chain (late half):\n"
                 "one blind pacemaker, three phase-locked followers")
    fig.tight_layout(); fig.savefig(LAB / "fig_chain4.png", dpi=130)
    print("wrote fig_chain4.png")

if __name__ == "__main__":
    main()
