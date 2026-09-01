"""Figure: the budding cascade (H97d) — population and locks over time."""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import h97_reproduce as H  # noqa: E402
from homeostasis.simulation import WallSimulation  # noqa: E402

def record(n=21600, lineage_seed=971):
    """Mirror H.run(reproduce=True) with BUDDING globals, recording series."""
    H.INHERIT_WIRING = True
    H.CLONE_TEST = True
    H.BUDDING = True
    rng = np.random.default_rng(lineage_seed)
    A = WallSimulation(wall_config=H.PACE_CFG, seed=H.PACE_SEED)
    agents = [H.Agent(dict(H.CHAMP["champion"]), H.CHAMP["champ_seed"], (15.0, 10.0))]
    D = np.full((n, H.CAP), np.nan)
    births = []
    for i in range(n):
        A.step()
        target = (A.env.x, A.env.y)
        for j, ag in enumerate(agents):
            D[i, j] = ag.step(target)
        if len(agents) < H.CAP:
            for ag in list(agents):
                if ag.lock_streak >= H.SPAWN_AFTER and ag.spawned == 0 and len(agents) < H.CAP:
                    child_g = dict(ag.genome)
                    child = H.Agent(child_g, H.CHAMP["champ_seed"], (ag.env.x, ag.env.y))
                    child.net.x = ag.net.x.copy()
                    child.net.targets = ag.net.targets.copy()
                    child.net.weights = ag.net.weights.copy()
                    child.net.spiked = ag.net.spiked.copy()
                    child.net._spiked_f = ag.net._spiked_f.copy()
                    child.env.heading = ag.env.heading
                    agents.append(child)
                    ag.spawned = 1
                    ag.lock_streak = 0
                    births.append(i)
    late = D[-3600:]
    locked = int(sum(bool(np.nanmean(late[:, j] < H.LOCK_D) >= 0.8)
                     for j in range(H.CAP) if not np.all(np.isnan(late[:, j]))))
    return D, births, len(agents), locked

def main():
    D, births, n_agents, locked = record()
    assert (n_agents, locked) == (6, 6), f"mirror mismatch: {n_agents}, {locked}"
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 5.4), sharex=True,
                                   height_ratios=[1, 2])
    t = np.arange(D.shape[0])
    pop = np.sum(~np.isnan(D), axis=1)
    ax1.step(t, pop, where="post", c="tab:green")
    for b in births:
        ax1.axvline(b, c="tab:green", lw=0.6, alpha=0.5)
    ax1.set_ylabel("population")
    ax1.set_title("The budding cascade: each locked agent spawns a full-state copy "
                  "(H97d — 6/6 locked)")
    colors = plt.cm.viridis(np.linspace(0, 0.9, 6))
    for j in range(6):
        ax2.plot(t[::12], D[::12, j], lw=0.9, color=colors[j],
                 label=f"agent {j}" if j < 3 else None)
    ax2.axhline(4.8, ls=":", c="gray", lw=0.8)
    ax2.set_ylabel("distance to pacemaker")
    ax2.set_xlabel("step")
    ax2.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    out = Path(__file__).resolve().parents[1] / "out" / "lab" / "fig_budding.png"
    fig.savefig(out, dpi=150)
    print("saved", out)

if __name__ == "__main__":
    main()
