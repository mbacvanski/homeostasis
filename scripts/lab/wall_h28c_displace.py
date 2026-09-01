"""H28c: displacement probe — is the settled orbit stability-selected?"""
from __future__ import annotations
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from common import make_configs  # noqa: F401
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from homeostasis.simulation import WALL_RESERVOIR_CONFIG, WallSimulation  # noqa: E402
from homeostasis.wall import WallConfig  # noqa: E402

LAB = Path(__file__).resolve().parents[1] / "out" / "lab"


def evaluate(task):
    seed, where = task
    sim = WallSimulation(WALL_RESERVOIR_CONFIG, WallConfig(), seed=seed)
    for _ in range(3600):
        sim.step()
    pre_hits = sim.env.hits
    if where == "wall":
        sim.env.x, sim.env.y = 2.0, 7.5
    else:
        sim.env.x, sim.env.y = 7.5, 7.5
    n = 1200
    hits = 0
    wall_dist = np.empty(n)
    absE = np.empty(n)
    for i in range(n):
        state, dh, h = sim.step()
        hits += h
        wall_dist[i] = min(sim.env.x, sim.env.y, 15 - sim.env.x, 15 - sim.env.y)
        absE[i] = float(np.mean(np.abs(state.error)))
    return dict(seed=seed, where=where, post_hits=int(hits),
                pre_hits=int(pre_hits),
                early_wall_dist=float(wall_dist[:200].mean()),
                late_wall_dist=float(wall_dist[-400:].mean()),
                absE_first200=float(absE[:200].mean()),
                absE_late=float(absE[-400:].mean()))


def main():
    tasks = [(s, w) for s in range(16) for w in ("wall", "flat")]
    with ProcessPoolExecutor(10) as pool:
        rows = list(pool.map(evaluate, tasks))
    (LAB / "wall_h28c.json").write_text(json.dumps(rows))
    for where in ("wall", "flat"):
        sel = [r for r in rows if r["where"] == where]
        print(f"══ teleport to {where} (16 seeds)")
        print(f"   post hits: median {np.median([r['post_hits'] for r in sel]):.0f} "
              f"mean {np.mean([r['post_hits'] for r in sel]):.1f}")
        print(f"   wall distance: first200 {np.mean([r['early_wall_dist'] for r in sel]):.2f} "
              f"-> late {np.mean([r['late_wall_dist'] for r in sel]):.2f}")
        print(f"   |E|: first200 {np.mean([r['absE_first200'] for r in sel]):.3f} "
              f"late {np.mean([r['absE_late'] for r in sel]):.3f}")


if __name__ == "__main__":
    main()
