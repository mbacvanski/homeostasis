"""H62b: sparse wall window completion — alive-and-clean across wlr."""
from __future__ import annotations
import dataclasses
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from homeostasis.simulation import WALL_RESERVOIR_CONFIG, WallSimulation  # noqa: E402

LAB = Path(__file__).resolve().parents[1] / "out" / "lab"

def run(task):
    p, wlr, seed = task
    res = dataclasses.replace(WALL_RESERVOIR_CONFIG, p_link=p, input_p_link=0.1,
                              weight_lr=wlr)
    sim = WallSimulation(res, seed=seed)
    net = sim.network
    orng = np.random.default_rng(seed + 880008)
    net.output_adjacency = orng.random((res.n_nodes, 2)) < 0.1
    net._rebuild_structure_caches()
    h = sim.run(3600)
    speed = float(np.hypot(np.diff(h.x[-1000:]), np.diff(h.y[-1000:])).mean())
    return dict(p=p, wlr=wlr, seed=seed, speed=speed,
                zero=bool(h.hit[-1000:].sum() == 0),
                late_hits=int(h.hit[-1000:].sum()),
                f=float(h.prop_spiked[-1000:].mean()))

def main():
    tasks = [(0.02, w, s) for w in (0.1, 0.3) for s in range(16)]
    tasks += [(0.1, 1.0, s) for s in range(16)]  # paper reference with same pins
    with ProcessPoolExecutor(10) as pool:
        rows = list(pool.map(run, tasks, chunksize=2))
    (LAB / "h62b_window.json").write_text(json.dumps(rows))
    for p, w in ((0.02, 0.1), (0.02, 0.3), (0.1, 1.0)):
        sel = [r for r in rows if r["p"] == p and r["wlr"] == w]
        alive = [r for r in sel if r["speed"] > 0.02]
        ac = sum(1 for r in alive if r["zero"])
        print(f"p={p:<5} wlr={w:<4} alive {len(alive):2d}/16  alive+clean {ac:2d}/16"
              f"  median late hits {np.median([r['late_hits'] for r in sel]):4.0f}"
              f"  f {np.mean([r['f'] for r in sel]):.3f}")

if __name__ == "__main__":
    main()
