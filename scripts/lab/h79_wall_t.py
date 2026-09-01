"""H79: freeze-T on the wall task's statue regime."""
from __future__ import annotations
import dataclasses
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from homeostasis.simulation import WALL_RESERVOIR_CONFIG, WallSimulation  # noqa: E402

LAB = Path(__file__).resolve().parents[1] / "out" / "lab"

def run(task):
    p, wlr, freeze_t, seed = task
    res = dataclasses.replace(WALL_RESERVOIR_CONFIG, p_link=p, input_p_link=0.1,
                              weight_lr=wlr,
                              target_lr=0.0 if freeze_t else WALL_RESERVOIR_CONFIG.target_lr)
    sim = WallSimulation(res, seed=seed)
    net = sim.network
    orng = np.random.default_rng(seed + 880008)
    net.output_adjacency = orng.random((res.n_nodes, 2)) < 0.1
    net._rebuild_structure_caches()
    h = sim.run(3600)
    speed = float(np.hypot(np.diff(h.x[-1000:]), np.diff(h.y[-1000:])).mean())
    return dict(freeze_t=freeze_t, seed=seed, speed=speed,
                alive=bool(speed > 0.02),
                clean=bool(h.hit[-1000:].sum() == 0),
                f=float(h.prop_spiked[-1000:].mean()))

def main():
    tasks = [(0.1, 0.03, ft, s) for ft in (False, True) for s in range(16)]
    tasks += [(0.02, 0.03, True, s) for s in range(16)]
    with ProcessPoolExecutor(10) as pool:
        rows = list(pool.map(run, tasks, chunksize=2))
    (LAB / "h79_wall_t.json").write_text(json.dumps(rows))
    for ft in (False, True, "sparse"):
        if ft == "sparse":
            sel = rows[32:]
        else:
            sel = [r for r in rows[:32] if r["freeze_t"] == ft]
        alive = [r for r in sel if r["alive"]]
        ac = sum(1 for r in alive if r["clean"])
        label = {False: "full    ", True: "freeze-T", "sparse": "sparse+frzT"}[ft]
        print(f"{label}: alive {len(alive):2d}/16"
              f"  alive+clean {ac:2d}/16  f {np.mean([r['f'] for r in sel]):.3f}")

if __name__ == "__main__":
    main()
