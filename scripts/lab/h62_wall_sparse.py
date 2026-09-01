"""H62: sparse pre-adaptation on wall avoidance."""
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
    late = h.hit[-1000:]
    return dict(p=p, wlr=wlr, seed=seed, late_hits=int(late.sum()),
                zero=bool(late.sum() == 0),
                f=float(h.prop_spiked[-1000:].mean()),
                sw=float(net.weights.sum(0).mean()))

def main():
    tasks = [(p, w, s) for p in (0.02, 0.1) for w in (0.03, 1.0) for s in range(16)]
    with ProcessPoolExecutor(10) as pool:
        rows = list(pool.map(run, tasks, chunksize=2))
    (LAB / "h62_wall_sparse.json").write_text(json.dumps(rows))
    for p in (0.02, 0.1):
        for w in (0.03, 1.0):
            sel = [r for r in rows if r["p"] == p and r["wlr"] == w]
            print(f"p={p:<5} wlr={w:<5} zero-late-hit {np.mean([r['zero'] for r in sel]):.2f}"
                  f"  median late hits {np.median([r['late_hits'] for r in sel]):4.0f}"
                  f"  f {np.mean([r['f'] for r in sel]):.3f}"
                  f"  Sigma-w {np.mean([r['sw'] for r in sel]):6.2f}")

if __name__ == "__main__":
    main()
