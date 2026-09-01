"""H58: sparsity laws on Pong (subcritical embodiment)."""
from __future__ import annotations
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import dataclasses
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from homeostasis.simulation import PONG_RESERVOIR_CONFIG, PongSimulation  # noqa: E402
from homeostasis.pong import PongConfig  # noqa: E402

LAB = Path(__file__).resolve().parents[1] / "out" / "lab"

def run(task):
    res = dataclasses.replace(PONG_RESERVOIR_CONFIG, **task["res"])
    sim = PongSimulation(res, PongConfig.published(), seed=task["seed"])
    h = sim.run(100_000, record=False)
    net = sim.network
    p, n = task["res"]["p_link"], res.n_nodes
    return dict(seed=task["seed"], p=p, wlr=task["res"]["weight_lr"],
                hit=h.hit_rate, wpn=float(net.weights.sum(0).mean()),
                f=float(net.spiked.mean()), T=float(net.targets.mean()))

def main():
    tasks = [dict(res={"p_link": p, "weight_lr": w, "target_lr": 0.01}, seed=s)
             for p in (0.02, 0.1) for w in (0.03, 0.1, 0.3, 1.0)
             for s in range(12)]
    with ProcessPoolExecutor(10) as pool:
        rows = list(pool.map(run, tasks, chunksize=1))
    (LAB / "h58_pong_sparse.json").write_text(json.dumps(rows))
    print("Pong (published cfg, tlr .01): hit (SD) | Sigma-w_in | f_end")
    for p in (0.02, 0.1):
        for w in (0.03, 0.1, 0.3, 1.0):
            sel = [r for r in rows if r["p"] == p and r["wlr"] == w]
            print(f"p={p:<5} wlr={w:<5} {np.mean([r['hit'] for r in sel]):.3f}"
                  f" ({np.std([r['hit'] for r in sel]):.3f})"
                  f" | {np.mean([r['wpn'] for r in sel]):7.2f}"
                  f" | {np.mean([r['f'] for r in sel]):.3f}")

if __name__ == "__main__":
    main()
