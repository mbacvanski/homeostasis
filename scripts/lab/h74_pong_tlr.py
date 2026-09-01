"""H74: target-channel role at the sparse-slow Pong optimum."""
from __future__ import annotations
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import dataclasses
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from homeostasis.simulation import PONG_RESERVOIR_CONFIG, PongSimulation  # noqa: E402
from homeostasis.pong import PongConfig  # noqa: E402

LAB = Path(__file__).resolve().parents[1] / "out" / "lab"

def run(task):
    res = dataclasses.replace(PONG_RESERVOIR_CONFIG, **task["res"])
    sim = PongSimulation(res, PongConfig.published(), seed=task["seed"])
    h = sim.run(100_000, record=False)
    return dict(seed=task["seed"], tlr=task["res"]["target_lr"], hit=h.hit_rate,
                f=float(sim.network.spiked.mean()))

def main():
    tasks = [dict(res={"p_link": 0.02, "weight_lr": 0.03, "target_lr": t}, seed=s)
             for t in (0.0, 0.01, 0.1) for s in range(12)]
    with ProcessPoolExecutor(10) as pool:
        rows = list(pool.map(run, tasks, chunksize=1))
    (LAB / "h74_pong_tlr.json").write_text(json.dumps(rows))
    for t in (0.0, 0.01, 0.1):
        sel = [r["hit"] for r in rows if r["tlr"] == t]
        print(f"tlr={t:<5} hit {np.mean(sel):.3f} (SD {np.std(sel):.3f})")

if __name__ == "__main__":
    main()
