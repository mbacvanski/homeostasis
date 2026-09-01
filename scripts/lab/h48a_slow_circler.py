"""H48a: find a naturally slow wall circler."""
from __future__ import annotations
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from common import make_configs  # noqa: F401
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import dataclasses  # noqa: E402
from homeostasis.simulation import WALL_RESERVOIR_CONFIG, run_wall  # noqa: E402

LAB = Path(__file__).resolve().parents[1] / "out" / "lab"

def evaluate(task):
    wlr, seed = task
    res = dataclasses.replace(WALL_RESERVOIR_CONFIG, weight_lr=wlr)
    h = run_wall(n_steps=3600, seed=seed, reservoir_config=res)
    late = slice(2600, None)
    return dict(wlr=wlr, seed=seed,
                late_hits=int(h.hit[late].sum()),
                omega=float(np.rad2deg(np.abs(h.d_heading[late]).mean())),
                omega_signed=float(np.rad2deg(np.abs(h.d_heading[late].mean()))),
                f=float(h.prop_spiked[late].mean()),
                speed=float(np.hypot(np.diff(h.x), np.diff(h.y))[2600:].mean()))

def main():
    tasks = [(wlr, s) for wlr in (0.05, 0.1, 0.2, 1.0) for s in range(8)]
    with ProcessPoolExecutor(10) as pool:
        rows = list(pool.map(evaluate, tasks, chunksize=2))
    (LAB / "h48a_slow.json").write_text(json.dumps(rows))
    print("wlr    clean  |omega|°/st  signed  f     v      r=v/w")
    for wlr in (0.05, 0.1, 0.2, 1.0):
        sel = [r for r in rows if r["wlr"] == wlr]
        clean = sum(r["late_hits"] == 0 for r in sel)
        om = np.mean([r["omega_signed"] for r in sel])
        v = np.mean([r["speed"] for r in sel])
        print(f"{wlr:<5}  {clean}/8   {np.mean([r['omega'] for r in sel]):6.2f}   "
              f"{om:6.2f}  {np.mean([r['f'] for r in sel]):.2f}  {v:.2f}   "
              f"{v / max(np.deg2rad(om), 1e-9):.1f}")
    # best individual slow clean circlers
    good = [r for r in rows if r["late_hits"] == 0 and r["omega_signed"] > 0.3]
    good.sort(key=lambda r: r["omega_signed"])
    print("slowest clean circlers:", [(r["wlr"], r["seed"], round(r["omega_signed"], 2)) for r in good[:5]])

if __name__ == "__main__":
    main()
