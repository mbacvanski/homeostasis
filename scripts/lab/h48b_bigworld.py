"""H48b: slow circlers via wide wheel base in a big arena (box 30)."""
from __future__ import annotations
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from common import make_configs  # noqa: F401
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from homeostasis.simulation import run_wall  # noqa: E402
from homeostasis.wall import WallConfig  # noqa: E402

LAB = Path(__file__).resolve().parents[1] / "out" / "lab"

def evaluate(task):
    wb, seed = task
    wc = WallConfig(box_size=30.0, initial_x=15.0, initial_y=15.0, wheel_base=wb)
    h = run_wall(n_steps=4800, seed=seed, wall_config=wc)
    late = slice(3300, None)
    return dict(wb=wb, seed=seed, late_hits=int(h.hit[late].sum()),
                omega=float(np.rad2deg(np.abs(h.d_heading[late].mean()))),
                speed=float(np.hypot(np.diff(h.x), np.diff(h.y))[3300:].mean()))

def main():
    tasks = [(wb, s) for wb in (1.0, 2.5, 4.0) for s in range(8)]
    with ProcessPoolExecutor(10) as pool:
        rows = list(pool.map(evaluate, tasks, chunksize=2))
    (LAB / "h48b_bigworld.json").write_text(json.dumps(rows))
    print("wb    clean  omega deg/st   v     r=v/w")
    for wb in (1.0, 2.5, 4.0):
        sel = [r for r in rows if r["wb"] == wb]
        clean = sum(r["late_hits"] == 0 for r in sel)
        om = np.mean([r["omega"] for r in sel]); v = np.mean([r["speed"] for r in sel])
        print(f"{wb:<4}  {clean}/8   {om:6.2f}     {v:.2f}   {v/max(np.deg2rad(om),1e-9):5.1f}")
    good = sorted([r for r in rows if r["late_hits"] == 0 and 0.5 < r["omega"] <= 3.0],
                  key=lambda r: r["omega"])
    print("band-compatible clean circlers (0.5-3 deg/step):",
          [(r["wb"], r["seed"], round(r["omega"], 2)) for r in good[:6]])

if __name__ == "__main__":
    main()
