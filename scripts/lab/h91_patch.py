"""H91: local size of an acquisition basin."""
from __future__ import annotations
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from h88_basin import run as basin_run, CX, CY, R0  # noqa: E402
from h50_depth import LAB  # noqa: E402

# winning cell: offset -6, angle index 2 of linspace(0, 2pi, 4) = pi
WIN_OFF, WIN_ANG = -6, np.pi

def run(task):
    dx, dy = task
    # reuse h88's run via a synthetic (off, ang) that lands at the perturbed point:
    x = CX + (R0 + WIN_OFF) * np.cos(WIN_ANG) + dx
    y = CY + (R0 + WIN_OFF) * np.sin(WIN_ANG) + dy
    r = np.hypot(x - CX, y - CY)
    ang = np.arctan2(y - CY, x - CX)
    return dict(dx=dx, dy=dy, **basin_run((r - R0, ang)))

def main():
    tasks = [(d * np.cos(a), d * np.sin(a))
             for d in (0.25, 0.5, 1.0, 2.0)
             for a in np.linspace(0, 2 * np.pi, 4, endpoint=False)]
    with ProcessPoolExecutor(10) as pool:
        rows = list(pool.map(run, tasks, chunksize=1))
    (LAB / "h91_patch.json").write_text(json.dumps(rows))
    for d in (0.25, 0.5, 1.0, 2.0):
        sel = [r for r in rows if abs(np.hypot(r["dx"], r["dy"]) - d) < 1e-6]
        print(f"delta {d}: acquire {sum(r['lock'] >= 0.8 for r in sel)}/4"
              f" (locks {[round(r['lock'], 2) for r in sel]})")

if __name__ == "__main__":
    main()
