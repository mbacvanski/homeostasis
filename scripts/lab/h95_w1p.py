"""H95: freeze-T-at-birth on the w1' configuration, long horizon."""
from __future__ import annotations
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from common import run_closed_loop
from b8_bias_carrier import w1p

LAB = Path(__file__).resolve().parents[1] / "out" / "lab"

def run(task):
    r = run_closed_loop(task)
    ss = r["seg_scores"]
    return dict(arm=task["_arm"], seed=task["seed"],
                early=float(np.mean(ss[5:10])), late=float(np.mean(ss[25:30])))

def main():
    res, trk = w1p()
    tasks = []
    for s in range(12):
        tasks.append(dict(res=res, trk=trk, seed=s, n_steps=21600, _arm="full"))
        tasks.append(dict(res=res, trk=trk, seed=s, n_steps=21600,
                          freeze_T_at=0, _arm="freezeT0"))
    with ProcessPoolExecutor(10) as pool:
        rows = list(pool.map(run, tasks, chunksize=2))
    (LAB / "h95_w1p.json").write_text(json.dumps(rows))
    for arm in ("full", "freezeT0"):
        sel = [r for r in rows if r["arm"] == arm]
        E = np.mean([r["early"] for r in sel]); L = np.mean([r["late"] for r in sel])
        print(f"w1' {arm:<9} early {E:.3f} -> late {L:.3f} (sag {L - E:+.3f})")
    fu = {r["seed"]: r["late"] for r in rows if r["arm"] == "full"}
    fz = {r["seed"]: r["late"] for r in rows if r["arm"] == "freezeT0"}
    d = np.array([fz[s] - fu[s] for s in fu])
    print(f"paired freezeT0 - full late: {d.mean():+.3f}"
          f" (t={d.mean() / (d.std(ddof=1) / np.sqrt(len(d))):+.2f})")

if __name__ == "__main__":
    main()
