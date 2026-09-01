"""H69: does mid-run freezing abolish long-horizon erosion?"""
from __future__ import annotations
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from common import run_closed_loop

LAB = Path(__file__).resolve().parents[1] / "out" / "lab"

def run(task):
    r = run_closed_loop(task)
    ss = r["seg_scores"]
    return dict(arm=task.get("arm", "full"), seed=task["seed"],
                early=float(np.mean(ss[5:10])), late=float(np.mean(ss[25:30])))

def main():
    res = {"weight_lr": 0.1, "target_lr": 0.01, "p_link": 0.1, "input_p_link": 0.1}
    tasks = [dict(res=res, seed=s, n_steps=21600, pin_output_p=0.1, arm=arm)
             for arm in ("full", "freeze-mid") for s in range(16)]
    with ProcessPoolExecutor(10) as pool:
        rows = list(pool.map(run, tasks, chunksize=2))
    (LAB / "h69_devfreeze.json").write_text(json.dumps(rows))
    for arm in ("full", "freeze-mid"):
        sel = [r for r in rows if r["arm"] == arm]
        E = np.mean([r["early"] for r in sel]); L = np.mean([r["late"] for r in sel])
        print(f"{arm:<11} early {E:.3f} -> late {L:.3f} (delta {L - E:+.3f})")
    fu = {r["seed"]: r for r in rows if r["arm"] == "full"}
    fz = {r["seed"]: r for r in rows if r["arm"] == "freeze-mid"}
    d = np.array([fz[s]["late"] - fu[s]["late"] for s in fu])
    print(f"paired freeze-late minus full-late: {d.mean():+.3f} "
          f"t={d.mean() / (d.std(ddof=1) / np.sqrt(len(d))):+.2f}")

if __name__ == "__main__":
    main()
