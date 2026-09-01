"""H71: is the long-horizon sag driven by target inflation?"""
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
    T = r["snaps"]["T_mean"]
    return dict(arm=task.get("arm", "full"), seed=task["seed"],
                early=float(np.mean(ss[5:10])), late=float(np.mean(ss[25:30])),
                T_early=float(np.mean(T[4:8])), T_late=float(np.mean(T[-5:])))

def main():
    res = {"weight_lr": 0.1, "target_lr": 0.01, "p_link": 0.1, "input_p_link": 0.1}
    tasks = [dict(res=res, seed=s, n_steps=21600, pin_output_p=0.1, arm=arm)
             for arm in ("full", "freeze-T-only") for s in range(12)]
    with ProcessPoolExecutor(10) as pool:
        rows = list(pool.map(run, tasks, chunksize=2))
    (LAB / "h71_slowstatue.json").write_text(json.dumps(rows))
    for arm in ("full", "freeze-T-only"):
        sel = [r for r in rows if r["arm"] == arm]
        E = np.mean([r["early"] for r in sel]); L = np.mean([r["late"] for r in sel])
        dT = np.mean([r["T_late"] - r["T_early"] for r in sel])
        print(f"{arm:<14} score {E:.3f} -> {L:.3f} (sag {L - E:+.3f}) | dT {dT:+.3f}")
    sel = [r for r in rows if r["arm"] == "full"]
    ds = np.array([r["late"] - r["early"] for r in sel])
    dt = np.array([r["T_late"] - r["T_early"] for r in sel])
    print(f"full: r(sag, dT) = {np.corrcoef(ds, dt)[0,1]:+.2f}")

if __name__ == "__main__":
    main()
