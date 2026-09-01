"""H76: freeze T after a calibration window."""
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
    return dict(seed=task["seed"], early=float(np.mean(ss[5:10])),
                late=float(np.mean(ss[25:30])))

def main():
    res = {"weight_lr": 0.1, "target_lr": 0.01, "p_link": 0.1, "input_p_link": 0.1}
    tasks = [dict(res=res, seed=s, n_steps=21600, pin_output_p=0.1, freeze_T_at=3600)
             for s in range(12)]
    with ProcessPoolExecutor(10) as pool:
        rows = list(pool.map(run, tasks, chunksize=2))
    (LAB / "h76_twindow.json").write_text(json.dumps(rows))
    old = json.load(open(LAB / "h71_slowstatue.json"))
    f0 = {r["seed"]: r["late"] for r in old if r["arm"] == "freeze-T-only"}
    fu = {r["seed"]: r["late"] for r in old if r["arm"] == "full"}
    f36 = {r["seed"]: r["late"] for r in rows}
    E = np.mean([r["early"] for r in rows]); L = np.mean([r["late"] for r in rows])
    print(f"freeze-T@3600: early {E:.3f} -> late {L:.3f}")
    d_full = np.array([f36[s] - fu[s] for s in f36])
    d_f0 = np.array([f36[s] - f0[s] for s in f36])
    print(f"vs full late:        {d_full.mean():+.3f} (t={d_full.mean()/(d_full.std(ddof=1)/np.sqrt(len(d_full))):+.2f})")
    print(f"vs freeze-at-0 late: {d_f0.mean():+.3f} (t={d_f0.mean()/(d_f0.std(ddof=1)/np.sqrt(len(d_f0))):+.2f})")

if __name__ == "__main__":
    main()
