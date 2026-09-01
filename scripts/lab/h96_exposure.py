"""H96: sag vs weight rate — the error-exposure account of the T-toxin."""
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
    return dict(wlr=task["res"]["weight_lr"], seed=task["seed"],
                early=float(np.mean(ss[5:10])), late=float(np.mean(ss[25:30])),
                absE=r["mean_abs_E"])

def main():
    tasks = [dict(res={"weight_lr": w, "target_lr": 0.01, "p_link": 0.1,
                       "input_p_link": 0.1}, seed=s, n_steps=21600,
                  pin_output_p=0.1)
             for w in (0.03, 0.1, 0.3, 1.0) for s in range(12)]
    with ProcessPoolExecutor(10) as pool:
        rows = list(pool.map(run, tasks, chunksize=2))
    (LAB / "h96_exposure.json").write_text(json.dumps(rows))
    for w in (0.03, 0.1, 0.3, 1.0):
        sel = [r for r in rows if r["wlr"] == w]
        E = np.mean([r["early"] for r in sel]); L = np.mean([r["late"] for r in sel])
        print(f"wlr={w:<5} early {E:.3f} -> late {L:.3f} (sag {L - E:+.3f})"
              f"  |E| {np.mean([r['absE'] for r in sel]):.3f}")

if __name__ == "__main__":
    main()
