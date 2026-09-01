"""H70: are wander dips dark excursions?"""
from __future__ import annotations
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from common import run_closed_loop

LAB = Path(__file__).resolve().parents[1] / "out" / "lab"
CELLS = {"ridge": {"p_link": 0.1, "weight_lr": 0.1},
         "sparse": {"p_link": 0.02, "weight_lr": 0.03}}

def run(task):
    r = run_closed_loop(task)
    return dict(cell=task["_cell"], seed=task["seed"],
                segs=r["seg_scores"], duty=r["seg_duty"], flow=r["seg_flow"])

def main():
    tasks = [dict(res={**res, "target_lr": 0.01, "input_p_link": 0.1},
                  seed=s, n_steps=21600, pin_output_p=0.1, _cell=name)
             for name, res in CELLS.items() for s in range(16)]
    with ProcessPoolExecutor(10) as pool:
        rows = list(pool.map(run, tasks, chunksize=2))
    (LAB / "h70_dips.json").write_text(json.dumps(rows))
    for name in CELLS:
        sel = [r for r in rows if r["cell"] == name]
        cors = []
        for r in sel:
            sc = np.array(r["segs"][5:])
            du = np.array(r["duty"][5:])
            if sc.std() > 1e-9 and du.std() > 1e-9:
                cors.append(np.corrcoef(sc, du)[0, 1])
        dut_e = np.mean([np.mean(r["duty"][5:10]) for r in sel])
        dut_l = np.mean([np.mean(r["duty"][25:30]) for r in sel])
        print(f"{name:<7} within-run r(score, duty): mean {np.mean(cors):+.3f} "
              f"(per-seed range {min(cors):+.2f}..{max(cors):+.2f}) | "
              f"duty early {dut_e:.3f} -> late {dut_l:.3f}")

if __name__ == "__main__":
    main()
