"""H80: reset-and-freeze T at 3600 — profile vs co-adapted-W damage."""
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
    tasks = [dict(res=res, seed=s, n_steps=21600, pin_output_p=0.1,
                  freeze_T_at=3600, reset_T_on_freeze=True) for s in range(12)]
    with ProcessPoolExecutor(10) as pool:
        rows = list(pool.map(run, tasks, chunksize=2))
    (LAB / "h80_reset.json").write_text(json.dumps(rows))
    E = np.mean([r["early"] for r in rows]); L = np.mean([r["late"] for r in rows])
    print(f"reset+freeze T@3600: early {E:.3f} -> late {L:.3f} (sag {L - E:+.3f})")
    print("(references: freezeT0 late 0.519/0.466, freezeT3600 late 0.400/0.376, full 0.336/0.351)")

if __name__ == "__main__":
    main()
