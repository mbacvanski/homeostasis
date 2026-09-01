"""H66: variance decomposition of the sparse cell's reliability."""
from __future__ import annotations
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from common import run_closed_loop

LAB = Path(__file__).resolve().parents[1] / "out" / "lab"
CELLS = {"sparse": {"p_link": 0.02, "weight_lr": 0.03},
         "ridge": {"p_link": 0.1, "weight_lr": 0.1},
         "statue-ref": {"p_link": 0.1, "weight_lr": 0.03}}

def run(task):
    r = run_closed_loop(task)
    ss = np.array(r["seg_scores"][5:])
    return dict(cell=task["_cell"], seed=task["seed"],
                mean=float(ss.mean()), wander=float(ss.std()),
                segs=r["seg_scores"],
                ac1=float(np.corrcoef(ss[:-1], ss[1:])[0, 1]) if ss.std() > 1e-9 else 0.0)

def main():
    tasks = [dict(res={**res, "target_lr": 0.01, "input_p_link": 0.1},
                  seed=s, n_steps=21600, pin_output_p=0.1, _cell=name)
             for name, res in CELLS.items() for s in range(12)]
    with ProcessPoolExecutor(10) as pool:
        rows = list(pool.map(run, tasks, chunksize=2))
    (LAB / "h66_wander.json").write_text(json.dumps(rows))
    print("cell        late mean (between-seed SD) | within-seed wander SD | lag-1 AC")
    for name in CELLS:
        sel = [r for r in rows if r["cell"] == name]
        means = [r["mean"] for r in sel]
        print(f"{name:<11} {np.mean(means):.3f} ({np.std(means):.3f})"
              f"           | {np.mean([r['wander'] for r in sel]):.3f}"
              f"                | {np.mean([r['ac1'] for r in sel]):+.2f}")
    print("early (segs 5-9) vs late (segs 25-29):")
    for name in CELLS:
        sel = [r for r in rows if r["cell"] == name]
        E = np.mean([np.mean(r["segs"][5:10]) for r in sel])
        L = np.mean([np.mean(r["segs"][25:30]) for r in sel])
        print(f"  {name:<11} {E:.3f} -> {L:.3f}  (delta {L - E:+.3f})")

if __name__ == "__main__":
    main()
