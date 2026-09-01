"""H83: stack all discovered advantages."""
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
    out = dict(sig=task.get("sensor_noise", 0.0), n=task["n_steps"],
               seed=task["seed"], score=r["score_late"], duty=r["input_duty"])
    if task["n_steps"] >= 21600:
        out["early"] = float(np.mean(ss[5:10])); out["late"] = float(np.mean(ss[25:30]))
    return out

def main():
    res = {"weight_lr": 0.03, "target_lr": 0.0, "p_link": 0.02, "input_p_link": 0.1}
    tasks = [dict(res=res, seed=s, n_steps=7200, pin_output_p=0.1, sensor_noise=sig)
             for sig in (0.0, 0.1) for s in range(16)]
    tasks += [dict(res=res, seed=s, n_steps=21600, pin_output_p=0.1) for s in range(16)]
    with ProcessPoolExecutor(10) as pool:
        rows = list(pool.map(run, tasks, chunksize=2))
    (LAB / "h83_stack.json").write_text(json.dumps(rows))
    for sig in (0.0, 0.1):
        sel = [r for r in rows if r["n"] == 7200 and r["sig"] == sig]
        sc = [r["score"] for r in sel]
        print(f"stack sigma={sig}: {np.mean(sc):.3f} (frac>=.35 {np.mean([x>=.35 for x in sc]):.2f})"
              f" duty {np.mean([r['duty'] for r in sel]):.2f}")
    sel = [r for r in rows if r["n"] == 21600]
    E = np.mean([r["early"] for r in sel]); L = np.mean([r["late"] for r in sel])
    print(f"long-run: early {E:.3f} -> late {L:.3f} (sag {L-E:+.3f})")

if __name__ == "__main__":
    main()
