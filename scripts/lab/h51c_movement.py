"""H51c: is the sigma=0.1 rescue of wlr=0.03 behavioral (un-stilling) rather than desaturation?"""
from __future__ import annotations
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from common import run_closed_loop

LAB = Path(__file__).resolve().parents[1] / "out" / "lab"

def run(t):
    r = run_closed_loop(t)
    return dict(sig=t.get("sensor_noise", 0.0), seed=t["seed"],
                score=r["score_late"], f=r["prop_spiked"],
                eff_diff=r["eff_diff"], eff_sat=r["eff_sat"],
                flow=r["input_flow"], duty=r["input_duty"],
                dir_agree=r["dir_agree"])

def main():
    tasks = [dict(res={"weight_lr": 0.03, "target_lr": 0.01}, seed=s,
                  sensor_noise=sig)
             for sig in (0.0, 0.1, 0.2) for s in range(16)]
    with ProcessPoolExecutor(10) as pool:
        rows = list(pool.map(run, tasks, chunksize=2))
    (LAB / "h51c_movement.json").write_text(json.dumps(rows))
    print("wlr=0.03 closed loop: score | eff_diff | eff_sat | flow | duty | dir_agree")
    for sig in (0.0, 0.1, 0.2):
        sel = [r for r in rows if r["sig"] == sig]
        m = lambda k: np.mean([r[k] for r in sel])
        print(f"σ={sig}:  {m('score'):.3f} | {m('eff_diff'):.3f} | {m('eff_sat'):.2f}"
              f" | {m('flow'):.2f} | {m('duty'):.2f} | {m('dir_agree'):.3f}")

if __name__ == "__main__":
    main()
