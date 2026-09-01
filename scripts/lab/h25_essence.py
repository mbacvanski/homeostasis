"""H25: transplant w1's core trio onto defaults."""
from __future__ import annotations
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from common import run_closed_loop

LAB = Path(__file__).resolve().parents[1] / "out" / "lab"

def run(t):
    r = run_closed_loop(t)
    r["t"] = t["_t"]
    return r

def main():
    arms = {
        "essence": (dict(leak=0.574), dict(gain=28.35)),
        "essence+wlr0.65": (dict(leak=0.574, weight_lr=0.65), dict(gain=28.35)),
        "leak-only": (dict(leak=0.574), {}),
        "gain-only": ({}, dict(gain=28.35)),
    }
    tasks = [dict(res=dict(r_), trk=dict(t_), seed=s, arm="full", snap_every=7200,
                  _t=name)
             for name, (r_, t_) in arms.items() for s in range(16)]
    with ProcessPoolExecutor(10) as pool:
        rows = list(pool.map(run, tasks, chunksize=2))
    (LAB / "h25_essence.json").write_text(json.dumps(
        [{k: v for k, v in r.items() if k != "snaps"} for r in rows]))
    for name in arms:
        v = np.array([r["score_late"] for r in rows if r["t"] == name])
        print(f"   {name:16s} {v.mean():.3f}±{v.std():.3f}  frac>=0.35 {np.mean(v>=0.35):.2f}  min {v.min():.2f}")

if __name__ == "__main__":
    main()
