"""H92: fine-grain test of the sensor-spacing resonance."""
from __future__ import annotations
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from common import run_open_loop

LAB = Path(__file__).resolve().parents[1] / "out" / "lab"

def run(task):
    r = run_open_loop(task)
    return dict(P=task["schedule"]["period"], seed=task["seed"],
                gain=r.get("recon_gain", 0.0))

def main():
    tasks = [dict(res={"weight_lr": 0.1, "target_lr": 0.01}, seed=s,
                  n_steps=4000, recon=True,
                  schedule={"kind": "sine", "amp": 20.0, "period": P})
             for P in (24, 30, 36, 42, 48, 60) for s in range(8)]
    with ProcessPoolExecutor(10) as pool:
        rows = list(pool.map(run, tasks, chunksize=2))
    (LAB / "h92_resonance.json").write_text(json.dumps(rows))
    for P in (24, 30, 36, 42, 48, 60):
        sel = [r["gain"] for r in rows if r["P"] == P]
        print(f"P={P:<4} peak speed {20*2*np.pi/P:.2f} deg/step  gain {np.mean(sel):.3f} (SD {np.std(sel):.3f})")

if __name__ == "__main__":
    main()
