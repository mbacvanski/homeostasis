"""H51b: does noise desaturate and restore information at wlr=0.03?"""
from __future__ import annotations
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from common import run_open_loop

LAB = Path(__file__).resolve().parents[1] / "out" / "lab"

def run(t):
    r = run_open_loop(t)
    return dict(sig=t.get("sensor_noise", 0.0), wlr=t["res"]["weight_lr"],
                seed=t["seed"], gain=r.get("recon_gain", 0.0),
                f=r["f_mean_late"])

def main():
    tasks = [dict(res={"weight_lr": w, "target_lr": 0.01}, seed=s, n_steps=4000,
                  recon=True, sensor_noise=sig,
                  schedule={"kind": "sine", "amp": 20.0, "period": 120})
             for w in (0.03, 0.1) for sig in (0.0, 0.1, 0.2) for s in range(8)]
    with ProcessPoolExecutor(10) as pool:
        rows = list(pool.map(run, tasks, chunksize=2))
    (LAB / "h51b_desat.json").write_text(json.dumps(rows))
    print("recon gain (P=120) / f_late")
    for w in (0.03, 0.1):
        line = f"wlr={w:<5}"
        for sig in (0.0, 0.1, 0.2):
            sel = [r for r in rows if r["wlr"] == w and r["sig"] == sig]
            line += f"  s={sig}: {np.mean([r['gain'] for r in sel]):.3f}/{np.mean([r['f'] for r in sel]):.2f}"
        print(line)

if __name__ == "__main__":
    main()
