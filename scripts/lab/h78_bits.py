"""H78: information rate and bits-per-spike across density and drive period."""
from __future__ import annotations
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from common import run_open_loop

LAB = Path(__file__).resolve().parents[1] / "out" / "lab"

def run(task):
    r = run_open_loop(task)
    g = min(max(r.get("recon_gain", 0.0), 0.0), 0.999)
    I = -0.5 * np.log2(1.0 - g * g)
    f = max(r["f_mean_late"], 1e-6)
    return dict(p=task["res"]["p_link"], P=task["schedule"]["period"],
                seed=task["seed"], gain=g, f=f, I=float(I),
                eff=float(I / (200 * f)))

def main():
    tasks = [dict(res={"weight_lr": 0.1, "target_lr": 0.01, "p_link": p,
                       "input_p_link": 0.1}, seed=s, n_steps=4000, recon=True,
                  schedule={"kind": "sine", "amp": 20.0, "period": P})
             for p in (0.02, 0.1, 0.4) for P in (60, 120, 240) for s in range(8)]
    with ProcessPoolExecutor(10) as pool:
        rows = list(pool.map(run, tasks, chunksize=2))
    (LAB / "h78_bits.json").write_text(json.dumps(rows))
    print("p      P    gain    f      I(bits/step)  bits/spike (x1e3)")
    for p in (0.02, 0.1, 0.4):
        for P in (60, 120, 240):
            sel = [r for r in rows if r["p"] == p and r["P"] == P]
            print(f"{p:<6} {P:<4} {np.mean([r['gain'] for r in sel]):.3f}"
                  f"  {np.mean([r['f'] for r in sel]):.3f}"
                  f"  {np.mean([r['I'] for r in sel]):.4f}"
                  f"        {np.mean([r['eff'] for r in sel])*1e3:.3f}")

if __name__ == "__main__":
    main()
