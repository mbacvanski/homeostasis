"""H45c: duty-law per-node check at N=2000 (open loop, slip 1)."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from common import run_open_loop

LAB = Path(__file__).resolve().parents[1] / "out" / "lab"

def main():
    r = run_open_loop(dict(res={"n_nodes": 2000, "p_link": 0.01, "input_p_link": 0.1,
                                "weight_lr": 0.1, "leak": 0.25},
                           seed=0, schedule={"kind": "slip", "speed": 1.0},
                           n_steps=3000, per_node=True))
    law = r["law"]
    drive = np.array(law["drive"]); f = np.array(law["f"]); T = np.array(law["T"])
    pred = np.clip((drive / np.maximum(T, 1e-9) - law["leak"]) / law["rho"], 0, 1)
    P, F = pred[2:].ravel(), f[2:].ravel()
    c = float(np.corrcoef(P, F)[0, 1])
    print(f"N=2000 duty law: corr {c:+.4f}  median|resid| {np.median(np.abs(F-P)):.4f} "
          f"n={len(F)}  f range {F.min():.2f}-{F.max():.2f}")
    (LAB / "h45_duty2000.json").write_text(json.dumps(dict(corr=c,
        med_resid=float(np.median(np.abs(F-P))))))

if __name__ == "__main__":
    main()
