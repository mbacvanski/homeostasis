"""H51: sensor-noise robustness of the ridge."""
from __future__ import annotations
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from common import run_closed_loop

LAB = Path(__file__).resolve().parents[1] / "out" / "lab"

def run(t):
    r = run_closed_loop(t)
    return dict(sig=t["sensor_noise"], wlr=t["res"]["weight_lr"], seed=t["seed"],
                score=r["score_late"], f=r["prop_spiked"])

def main():
    tasks = [dict(res={"weight_lr": w, "leak": 0.25}, trk={}, seed=s, arm="full",
                  snap_every=7200, sensor_noise=sig)
             for sig in (0.0, 0.1, 0.2) for w in (0.03, 0.1, 0.3, 1.0) for s in range(16)]
    with ProcessPoolExecutor(10) as pool:
        rows = list(pool.map(run, tasks, chunksize=2))
    (LAB / "h51_noise.json").write_text(json.dumps(rows))
    print("score_late (16 seeds; frac>=0.35) | prop_spiked")
    print("        wlr=0.03      wlr=0.1       wlr=0.3       wlr=1.0")
    for sig in (0.0, 0.1, 0.2):
        cells = []
        fs = []
        for w in (0.03, 0.1, 0.3, 1.0):
            v = np.array([r["score"] for r in rows if r["sig"] == sig and r["wlr"] == w])
            f = np.mean([r["f"] for r in rows if r["sig"] == sig and r["wlr"] == w])
            cells.append(f"{v.mean():.3f}({np.mean(v>=0.35):.2f})")
            fs.append(f"{f:.2f}")
        print(f"σ={sig:<4} " + "  ".join(f"{c:>12s}" for c in cells) + "   f: " + " ".join(fs))

if __name__ == "__main__":
    main()
