"""H40b: powered early-dynamics lottery predictor (300 wirings)."""
from __future__ import annotations
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from h40_h41_lottery import h40_run

LAB = Path(__file__).resolve().parents[1] / "out" / "lab"

def main():
    champ = json.loads((LAB / "h34_joint.json").read_text())[-1]["champion"]
    with ProcessPoolExecutor(10) as pool:
        rows = list(pool.map(h40_run, [(champ, 5000 + s) for s in range(300)], chunksize=4))
    (LAB / "h40b_powered.json").write_text(json.dumps(rows))
    lock = np.array([r["near_final"] >= 0.8 for r in rows])
    print(f"{lock.sum()}/300 wirings lock")
    feats = ("early_flow", "early_flow_trend", "early_rev", "early_dist_trend")
    for k in feats:
        v = np.array([r[k] for r in rows])
        best_acc, best_t, best_dir = 0, 0, ""
        for t in np.percentile(v, np.linspace(2, 98, 97)):
            a1 = ((v < t) == lock).mean(); a2 = ((v >= t) == lock).mean()
            if max(a1, a2) > best_acc:
                best_acc, best_t, best_dir = max(a1, a2), t, "<" if a1 > a2 else ">="
        base = max(lock.mean(), 1 - lock.mean())
        # balanced accuracy at that threshold
        pred = (v < best_t) if best_dir == "<" else (v >= best_t)
        tpr = pred[lock].mean() if lock.sum() else 0
        tnr = (~pred)[~lock].mean() if (~lock).sum() else 0
        print(f"   {k:18s} lockers {v[lock].mean():+8.3f} vs {v[~lock].mean():+8.3f}"
              f"   bal.acc {(tpr+tnr)/2:.2f} (base rate guard {base:.2f})")

if __name__ == "__main__":
    main()
