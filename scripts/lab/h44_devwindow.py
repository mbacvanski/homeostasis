"""H44: developmental-window structural plasticity."""
from __future__ import annotations
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from h42b_h43 import pursuit_variant, tracking_arm

LAB = Path(__file__).resolve().parents[1] / "out" / "lab"

def main():
    champ = json.loads((LAB / "h34_joint.json").read_text())[-1]["champion"]
    with ProcessPoolExecutor(10) as pool:
        prows = list(pool.map(pursuit_variant,
                              [(champ, 3000 + s, "devwindow", 120, 0.05, False)
                               for s in range(16)], chunksize=1))
        trows = list(pool.map(tracking_arm, [(s, "dev") for s in range(24)], chunksize=2))
    (LAB / "h44_devwindow.json").write_text(json.dumps(dict(pursuit=prows, tracking=trows)))
    locked = sum(r["near_late"] >= 0.8 for r in prows)
    print(f"H44 pursuit devwindow: locked {locked}/16  mean {np.mean([r['near_late'] for r in prows]):.2f}")
    v = np.array([r["score_late"] for r in trows])
    print(f"H44 tracking devwindow: score {v.mean():.3f}  frac>=0.35 {np.mean(v >= 0.35):.2f}")
    print("(refs: tracking baseline 0.491/0.88, lifelong-structural 0.284/0.00; pursuit lifelong 3/16)")

if __name__ == "__main__":
    main()
