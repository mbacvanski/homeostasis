"""H41b: flow-seeking annealing (corrected trigger)."""
from __future__ import annotations
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from h40_h41_lottery import h41_run

LAB = Path(__file__).resolve().parents[1] / "out" / "lab"

def main():
    champ = json.loads((LAB / "h34_joint.json").read_text())[-1]["champion"]
    with ProcessPoolExecutor(10) as pool:
        rows = list(pool.map(h41_run,
                             [(champ, 3000 + s, a) for a in (False, True) for s in range(16)],
                             chunksize=1))
    (LAB / "h41b_flowseek.json").write_text(json.dumps(rows))
    for a in (False, True):
        sel = [r for r in rows if r["anneal"] == a]
        locked = sum(r["near_late"] >= 0.8 for r in sel)
        print(f"anneal={a}: locked {locked}/16  near_late mean {np.mean([r['near_late'] for r in sel]):.2f} "
              f"median {np.median([r['near_late'] for r in sel]):.2f}  "
              f"shuffles {np.mean([r['shuffles'] for r in sel]):.1f}")

if __name__ == "__main__":
    main()
