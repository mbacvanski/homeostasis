"""Verify the H33 pursuit champion on fresh seeds."""
from __future__ import annotations
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from h33_evolve_pursuit import evaluate

LAB = Path(__file__).resolve().parents[1] / "out" / "lab"

def main():
    champ = json.loads((LAB / "h33_evolve_pursuit.json").read_text())[-1]["champion"]
    tasks = [(champ, 1000 + s) for s in range(16)]
    with ProcessPoolExecutor(10) as pool:
        rows = list(pool.map(evaluate, tasks))
    (LAB / "h33b_verify.json").write_text(json.dumps(rows))
    near = sorted([round(r["near3"], 2) for r in rows], reverse=True)
    print("champion on 16 FRESH seeds:")
    print("  near3 sorted:", near)
    print(f"  near3 mean {np.mean([r['near3'] for r in rows]):.2f}  median {np.median([r['near3'] for r in rows]):.2f}")
    print(f"  dist mean {np.mean([r['dist'] for r in rows]):.2f}  (still ref 4.50; hand best 6.06)")

if __name__ == "__main__":
    main()
