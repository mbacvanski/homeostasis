"""H23b: dual-scaled N-line (see LEDGER)."""
from __future__ import annotations
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from common import run_closed_loop

LAB = Path(__file__).resolve().parents[1] / "out" / "lab"

def run(t):
    r = run_closed_loop(t)
    r["n"] = t["res"]["n_nodes"]; r["scal"] = t["_s"]
    return r

def main():
    tasks = []
    for n in (50, 100, 200, 400):
        tasks.append(("p", dict(n_nodes=n, p_link=20.0 / n, weight_lr=0.1)))
        tasks.append(("w", dict(n_nodes=n, weight_init_mean=15.0 / (n * 0.1), weight_lr=0.1)))
    jobs = [dict(res=dict(res), trk={}, seed=s, arm="full", snap_every=7200, _s=sc)
            for sc, res in tasks for s in range(12)]
    with ProcessPoolExecutor(10) as pool:
        rows = list(pool.map(run, jobs, chunksize=2))
    (LAB / "h23b_dual.json").write_text(json.dumps(
        [{k: v for k, v in r.items() if k != "snaps"} for r in rows]))
    print("H23b dual-scaled N-line (score_late, 12 seeds; frac>=0.35):")
    print("          p=20/N          w0=15/(0.1N)")
    for n in (50, 100, 200, 400):
        cells = []
        for sc in ("p", "w"):
            v = np.array([r["score_late"] for r in rows if r["n"] == n and r["scal"] == sc])
            cells.append(f"{v.mean():.3f} ({np.mean(v>=0.35):.2f})")
        print(f"   N={n:<4} " + "   ".join(f"{c:>13s}" for c in cells))

if __name__ == "__main__":
    main()
