"""H23c: N-line with ONLY recurrent density scaled (input_p_link pinned 0.1).
Output pools still scale with p_link (structural, unavoidable without core
changes) - noted. 12 seeds, wlr=0.1."""
from __future__ import annotations
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from common import run_closed_loop

LAB = Path(__file__).resolve().parents[1] / "out" / "lab"

def run(t):
    r = run_closed_loop(t)
    r["n"] = t["res"]["n_nodes"]
    return r

def main():
    jobs = [dict(res=dict(n_nodes=n, p_link=20.0 / n, input_p_link=0.1,
                          weight_lr=0.1), trk={}, seed=s, arm="full",
                 snap_every=7200)
            for n in (50, 100, 200, 400, 800) for s in range(12)]
    with ProcessPoolExecutor(10) as pool:
        rows = list(pool.map(run, jobs, chunksize=2))
    (LAB / "h23c_clean.json").write_text(json.dumps(
        [{k: v for k, v in r.items() if k != "snaps"} for r in rows]))
    print("H23c: recurrent-p scaled (in-degree=20), input wiring fixed at 0.1:")
    for n in (50, 100, 200, 400, 800):
        v = np.array([r["score_late"] for r in rows if r["n"] == n])
        f = np.mean([r["prop_spiked"] for r in rows if r["n"] == n])
        print(f"   N={n:<4} score {v.mean():.3f}±{v.std():.3f}  frac>=0.35 {np.mean(v>=0.35):.2f}  prop_spiked {f:.3f}")

if __name__ == "__main__":
    main()
