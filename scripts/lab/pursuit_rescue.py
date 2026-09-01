"""Pursuit rescues: (A) aliveness-everywhere via big input weights (Law 1
as a design rule: keep the whole arena above the dead boundary);
(B) bearing-only retina (intensity falloff off - flow cannot starve with
distance, but approach is no longer flow-rewarded)."""
from __future__ import annotations
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from common import make_configs  # noqa: F401
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from homeostasis.reservoir import ReservoirConfig  # noqa: E402
from homeostasis.simulation import run_pursuit  # noqa: E402
from homeostasis.pursuit import PursuitConfig  # noqa: E402

LAB = Path(__file__).resolve().parents[1] / "out" / "lab"


def evaluate(task):
    name, w_in, wlr, iscale, seed = task
    res = ReservoirConfig(n_inputs=62, input_weight=w_in, weight_lr=wlr)
    pc = PursuitConfig(intensity_scale=iscale)
    h = run_pursuit(n_steps=3600, seed=seed, reservoir_config=res, pursuit_config=pc)
    late = slice(1800, None)
    br = np.abs(h.bearing[late])
    return dict(name=name, seed=seed,
                dist=float(h.dist[late].mean()),
                near3=float((h.dist[late] < 3).mean()),
                oriented=float((br <= 45).mean()),
                hits=int(h.hit.sum()), f=float(h.prop_spiked[late].mean()),
                flow=float(h.flow[late].mean()))


def main():
    arms = [
        ("w12", 12.0, 0.3, 3.0), ("w20", 20.0, 0.3, 3.0), ("w20-wlr1", 20.0, 1.0, 3.0),
        ("bearing-only", 4.0, 0.3, 1e9), ("bearing-only-w2", 2.0, 0.3, 1e9),
        ("gentle-fall", 4.0, 0.3, 8.0),
    ]
    tasks = [(n, w, l, i, s) for (n, w, l, i) in arms for s in range(8)]
    with ProcessPoolExecutor(10) as pool:
        rows = list(pool.map(evaluate, tasks, chunksize=2))
    (LAB / "pursuit_rescue.json").write_text(json.dumps(rows))
    print("arm            dist   near3  oriented  hits   f     flow")
    for (n, *_ ) in arms:
        sel = [r for r in rows if r["name"] == n]
        print(f"{n:14s} {np.mean([r['dist'] for r in sel]):5.2f}  "
              f"{np.mean([r['near3'] for r in sel]):.2f}   {np.mean([r['oriented'] for r in sel]):.2f}     "
              f"{np.mean([r['hits'] for r in sel]):4.0f}  {np.mean([r['f'] for r in sel]):.2f}  "
              f"{np.mean([r['flow'] for r in sel]):.2f}")
    print("(references: still 4.50 dist; random 7.16; P-controller 0.81 / near3 1.00)")


if __name__ == "__main__":
    main()
