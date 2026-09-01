"""Pursuit rescue round 2: delete the dark-pose basin with a 360-degree
retina (eye_offsets=(0,), 91 sensors x 4 deg = +/-180). If the basin was
the killer, activity persists and orientation is trivial; the open question
becomes whether APPROACH emerges (falloff on) vs mere orientation
(bearing-only)."""
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
R360 = dict(eye_offsets=(0.0,), sensors_per_eye=91)


def evaluate(task):
    name, w_in, wlr, iscale, seed = task
    pc = PursuitConfig(intensity_scale=iscale, **R360)
    res = ReservoirConfig(n_inputs=pc.n_sensors, input_weight=w_in, weight_lr=wlr)
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
    arms = [("r360-w8-fall", 8.0, 0.3, 3.0),
            ("r360-w4-fall", 4.0, 0.3, 3.0),
            ("r360-w8-bear", 8.0, 0.3, 1e9),
            ("r360-w8-wlr1", 8.0, 1.0, 3.0)]
    tasks = [(n, w, l, i, s) for (n, w, l, i) in arms for s in range(8)]
    with ProcessPoolExecutor(10) as pool:
        rows = list(pool.map(evaluate, tasks, chunksize=2))
    (LAB / "pursuit_rescue2.json").write_text(json.dumps(rows))
    print("arm            dist   near3  oriented  hits   f     flow")
    for (n, *_ ) in arms:
        sel = [r for r in rows if r["name"] == n]
        print(f"{n:14s} {np.mean([r['dist'] for r in sel]):5.2f}  "
              f"{np.mean([r['near3'] for r in sel]):.2f}   {np.mean([r['oriented'] for r in sel]):.2f}     "
              f"{np.mean([r['hits'] for r in sel]):4.0f}  {np.mean([r['f'] for r in sel]):.2f}  "
              f"{np.mean([r['flow'] for r in sel]):.2f}")
    print("(refs: still 4.50; random 7.16; P-ceiling 0.81/1.00)")


if __name__ == "__main__":
    main()
