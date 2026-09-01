"""Pursuit rescue round 3: wheel_base (motor grain) x retina, 8 seeds."""
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
    name, wb, w_in, wlr, iscale, seed = task
    pc = PursuitConfig(intensity_scale=iscale, wheel_base=wb, **R360)
    res = ReservoirConfig(n_inputs=pc.n_sensors, input_weight=w_in, weight_lr=wlr)
    h = run_pursuit(n_steps=3600, seed=seed, reservoir_config=res, pursuit_config=pc)
    late = slice(1800, None)
    br = np.abs(h.bearing[late])
    return dict(name=name, seed=seed, dist=float(h.dist[late].mean()),
                near3=float((h.dist[late] < 3).mean()),
                oriented=float((br <= 45).mean()),
                hits=int(h.hit.sum()), f=float(h.prop_spiked[late].mean()))


def main():
    arms = [(f"wb{wb}-wlr{l}", wb, 8.0, l, 3.0)
            for wb in (1.0, 4.0, 8.0, 16.0) for l in (0.1, 0.3, 1.0)]
    tasks = [(n, wb, w, l, i, s) for (n, wb, w, l, i) in arms for s in range(8)]
    with ProcessPoolExecutor(10) as pool:
        rows = list(pool.map(evaluate, tasks, chunksize=2))
    (LAB / "pursuit_rescue3.json").write_text(json.dumps(rows))
    print("arm             dist   near3  oriented  hits   f")
    for (n, *_ ) in arms:
        sel = [r for r in rows if r["name"] == n]
        print(f"{n:15s} {np.mean([r['dist'] for r in sel]):5.2f}  "
              f"{np.mean([r['near3'] for r in sel]):.2f}   {np.mean([r['oriented'] for r in sel]):.2f}     "
              f"{np.mean([r['hits'] for r in sel]):4.0f}  {np.mean([r['f'] for r in sel]):.2f}")
    print("(refs: still 4.50; random 7.16; P-ceiling 0.81/1.00; max turn deg/step = 57.3/wb)")


if __name__ == "__main__":
    main()
