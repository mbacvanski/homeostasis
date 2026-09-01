"""Pursuit round 5 (corrected): velocity-matching test ON THE ALIVE BASE
(360-deg retina, w_in 8, wb 4). wlr x stimulus_speed."""
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
    wlr, sspeed, seed = task
    pc = PursuitConfig(wheel_base=4.0, stimulus_speed=sspeed, **R360)
    res = ReservoirConfig(n_inputs=pc.n_sensors, input_weight=8.0, weight_lr=wlr)
    h = run_pursuit(n_steps=3600, seed=seed, reservoir_config=res, pursuit_config=pc)
    late = slice(1800, None)
    br = np.abs(h.bearing[late])
    speed = np.hypot(np.diff(h.x), np.diff(h.y))
    return dict(wlr=wlr, sspeed=sspeed, seed=seed,
                dist=float(h.dist[late].mean()),
                near3=float((h.dist[late] < 3).mean()),
                oriented=float((br <= 45).mean()),
                cruise=float(speed[1800:].mean()),
                f=float(h.prop_spiked[late].mean()))


def main():
    tasks = [(l, s, seed) for l in (0.05, 0.1, 0.3) for s in (0.15, 0.3, 0.5)
             for seed in range(8)]
    with ProcessPoolExecutor(10) as pool:
        rows = list(pool.map(evaluate, tasks, chunksize=2))
    (LAB / "pursuit_rescue5.json").write_text(json.dumps(rows))
    print("wlr    sspeed  dist  near3  orient  cruise   f")
    for l in (0.05, 0.1, 0.3):
        for s in (0.15, 0.3, 0.5):
            sel = [r for r in rows if r["wlr"] == l and r["sspeed"] == s]
            print(f"{l:<5}  {s:<5}  {np.mean([r['dist'] for r in sel]):5.2f} "
                  f"{np.mean([r['near3'] for r in sel]):.2f}   {np.mean([r['oriented'] for r in sel]):.2f}    "
                  f"{np.mean([r['cruise'] for r in sel]):.2f}   {np.mean([r['f'] for r in sel]):.2f}")


if __name__ == "__main__":
    main()
