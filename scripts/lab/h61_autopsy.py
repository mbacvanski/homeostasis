"""H61: why vision harms on loitering motion — motor stereotypy autopsy."""
from __future__ import annotations
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from homeostasis.reservoir import ReservoirConfig  # noqa: E402
from homeostasis.simulation import run_pursuit  # noqa: E402
from homeostasis.pursuit import PursuitConfig  # noqa: E402
from h55_intercept import H34_CHAMP  # noqa: E402

LAB = Path(__file__).resolve().parents[1] / "out" / "lab"
RES_KEYS = ("n_nodes", "p_link", "input_weight", "weight_init_mean",
            "leak", "target_lr", "threshold_ratio", "weight_lr")

def evaluate(task):
    genome, seed, motion = task
    pc = PursuitConfig(eye_offsets=(0.0,), sensors_per_eye=91,
                       stimulus_motion=motion, stimulus_speed=0.04 if motion == "waypoint" else 0.15,
                       wheel_base=genome["wheel_base"],
                       intensity_scale=genome["intensity_scale"])
    res = ReservoirConfig(n_inputs=pc.n_sensors, **{k: genome[k] for k in RES_KEYS})
    h = run_pursuit(n_steps=7200, seed=seed, reservoir_config=res, pursuit_config=pc)
    late = slice(3600, None)
    dh = np.diff(np.unwrap(h.heading))[3600:]
    wall = float(np.mean((np.minimum(h.x[late], 15 - h.x[late]) < 2)
                         | (np.minimum(h.y[late], 15 - h.y[late]) < 2)))
    same_sign = max(float(np.mean(dh > 0)), float(np.mean(dh < 0)))
    return dict(motion=motion, seed=seed,
                turn=float(np.median(np.abs(np.rad2deg(dh)))),
                one_sided=same_sign, wall=wall,
                dist=float(h.dist[late].mean()),
                speed=float(np.hypot(np.diff(h.x), np.diff(h.y))[3600:].mean()))

def main():
    combos = [("champ-orbit", H34_CHAMP, "orbit"),
              ("champ-way", H34_CHAMP, "waypoint"),
              ("blind-way", {**H34_CHAMP, "input_weight": 1e-6}, "waypoint")]
    out = {}
    with ProcessPoolExecutor(10) as pool:
        for label, g, m in combos:
            rows = list(pool.map(evaluate, [(g, s, m) for s in range(41, 49)]))
            out[label] = rows
            print(f"{label:<12} turn {np.mean([r['turn'] for r in rows]):5.2f} deg/step"
                  f" | one-sided {np.mean([r['one_sided'] for r in rows]):.2f}"
                  f" | wall-time {np.mean([r['wall'] for r in rows]):.2f}"
                  f" | dist {np.mean([r['dist'] for r in rows]):.2f}"
                  f" | speed {np.mean([r['speed'] for r in rows]):.3f}")
    (LAB / "h61_autopsy.json").write_text(json.dumps(out))

if __name__ == "__main__":
    main()
