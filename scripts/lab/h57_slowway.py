"""H57: does slowing aperiodic waypoint motion past the lock horizon unlock pursuit?"""
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
    genome, seed, speed = task
    pc = PursuitConfig(eye_offsets=(0.0,), sensors_per_eye=91,
                       stimulus_motion="waypoint", stimulus_speed=speed,
                       wheel_base=genome["wheel_base"],
                       intensity_scale=genome["intensity_scale"])
    res = ReservoirConfig(n_inputs=pc.n_sensors, **{k: genome[k] for k in RES_KEYS})
    h = run_pursuit(n_steps=7200, seed=seed, reservoir_config=res, pursuit_config=pc)
    late = slice(3600, None)
    return dict(near3=float((h.dist[late] < 3).mean()),
                dist=float(h.dist[late].mean()))

def main():
    out = {}
    with ProcessPoolExecutor(10) as pool:
        for speed in (0.15, 0.08, 0.04):
            for label, g in (("champ", H34_CHAMP),
                             ("blind", {**H34_CHAMP, "input_weight": 1e-6})):
                rows = list(pool.map(evaluate, [(g, s, speed) for s in range(41, 49)]))
                out[f"{label}@{speed}"] = rows
                print(f"speed {speed} {label}: near3 {np.mean([r['near3'] for r in rows]):.3f}"
                      f"  dist {np.mean([r['dist'] for r in rows]):.2f}")
    (LAB / "h57_slowway.json").write_text(json.dumps(out))
    for speed in (0.15, 0.08, 0.04):
        gap = (np.mean([r["near3"] for r in out[f"champ@{speed}"]])
               - np.mean([r["near3"] for r in out[f"blind@{speed}"]]))
        print(f"speed {speed}: skill gap {gap:+.3f}")

if __name__ == "__main__":
    main()
