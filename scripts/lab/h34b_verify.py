"""H34b: champion-pair checks — genome lottery, long horizon, waypoint generalization."""
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
RES_KEYS = ("n_nodes", "p_link", "input_weight", "weight_init_mean",
            "leak", "target_lr", "threshold_ratio", "weight_lr")


def run_one(task):
    champ, seed, motion, n = task
    pc = PursuitConfig(eye_offsets=(0.0,), sensors_per_eye=91,
                       wheel_base=champ["wheel_base"],
                       intensity_scale=champ["intensity_scale"],
                       stimulus_motion=motion)
    res = ReservoirConfig(n_inputs=pc.n_sensors, **{k: champ[k] for k in RES_KEYS})
    h = run_pursuit(n_steps=n, seed=seed, reservoir_config=res, pursuit_config=pc)
    late = slice(n // 2, None)
    return dict(seed=seed, motion=motion, n=n,
                dist=float(h.dist[late].mean()),
                near3=float((h.dist[late] < 3).mean()),
                hits=int(h.hit.sum()))


def main():
    log = json.loads((LAB / "h34_joint.json").read_text())[-1]
    champ, cseed = log["champion"], log["champ_seed"]
    tasks = ([(champ, 2000 + s, "orbit", 3600) for s in range(16)]      # genome lottery
             + [(champ, cseed, "orbit", 14400)]                          # long horizon
             + [(champ, cseed, "waypoint", 7200)]                        # generalization
             + [(champ, 2000 + s, "waypoint", 3600) for s in range(4)])
    with ProcessPoolExecutor(10) as pool:
        rows = list(pool.map(run_one, tasks))
    (LAB / "h34b_verify.json").write_text(json.dumps(rows))
    lot = [r for r in rows if r["motion"] == "orbit" and r["n"] == 3600]
    print("genome x 16 fresh wirings (orbit): near3 sorted",
          sorted([round(r["near3"], 2) for r in lot], reverse=True))
    print(f"   median {np.median([r['near3'] for r in lot]):.2f}")
    lh = next(r for r in rows if r["n"] == 14400)
    print(f"champion pair, 14400 steps: near3 {lh['near3']:.2f} dist {lh['dist']:.2f} hits {lh['hits']}")
    wp = next(r for r in rows if r["seed"] == cseed and r["motion"] == "waypoint")
    print(f"champion pair, WAYPOINT motion: near3 {wp['near3']:.2f} dist {wp['dist']:.2f} hits {wp['hits']}")
    wl = [r for r in rows if r["motion"] == "waypoint" and r["seed"] != cseed]
    print("genome x fresh wirings (waypoint): near3",
          [round(r["near3"], 2) for r in wl])


if __name__ == "__main__":
    main()
