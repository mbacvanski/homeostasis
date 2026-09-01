"""H37: followability vs target-motion persistence (champion pair)."""
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
# T_c ~ 1/sigma^2 (rad^2): sigma for T_c in {inf, 3000, 1000, 300, 100, 30}
SIGMAS = [(0.0, "inf"), (0.018, "3000"), (0.032, "1000"), (0.058, "300"),
          (0.1, "100"), (0.18, "30")]


def evaluate(task):
    champ, cseed, sigma, label, rep = task
    pc = PursuitConfig(eye_offsets=(0.0,), sensors_per_eye=91,
                       wheel_base=champ["wheel_base"],
                       intensity_scale=champ["intensity_scale"],
                       stimulus_motion="wander", wander_sigma=sigma)
    res = ReservoirConfig(n_inputs=pc.n_sensors, **{k: champ[k] for k in RES_KEYS})
    # same wiring seed; vary the trajectory by burning rep draws from the env rng
    h = run_pursuit(n_steps=7200 + rep, seed=cseed, reservoir_config=res,
                    pursuit_config=pc)
    late = slice((7200 + rep) // 2, None)
    return dict(label=label, rep=rep, dist=float(h.dist[late].mean()),
                near3=float((h.dist[late] < 3).mean()), hits=int(h.hit.sum()))


def main():
    log = json.loads((LAB / "h34_joint.json").read_text())[-1]
    champ, cseed = log["champion"], log["champ_seed"]
    tasks = [(champ, cseed, s, lab, rep) for (s, lab) in SIGMAS for rep in range(6)]
    with ProcessPoolExecutor(10) as pool:
        rows = list(pool.map(evaluate, tasks, chunksize=2))
    (LAB / "h37_followability.json").write_text(json.dumps(rows))
    print("T_c (steps)   near3 mean±sd     dist    hits")
    for (_, lab) in SIGMAS:
        sel = [r for r in rows if r["label"] == lab]
        print(f"   {lab:>5s}     {np.mean([r['near3'] for r in sel]):.2f}±"
              f"{np.std([r['near3'] for r in sel]):.2f}     "
              f"{np.mean([r['dist'] for r in sel]):5.2f}   "
              f"{np.mean([r['hits'] for r in sel]):4.0f}")


if __name__ == "__main__":
    main()
