"""H35: reward-gated weight plasticity on waypoint pursuit."""
from __future__ import annotations
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from common import make_configs  # noqa: F401
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from homeostasis.reservoir import HomeostaticReservoir, ReservoirConfig  # noqa: E402
from homeostasis.pursuit import PursuitConfig, PursuitEnv  # noqa: E402

LAB = Path(__file__).resolve().parents[1] / "out" / "lab"
RES_KEYS = ("n_nodes", "p_link", "input_weight", "weight_init_mean",
            "leak", "target_lr", "threshold_ratio", "weight_lr")


def run_gated(task):
    champ, seed, beta = task
    pc = PursuitConfig(eye_offsets=(0.0,), sensors_per_eye=91,
                       wheel_base=champ["wheel_base"],
                       intensity_scale=champ["intensity_scale"],
                       stimulus_motion="waypoint")
    res = ReservoirConfig(n_inputs=pc.n_sensors, **{k: champ[k] for k in RES_KEYS})
    net = HomeostaticReservoir(res, seed=seed)
    env = PursuitEnv(pc, rng=net.rng)
    n = 7200
    dist = np.empty(n)
    hits = 0
    prev_d = env.distance()
    for i in range(n):
        dist[i] = env.distance()
        w_pre = net.weights.copy()
        state = net.step(env.sense())
        if beta != 0.0:
            r = float(np.clip((prev_d - env.distance()) / 0.15, -1.0, 1.0))
            net.weights = w_pre + (1.0 + beta * r) * (net.weights - w_pre)
        prev_d = env.distance()
        _, h = env.apply_action(*map(float, state.outputs))
        hits += h
        env.advance_stimulus()
    late = slice(n // 2, None)
    return dict(seed=seed, beta=beta, dist=float(dist[late].mean()),
                near3=float((dist[late] < 3).mean()), hits=int(hits))


def main():
    champ = json.loads((LAB / "h34_joint.json").read_text())[-1]["champion"]
    tasks = [(champ, 2000 + s, b) for b in (0.0, 2.0, 5.0) for s in range(16)]
    with ProcessPoolExecutor(10) as pool:
        rows = list(pool.map(run_gated, tasks, chunksize=2))
    (LAB / "h35_reward_gate.json").write_text(json.dumps(rows))
    for b in (0.0, 2.0, 5.0):
        sel = [r for r in rows if r["beta"] == b]
        nn = sorted([round(r["near3"], 2) for r in sel], reverse=True)[:5]
        print(f"beta={b}: near3 mean {np.mean([r['near3'] for r in sel]):.2f} "
              f"median {np.median([r['near3'] for r in sel]):.2f}  dist {np.mean([r['dist'] for r in sel]):.2f}  "
              f"top5 {nn}")


if __name__ == "__main__":
    main()
