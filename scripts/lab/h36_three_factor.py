"""H36: three-factor reward-directed credit on waypoint pursuit."""
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


def run_three_factor(task):
    champ, seed, eta, shuffle = task
    pc = PursuitConfig(eye_offsets=(0.0,), sensors_per_eye=91,
                       wheel_base=champ["wheel_base"],
                       intensity_scale=champ["intensity_scale"],
                       stimulus_motion="waypoint")
    res = ReservoirConfig(n_inputs=pc.n_sensors, **{k: champ[k] for k in RES_KEYS})
    net = HomeostaticReservoir(res, seed=seed)
    env = PursuitEnv(pc, rng=net.rng)
    n = 7200
    shuf_rng = np.random.default_rng(seed + 777)
    dist = np.empty(n)
    hits = 0
    prev_d = env.distance()
    rewards = np.empty(n)
    adj = net.adjacency
    for i in range(n):
        dist[i] = env.distance()
        pre = net.spiked.copy()
        state = net.step(env.sense())
        r = float(np.clip((prev_d - env.distance()) / 0.15, -1.0, 1.0))
        rewards[i] = r
        if eta != 0.0:
            r_use = float(shuf_rng.choice(rewards[: i + 1])) if shuffle else r
            if pre.any() and state.spiked.any():
                net.weights += (eta * r_use) * (np.outer(pre, state.spiked) * adj)
        prev_d = env.distance()
        _, h = env.apply_action(*map(float, state.outputs))
        hits += h
        env.advance_stimulus()
    late = slice(n // 2, None)
    return dict(seed=seed, eta=eta, shuffle=bool(shuffle),
                dist=float(dist[late].mean()),
                near3=float((dist[late] < 3).mean()), hits=int(hits))


def main():
    champ = json.loads((LAB / "h34_joint.json").read_text())[-1]["champion"]
    tasks = [(champ, 2000 + s, e, False)
             for e in (0.0, 0.01, 0.05, 0.2) for s in range(12)]
    with ProcessPoolExecutor(10) as pool:
        rows = list(pool.map(run_three_factor, tasks, chunksize=2))
        best_eta = max((e for e in (0.01, 0.05, 0.2)),
                       key=lambda e: np.median([r["near3"] for r in rows if r["eta"] == e]))
        ctrl = list(pool.map(run_three_factor,
                             [(champ, 2000 + s, best_eta, True) for s in range(12)],
                             chunksize=2))
    (LAB / "h36_three_factor.json").write_text(json.dumps(rows + ctrl))
    for e in (0.0, 0.01, 0.05, 0.2):
        sel = [r for r in rows if r["eta"] == e]
        print(f"eta={e:<5} near3 mean {np.mean([r['near3'] for r in sel]):.2f} "
              f"median {np.median([r['near3'] for r in sel]):.2f}  "
              f"dist {np.mean([r['dist'] for r in sel]):.2f}  "
              f"top3 {sorted([round(r['near3'],2) for r in sel], reverse=True)[:3]}")
    print(f"shuffled-r control at eta={best_eta}: "
          f"median {np.median([r['near3'] for r in ctrl]):.2f} "
          f"mean {np.mean([r['near3'] for r in ctrl]):.2f}")


if __name__ == "__main__":
    main()
