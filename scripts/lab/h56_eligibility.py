"""H56: eligibility-traced reward learning vs the homeostatic absorbability barrier."""
from __future__ import annotations
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from homeostasis.reservoir import HomeostaticReservoir, ReservoirConfig  # noqa: E402
from homeostasis.pursuit import PursuitConfig, PursuitEnv  # noqa: E402

LAB = Path(__file__).resolve().parents[1] / "out" / "lab"
RES_KEYS = ("n_nodes", "p_link", "input_weight", "weight_init_mean",
            "leak", "target_lr", "threshold_ratio", "weight_lr")

def run(task):
    champ, seed, lam, eta, mode, shuffle = task
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
    rewards = np.empty(n)
    adj = net.adjacency
    trace = np.zeros_like(net.weights)
    prev_d = env.distance()
    for i in range(n):
        if mode == "frozen-half" and i == 3600:
            net.learning_enabled = False
        dist[i] = env.distance()
        pre = net.spiked.copy()
        state = net.step(env.sense())
        r = float(np.clip((prev_d - env.distance()) / 0.15, -1.0, 1.0))
        rewards[i] = r
        if eta != 0.0:
            if lam > 0.0:
                trace *= lam
                if pre.any() and state.spiked.any():
                    trace += np.outer(pre, state.spiked) * adj
                el = trace
            else:
                el = (np.outer(pre, state.spiked) * adj
                      if pre.any() and state.spiked.any() else None)
            if el is not None:
                r_use = float(shuf_rng.choice(rewards[: i + 1])) if shuffle else r
                net.weights += (eta * r_use) * el
        prev_d = env.distance()
        env.apply_action(*map(float, state.outputs))
        env.advance_stimulus()
    late = slice(n // 2, None)
    return dict(seed=seed, lam=lam, eta=eta, mode=mode, shuffle=bool(shuffle),
                dist=float(dist[late].mean()), near3=float((dist[late] < 3).mean()))

def main():
    champ = json.loads((LAB / "h34_joint.json").read_text())[-1]["champion"]
    cells = ([(lam, eta, mode) for lam in (0.9, 0.97) for eta in (0.003, 0.01, 0.03)
              for mode in ("full", "frozen-half")]
             + [(0.0, 0.01, "full"), (0.0, 0.0, "full"), (0.0, 0.0, "frozen-half")])
    tasks = [(champ, 2000 + s, lam, eta, mode, False)
             for (lam, eta, mode) in cells for s in range(12)]
    with ProcessPoolExecutor(10) as pool:
        rows = list(pool.map(run, tasks, chunksize=2))
        best = max(((lam, eta, mode) for (lam, eta, mode) in cells if eta > 0),
                   key=lambda c: np.median([r["near3"] for r in rows
                                            if (r["lam"], r["eta"], r["mode"]) == c]))
        ctrl = list(pool.map(run, [(champ, 2000 + s, best[0], best[1], best[2], True)
                                   for s in range(12)], chunksize=2))
    (LAB / "h56_eligibility.json").write_text(json.dumps(rows + ctrl))
    print("lam   eta    mode         near3 med (mean)   dist")
    for (lam, eta, mode) in cells:
        sel = [r for r in rows if (r["lam"], r["eta"], r["mode"]) == (lam, eta, mode)]
        print(f"{lam:<5} {eta:<6} {mode:<12} {np.median([r['near3'] for r in sel]):.2f}"
              f" ({np.mean([r['near3'] for r in sel]):.2f})   "
              f"{np.mean([r['dist'] for r in sel]):.2f}")
    print(f"shuffled control at {best}: med {np.median([r['near3'] for r in ctrl]):.2f}")

if __name__ == "__main__":
    main()
