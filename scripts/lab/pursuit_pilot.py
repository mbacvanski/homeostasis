"""Pursuit pilot (seeds 0-7, orbit): tune (input_weight x wlr), with null
baselines (random-turn, still, P-pursuit ceiling). Tuning only — the
preregistered battery runs on FRESH seeds afterwards."""
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
from homeostasis.pursuit import PursuitConfig, PursuitEnv  # noqa: E402

LAB = Path(__file__).resolve().parents[1] / "out" / "lab"


def null_run(task):
    policy, seed = task
    env = PursuitEnv(PursuitConfig(), rng=np.random.default_rng(seed))
    rng = np.random.default_rng(seed + 1)
    n = 3600
    dist = np.empty(n); hits = 0
    for i in range(n):
        dist[i] = env.distance()
        if policy == "still":
            e1 = e2 = 0.0
        elif policy == "random":
            e1, e2 = rng.uniform(0, 1, 2)
        else:  # P-pursuit ceiling: turn toward bearing, full speed
            b = np.deg2rad(env.stimulus_bearing_deg())
            omega = float(np.clip(b, -1.0, 1.0))
            e1 = np.clip(0.5 - omega / 2, 0, 1)
            e2 = np.clip(0.5 + omega / 2, 0, 1)
            e1, e2 = e1 * 0.9 + 0.05, e2 * 0.9 + 0.05
        _, h = env.apply_action(float(e1), float(e2))
        hits += h
        env.advance_stimulus()
    late = dist[1800:]
    return dict(kind=policy, seed=seed, dist=float(late.mean()),
                near3=float((late < 3).mean()), hits=int(hits))


def evaluate(task):
    w_in, wlr, seed = task
    res = ReservoirConfig(n_inputs=62, input_weight=w_in, weight_lr=wlr)
    h = run_pursuit(n_steps=3600, seed=seed, reservoir_config=res)
    late = slice(1800, None)
    return dict(w_in=w_in, wlr=wlr, seed=seed,
                dist=float(h.dist[late].mean()),
                near3=float((h.dist[late] < 3).mean()),
                hits=int(h.hit.sum()), f=float(h.prop_spiked[late].mean()),
                flow=float(h.flow[late].mean()))


def main():
    tasks = [(w, l, s) for w in (2.0, 4.0, 8.0) for l in (0.1, 0.3, 1.0) for s in range(8)]
    nulls = [(p, s) for p in ("still", "random", "pursuitP") for s in range(8)]
    with ProcessPoolExecutor(10) as pool:
        rows = list(pool.map(evaluate, tasks, chunksize=2))
        nrows = list(pool.map(null_run, nulls, chunksize=2))
    (LAB / "pursuit_pilot.json").write_text(json.dumps(dict(grid=rows, nulls=nrows)))
    print("nulls (late mean dist / frac within 3 / hits):")
    for p in ("still", "random", "pursuitP"):
        sel = [r for r in nrows if r["kind"] == p]
        print(f"   {p:9s} {np.mean([r['dist'] for r in sel]):.2f}  "
              f"{np.mean([r['near3'] for r in sel]):.2f}  {np.mean([r['hits'] for r in sel]):.0f}")
    print("\ngrid (late dist / near3 / f):")
    for w in (2.0, 4.0, 8.0):
        line = f"   w_in={w:<4}"
        for l in (0.1, 0.3, 1.0):
            sel = [r for r in rows if r["w_in"] == w and r["wlr"] == l]
            line += (f"  wlr{l}: {np.mean([r['dist'] for r in sel]):.2f}/"
                     f"{np.mean([r['near3'] for r in sel]):.2f}/"
                     f"{np.mean([r['f'] for r in sel]):.2f}")
        print(line)


if __name__ == "__main__":
    main()
