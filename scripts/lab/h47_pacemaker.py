"""H47: evolve a follower of a recorded wall-circler trajectory."""
from __future__ import annotations
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from common import make_configs  # noqa: F401
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from homeostasis.reservoir import HomeostaticReservoir, ReservoirConfig  # noqa: E402
from homeostasis.simulation import run_wall  # noqa: E402
from homeostasis.pursuit import PursuitConfig, PursuitEnv  # noqa: E402
from h33_evolve_pursuit import random_genome, mutate, crossover, tournament  # noqa: E402

LAB = Path(__file__).resolve().parents[1] / "out" / "lab"
RES_KEYS = ("n_nodes", "p_link", "input_weight", "weight_init_mean",
            "leak", "target_lr", "threshold_ratio", "weight_lr")

_wall = run_wall(n_steps=3600, seed=0)
# one clean orbital period from the settled circler: find the lap length by
# minimizing return distance around the estimated period
_xy = np.stack([_wall.x, _wall.y], axis=1)
_start = 3000
_best_T, _best_gap = None, 1e9
for T in range(30, 80):
    gap = float(np.hypot(*(_xy[_start] - _xy[_start + T])))
    if gap < _best_gap:
        _best_gap, _best_T = gap, T
_loop = _xy[_start:_start + _best_T]
# replay at 1/3 speed via linear interpolation (band-limited pacemaker test)
_t = np.linspace(0, len(_loop), len(_loop) * 3, endpoint=False)
_i0 = np.floor(_t).astype(int) % len(_loop)
_i1 = (_i0 + 1) % len(_loop)
_fr = (_t - np.floor(_t))[:, None]
TRAJ = _loop[_i0] * (1 - _fr) + _loop[_i1] * _fr
print(f"pacemaker loop: period {len(TRAJ)} steps (x3 slowed), closure gap {_best_gap:.3f}")


def evaluate(task):
    genome, seed = task
    pc = PursuitConfig(eye_offsets=(0.0,), sensors_per_eye=91,
                       wheel_base=genome["wheel_base"],
                       intensity_scale=genome["intensity_scale"])
    res = ReservoirConfig(n_inputs=pc.n_sensors, **{k: genome[k] for k in RES_KEYS})
    net = HomeostaticReservoir(res, seed=seed)
    env = PursuitEnv(pc, rng=net.rng)
    n = 3600
    dist = np.empty(n)
    for i in range(n):
        env.sx, env.sy = TRAJ[i % len(TRAJ)]
        dist[i] = env.distance()
        state = net.step(env.sense())
        env.apply_action(*map(float, state.outputs))
        env.steps += 1
    late = slice(1800, None)
    d = float(dist[late].mean())
    near = float((dist[late] < 3).mean())
    return dict(fit=near - d / 15.0, near3=near, dist=d)


def main():
    rng = np.random.default_rng(71)
    pop = [(random_genome(rng), int(rng.integers(0, 100000))) for _ in range(24)]
    best = None
    with ProcessPoolExecutor(10) as pool:
        for gen in range(10):
            rows = list(pool.map(evaluate, pop, chunksize=2))
            fits = [r["fit"] for r in rows]
            bi = int(np.argmax(fits))
            if best is None or rows[bi]["fit"] > best["fit"]:
                best = dict(rows[bi], gen=gen, champion=pop[bi][0], champ_seed=pop[bi][1])
            print(f"gen {gen:02d}  best near3 {rows[bi]['near3']:.2f} dist {rows[bi]['dist']:.2f}",
                  flush=True)
            elite = pop[bi]
            new = [elite]
            while len(new) < len(pop):
                pa = tournament(pop, fits, rng); pb = tournament(pop, fits, rng)
                g = mutate(crossover(pa[0], pb[0], rng), rng)
                s = pa[1] if rng.random() < 0.5 else pb[1]
                if rng.random() < 0.25:
                    s = int(rng.integers(0, 100000))
                new.append((g, int(s)))
            pop = new
    (LAB / "h47_pacemaker.json").write_text(json.dumps(best))
    print(f"BEST: near3 {best['near3']:.2f} dist {best['dist']:.2f} (gen {best['gen']})")


if __name__ == "__main__":
    main()
