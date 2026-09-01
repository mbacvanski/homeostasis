"""H38: joint GA on ellipse vs shuttle motion."""
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
from h33_evolve_pursuit import GENOME, random_genome, mutate, crossover, tournament  # noqa: E402

LAB = Path(__file__).resolve().parents[1] / "out" / "lab"
RES_KEYS = ("n_nodes", "p_link", "input_weight", "weight_init_mean",
            "leak", "target_lr", "threshold_ratio", "weight_lr")


def evaluate(task):
    genome, seed, motion = task
    kw = dict(stimulus_motion=motion)
    if motion == "shuttle":
        kw = dict(stimulus_motion="wander", wander_sigma=0.0)
    pc = PursuitConfig(eye_offsets=(0.0,), sensors_per_eye=91,
                       wheel_base=genome["wheel_base"],
                       intensity_scale=genome["intensity_scale"], **kw)
    res = ReservoirConfig(n_inputs=pc.n_sensors, **{k: genome[k] for k in RES_KEYS})
    h = run_pursuit(n_steps=3600, seed=seed, reservoir_config=res, pursuit_config=pc)
    late = slice(1800, None)
    d = float(h.dist[late].mean())
    near = float((h.dist[late] < 3).mean())
    return dict(fit=near - d / 15.0, near3=near, dist=d)


def ga(motion, gens=10, seed0=31):
    rng = np.random.default_rng(seed0)
    pop = [(random_genome(rng), int(rng.integers(0, 100000))) for _ in range(24)]
    best_log = []
    with ProcessPoolExecutor(10) as pool:
        for gen in range(gens):
            rows = list(pool.map(evaluate, [(g, s, motion) for g, s in pop], chunksize=2))
            fits = [r["fit"] for r in rows]
            nears = [r["near3"] for r in rows]
            b = int(np.argmax(fits))
            best_log.append(dict(gen=gen, best_near=float(nears[b]),
                                 best_dist=float(rows[b]["dist"]),
                                 champion=pop[b][0], champ_seed=pop[b][1]))
            print(f"  {motion} gen {gen:02d}  best near3 {nears[b]:.2f} "
                  f"dist {rows[b]['dist']:.2f}", flush=True)
            elite = pop[b]
            new = [elite]
            while len(new) < len(pop):
                pa = tournament(pop, fits, rng); pb = tournament(pop, fits, rng)
                g = mutate(crossover(pa[0], pb[0], rng), rng)
                s = pa[1] if rng.random() < 0.5 else pb[1]
                if rng.random() < 0.25:
                    s = int(rng.integers(0, 100000))
                new.append((g, int(s)))
            pop = new
    return best_log


def main():
    out = {}
    for motion in ("ellipse", "shuttle"):
        print(f"== {motion}")
        out[motion] = ga(motion)
    (LAB / "h38_manifold.json").write_text(json.dumps(out))
    for m, log in out.items():
        print(f"{m}: final best near3 {log[-1]['best_near']:.2f}  "
              f"max over gens {max(l['best_near'] for l in log):.2f}")


if __name__ == "__main__":
    main()
