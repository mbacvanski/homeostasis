"""H38c: joint GA on a nearly-circular ellipse (a=4.5, b=4.0)."""
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
from h33_evolve_pursuit import random_genome, mutate, crossover, tournament  # noqa: E402

LAB = Path(__file__).resolve().parents[1] / "out" / "lab"
RES_KEYS = ("n_nodes", "p_link", "input_weight", "weight_init_mean",
            "leak", "target_lr", "threshold_ratio", "weight_lr")


def evaluate(task):
    genome, seed = task
    pc = PursuitConfig(eye_offsets=(0.0,), sensors_per_eye=91,
                       wheel_base=genome["wheel_base"],
                       intensity_scale=genome["intensity_scale"],
                       stimulus_motion="ellipse", ellipse_a=4.5, ellipse_b=4.0)
    res = ReservoirConfig(n_inputs=pc.n_sensors, **{k: genome[k] for k in RES_KEYS})
    h = run_pursuit(n_steps=3600, seed=seed, reservoir_config=res, pursuit_config=pc)
    late = slice(1800, None)
    d = float(h.dist[late].mean())
    near = float((h.dist[late] < 3).mean())
    speed = float(np.hypot(np.diff(h.x), np.diff(h.y))[1800:].mean())
    spread = float(np.hypot(h.x[late].std(), h.y[late].std()))
    return dict(fit=near - d / 15.0, near3=near, dist=d, speed=speed, spread=spread)


def main():
    rng = np.random.default_rng(41)
    pop = [(random_genome(rng), int(rng.integers(0, 100000))) for _ in range(24)]
    log = []
    with ProcessPoolExecutor(10) as pool:
        for gen in range(10):
            rows = list(pool.map(evaluate, pop, chunksize=2))
            fits = [r["fit"] for r in rows]
            b = int(np.argmax(fits))
            log.append(dict(gen=gen, **{k: rows[b][k] for k in ("near3", "dist", "speed", "spread")},
                            champion=pop[b][0], champ_seed=pop[b][1]))
            print(f"gen {gen:02d}  near3 {rows[b]['near3']:.2f} dist {rows[b]['dist']:.2f} "
                  f"speed {rows[b]['speed']:.3f} spread {rows[b]['spread']:.2f}", flush=True)
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
    (LAB / "h38c_interp.json").write_text(json.dumps(log))
    best = max(log, key=lambda l: l["near3"])
    verdict = "FOLLOWER" if best["speed"] > 0.05 and best["spread"] > 1 and best["near3"] >= 0.8 else \
              ("partial" if best["near3"] >= 0.5 else "fail")
    print(f"best: near3 {best['near3']:.2f} speed {best['speed']:.3f} spread {best['spread']:.2f} -> {verdict}")


if __name__ == "__main__":
    main()
