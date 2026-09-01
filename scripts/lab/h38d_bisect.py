"""H38d: bisect the constant-control tolerance (curvature ratios 1.6, 2.5)."""
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
AB = {"r1.6": (4.5, 2.8125), "r2.5": (5.0, 2.0)}


def evaluate(task):
    genome, seed, name = task
    a, b = AB[name]
    pc = PursuitConfig(eye_offsets=(0.0,), sensors_per_eye=91,
                       wheel_base=genome["wheel_base"],
                       intensity_scale=genome["intensity_scale"],
                       stimulus_motion="ellipse", ellipse_a=a, ellipse_b=b)
    res = ReservoirConfig(n_inputs=pc.n_sensors, **{k: genome[k] for k in RES_KEYS})
    h = run_pursuit(n_steps=3600, seed=seed, reservoir_config=res, pursuit_config=pc)
    late = slice(1800, None)
    d = float(h.dist[late].mean())
    near = float((h.dist[late] < 3).mean())
    speed = float(np.hypot(np.diff(h.x), np.diff(h.y))[1800:].mean())
    spread = float(np.hypot(h.x[late].std(), h.y[late].std()))
    return dict(fit=near - d / 15.0, near3=near, dist=d, speed=speed, spread=spread)


def ga(name, gens=10, seed0=53):
    rng = np.random.default_rng(seed0)
    pop = [(random_genome(rng), int(rng.integers(0, 100000))) for _ in range(24)]
    best = None
    with ProcessPoolExecutor(10) as pool:
        for gen in range(gens):
            rows = list(pool.map(evaluate, [(g, s, name) for g, s in pop], chunksize=2))
            fits = [r["fit"] for r in rows]
            bi = int(np.argmax(fits))
            cand = dict(gen=gen, **{k: rows[bi][k] for k in ("near3", "dist", "speed", "spread")})
            if best is None or cand["near3"] > best["near3"]:
                best = cand
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
    return best


def main():
    out = {}
    for name in ("r1.6", "r2.5"):
        out[name] = ga(name)
        b = out[name]
        kind = "FOLLOWER" if b["speed"] > 0.05 and b["spread"] > 1 and b["near3"] >= 0.8 else "toll-booth/partial"
        print(f"{name}: near3 {b['near3']:.2f} dist {b['dist']:.2f} speed {b['speed']:.3f} "
              f"spread {b['spread']:.2f} -> {kind}", flush=True)
    (LAB / "h38d_bisect.json").write_text(json.dumps(out))


if __name__ == "__main__":
    main()
