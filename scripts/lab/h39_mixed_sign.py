"""H39: mixed-sign embodiment — joint GA with vs without wall sensors,
stimulus orbit radius 6.0 (wall-adjacent conflict)."""
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


def make(genome, walls):
    pc = PursuitConfig(eye_offsets=(0.0,), sensors_per_eye=91,
                       wheel_base=genome["wheel_base"],
                       intensity_scale=genome["intensity_scale"],
                       stimulus_motion="orbit", orbit_radius=6.0,
                       wall_sensors=walls)
    res = ReservoirConfig(n_inputs=pc.n_sensors, **{k: genome[k] for k in RES_KEYS})
    return pc, res


def evaluate(task):
    genome, seed, walls = task
    pc, res = make(genome, walls)
    h = run_pursuit(n_steps=3600, seed=seed, reservoir_config=res, pursuit_config=pc)
    late = slice(1800, None)
    d = float(h.dist[late].mean())
    near = float((h.dist[late] < 3).mean())
    return dict(fit=near - d / 15.0, near3=near, dist=d, hits=int(h.hit.sum()))


def ga(walls, gens=10, seed0=61):
    rng = np.random.default_rng(seed0)
    pop = [(random_genome(rng), int(rng.integers(0, 100000))) for _ in range(24)]
    best = None
    with ProcessPoolExecutor(10) as pool:
        for gen in range(gens):
            rows = list(pool.map(evaluate, [(g, s, walls) for g, s in pop], chunksize=2))
            fits = [r["fit"] for r in rows]
            bi = int(np.argmax(fits))
            if best is None or rows[bi]["fit"] > best["fit"]:
                best = dict(rows[bi], champion=pop[bi][0], champ_seed=pop[bi][1], gen=gen)
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


def compromise(best, walls):
    """corr(agent-stimulus distance, stimulus wall proximity) over late orbit."""
    pc, res = make(best["champion"], walls)
    h = run_pursuit(n_steps=7200, seed=best["champ_seed"], reservoir_config=res,
                    pursuit_config=pc)
    late = slice(3600, None)
    swall = np.minimum.reduce([h.sx[late], h.sy[late], 15 - h.sx[late], 15 - h.sy[late]])
    d = h.dist[late]
    if d.std() < 1e-9 or swall.std() < 1e-9:
        return 0.0, float(d.mean()), int(h.hit.sum())
    return (float(np.corrcoef(-swall, d)[0, 1]), float(d.mean()), int(h.hit.sum()))


def main():
    out = {}
    for walls in (False, True):
        b = ga(walls)
        c, dmean, hits = compromise(b, walls)
        out[str(walls)] = dict(best=b, compromise_corr=c, long_dist=dmean, long_hits=hits)
        print(f"wall_sensors={walls}: best near3 {b['near3']:.2f} dist {b['dist']:.2f} "
              f"hits {b['hits']} (gen {b['gen']})  | long-run compromise corr "
              f"{c:+.2f}  dist {dmean:.2f} hits {hits}", flush=True)
    (LAB / "h39_mixed.json").write_text(json.dumps(out))


if __name__ == "__main__":
    main()
