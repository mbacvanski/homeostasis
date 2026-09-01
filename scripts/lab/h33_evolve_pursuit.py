"""H33: evolve pursuit (GA per evolve_viability protocol)."""
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
GENOME = {
    "n_nodes": (64, 320, True, True),
    "p_link": (0.03, 0.3, True, False),
    "input_weight": (1.0, 16.0, True, False),
    "weight_init_mean": (0.1, 4.0, True, False),
    "leak": (0.05, 0.7, False, False),
    "target_lr": (0.001, 0.1, True, False),
    "threshold_ratio": (1.2, 4.0, False, False),
    "weight_lr": (0.02, 2.0, True, False),
    "wheel_base": (1.0, 16.0, True, False),
    "intensity_scale": (1.0, 9.0, True, False),
}


def random_genome(rng):
    g = {}
    for k, (lo, hi, log, integer) in GENOME.items():
        v = np.exp(rng.uniform(np.log(lo), np.log(hi))) if log else rng.uniform(lo, hi)
        g[k] = int(round(v)) if integer else float(v)
    return g


def mutate(g, rng, rate=0.45):
    out = dict(g)
    for k, (lo, hi, log, integer) in GENOME.items():
        if rng.random() < rate:
            v = out[k]
            v = float(np.exp(np.log(v) + rng.normal(0, 0.3))) if log else v + rng.normal(0, (hi - lo) * 0.15)
            v = min(max(v, lo), hi)
            out[k] = int(round(v)) if integer else v
    return out


def crossover(a, b, rng):
    return {k: (a if rng.random() < 0.5 else b)[k] for k in GENOME}


def tournament(pop, fits, rng, k=3):
    idx = rng.choice(len(pop), size=k, replace=False)
    return pop[int(idx[np.argmax(np.asarray(fits)[idx])])]


def evaluate(task):
    genome, seed = task
    pc = PursuitConfig(eye_offsets=(0.0,), sensors_per_eye=91,
                       wheel_base=genome["wheel_base"],
                       intensity_scale=genome["intensity_scale"])
    res_keys = ("n_nodes", "p_link", "input_weight", "weight_init_mean",
                "leak", "target_lr", "threshold_ratio", "weight_lr")
    res = ReservoirConfig(n_inputs=pc.n_sensors,
                          **{k: genome[k] for k in res_keys})
    h = run_pursuit(n_steps=3600, seed=seed, reservoir_config=res, pursuit_config=pc)
    late = slice(1800, None)
    d = float(h.dist[late].mean())
    near = float((h.dist[late] < 3).mean())
    return dict(fit=near - d / 15.0, near3=near, dist=d)


def main():
    rng = np.random.default_rng(11)
    pop = [random_genome(rng) for _ in range(24)]
    log = []
    with ProcessPoolExecutor(10) as pool:
        for gen in range(14):
            seeds = rng.integers(0, 10_000, size=3)
            tasks = [(g, int(s)) for g in pop for s in seeds]
            rows = list(pool.map(evaluate, tasks, chunksize=2))
            fits, nears, dists = [], [], []
            for i in range(len(pop)):
                sub = rows[i * 3:(i + 1) * 3]
                fits.append(np.mean([r["fit"] for r in sub]))
                nears.append(np.mean([r["near3"] for r in sub]))
                dists.append(np.mean([r["dist"] for r in sub]))
            b = int(np.argmax(fits))
            log.append(dict(gen=gen, best_fit=float(fits[b]), best_near=float(nears[b]),
                            best_dist=float(dists[b]), mean_near=float(np.mean(nears)),
                            champion=pop[b]))
            print(f"gen {gen:02d}  best near3 {nears[b]:.2f} dist {dists[b]:.2f}  "
                  f"pop near3 {np.mean(nears):.2f}", flush=True)
            elite = pop[b]
            new = [elite]
            while len(new) < len(pop):
                new.append(mutate(crossover(tournament(pop, fits, rng),
                                            tournament(pop, fits, rng), rng), rng))
            pop = new
    (LAB / "h33_evolve_pursuit.json").write_text(json.dumps(log))
    print("champion:", json.dumps(log[-1]["champion"]))


if __name__ == "__main__":
    main()
