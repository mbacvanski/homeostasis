"""H30: evolve for flank-band occupancy (GA helpers per evolve_viability.py)."""
from __future__ import annotations
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from common import ERR_EDGES, run_closed_loop

LAB = Path(__file__).resolve().parents[1] / "out" / "lab"
CENTERS = (ERR_EDGES[:-1] + ERR_EDGES[1:]) / 2
FLANK = (np.abs(CENTERS) >= 50) & (np.abs(CENTERS) <= 90)
FOVEA = np.abs(CENTERS) <= 45

GENOME = {   # name: (low, high, log, integer)
    "n_nodes": (64, 320, True, True),
    "p_link": (0.03, 0.3, True, False),
    "input_weight": (0.2, 3.0, True, False),
    "weight_init_mean": (0.1, 3.0, True, False),
    "leak": (0.05, 0.7, False, False),
    "target_lr": (0.001, 0.1, True, False),
    "threshold_ratio": (1.2, 4.0, False, False),
    "weight_lr": (0.01, 2.0, True, False),
    "gain": (2.0, 40.0, True, False),
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
    res = {k: v for k, v in genome.items() if k != "gain"}
    r = run_closed_loop(dict(res=res, trk={"gain": genome["gain"]}, seed=seed,
                             arm="full", n_steps=3600, snap_every=3600))
    cnt = np.array(r["policy"]["count"], float)
    tot = max(cnt.sum(), 1)
    return dict(flank=float(cnt[FLANK].sum() / tot),
                fovea=float(cnt[FOVEA].sum() / tot),
                within45=r["score_late"])


def main():
    rng = np.random.default_rng(7)
    pop = [random_genome(rng) for _ in range(24)]
    log = []
    with ProcessPoolExecutor(10) as pool:
        for gen in range(12):
            seeds = rng.integers(0, 10_000, size=3)
            tasks = [(g, int(s)) for g in pop for s in seeds]
            rows = list(pool.map(evaluate, tasks, chunksize=2))
            fits, w45s = [], []
            for i in range(len(pop)):
                sub = rows[i * 3:(i + 1) * 3]
                fits.append(np.mean([r["flank"] for r in sub]))
                w45s.append(np.mean([r["within45"] for r in sub]))
            best = int(np.argmax(fits))
            log.append(dict(gen=gen, mean=float(np.mean(fits)), best=float(fits[best]),
                            best_w45=float(w45s[best]), champion=pop[best]))
            print(f"gen {gen:02d}  mean flank {np.mean(fits):.3f}  best {fits[best]:.3f} "
                  f"(within45 {w45s[best]:.3f})", flush=True)
            elite = pop[best]
            new = [elite]
            while len(new) < len(pop):
                child = crossover(tournament(pop, fits, rng), tournament(pop, fits, rng), rng)
                new.append(mutate(child, rng))
            pop = new
    (LAB / "h30_evolve_flank.json").write_text(json.dumps(log))
    print("champion:", json.dumps(log[-1]["champion"]))


if __name__ == "__main__":
    main()
