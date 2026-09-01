"""H34: joint (genome, wiring-seed) evolution for pursuit."""
from __future__ import annotations
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from h33_evolve_pursuit import GENOME, random_genome, mutate, crossover, tournament, evaluate

LAB = Path(__file__).resolve().parents[1] / "out" / "lab"


def main():
    rng = np.random.default_rng(23)
    pop = [(random_genome(rng), int(rng.integers(0, 100000))) for _ in range(24)]
    log = []
    with ProcessPoolExecutor(10) as pool:
        for gen in range(12):
            rows = list(pool.map(evaluate, pop, chunksize=2))
            fits = [r["fit"] for r in rows]
            nears = [r["near3"] for r in rows]
            b = int(np.argmax(fits))
            log.append(dict(gen=gen, best_near=float(nears[b]),
                            best_dist=float(rows[b]["dist"]),
                            mean_near=float(np.mean(nears)),
                            champion=pop[b][0], champ_seed=pop[b][1]))
            print(f"gen {gen:02d}  best near3 {nears[b]:.2f} dist {rows[b]['dist']:.2f}  "
                  f"pop near3 {np.mean(nears):.2f}", flush=True)
            elite = pop[b]
            genomes = [p[0] for p in pop]
            new = [elite]
            while len(new) < len(pop):
                pa = tournament(pop, fits, rng)
                pb = tournament(pop, fits, rng)
                g = mutate(crossover(pa[0], pb[0], rng), rng)
                s = pa[1] if rng.random() < 0.5 else pb[1]
                if rng.random() < 0.25:
                    s = int(rng.integers(0, 100000))
                new.append((g, int(s)))
            pop = new
    (LAB / "h34_joint.json").write_text(json.dumps(log))
    print("champion seed:", log[-1]["champ_seed"])
    print("champion:", json.dumps(log[-1]["champion"]))


if __name__ == "__main__":
    main()
