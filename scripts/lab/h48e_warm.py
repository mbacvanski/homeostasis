"""H48e: warm-started GA on the live chain."""
from __future__ import annotations
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from h33_evolve_pursuit import random_genome, mutate, crossover, tournament
from h48c_live_chain import cosim, LAB


def evaluate(task):
    genome, seed = task
    r = cosim(genome, seed)
    return dict(fit=r["near4"] - r["dist"] / 30.0, **r)


def main():
    rng = np.random.default_rng(97)
    warm = json.loads((LAB / "h34_joint.json").read_text())[-1]
    wg, ws = warm["champion"], warm["champ_seed"]
    pop = [(dict(wg), ws)] + \
          [(mutate(dict(wg), rng), ws) for _ in range(5)] + \
          [(mutate(dict(wg), rng), int(rng.integers(0, 100000))) for _ in range(6)] + \
          [(random_genome(rng), int(rng.integers(0, 100000))) for _ in range(12)]
    best = None
    with ProcessPoolExecutor(10) as pool:
        for gen in range(12):
            rows = list(pool.map(evaluate, pop, chunksize=1))
            fits = [r["fit"] for r in rows]
            bi = int(np.argmax(fits))
            if best is None or rows[bi]["fit"] > best["fit"]:
                best = dict(rows[bi], gen=gen, champion=pop[bi][0], champ_seed=pop[bi][1])
            print(f"gen {gen:02d}  near4 {rows[bi]['near4']:.2f} dist {rows[bi]['dist']:.2f}",
                  flush=True)
            elite = pop[bi]
            new = [elite]
            while len(new) < len(pop):
                pa = tournament(pop, fits, rng); pb = tournament(pop, fits, rng)
                g = mutate(crossover(pa[0], pb[0], rng), rng)
                s = pa[1] if rng.random() < 0.5 else pb[1]
                if rng.random() < 0.2:
                    s = int(rng.integers(0, 100000))
                new.append((g, int(s)))
            pop = new
    long = cosim(best["champion"], best["champ_seed"], n=10800, record=True)
    best["long_near4"] = long["near4"]; best["long_dist"] = long["dist"]
    (LAB / "h48e_warm.json").write_text(json.dumps(
        {k: v for k, v in best.items() if k != "traj"}))
    print(f"BEST gen {best['gen']}: near4 {best['near4']:.2f} | LONG: near4 {long['near4']:.2f} "
          f"dist {long['dist']:.2f}")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tr = long["traj"]
    half = len(tr["ax"]) // 2
    fig, ax = plt.subplots(figsize=(6.6, 6.6))
    ax.plot(tr["ax"][half:], tr["ay"][half:], "-", lw=1.8, color="tab:red",
            label="pacemaker (wall circler, blind)")
    ax.plot(tr["bx"][half:], tr["by"][half:], "-", lw=0.8, color="tab:blue",
            label="follower (senses pacemaker)")
    ax.set_xlim(0, 30); ax.set_ylim(0, 30); ax.set_aspect("equal")
    ax.legend(fontsize=8)
    ax.set_title(f"Live two-agent homeostatic chain (late half): dist {long['dist']:.2f}, "
                 f"near4 {long['near4']:.2f}")
    fig.tight_layout(); fig.savefig(LAB / "fig_live_chain.png", dpi=130)
    print("wrote fig_live_chain.png")


if __name__ == "__main__":
    main()
