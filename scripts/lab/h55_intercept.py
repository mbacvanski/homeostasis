"""H55: ballistic interception (the baseball test). Baselines then GA."""
from __future__ import annotations
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from homeostasis.reservoir import ReservoirConfig  # noqa: E402
from homeostasis.simulation import run_pursuit  # noqa: E402
from homeostasis.pursuit import PursuitConfig  # noqa: E402
from h33_evolve_pursuit import (  # noqa: E402
    GENOME, random_genome, mutate, crossover, tournament)

LAB = Path(__file__).resolve().parents[1] / "out" / "lab"
CATCH_R = 1.5
H34_CHAMP = {"n_nodes": 64, "p_link": 0.11289832502578351, "input_weight": 16.0,
             "weight_init_mean": 0.14000597018424724, "leak": 0.5922583933802147,
             "target_lr": 0.010784946575662858, "threshold_ratio": 3.88785988551928,
             "weight_lr": 0.04977769216516565, "wheel_base": 2.334271572748883,
             "intensity_scale": 5.307718558307296}


def crossing_stats(h):
    """Split the run at respawn jumps; per-crossing catch + closing-geometry."""
    jump = np.hypot(np.diff(h.sx), np.diff(h.sy)) > 1.0
    bounds = [0] + (np.flatnonzero(jump) + 1).tolist() + [len(h.sx)]
    catches, n = 0, 0
    for a, b in zip(bounds[:-1], bounds[1:]):
        if b - a < 20:
            continue
        n += 1
        if float(h.dist[a:b].min()) < CATCH_R:
            catches += 1
    return (catches / max(n, 1)), n


def evaluate(task):
    genome, seed = task
    pc = PursuitConfig(eye_offsets=(0.0,), sensors_per_eye=91,
                       stimulus_motion="ballistic",
                       wheel_base=genome["wheel_base"],
                       intensity_scale=genome["intensity_scale"])
    res_keys = ("n_nodes", "p_link", "input_weight", "weight_init_mean",
                "leak", "target_lr", "threshold_ratio", "weight_lr")
    res = ReservoirConfig(n_inputs=pc.n_sensors,
                          **{k: genome[k] for k in res_keys})
    h = run_pursuit(n_steps=3600, seed=seed, reservoir_config=res, pursuit_config=pc)
    catch, n = crossing_stats(h)
    d = float(h.dist[1800:].mean())
    return dict(fit=catch - d / 30.0, catch=catch, n_cross=n, dist=d)


def main():
    with ProcessPoolExecutor(10) as pool:
        # baselines: H34 orbital champion transfer + blinded chance anchor
        for label, g in (("h34-transfer", H34_CHAMP),
                         ("blind", {**H34_CHAMP, "input_weight": 1e-6})):
            rows = list(pool.map(evaluate, [(g, s) for s in range(41, 49)]))
            print(f"{label:<12} catch {np.mean([r['catch'] for r in rows]):.3f}"
                  f" (n_cross ~{np.mean([r['n_cross'] for r in rows]):.0f})"
                  f" dist {np.mean([r['dist'] for r in rows]):.2f}")
        # GA
        rng = np.random.default_rng(55)
        pop = [random_genome(rng) for _ in range(24)]
        pop[0] = dict(H34_CHAMP)  # seed the population with the orbital champion
        log = []
        for gen in range(10):
            seeds = rng.integers(0, 10_000, size=3)
            tasks = [(g, int(s)) for g in pop for s in seeds]
            rows = list(pool.map(evaluate, tasks, chunksize=2))
            fits = [np.mean([r["fit"] for r in rows[i * 3:(i + 1) * 3]])
                    for i in range(len(pop))]
            catches = [np.mean([r["catch"] for r in rows[i * 3:(i + 1) * 3]])
                       for i in range(len(pop))]
            b = int(np.argmax(fits))
            log.append(dict(gen=gen, best_catch=float(catches[b]),
                            mean_catch=float(np.mean(catches)),
                            champion=pop[b]))
            print(f"gen {gen}: best catch {catches[b]:.3f} mean {np.mean(catches):.3f}")
            elite = [pop[b]]
            nxt = list(elite)
            while len(nxt) < len(pop):
                a, c = tournament(pop, fits, rng), tournament(pop, fits, rng)
                nxt.append(mutate(crossover(a, c, rng), rng))
            pop = nxt
        (LAB / "h55_intercept.json").write_text(json.dumps(log))
        # champion re-verify on 8 fresh seeds
        champ = log[-1]["champion"]
        rows = list(pool.map(evaluate, [(champ, s) for s in range(101, 109)]))
        print(f"champion fresh-seed catch: {np.mean([r['catch'] for r in rows]):.3f}"
              f" +- {np.std([r['catch'] for r in rows]):.3f}")
        (LAB / "h55_champion.json").write_text(json.dumps(
            dict(champion=champ, fresh=[r["catch"] for r in rows])))

if __name__ == "__main__":
    main()
