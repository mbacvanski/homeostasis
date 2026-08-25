"""Evolve Pong hyperparameters on END-WEIGHTED INPUT FLOW.

Fitness = exponentially end-weighted mean sensory flow (weight exp((t-T)/tau),
tau = T/4), so the final quarter of life dominates and the early learning
transient counts for ~2%. Hit rate is recorded for every evaluation but never
selected on. Same protocol as the tracking evolutions: evaluation seeds are
resampled every generation, so lineages cannot overfit a wiring draw.

Outputs into scripts/out/evolution_pong/: per-generation log, evolution.json,
champions.json (visualizer loadouts), and three PNGs.

Usage: python scripts/evolve_pong.py [--generations 16] [--pop 22]
"""

from __future__ import annotations

import os

for _v in ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import dataclasses
import json
import pathlib
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from homeostasis import PONG_RESERVOIR_CONFIG, PongConfig, PongSimulation

# ---- genome ----------------------------------------------------------------
GENOME = {
    "n_nodes":             (200,  700,  False, True),
    "p_link":              (0.03, 0.30, True,  False),
    "input_weight":        (0.5,  8.0,  True,  False),
    "weight_init_mean":    (-0.5, 0.5,  False, False),
    "weight_init_sd":      (0.05, 0.6,  True,  False),
    "inhibitory_fraction": (0.0,  0.5,  False, False),
    "leak":                (0.05, 0.6,  False, False),
    "target_lr":           (0.01, 0.4,  True,  False),
    "threshold_ratio":     (1.2,  4.0,  False, False),
    "gain":                (20.0, 400.0, True, False),
}
PAPER = {
    "n_nodes": 500, "p_link": 0.1, "input_weight": 2.75,
    "weight_init_mean": 0.0, "weight_init_sd": 0.2, "inhibitory_fraction": 0.25,
    "leak": 0.25, "target_lr": 0.1, "threshold_ratio": 2.0, "gain": 100.0,
}
PUBLISHED_HIT_RATE = 0.582
CHANCE = 0.20
RESERVOIR_KEYS = tuple(k for k in GENOME if k != "gain")


def evaluate(task):
    """(genome, seed, steps, condition) -> end-weighted flow + recorded stats."""
    genome, seed, steps, condition = task
    r_cfg = dataclasses.replace(
        PONG_RESERVOIR_CONFIG, **{k: genome[k] for k in RESERVOIR_KEYS}
    )
    sim = PongSimulation(r_cfg, PongConfig(gain=genome["gain"]), seed=seed)
    net = sim.network
    if condition == "no-learn":
        net.learning_enabled = False
    elif condition == "lesion":
        net.adjacency[:] = False
        net.weights[:] = 0.0
        net._rebuild_structure_caches()

    tau = steps / 4.0
    w_sum = f_sum = wf_sum = 0.0
    spiked = 0.0
    for t in range(steps):
        state, _, _ = sim.step()
        flow = float(state.inputs.sum())
        w = float(np.exp((t - steps) / tau))
        wf_sum += w * flow
        w_sum += w
        f_sum += flow
        spiked += state.prop_spiked
    hits = np.asarray(sim.env.hits, dtype=float)
    return {
        "fitness": wf_sum / w_sum,               # end-weighted flow (maximize)
        "flow_mean": f_sum / steps,
        "hit_rate": float(hits.mean()) if hits.size else 0.0,
        "hits_first20": float(hits[:20].mean()) if hits.size >= 20 else float("nan"),
        "hits_last20": float(hits[-20:].mean()) if hits.size >= 20 else float("nan"),
        "n_opps": int(hits.size),
        "prop_spiked": spiked / steps,
        "seed": seed,
    }


def random_genome(rng):
    g = {}
    for k, (lo, hi, log, integer) in GENOME.items():
        v = float(np.exp(rng.uniform(np.log(lo), np.log(hi)))) if log else float(rng.uniform(lo, hi))
        g[k] = int(round(v)) if integer else v
    return g


def mutate(genome, rng, rate=0.45):
    g = dict(genome)
    for k, (lo, hi, log, integer) in GENOME.items():
        if rng.random() > rate:
            continue
        v = g[k] * float(np.exp(rng.normal(0, 0.25))) if log else g[k] + float(rng.normal(0, 0.15 * (hi - lo)))
        v = min(max(v, lo), hi)
        g[k] = int(round(v)) if integer else v
    return g


def crossover(a, b, rng):
    return {k: (a if rng.random() < 0.5 else b)[k] for k in GENOME}


def tournament(pop, fits, rng, k=3):
    idx = rng.integers(0, len(pop), size=k)
    return pop[int(idx[np.argmax([fits[i] for i in idx])])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations", type=int, default=16)
    ap.add_argument("--pop", type=int, default=22)
    ap.add_argument("--replicates", type=int, default=2)
    ap.add_argument("--seeds-per-genome", type=int, default=2)
    ap.add_argument("--steps", type=int, default=15000)
    ap.add_argument("--holdout-steps", type=int, default=30000)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--out", type=str, default="scripts/out/evolution_pong")
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    all_evals, history, champions = [], {}, []

    with ProcessPoolExecutor(max_workers=args.workers) as pool:

        def eval_population(pop, seeds, condition="full", steps=None):
            steps = steps or args.steps
            tasks = [(g, s, steps, condition) for g in pop for s in seeds]
            res = list(pool.map(evaluate, tasks, chunksize=1))
            agg = []
            for i in range(len(pop)):
                rs = res[i * len(seeds):(i + 1) * len(seeds)]
                a = {k: float(np.nanmean([r[k] for r in rs])) for k in rs[0] if k != "seed"}
                a["hit_sd"] = float(np.std([r["hit_rate"] for r in rs]))
                agg.append(a)
            return agg

        for rep in range(args.replicates):
            rng = np.random.default_rng(3000 + rep)
            pop = [random_genome(rng) for _ in range(args.pop)]
            hist = []
            for gen in range(args.generations):
                seeds = [int(s) for s in rng.integers(0, 2**31 - 1, size=args.seeds_per_genome)]
                stats = eval_population(pop, seeds)
                fits = [s["fitness"] for s in stats]
                order = np.argsort(fits)[::-1]  # maximize
                best = stats[int(order[0])]
                for g, s in zip(pop, stats):
                    all_evals.append({"rep": rep, "gen": gen, **s, "genome": g})
                row = {
                    "gen": gen,
                    "best_fitness": best["fitness"],
                    "mean_fitness": float(np.mean(fits)),
                    "best_hit_rate": best["hit_rate"],
                    "max_hit_rate": float(np.max([s["hit_rate"] for s in stats])),
                    "mean_hit_rate": float(np.mean([s["hit_rate"] for s in stats])),
                    "best_prop_spiked": best["prop_spiked"],
                    "corr_fit_hit": float(np.corrcoef(fits, [s["hit_rate"] for s in stats])[0, 1]),
                    "param_mean": {k: float(np.mean([g[k] for g in pop])) for k in GENOME},
                    "param_sd": {k: float(np.std([g[k] for g in pop])) for k in GENOME},
                    "best_genome": pop[int(order[0])],
                }
                hist.append(row)
                print(f"rep{rep} gen{gen:02d} | end-flow best {best['fitness']:.3f} "
                      f"pop {row['mean_fitness']:.3f} | hit best-fit {best['hit_rate']:.2f} "
                      f"popmax {row['max_hit_rate']:.2f} | spike {best['prop_spiked']:.2f} "
                      f"| r(F,H) {row['corr_fit_hit']:+.2f} | {time.perf_counter()-t0:.0f}s",
                      flush=True)
                elite = [pop[int(i)] for i in order[:2]]
                children = []
                while len(children) < args.pop - 2:
                    children.append(mutate(crossover(
                        tournament(pop, fits, rng), tournament(pop, fits, rng), rng), rng))
                pop = elite + children
            history[rep] = hist
            champions.append(hist[-1]["best_genome"])

        print(f"\nheld-out ({args.holdout_steps} steps, 12 fresh seeds)...", flush=True)
        finalists = champions + [dict(PAPER)]
        held = eval_population(finalists, seeds=list(range(900, 912)),
                               steps=args.holdout_steps)
        labels = [f"champion rep{r}" for r in range(len(champions))] + ["paper"]
        for lbl, h in zip(labels, held):
            print(f"  {lbl:>13}: hit {h['hit_rate']:.3f} ± {h['hit_sd']:.3f} "
                  f"(first20 {h['hits_first20']:.2f} last20 {h['hits_last20']:.2f}, "
                  f"{h['n_opps']:.0f} opps) | end-flow {h['fitness']:.3f} "
                  f"| spike {h['prop_spiked']:.2f}", flush=True)

        best_i = int(np.argmax([h["fitness"] for h in held[:-1]]))
        champ = finalists[best_i]
        print(f"\nautopsy of champion rep{best_i} (6 seeds, {args.steps} steps):", flush=True)
        autopsy = {}
        for cond in ("full", "no-learn", "lesion"):
            (r,) = eval_population([champ], seeds=list(range(950, 956)), condition=cond)
            autopsy[cond] = r
            print(f"  {cond:>9}: hit {r['hit_rate']:.3f} | end-flow {r['fitness']:.3f} "
                  f"| spike {r['prop_spiked']:.2f}", flush=True)

    payload = {
        "steps": args.steps, "holdout_steps": args.holdout_steps,
        "history": {str(k): v for k, v in history.items()},
        "held_out": [{"label": l, **h, "genome": g}
                     for l, h, g in zip(labels, held, finalists)],
        "autopsy": autopsy, "evals": all_evals,
    }
    (out / "evolution.json").write_text(json.dumps(payload))

    champ_entries = []
    for r, (g, h) in enumerate(zip(champions, held[:len(champions)])):
        champ_entries.append({
            "id": f"pongEvo{r}",
            "label": f"flow-evolved rep{r} (hit {h['hit_rate']:.2f})",
            "metrics": {"hit_rate": round(h["hit_rate"], 3),
                        "end_flow": round(h["fitness"], 3),
                        "prop_spiked": round(h["prop_spiked"], 2)},
            "params": {k: (g[k] if isinstance(g[k], int) else round(float(g[k]), 4))
                       for k in GENOME},
        })
    (out / "champions.json").write_text(json.dumps(champ_entries, indent=1))
    print(f"saved evolution.json + champions.json to {out}", flush=True)
    make_plots(history, all_evals, held, out)
    print(f"total {time.perf_counter()-t0:.0f}s", flush=True)


def make_plots(history, evals, held, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = ["tab:blue", "tab:orange"]
    paper = held[-1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4))
    for r, h in history.items():
        gens = [row["gen"] for row in h]
        ax1.plot(gens, [row["best_fitness"] for row in h], color=colors[r], lw=2, label=f"rep {r} best")
        ax1.plot(gens, [row["mean_fitness"] for row in h], color=colors[r], lw=1, ls="--", alpha=0.6)
        ax2.plot(gens, [row["best_hit_rate"] for row in h], color=colors[r], lw=2,
                 label=f"rep {r} fittest genome")
        ax2.plot(gens, [row["mean_hit_rate"] for row in h], color=colors[r], lw=1, ls="--", alpha=0.6)
    ax1.axhline(paper["fitness"], color="black", ls=":", label="paper config")
    ax1.set_xlabel("generation"); ax1.set_ylabel("end-weighted input flow (fitness)")
    ax1.set_title("What selection sees"); ax1.legend(fontsize=8)
    ax2.axhline(paper["hit_rate"], color="black", ls=":", label=f"paper config ({paper['hit_rate']:.2f})")
    ax2.axhline(PUBLISHED_HIT_RATE, color="tab:red", ls="--", lw=1, label="published 0.582")
    ax2.axhline(CHANCE, color="gray", ls=":", lw=1, label="chance")
    ax2.set_xlabel("generation"); ax2.set_ylabel("hit rate")
    ax2.set_title("What selection never sees: Pong performance"); ax2.legend(fontsize=8)
    fig.suptitle("Pong evolution on end-weighted input flow")
    fig.tight_layout(); fig.savefig(out / "evo_trajectories.png", dpi=150)

    fig, ax = plt.subplots(figsize=(7.5, 5))
    f = [e["fitness"] for e in evals]; s = [e["hit_rate"] for e in evals]
    g = [e["gen"] for e in evals]
    sc = ax.scatter(f, s, c=g, s=14, cmap="viridis", alpha=0.7)
    fig.colorbar(sc, label="generation")
    ax.scatter([paper["fitness"]], [paper["hit_rate"]], marker="s", s=110, color="black",
               zorder=5, label="paper config")
    for i, h in enumerate(held[:-1]):
        ax.scatter([h["fitness"]], [h["hit_rate"]], marker="*", s=240, color=colors[i],
                   edgecolor="black", zorder=5, label=f"champion rep{i} (held-out)")
    ax.axhline(CHANCE, color="gray", ls=":", lw=1)
    ax.set_xlabel("end-weighted input flow (fitness)")
    ax.set_ylabel("hit rate (never selected on)")
    ax.set_title("Every genome evaluated"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "evo_flow_vs_hits.png", dpi=150)

    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    for ax, k in zip(axes.ravel(), GENOME):
        for r, h in history.items():
            gens = [row["gen"] for row in h]
            m = np.array([row["param_mean"][k] for row in h])
            sd = np.array([row["param_sd"][k] for row in h])
            ax.plot(gens, m, color=colors[r], lw=1.5)
            ax.fill_between(gens, m - sd, m + sd, color=colors[r], alpha=0.15)
        ax.axhline(PAPER[k], color="black", ls=":", lw=1)
        if GENOME[k][2]:
            ax.set_yscale("log")
        ax.set_title(k, fontsize=9)
    fig.suptitle("Genome trajectories (mean ± sd; dotted = paper value)")
    fig.tight_layout(); fig.savefig(out / "evo_params.png", dpi=150)
    print("plots saved", flush=True)


if __name__ == "__main__":
    main()
