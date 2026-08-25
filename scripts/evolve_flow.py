"""Evolve tracking-model hyperparameters on INPUT FLOW.

Fitness = mean total sensory drive per step (post-settling): the crudest
boundary quantity, and exactly the "consistent flow of input" the paper's
mechanism narrative appeals to. Chosen because the metric screen
(scripts/screen_metrics.py) found it to be the only internal metric that
strongly correlates with tracking score across 241 configurations
(Spearman rho = 0.77; top-10-by-flow mean score 0.605) while ranking the
known degenerate solutions (statues, autarkic hums) below average.
Score is recorded but never selected on.
"""

from __future__ import annotations

import os

for _v in ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import pathlib
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from homeostasis import ReservoirConfig, TrackingConfig, TrackingSimulation
from homeostasis.tracking import angular_difference

# ---- genome ----------------------------------------------------------------
# name: (low, high, log-scale?, integer?)
GENOME = {
    "n_nodes":          (80,    250,  False, True),
    "p_link":           (0.03,  0.30, True,  False),
    "input_weight":     (0.25,  2.50, True,  False),
    "weight_init_mean": (0.0,   1.50, False, False),
    "weight_init_sd":   (0.02,  1.00, True,  False),
    "leak":             (0.05,  0.60, False, False),
    "target_lr":        (0.001, 0.10, True,  False),
    "threshold_ratio":  (1.2,   4.0,  False, False),
    "gain":             (2.0,   40.0, True,  False),
}
PAPER = {"n_nodes": 200, "p_link": 0.1, "input_weight": 0.75,
         "weight_init_mean": 0.75, "weight_init_sd": 0.1, "leak": 0.25,
         "target_lr": 0.01, "threshold_ratio": 2.0, "gain": 10.0}

STEPS = 3600        # inner-loop run length (5 reversal epochs)
SETTLE = 720
RATE_WINDOW = 120   # firing-rate window, in steps
RATE_LO = 0.05      # viable band on per-node firing rate over a window:
RATE_HI = 0.90      # below = starving, above = saturated


NARROW_TRACKING = dict(eye_offsets=(10.0, -10.0), sensors_per_eye=8)


def evaluate(task):
    """One (genome, seed) run -> viability + behavior + regime metrics."""
    genome, seed, steps, condition, aperture = task
    r_kwargs = {k: genome[k] for k in GENOME if k != "gain"}
    if aperture == "narrow":
        t_cfg = TrackingConfig(gain=genome["gain"], **NARROW_TRACKING)
        r_kwargs["n_inputs"] = t_cfg.n_sensors
    else:
        t_cfg = TrackingConfig(gain=genome["gain"])
    sim = TrackingSimulation(ReservoirConfig(**r_kwargs), t_cfg, seed=seed)
    net = sim.network
    if condition == "no-learn":
        net.learning_enabled = False
    elif condition == "lesion":
        net.adjacency[:] = False
        net.weights[:] = 0.0
        net._rebuild_structure_caches()

    n = net.config.n_nodes
    n_sens = len(sim.env.sense())
    sum_abs_e = spiked = 0.0
    in45 = agree_n = 0
    dh_window = []
    kernel_n = 50
    count = 0
    spike_counts = np.zeros(n)
    sensor_sums = np.zeros(n_sens)
    window_len = 0
    band_sum = 0.0
    starve_sum = 0.0
    saturate_sum = 0.0
    s_band_sum = s_starve_sum = s_sat_sum = 0.0
    abs_dh_sum = 0.0
    flow_sum = 0.0
    n_windows = 0
    for t in range(steps):
        err_deg = sim.env.heading_error()
        direction = sim.env.stimulus_direction
        state, dh = sim.step()
        dh_window.append(dh)
        if len(dh_window) > kernel_n:
            dh_window.pop(0)
        if t >= SETTLE:
            sum_abs_e += float(np.mean(np.abs(state.error)))
            spiked += state.prop_spiked
            in45 += 1 if abs(err_deg) <= 45.0 else 0
            agree_n += 1 if np.sign(sum(dh_window)) == direction else 0
            abs_dh_sum += abs(dh)
            count += 1
            spike_counts += state.spiked
            sensor_sums += state.inputs
            flow_sum += float(state.inputs.sum())
            window_len += 1
            if window_len == RATE_WINDOW:
                rates = spike_counts / RATE_WINDOW
                band_sum += float(np.mean((rates > RATE_LO) & (rates < RATE_HI)))
                starve_sum += float(np.mean(rates <= RATE_LO))
                saturate_sum += float(np.mean(rates >= RATE_HI))
                # boundary: per-sensor mean activation over the same window
                s = sensor_sums / RATE_WINDOW
                s_band_sum += float(np.mean((s > RATE_LO) & (s < RATE_HI)))
                s_starve_sum += float(np.mean(s <= RATE_LO))
                s_sat_sum += float(np.mean(s >= RATE_HI))
                spike_counts[:] = 0.0
                sensor_sums[:] = 0.0
                window_len = 0
                n_windows += 1
    n_windows = max(n_windows, 1)
    rate_band = band_sum / n_windows
    sensor_band = s_band_sum / n_windows
    return {
        "viability": -(flow_sum / count),        # negative input flow (fitness)
        "input_flow": flow_sum / count,
        "sensor_band": sensor_band,              # fraction of sensor-windows viable
        "sensor_starving": s_starve_sum / n_windows,
        "sensor_saturated": s_sat_sum / n_windows,
        "rate_band": rate_band,                  # interior band (recorded only)
        "starving": starve_sum / n_windows,
        "saturated": saturate_sum / n_windows,
        "mean_abs_E": sum_abs_e / count,
        "score": in45 / count,                   # within-45 fraction
        "dir_agree": agree_n / count,
        "prop_spiked": spiked / count,
        "mean_abs_dh": abs_dh_sum / count,       # turn magnitude (scanning signature)
        "seed": seed,
    }


# ---- GA operators ----------------------------------------------------------

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
        if log:
            v = g[k] * float(np.exp(rng.normal(0.0, 0.25)))
        else:
            v = g[k] + float(rng.normal(0.0, 0.15 * (hi - lo)))
        v = min(max(v, lo), hi)
        g[k] = int(round(v)) if integer else v
    return g


def crossover(a, b, rng):
    return {k: (a if rng.random() < 0.5 else b)[k] for k in GENOME}


def tournament(pop, fits, rng, k=3):
    idx = rng.integers(0, len(pop), size=k)
    return pop[int(idx[np.argmin([fits[i] for i in idx])])]  # min viability wins


# ---- main loop -------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations", type=int, default=30)
    ap.add_argument("--pop", type=int, default=28)
    ap.add_argument("--replicates", type=int, default=2)
    ap.add_argument("--seeds-per-genome", type=int, default=3)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--aperture", choices=["wide", "narrow"], default="wide")
    ap.add_argument("--out", type=str, default="scripts/out/evolution_flow")
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    t_start = time.perf_counter()
    all_evals = []          # every genome evaluation ever (mean over seeds)
    history = {}            # per replicate: per-generation aggregates
    champions = []

    with ProcessPoolExecutor(max_workers=args.workers) as pool:

        def eval_population(pop, seeds, condition="full", steps=STEPS):
            tasks = [(g, s, steps, condition, args.aperture) for g in pop for s in seeds]
            results = list(pool.map(evaluate, tasks, chunksize=2))
            per_genome = []
            for i in range(len(pop)):
                runs = results[i * len(seeds):(i + 1) * len(seeds)]
                agg = {k: float(np.mean([r[k] for r in runs]))
                       for k in runs[0] if k != "seed"}
                agg["score_sd"] = float(np.std([r["score"] for r in runs]))
                per_genome.append(agg)
            return per_genome

        for rep in range(args.replicates):
            rng = np.random.default_rng(1000 + rep)
            pop = [random_genome(rng) for _ in range(args.pop)]
            hist = []
            for gen in range(args.generations):
                seeds = [int(s) for s in rng.integers(0, 2**31 - 1, size=args.seeds_per_genome)]
                stats = eval_population(pop, seeds)
                fits = [s["viability"] for s in stats]
                order = np.argsort(fits)  # ascending viability = best first

                for g, s in zip(pop, stats):
                    all_evals.append({"rep": rep, "gen": gen, **s, "genome": g})
                best = stats[order[0]]
                gen_row = {
                    "gen": gen,
                    "best_viability": best["viability"],
                    "mean_viability": float(np.mean(fits)),
                    "best_score": best["score"],
                    "best_genome_score_sd": best["score_sd"],
                    "mean_score": float(np.mean([s["score"] for s in stats])),
                    "max_score": float(np.max([s["score"] for s in stats])),
                    "best_prop_spiked": best["prop_spiked"],
                    "best_in_band": best["input_flow"],
                    "best_interior_band": best["rate_band"],
                    "best_mean_abs_dh": best["mean_abs_dh"],
                    "best_starving": best["starving"],
                    "best_saturated": best["saturated"],
                    "best_mean_abs_E": best["mean_abs_E"],
                    "corr_viability_score": float(np.corrcoef(
                        fits, [s["score"] for s in stats])[0, 1]),
                    "param_mean": {k: float(np.mean([g[k] for g in pop])) for k in GENOME},
                    "param_sd": {k: float(np.std([g[k] for g in pop])) for k in GENOME},
                    "best_genome": pop[int(order[0])],
                }
                hist.append(gen_row)
                print(f"rep{rep} gen{gen:02d} | flow best {best['input_flow']:.2f} "
                      f"pop {-gen_row['mean_viability']:.2f} | score best-fit {best['score']:.2f} "
                      f"popmax {gen_row['max_score']:.2f} | spike {best['prop_spiked']:.2f} "
                      f"int-band {best['rate_band']:.2f} |dH| {best['mean_abs_dh']:.2f} "
                      f"| r(V,S) {gen_row['corr_viability_score']:+.2f} "
                      f"| {time.perf_counter() - t_start:.0f}s", flush=True)

                # next generation: elitism + tournament/crossover/mutation
                elite = [pop[int(i)] for i in order[:2]]
                children = []
                while len(children) < args.pop - len(elite):
                    child = crossover(tournament(pop, fits, rng), tournament(pop, fits, rng), rng)
                    children.append(mutate(child, rng))
                pop = elite + children
            history[rep] = hist
            champions.append(hist[-1]["best_genome"])

        # ---- held-out verdict: champions + paper config, full-length runs --
        print("\nheld-out evaluation (16 fresh seeds, 7200 steps)...", flush=True)
        finalists = champions + [dict(PAPER)]
        held = eval_population(finalists, seeds=list(range(500, 516)), steps=7200)
        for label, h in zip([f"champion rep{r}" for r in range(len(champions))] + ["paper"], held):
            print(f"  {label:>14}: flow {h['input_flow']:.2f} | score {h['score']:.3f} "
                  f"± {h['score_sd']:.3f} | dir {h['dir_agree']:.2f} | "
                  f"spike {h['prop_spiked']:.2f} | int-band {h['rate_band']:.2f} | |dH| {h['mean_abs_dh']:.2f}", flush=True)

        # ---- mechanism autopsy on the overall champion ---------------------
        best_i = int(np.argmin([h["viability"] for h in held[:-1]]))
        champ = finalists[best_i]
        print(f"\nautopsy of champion rep{best_i} (8 seeds, {STEPS} steps):", flush=True)
        autopsy = {}
        for condition in ("full", "no-learn", "lesion"):
            (r,) = eval_population([champ], seeds=list(range(700, 708)), condition=condition)
            autopsy[condition] = r
            print(f"  {condition:>9}: score {r['score']:.3f} | flow {r['input_flow']:.2f} "
                  f"| spike {r['prop_spiked']:.2f}", flush=True)

    # ---- save everything ---------------------------------------------------
    payload = {
        "steps": STEPS, "settle": SETTLE,
        "rate_window": RATE_WINDOW, "rate_band": [RATE_LO, RATE_HI],
        "history": {str(k): v for k, v in history.items()},
        "held_out": [
            {"label": lbl, **h, "genome": g}
            for lbl, h, g in zip(
                [f"champion rep{r}" for r in range(len(champions))] + ["paper"],
                held, finalists)
        ],
        "autopsy": autopsy,
        "evals": all_evals,
    }
    (out / "evolution.json").write_text(json.dumps(payload))

    champ_entries = []
    for r, (g, h) in enumerate(zip(champions, held[:-1])):
        champ_entries.append({
            "id": f"evoF{r}",
            "label": f"flow champion rep{r}",
            "metrics": {"score": round(h["score"], 3),
                        "dir_agree": round(h["dir_agree"], 2),
                        "prop_spiked": round(h["prop_spiked"], 2),
                        "input_flow": round(h["input_flow"], 2),
                        "rate_band": round(h["rate_band"], 3),
                        "mean_abs_E": round(h["mean_abs_E"], 4)},
            "params": {k: (g[k] if isinstance(g[k], int) else round(g[k], 4)) for k in GENOME},
        })
    (out / "champions.json").write_text(json.dumps(champ_entries, indent=1))
    print(f"\nsaved evolution.json + champions.json to {out}", flush=True)

    make_plots(history, all_evals, held, finalists, out)
    print(f"total {time.perf_counter() - t_start:.0f}s", flush=True)


def make_plots(history, evals, held, finalists, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    reps = sorted(history)
    colors = ["tab:blue", "tab:orange", "tab:green"]
    paper_held = held[-1]

    # 1. fitness (viability) and score across generations
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4))
    for r in reps:
        h = history[r]
        gens = [row["gen"] for row in h]
        ax1.plot(gens, [row["best_viability"] for row in h], color=colors[r], lw=2,
                 label=f"rep {r} best")
        ax1.plot(gens, [row["mean_viability"] for row in h], color=colors[r], lw=1,
                 ls="--", alpha=0.6, label=f"rep {r} pop mean")
        ax2.plot(gens, [row["best_score"] for row in h], color=colors[r], lw=2,
                 label=f"rep {r} fittest genome")
        ax2.plot(gens, [row["mean_score"] for row in h], color=colors[r], lw=1,
                 ls="--", alpha=0.6)
    ax1.axhline(paper_held["viability"], color="black", ls=":", label="paper config")
    ax1.set_xlabel("generation")
    ax1.set_ylabel("negative input flow (fitness, lower = better)")
    ax1.set_title("What selection sees: input flow")
    ax1.legend(fontsize=8)
    ax2.axhline(paper_held["score"], color="black", ls=":", label="paper config")
    ax2.axhline(0.25, color="gray", ls=":", lw=1, label="chance")
    ax2.set_xlabel("generation"); ax2.set_ylabel("within-45° score")
    ax2.set_title("What selection never sees: tracking")
    ax2.legend(fontsize=8)
    fig.suptitle("Evolution on input flow - score recorded, never selected on")
    fig.tight_layout()
    fig.savefig(out / "evo_trajectories.png", dpi=150)

    # 2. viability vs score, every genome ever evaluated
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    v = [e["viability"] for e in evals]
    s = [e["score"] for e in evals]
    g = [e["gen"] for e in evals]
    sc = ax.scatter(v, s, c=g, s=12, cmap="viridis", alpha=0.65)
    fig.colorbar(sc, label="generation")
    ax.scatter([paper_held["viability"]], [paper_held["score"]], marker="s", s=90,
               color="black", label="paper config", zorder=5)
    for i, h in enumerate(held[:-1]):
        ax.scatter([h["viability"]], [h["score"]], marker="*", s=220,
                   color=colors[i], edgecolor="black", zorder=5,
                   label=f"champion rep{i} (held-out)")
    # Spearman rho via numpy (rank-transform then Pearson); no scipy needed.
    def _ranks(a):
        order = np.argsort(a)
        r = np.empty(len(a))
        r[order] = np.arange(len(a))
        return r

    rho = float(np.corrcoef(_ranks(np.array(v)), _ranks(np.array(s)))[0, 1])
    note = f"Spearman ρ = {rho:.2f}"
    ax.set_xlabel("negative input flow (selection minimizes this)")
    ax.set_ylabel("within-45° score (never selected on)")
    ax.set_title(f"Does homeostatic viability buy behavior? {note}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "evo_viability_vs_score.png", dpi=150)

    # 3. parameter trajectories
    fig, axes = plt.subplots(3, 3, figsize=(11, 8))
    for ax, k in zip(axes.ravel(), GENOME):
        for r in reps:
            h = history[r]
            gens = [row["gen"] for row in h]
            mean = np.array([row["param_mean"][k] for row in h])
            sd = np.array([row["param_sd"][k] for row in h])
            ax.plot(gens, mean, color=colors[r], lw=1.6)
            ax.fill_between(gens, mean - sd, mean + sd, color=colors[r], alpha=0.15)
            ax.plot(gens, [row["best_genome"][k] for row in h], color=colors[r],
                    lw=0.8, ls=":", alpha=0.8)
        ax.axhline(PAPER[k], color="black", ls=":", lw=1)
        lo, hi, log, _ = GENOME[k]
        if log:
            ax.set_yscale("log")
        ax.set_title(k, fontsize=10)
    fig.suptitle("Genome trajectories (mean ± sd; dotted = fittest; black = paper value)")
    fig.tight_layout()
    fig.savefig(out / "evo_params.png", dpi=150)

    # 4. regime metrics of the fittest genome over generations
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    for r in reps:
        h = history[r]
        gens = [row["gen"] for row in h]
        ax.plot(gens, [row["best_prop_spiked"] for row in h], color=colors[r], lw=1.8,
                label=f"rep {r} prop spiked")
        ax.plot(gens, [row["best_in_band"] for row in h], color=colors[r], lw=1.2,
                ls="--", label=f"rep {r} input flow (fitness)")
        ax.plot(gens, [row["best_interior_band"] for row in h], color=colors[r], lw=1.0,
                ls=":", label=f"rep {r} interior band (recorded)")
    ax.set_xlabel("generation"); ax.set_ylabel("fraction")
    ax.set_ylim(0, 1)
    ax.set_title("Regime of the fittest genome")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out / "evo_regime.png", dpi=150)
    print("plots: evo_trajectories.png, evo_viability_vs_score.png, "
          "evo_params.png, evo_regime.png", flush=True)


if __name__ == "__main__":
    main()
