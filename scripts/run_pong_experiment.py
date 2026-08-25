"""Validate the Pong re-implementation (Falandays et al. 2024, case study 2)
against the published results.

Runs three conditions in parallel and compares each to the paper:

  baseline     egocentric sensors, homeostatic updating on   paper: 0.582 (SD 0.100)
  no-learning  targets and weights frozen at initialization  paper: 0.43  (SD 0.138)
  allocentric  sensors encode the ball's absolute height     paper: 0.216 (SD 0.021)

Chance is 0.20 (a 100 px paddle in a 500 px field). Also reports the first
and last 50 scoring opportunities per run, which the paper uses to argue that
performance is at ceiling from the start (M = 0.5786 for both).

Usage: python scripts/run_pong_experiment.py [--runs 120] [--steps 100000]
"""

from __future__ import annotations

import os

for _var in ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import dataclasses
import json
import pathlib
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from homeostasis import PONG_RESERVOIR_CONFIG, PongConfig, pong_metrics, run_pong

# Published values to compare against; None where the paper has no counterpart.
PAPER = {
    "published": (0.582, 0.0995),
    "fixed-sensors": None,
    "no-learning": (0.43, 0.138),
    "allocentric": (0.216, 0.0207),
}
CHANCE = 0.20

ALLOCENTRIC_PONG = PongConfig.allocentric()
# The allocentric control feeds 50 sensors instead of 46; the network is
# otherwise identical to the published one.
ALLOCENTRIC_RESERVOIR = dataclasses.replace(
    PONG_RESERVOIR_CONFIG, n_inputs=ALLOCENTRIC_PONG.n_sensors
)


def evaluate(task: tuple[str, int, int]) -> dict:
    """One run. Conditions:

    published      released sensors (strict `< 2`, blind straight ahead)
    fixed-sensors  our default (`<= 2`), the only change from `published`
    no-learning    released sensors, homeostatic updating frozen
    allocentric    ball height instead of egocentric angle
    """
    condition, seed, n_steps = task
    pong_config = PongConfig.published()
    reservoir_config = PONG_RESERVOIR_CONFIG
    learning = True
    if condition == "fixed-sensors":
        pong_config = PongConfig()
    elif condition == "no-learning":
        learning = False
    elif condition == "allocentric":
        pong_config = ALLOCENTRIC_PONG
        reservoir_config = ALLOCENTRIC_RESERVOIR

    h = run_pong(
        n_steps=n_steps,
        seed=seed,
        learning_enabled=learning,
        reservoir_config=reservoir_config,
        pong_config=pong_config,
    )
    m = pong_metrics(h)
    m["condition"] = condition
    m["seed"] = seed
    return m


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=40)
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 4))
    parser.add_argument("--conditions", type=str,
                        default="published,fixed-sensors,no-learning,allocentric")
    parser.add_argument("--out", type=str, default="scripts/out/pong")
    args = parser.parse_args()
    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    tasks = [(c, seed, args.steps) for c in conditions for seed in range(args.runs)]
    print(f"{len(tasks)} runs of {args.steps:,} steps on {args.workers} workers")

    results: list[dict] = []
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(evaluate, t) for t in tasks]
        for i, fut in enumerate(as_completed(futures), 1):
            results.append(fut.result())
            if i % 20 == 0 or i == len(tasks):
                rate = i / (time.perf_counter() - t0)
                print(f"  {i}/{len(tasks)} ({rate * 60:.0f} runs/min, "
                      f"~{(len(tasks) - i) / rate / 60:.1f} min left)")

    by_condition: dict[str, list[dict]] = {}
    for r in results:
        by_condition.setdefault(r["condition"], []).append(r)

    print(f"\n=== hit rate, {args.runs} runs x {args.steps:,} steps "
          f"(chance {CHANCE:.2f}) ===")
    print(f"{'condition':>14} {'ours':>16} {'paper':>16} {'first50':>9} "
          f"{'last50':>8} {'opps':>6}")
    summary = {}
    for condition in conditions:
        runs = by_condition.get(condition, [])
        if not runs:
            continue
        rate = np.array([r["hit_rate"] for r in runs])
        first = np.array([r["first"] for r in runs])
        last = np.array([r["last"] for r in runs])
        opps = np.array([r["n_opportunities"] for r in runs])
        paper = PAPER.get(condition)
        paper_txt = f"{paper[0]:.3f} ± {paper[1]:.3f}" if paper else "       —      "
        print(f"{condition:>14} {rate.mean():.3f} ± {rate.std():.3f}  "
              f"{paper_txt}  {first.mean():>8.3f} {last.mean():>8.3f} "
              f"{opps.mean():>6.0f}")
        summary[condition] = {
            "hit_rate_mean": float(rate.mean()), "hit_rate_sd": float(rate.std()),
            "first50": float(first.mean()), "last50": float(last.mean()),
            "n_opportunities_mean": float(opps.mean()),
            "paper_mean": paper[0] if paper else None,
            "paper_sd": paper[1] if paper else None,
            "hit_rates": rate.tolist(),
        }

    if "published" in summary and "fixed-sensors" in summary:
        a = np.array(summary["published"]["hit_rates"])
        b = np.array(summary["fixed-sensors"]["hit_rates"])
        # Same seeds in both conditions, so compare them pairwise.
        d = b - a
        se = d.std(ddof=1) / np.sqrt(d.size) if d.size > 1 else float("nan")
        print(f"\nsensor fix (<= 2) vs published (< 2), paired over {d.size} seeds: "
              f"{d.mean():+.4f} ± {se:.4f} (s.e.)")

    if "published" in summary:
        b = summary["published"]
        z = (b["hit_rate_mean"] - CHANCE) / (b["hit_rate_sd"] / np.sqrt(args.runs))
        print(f"\nbaseline vs chance: +{b['hit_rate_mean'] - CHANCE:.3f} (z = {z:.0f})")
        print(f"baseline first50 {b['first50']:.3f} vs last50 {b['last50']:.3f} "
              f"(paper: 0.579 vs 0.579 - no improvement over the run)")

    payload = {"runs": args.runs, "steps": args.steps, "chance": CHANCE,
               "summary": summary}
    (out_dir / "results.json").write_text(json.dumps(payload))
    print(f"\nsaved {out_dir / 'results.json'}")

    make_plot(summary, conditions, out_dir)


def make_plot(summary, conditions, out_dir) -> None:
    if not summary:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.6))
    rng = np.random.default_rng(0)
    for i, condition in enumerate([c for c in conditions if c in summary]):
        rates = np.array(summary[condition]["hit_rates"])
        ax.scatter(i + rng.uniform(-0.12, 0.12, rates.size), rates, s=9,
                   alpha=0.45, color="tab:blue", zorder=2)
        ax.errorbar(i, rates.mean(), yerr=rates.std(), fmt="o", color="black",
                    capsize=5, zorder=3, label="ours (mean ± sd)" if i == 0 else None)
        p_mean, p_sd = summary[condition]["paper_mean"], summary[condition]["paper_sd"]
        if p_mean is not None:  # no published counterpart for fixed-sensors
            ax.errorbar(i + 0.28, p_mean, yerr=p_sd, fmt="s", color="tab:red",
                        capsize=5, zorder=3, label="paper" if i == 0 else None)
    ax.axhline(CHANCE, color="gray", ls=":", label=f"chance ({CHANCE:.2f})")
    ax.set_xticks(range(len([c for c in conditions if c in summary])))
    ax.set_xticklabels([c for c in conditions if c in summary])
    ax.set_ylabel("proportion of hits")
    ax.set_title("Pong: re-implementation vs. published results (cf. paper Fig. 7D)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "pong_validation.png", dpi=150)
    print(f"plot saved to {out_dir / 'pong_validation.png'}")


if __name__ == "__main__":
    main()
