"""Derived figures from existing screen/evolution JSONs - no new simulation.

Reads scripts/out/{metric_screen,cross_task_screen}.json and the four
tracking evolution arms plus the Pong arm, and produces:

  out/four_arm_evolution.png   - all evolution arms on one axis (score of the
                                 fittest genome per generation, never selected
                                 on) + per-generation fitness-score correlation
  out/metric_league.png        - Spearman rho league table for all screened
                                 internal metrics, with the Goodhart check
                                 (where the known cheaters rank per metric)
  out/cross_embodiment_rho.png - per-metric rho on tracking vs rho on Pong

Usage: python scripts/plot_derived_figures.py
"""

from __future__ import annotations

import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = pathlib.Path(__file__).resolve().parent / "out"

TRACKING_ARMS = {
    # dir name -> (legend label, color)
    "evolution_flow": ("input flow (boundary throughput)", "tab:blue"),
    "evolution": ("firing-rate band (interior comfort)", "tab:red"),
    "evolution_boundary": ("sensor band (boundary comfort)", "tab:orange"),
    "evolution_boundary_narrow": ("sensor band, narrow", "tab:gray"),
}
CHANCE = 0.25


def _ranks(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x))
    ranks[order] = np.arange(len(x), dtype=float)
    # average ties
    vals, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    sums = np.zeros(len(vals))
    np.add.at(sums, inv, ranks)
    return sums[inv] / counts[inv]


def spearman(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    rx, ry = _ranks(x), _ranks(y)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    denom = np.sqrt((rx**2).sum() * (ry**2).sum())
    return float((rx * ry).sum() / denom) if denom else 0.0


def fig_four_arms() -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    paper_score = None
    for arm, (label, color) in TRACKING_ARMS.items():
        ev = json.load(open(OUT / arm / "evolution.json"))
        reps = [ev["history"][k] for k in sorted(ev["history"])]
        gens = np.array([g["gen"] for g in reps[0]])
        scores = np.array([[g["best_score"] for g in rep] for rep in reps])
        corrs = np.array([[-g["corr_viability_score"] for g in rep] for rep in reps])
        for rep_scores in scores:
            ax1.plot(gens, rep_scores, color=color, alpha=0.25, lw=1)
        ax1.plot(gens, scores.mean(axis=0), color=color, lw=2.2, label=label)
        ax2.plot(gens, np.nanmean(corrs, axis=0), color=color, lw=2)
        champs = [h["score"] for h in ev["held_out"] if "champion" in h["label"]]
        ax1.scatter([gens[-1] + 1.5] * len(champs), champs, marker="*", s=140,
                    color=color, edgecolor="black", linewidth=0.5, zorder=5)
        paper = [h["score"] for h in ev["held_out"] if h["label"] == "paper"]
        if arm == "evolution_flow" and paper:
            paper_score = paper[0]

    if paper_score is not None:
        ax1.axhline(paper_score, color="black", ls=":", label="paper config (held-out)")
    ax1.axhline(CHANCE, color="gray", ls=":", lw=1, label="chance")
    ax1.set_xlabel("generation")
    ax1.set_ylabel("within-45° score of fittest genome")
    ax1.set_title("What selection never sees: tracking\n(stars = held-out champions)")
    ax1.legend(fontsize=8, loc="upper left")

    ax2.axhline(0, color="gray", lw=1)
    ax2.set_xlabel("generation")
    ax2.set_ylabel("r(fitness, score) within population\n(+ = fitter genomes track better)")
    ax2.set_title("Fitness-score alignment per generation")

    fig.suptitle("Evolving on internal quantities only (score recorded, never selected on): "
                 "boundary throughput works, comfort bands are gamed")
    fig.tight_layout()
    fig.savefig(OUT / "four_arm_evolution.png", dpi=150)
    plt.close(fig)


def fig_metric_league() -> None:
    rows = json.load(open(OUT / "metric_screen.json"))
    randoms = [r for r in rows if r["kind"] == "random"]
    cheats = [r for r in rows if r["kind"] in ("hum", "statue")]
    metrics = [k for k in rows[0] if k not in ("kind", "name", "score")]

    league = []
    for m in metrics:
        xs = np.array([r[m] for r in randoms])
        rho = spearman(xs, [r["score"] for r in randoms])
        # Goodhart check: percentile of cheaters under this metric, oriented so
        # that a high percentile means "looks fit" in the direction rho favors.
        pcts = []
        for c in cheats:
            frac_below = float((xs < c[m]).mean())
            pcts.append(frac_below if rho >= 0 else 1 - frac_below)
        league.append((m, rho, 100 * float(np.mean(pcts))))
    league.sort(key=lambda t: t[1])

    fig, ax = plt.subplots(figsize=(9, 6))
    names = [t[0] for t in league]
    rhos = [t[1] for t in league]
    colors = ["tab:blue" if p < 50 else "tab:red" for _, _, p in league]
    bars = ax.barh(names, rhos, color=colors, alpha=0.85)
    for bar, (_, rho, pct) in zip(bars, league):
        # annotate to the right of the bar tip; for negative bars the region
        # right of zero is free, so anchor there instead
        ax.text(max(rho, 0) + 0.02, bar.get_y() + bar.get_height() / 2,
                f"cheaters @ {pct:.0f}th pctile", va="center", ha="left", fontsize=8)
    ax.axvline(0, color="black", lw=1)
    ax.set_xlabel("Spearman rho with tracking score (random configs, n=%d)" % len(randoms))
    ax.set_xlim(-0.75, 1.05)
    ax.set_title("Which internal metrics predict behavior - and which can be gamed?\n"
                 "blue = cheaters rank low under the metric (usable fitness); "
                 "red = cheaters look fit (Goodharted)")
    fig.tight_layout()
    fig.savefig(OUT / "metric_league.png", dpi=150)
    plt.close(fig)


def fig_cross_embodiment() -> None:
    ct = json.load(open(OUT / "cross_task_screen.json"))
    shared = sorted(
        (set(ct["tracking"][0]) & set(ct["pong"][0])) - {"score", "opps"})
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    offsets = {  # hand-placed to keep the mid-cluster labels apart
        "input_duty": (-8, -12), "input_flow": (8, -4),
        "prop_spiked": (8, 4), "act_dynamism": (8, 8), "mean_jump": (-8, -4),
        "rate_band": (8, -12), "smooth_presence": (8, -10), "mean_abs_E": (8, -4),
    }
    for m in shared:
        rt = spearman([r[m] for r in ct["tracking"]],
                      [r["score"] for r in ct["tracking"]])
        rp = spearman([r[m] for r in ct["pong"]],
                      [r["score"] for r in ct["pong"]])
        boundary = m in ("input_flow", "input_duty")
        ax.scatter(rt, rp, s=90 if boundary else 55,
                   color="tab:blue" if boundary else "tab:gray", zorder=5)
        dx, dy = offsets.get(m, (6, 4))
        ax.annotate(m, (rt, rp), xytext=(dx, dy), textcoords="offset points",
                    fontsize=9, ha="left" if dx > 0 else "right")
    lim = 1.05
    ax.plot([-lim, lim], [-lim, lim], color="gray", lw=0.8, ls="--")
    ax.axhline(0, color="black", lw=0.8)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("Spearman rho with tracking score (n=%d configs)" % len(ct["tracking"]))
    ax.set_ylabel("Spearman rho with Pong hit rate (n=%d configs)" % len(ct["pong"]))
    ax.set_title("Do internal metrics generalize across embodiments?\n"
                 "boundary throughput (blue) transfers; smoothness does not",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "cross_embodiment_rho.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    fig_four_arms()
    fig_metric_league()
    fig_cross_embodiment()
    for f in ("four_arm_evolution", "metric_league", "cross_embodiment_rho"):
        print(OUT / f"{f}.png")
