"""Random-search hyperparameter sweep for the tracking task.

Samples model configurations (the task itself — stimulus speed, reversal
period, sensor geometry — stays as published), runs each on several seeds in
parallel, and scores the fraction of steps the agent spends within 45 degrees
of the stimulus over the full 7200-step session. The top configurations are
re-scored on held-out seeds (guarding against winner's-curse), and every run
records two "loss curves":

- per-epoch: within-45 fraction for each of the ten 720-step reversal epochs
  (does the agent improve over the session?);
- reversal-aligned: within-45 fraction as a function of step-within-epoch,
  averaged over epochs (how quickly does it re-lock after each flip?).

Outputs: console table, scripts/out/sweep/results.json, and three PNGs
(scores, marginals, curves).

Usage: python scripts/sweep_tracking.py [--configs 240] [--workers 10]
"""

from __future__ import annotations

import os

# Keep BLAS single-threaded in the workers; parallelism comes from processes.
for var in ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(var, "1")

import argparse
import json
import pathlib
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from homeostasis import ReservoirConfig, TrackingConfig, run_tracking
from homeostasis.tracking import angular_difference

N_STEPS = 7200
EPOCH = 720
SEARCH_SEEDS = list(range(6))
EVAL_SEEDS = list(range(100, 116))
BAND_DEG = 45.0
CURVE_DS = 10  # downsample factor for the reversal-aligned curve

RESERVOIR_KEYS = (
    "n_nodes", "p_link", "input_weight", "weight_init_mean",
    "weight_init_sd", "leak", "target_lr", "threshold_ratio",
)

PAPER_CONFIG = {
    "n_nodes": 200, "p_link": 0.1, "input_weight": 0.75,
    "weight_init_mean": 0.75, "weight_init_sd": 0.1, "leak": 0.25,
    "target_lr": 0.01, "threshold_ratio": 2.0, "gain": 10.0,
}


def sample_config(rng: np.random.Generator) -> dict:
    log_u = lambda lo, hi: float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
    return {
        "n_nodes": int(rng.choice([100, 200, 300])),
        "p_link": log_u(0.03, 0.3),
        "input_weight": log_u(0.25, 2.5),
        "weight_init_mean": float(rng.uniform(0.0, 1.5)),
        "weight_init_sd": log_u(0.02, 1.0),
        "leak": float(rng.uniform(0.05, 0.6)),
        "target_lr": log_u(0.001, 0.1),
        "threshold_ratio": float(rng.uniform(1.2, 4.0)),
        "gain": log_u(2.0, 40.0),
    }


def evaluate(task: tuple[int, dict, int]) -> dict:
    """Run one (config, seed) pair; returns score + loss curves."""
    config_id, cfg, seed = task
    r_cfg = ReservoirConfig(**{k: cfg[k] for k in RESERVOIR_KEYS})
    t_cfg = TrackingConfig(gain=cfg["gain"])
    h = run_tracking(
        n_steps=N_STEPS, seed=seed,
        reservoir_config=r_cfg, tracking_config=t_cfg, record_spikes=False,
    )
    in_band = np.abs(angular_difference(h.stimulus_angle, h.heading)) <= BAND_DEG
    by_epoch = in_band.reshape(-1, EPOCH)

    # direction-agreement sanity metric (chance 0.5; 0 = never turns)
    kernel = np.ones(50) / 50.0
    smoothed = np.convolve(h.d_heading, kernel, mode="same")
    agree = float(np.mean(np.sign(smoothed[EPOCH:]) == h.stimulus_direction[EPOCH:]))

    return {
        "config_id": config_id,
        "seed": seed,
        "score": float(in_band.mean()),
        "score_post_settle": float(in_band[EPOCH:].mean()),
        "dir_agree": agree,
        "prop_spiked": float(h.prop_spiked.mean()),
        "epoch_curve": by_epoch.mean(axis=1).tolist(),
        "aligned_curve": by_epoch.mean(axis=0).reshape(-1, CURVE_DS).mean(axis=1).tolist(),
    }


def run_batch(tasks, workers, label):
    results = []
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(evaluate, t) for t in tasks]
        for i, fut in enumerate(as_completed(futures), 1):
            results.append(fut.result())
            if i % 100 == 0 or i == len(tasks):
                rate = i / (time.perf_counter() - t0)
                print(f"  {label}: {i}/{len(tasks)} runs "
                      f"({rate:.0f} runs/s, ~{(len(tasks) - i) / rate:.0f}s left)")
    return results


def aggregate(runs: list[dict]) -> dict:
    """Mean metrics across seeds for one config."""
    return {
        "score": float(np.mean([r["score"] for r in runs])),
        "score_sd": float(np.std([r["score"] for r in runs])),
        "score_post_settle": float(np.mean([r["score_post_settle"] for r in runs])),
        "dir_agree": float(np.mean([r["dir_agree"] for r in runs])),
        "prop_spiked": float(np.mean([r["prop_spiked"] for r in runs])),
        "epoch_curve": np.mean([r["epoch_curve"] for r in runs], axis=0).tolist(),
        "epoch_curve_sd": np.std([r["epoch_curve"] for r in runs], axis=0).tolist(),
        "aligned_curve": np.mean([r["aligned_curve"] for r in runs], axis=0).tolist(),
        "n_seeds": len(runs),
    }


def describe(cfg: dict) -> str:
    return (f"N={cfg['n_nodes']} p={cfg['p_link']:.3f} w_in={cfg['input_weight']:.2f} "
            f"w0={cfg['weight_init_mean']:.2f}±{cfg['weight_init_sd']:.2f} "
            f"leak={cfg['leak']:.2f} lr={cfg['target_lr']:.4f} "
            f"thr={cfg['threshold_ratio']:.2f} gain={cfg['gain']:.1f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", type=int, default=240)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--out", type=str, default="scripts/out/sweep")
    args = parser.parse_args()
    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)
    configs = [dict(PAPER_CONFIG)] + [sample_config(rng) for _ in range(args.configs)]
    print(f"searching {len(configs)} configs (id 0 = paper defaults) x "
          f"{len(SEARCH_SEEDS)} seeds on {args.workers} workers")

    tasks = [(i, cfg, s) for i, cfg in enumerate(configs) for s in SEARCH_SEEDS]
    search_runs = run_batch(tasks, args.workers, "search")

    by_config: dict[int, list[dict]] = {}
    for r in search_runs:
        by_config.setdefault(r["config_id"], []).append(r)
    search_agg = {i: aggregate(runs) for i, runs in by_config.items()}

    order = sorted(search_agg, key=lambda i: search_agg[i]["score"], reverse=True)
    finalists = order[: args.top]
    if 0 not in finalists:  # always re-evaluate the paper baseline alongside
        finalists = finalists + [0]

    print(f"\nre-evaluating top {args.top} + baseline on {len(EVAL_SEEDS)} held-out seeds")
    eval_tasks = [(i, configs[i], s) for i in finalists for s in EVAL_SEEDS]
    eval_runs = run_batch(eval_tasks, args.workers, "eval")
    eval_by_config: dict[int, list[dict]] = {}
    for r in eval_runs:
        eval_by_config.setdefault(r["config_id"], []).append(r)
    eval_agg = {i: aggregate(runs) for i, runs in eval_by_config.items()}

    # ---- report -----------------------------------------------------------
    final_order = sorted(
        (i for i in eval_agg if i != 0),
        key=lambda i: eval_agg[i]["score"], reverse=True,
    )
    base = eval_agg[0]
    print("\n=== held-out results (16 seeds, mean ± sd of within-45) ===")
    print(f"{'rank':>4} {'id':>4} {'score':>13} {'dir':>5} {'spike':>6}  config")
    print(f"{'BASE':>4} {0:>4} {base['score']:.3f} ± {base['score_sd']:.3f} "
          f"{base['dir_agree']:.2f} {base['prop_spiked']:.3f}  {describe(PAPER_CONFIG)}")
    for rank, i in enumerate(final_order, 1):
        a = eval_agg[i]
        print(f"{rank:>4} {i:>4} {a['score']:.3f} ± {a['score_sd']:.3f} "
              f"{a['dir_agree']:.2f} {a['prop_spiked']:.3f}  {describe(configs[i])}")

    best = final_order[0]
    print(f"\nbaseline {base['score']:.3f} -> best {eval_agg[best]['score']:.3f} "
          f"({(eval_agg[best]['score'] - base['score']) / base['score'] * +100:.0f}% relative)")

    payload = {
        "n_steps": N_STEPS, "band_deg": BAND_DEG,
        "search_seeds": SEARCH_SEEDS, "eval_seeds": EVAL_SEEDS,
        "configs": configs,
        "search": {str(i): a for i, a in search_agg.items()},
        "eval": {str(i): a for i, a in eval_agg.items()},
        "final_order": final_order,
    }
    (out_dir / "results.json").write_text(json.dumps(payload))
    print(f"results saved to {out_dir / 'results.json'}")

    make_plots(configs, search_agg, eval_agg, final_order, out_dir)


def make_plots(configs, search_agg, eval_agg, final_order, out_dir) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    base = eval_agg[0]

    # 1. sorted search scores + baseline reference
    fig, ax = plt.subplots(figsize=(9, 4.5))
    scores = sorted((a["score"] for a in search_agg.values()), reverse=True)
    ax.plot(scores, ".-", ms=3, lw=0.6, color="tab:blue", label="configs (search, 6 seeds)")
    ax.axhline(search_agg[0]["score"], color="black", ls="--",
               label=f"paper defaults ({search_agg[0]['score']:.2f})")
    ax.axhline(0.25, color="gray", ls=":", label="chance (0.25)")
    ax.set_xlabel("config rank")
    ax.set_ylabel("within-45° fraction (full session)")
    ax.set_title("Random search: sorted config scores")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "sweep_scores.png", dpi=150)

    # 2. marginals: each parameter vs score
    params = list(PAPER_CONFIG)
    fig, axes = plt.subplots(3, 3, figsize=(11, 8.5))
    for ax, p in zip(axes.ravel(), params):
        xs = [configs[i][p] for i in search_agg if i != 0]
        ys = [search_agg[i]["score"] for i in search_agg if i != 0]
        ax.scatter(xs, ys, s=8, alpha=0.55, color="tab:blue")
        ax.scatter([PAPER_CONFIG[p]], [search_agg[0]["score"]], marker="*",
                   s=140, color="black", zorder=5, label="paper")
        if p in ("p_link", "input_weight", "weight_init_sd", "target_lr", "gain"):
            ax.set_xscale("log")
        ax.set_title(p, fontsize=10)
        ax.set_ylim(0, max(ys) + 0.05)
    axes[0, 0].set_ylabel("within-45°")
    axes[1, 0].set_ylabel("within-45°")
    axes[2, 0].set_ylabel("within-45°")
    fig.suptitle("Score vs. each parameter (search runs; ★ = paper default)")
    fig.tight_layout()
    fig.savefig(out_dir / "sweep_marginals.png", dpi=150)

    # 3. loss curves: per-epoch and reversal-aligned, baseline vs top 5
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    epochs = np.arange(1, 11)
    curve = np.array(base["epoch_curve"])
    sd = np.array(base["epoch_curve_sd"])
    ax1.errorbar(epochs, curve, yerr=sd, color="black", lw=2, capsize=3,
                 label="paper defaults")
    for rank, i in enumerate(final_order[:5], 1):
        ax1.plot(epochs, eval_agg[i]["epoch_curve"], marker="o", ms=3,
                 label=f"#{rank} (id {i})")
    ax1.axhline(0.25, color="gray", ls=":", lw=1)
    ax1.set_xlabel("reversal epoch (720 steps each)")
    ax1.set_ylabel("within-45° fraction")
    ax1.set_title("Across the session: per-epoch score")
    ax1.set_xticks(epochs)
    ax1.legend(fontsize=8)

    x = (np.arange(len(base["aligned_curve"])) + 0.5) * CURVE_DS
    ax2.plot(x, base["aligned_curve"], color="black", lw=2, label="paper defaults")
    for rank, i in enumerate(final_order[:5], 1):
        ax2.plot(x, eval_agg[i]["aligned_curve"], lw=1.2, label=f"#{rank} (id {i})")
    ax2.axhline(0.25, color="gray", ls=":", lw=1)
    ax2.set_xlabel("steps since direction reversal")
    ax2.set_ylabel("within-45° fraction")
    ax2.set_title("Within an epoch: re-locking after each reversal")
    ax2.legend(fontsize=8)
    fig.suptitle("Loss curves (held-out seeds, mean over 16 seeds)")
    fig.tight_layout()
    fig.savefig(out_dir / "sweep_curves.png", dpi=150)
    print(f"plots saved to {out_dir}/sweep_scores.png, sweep_marginals.png, sweep_curves.png")


if __name__ == "__main__":
    main()
