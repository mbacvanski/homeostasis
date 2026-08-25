"""Run the moving-object tracking experiment (Falandays et al. 2024, case
study 1) across seeds, report tracking metrics, and produce Fig. 4-style
verification plots for a representative run.

Usage: python scripts/run_tracking_experiment.py [--seeds N] [--steps N]
"""

from __future__ import annotations

import argparse

import numpy as np

from homeostasis import run_tracking, tracking_metrics
from homeostasis.tracking import angular_difference


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--steps", type=int, default=7200)
    parser.add_argument("--plot-seed", type=int, default=None,
                        help="seed to plot (default: best within45)")
    parser.add_argument("--out", type=str, default="scripts/out")
    args = parser.parse_args()

    rows = []
    histories = {}
    for seed in range(args.seeds):
        h = run_tracking(n_steps=args.steps, seed=seed)
        m = tracking_metrics(h)
        h_off = run_tracking(n_steps=args.steps, seed=seed, learning_enabled=False)
        m_off = tracking_metrics(h_off)
        rows.append((seed, m, m_off))
        histories[seed] = h
        print(
            f"seed {seed:2d} | within45 {m['within45']:.2f} "
            f"(no-learn {m_off['within45']:.2f}) | "
            f"median|err| {m['median_abs_error']:6.1f} deg "
            f"(no-learn {m_off['median_abs_error']:6.1f}) | "
            f"dir-agree {m['direction_agreement']:.2f} "
            f"(no-learn {m_off['direction_agreement']:.2f}) | "
            f"spiking {m['prop_spiked_mean']:.2f}"
        )

    w45 = np.array([r[1]["within45"] for r in rows])
    w45_off = np.array([r[2]["within45"] for r in rows])
    dir_on = np.array([r[1]["direction_agreement"] for r in rows])
    print(
        f"\nlearning ON : within45 median {np.median(w45):.2f} "
        f"(min {w45.min():.2f}, max {w45.max():.2f}), "
        f"dir-agree median {np.median(dir_on):.2f}"
    )
    print(
        f"learning OFF: within45 median {np.median(w45_off):.2f} "
        f"(min {w45_off.min():.2f}, max {w45_off.max():.2f})"
    )

    # ---- Fig. 4-style plots for a representative run ----------------------
    import pathlib

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_seed = args.plot_seed if args.plot_seed is not None else int(np.argmax(w45))
    h = histories[plot_seed]
    t = np.arange(len(h))

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), height_ratios=[2, 1, 1], sharex=True)
    axes[0].scatter(t, h.stimulus_angle, s=1.5, c="black", label="Stimulus")
    axes[0].scatter(t, h.heading, s=1.5, c="red", label="Agent")
    axes[0].set_ylabel("Heading (deg.)")
    axes[0].set_yticks([0, 90, 180, 270, 360])
    axes[0].legend(loc="upper right", markerscale=6)
    axes[0].set_title(f"Moving-object tracking, seed {plot_seed} (cf. paper Fig. 4A)")

    axes[1].plot(t, h.prop_spiked, lw=0.6, c="black")
    axes[1].set_ylabel("Prop. spiked")
    axes[1].set_ylim(0, 1)

    err = angular_difference(h.stimulus_angle, h.heading)
    axes[2].plot(t, err, lw=0.6, c="tab:blue")
    axes[2].axhline(0, c="gray", lw=0.5)
    axes[2].set_ylabel("Heading error (deg.)")
    axes[2].set_xlabel("Time")
    axes[2].set_ylim(-185, 185)
    for ax in axes:
        for flip in range(720, len(h), 720):
            ax.axvline(flip, color="gray", lw=0.4, alpha=0.5)

    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_dir / "tracking_run.png", dpi=150)
    print(f"\nplot saved to {out_dir / 'tracking_run.png'} (seed {plot_seed})")


if __name__ == "__main__":
    main()
