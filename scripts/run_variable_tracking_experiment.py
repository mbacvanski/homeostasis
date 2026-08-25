"""Run and plot the opt-in irregular-motion tracking task.

Usage:
    .venv/bin/python scripts/run_variable_tracking_experiment.py

The published constant-speed task and its output files are left untouched.
This script writes ``scripts/out/variable_tracking_run.pdf`` by default.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from homeostasis import VariableTrackingConfig, run_variable_tracking, tracking_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=7200)
    parser.add_argument("--initial-speed", type=float, default=1.0)
    parser.add_argument("--speed-min", type=float, default=0.5)
    parser.add_argument("--speed-max", type=float, default=1.5)
    parser.add_argument("--speed-smoothing", type=float, default=0.02)
    parser.add_argument("--speed-change-min", type=int, default=180)
    parser.add_argument("--speed-change-max", type=int, default=540)
    parser.add_argument("--reverse-min", type=int, default=480)
    parser.add_argument("--reverse-max", type=int, default=960)
    parser.add_argument("--out", type=Path, default=Path("scripts/out/variable_tracking_run.pdf"))
    args = parser.parse_args()

    config = VariableTrackingConfig(
        stimulus_speed=args.initial_speed,
        stimulus_speed_min=args.speed_min,
        stimulus_speed_max=args.speed_max,
        speed_smoothing=args.speed_smoothing,
        speed_change_min_steps=args.speed_change_min,
        speed_change_max_steps=args.speed_change_max,
        reverse_min_steps=args.reverse_min,
        reverse_max_steps=args.reverse_max,
    )
    history = run_variable_tracking(
        n_steps=args.steps,
        seed=args.seed,
        tracking_config=config,
        record_spikes=False,
    )
    metrics = tracking_metrics(history)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.arange(len(history))
    flips = np.flatnonzero(np.diff(history.stimulus_direction) != 0) + 1
    fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)

    axes[0].scatter(t, history.stimulus_angle, s=1.2, c="black", label="Stimulus")
    axes[0].scatter(t, history.heading, s=1.2, c="red", label="Agent")
    axes[0].set_ylabel("Heading (deg.)")
    axes[0].set_yticks([0, 90, 180, 270, 360])
    axes[0].legend(loc="upper right", markerscale=6)
    axes[0].set_title(f"Irregular-motion tracking, seed {args.seed}")

    axes[1].plot(t, history.stimulus_speed, color="tab:green", linewidth=0.8)
    axes[1].set_ylabel("Stimulus speed\n(deg./step)")
    axes[1].set_ylim(bottom=0)

    axes[2].plot(t, history.error, color="tab:blue", linewidth=0.7)
    axes[2].axhspan(-45, 45, color="tab:green", alpha=0.1)
    axes[2].axhline(0, color="gray", linewidth=0.5)
    axes[2].set_ylabel("Heading error\n(deg.)")
    axes[2].set_ylim(-185, 185)

    axes[3].plot(t, history.prop_spiked, color="black", linewidth=0.7)
    axes[3].set_ylabel("Prop. spiked")
    axes[3].set_xlabel("Time step")
    axes[3].set_ylim(0, 1)

    for axis in axes:
        for flip in flips:
            axis.axvline(flip, color="gray", linewidth=0.35, alpha=0.45)

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    print(
        f"within45={metrics['within45']:.3f} "
        f"median_abs_error={metrics['median_abs_error']:.2f} "
        f"direction_agreement={metrics['direction_agreement']:.3f}"
    )
    print(f"plot saved to {args.out}")


if __name__ == "__main__":
    main()
