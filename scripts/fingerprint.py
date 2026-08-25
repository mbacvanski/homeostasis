"""Print the network-state fingerprint (sums of x, targets, weights) after
running the tracking simulation for a given number of steps.

The visualizer displays the same fingerprint live; matching values confirm
that what you watched in the browser is bit-for-bit the same trajectory the
batch code produces (same seed, same step count, default parameters).

Usage: python scripts/fingerprint.py --seed 0 --steps 500
"""

from __future__ import annotations

import argparse

import numpy as np

from homeostasis import TrackingSimulation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=500)
    args = parser.parse_args()

    sim = TrackingSimulation(seed=args.seed)
    for _ in range(args.steps):
        sim.step()
    net = sim.network
    print(
        f"seed {args.seed}, t={sim.t}: "
        f"Σx={np.sum(net.x):.6f} "
        f"ΣT={np.sum(net.targets):.6f} "
        f"ΣW={np.sum(net.weights):.6f} "
        f"| heading={sim.env.heading:.4f} stim={sim.env.stimulus_angle:.4f}"
    )


if __name__ == "__main__":
    main()
