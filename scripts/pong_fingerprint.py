"""Print the network-state fingerprint after running the Pong simulation for a
given number of steps, with the published default parameters.

The visualizer at /pong displays the same fingerprint live; matching values
confirm that what you watched in the browser is bit-for-bit the trajectory the
batch code produces.

Usage: python scripts/pong_fingerprint.py --seed 0 --steps 2000
"""

from __future__ import annotations

import argparse

import numpy as np

from homeostasis import PongSimulation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=2000)
    args = parser.parse_args()

    sim = PongSimulation(seed=args.seed)
    for _ in range(args.steps):
        sim.step()
    net, env = sim.network, sim.env
    hits = np.asarray(env.hits, dtype=float)
    print(
        f"seed {args.seed}, t={sim.t}: "
        f"Σx={np.sum(net.x):.6f} "
        f"ΣT={np.sum(net.targets):.6f} "
        f"ΣW={np.sum(net.weights):.6f} "
        f"opps={hits.size} "
        f"| hit rate {hits.mean() if hits.size else float('nan'):.4f} "
        f"| ball=({env.ball_x:.2f}, {env.ball_y:.2f}) paddle={env.paddle_y:.2f}"
    )


if __name__ == "__main__":
    main()
