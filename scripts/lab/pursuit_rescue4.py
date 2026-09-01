"""Pursuit round 4: the velocity-matching account. (A) lower churn speed
(lower f via wlr/rho); (B) raise stimulus speed toward the agent's cruise -
prediction: FASTER stimuli are easier (cannot entrain below own minimum
speed). 62-sensor forward retina, wb=4, w_in=8, falloff 3."""
from __future__ import annotations
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from common import make_configs  # noqa: F401
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from homeostasis.reservoir import ReservoirConfig  # noqa: E402
from homeostasis.simulation import run_pursuit  # noqa: E402
from homeostasis.pursuit import PursuitConfig  # noqa: E402

LAB = Path(__file__).resolve().parents[1] / "out" / "lab"


def evaluate(task):
    name, wlr, rho, sspeed, seed = task
    pc = PursuitConfig(wheel_base=4.0, stimulus_speed=sspeed)
    res = ReservoirConfig(n_inputs=62, input_weight=8.0, weight_lr=wlr,
                          threshold_ratio=rho)
    h = run_pursuit(n_steps=3600, seed=seed, reservoir_config=res, pursuit_config=pc)
    late = slice(1800, None)
    br = np.abs(h.bearing[late])
    speed = np.hypot(np.diff(h.x), np.diff(h.y))
    return dict(name=name, seed=seed, dist=float(h.dist[late].mean()),
                near3=float((h.dist[late] < 3).mean()),
                oriented=float((br <= 45).mean()),
                cruise=float(speed[1800:].mean()),
                hits=int(h.hit.sum()), f=float(h.prop_spiked[late].mean()))


def main():
    arms = ([(f"A:wlr{l}-rho{r}", l, r, 0.15) for l in (0.05, 0.1) for r in (2.0, 3.0)]
            + [(f"B:speed{s}", 0.1, 2.0, s) for s in (0.3, 0.5)])
    tasks = [(n, l, r, s, seed) for (n, l, r, s) in arms for seed in range(8)]
    with ProcessPoolExecutor(10) as pool:
        rows = list(pool.map(evaluate, tasks, chunksize=2))
    (LAB / "pursuit_rescue4.json").write_text(json.dumps(rows))
    print("arm              dist  near3  orient  cruise  hits   f")
    for (n, *_ ) in arms:
        sel = [r for r in rows if r["name"] == n]
        print(f"{n:16s} {np.mean([r['dist'] for r in sel]):5.2f} {np.mean([r['near3'] for r in sel]):.2f}   "
              f"{np.mean([r['oriented'] for r in sel]):.2f}    {np.mean([r['cruise'] for r in sel]):.2f}   "
              f"{np.mean([r['hits'] for r in sel]):4.0f}  {np.mean([r['f'] for r in sel]):.2f}")
    print("(stimulus speeds: A arms 0.15; B arms as named. still-agent dist ref 4.50 at 0.15)")


if __name__ == "__main__":
    main()
