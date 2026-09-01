"""B6b: re-entrainment time constant after stimulus reversals.

Windowed mean dH aligned on the 9 reversals, sign-normalized so +1 = fully
entrained to the NEW direction; fit tau by 63% crossing. Prediction (H19b):
tau shrinks as wlr grows (bias is re-learned at the absorption rate);
w1' (high leak, wlr=1) fastest.
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from common import make_configs
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from homeostasis.simulation import run_tracking  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "out"
LAB = OUT / "lab"
SEG, WIN = 720, 30


def w1p():
    cfg = json.loads((OUT / "sweep" / "results.json").read_text())["configs"][236]
    return (dict(n_nodes=cfg["n_nodes"], p_link=cfg["p_link"],
                 input_weight=cfg["input_weight"], weight_init_mean=0.75,
                 weight_init_sd=cfg["weight_init_sd"], leak=cfg["leak"],
                 target_lr=cfg["target_lr"], threshold_ratio=cfg["threshold_ratio"]),
            dict(gain=cfg["gain"]))


def evaluate(task):
    name, res_over, trk_over, seed = task
    rcfg, tcfg = make_configs(res_over, trk_over)
    h = run_tracking(n_steps=7200, seed=seed, reservoir_config=rcfg,
                     tracking_config=tcfg, record_spikes=False)
    dh = h.d_heading
    sd = h.stimulus_direction
    curves = []
    for rev in range(1, 9):  # reversals at 720*rev; skip t=0 start
        t0 = SEG * rev
        new_dir = sd[t0 + 5]
        seg = dh[t0:t0 + SEG] * new_dir  # + = entrained to new direction
        curve = seg[: (SEG // WIN) * WIN].reshape(-1, WIN).mean(axis=1)
        curves.append(curve)
    return dict(name=name, seed=seed, curve=np.mean(curves, axis=0).tolist(),
                score=float(np.mean(np.abs(h.error) <= 45)))


def main():
    w1p_res, w1p_trk = w1p()
    variants = {"wlr0.03": ({"weight_lr": 0.03}, {}),
                "wlr0.1": ({"weight_lr": 0.1}, {}),
                "wlr0.3": ({"weight_lr": 0.3}, {}),
                "wlr1.0": ({}, {}),
                "w1prime": (w1p_res, w1p_trk)}
    tasks = [(n, r, t, s) for n, (r, t) in variants.items() for s in range(8)]
    with ProcessPoolExecutor(10) as pool:
        rows = list(pool.map(evaluate, tasks))
    (LAB / "b6b_reentrainment.json").write_text(json.dumps(rows))

    print("Re-entrainment curves (windowed mean dH toward NEW direction, deg/step;")
    print(f"windows of {WIN} steps after each reversal, averaged over 8 reversals x 8 seeds)\n")
    for name in variants:
        c = np.mean([r["curve"] for r in rows if r["name"] == name], axis=0)
        sc = np.mean([r["score"] for r in rows if r["name"] == name])
        asym = c[-8:].mean()
        tau = next((i * WIN for i, v in enumerate(c) if asym > 0 and v >= 0.63 * asym), -1)
        head = " ".join(f"{v:+.2f}" for v in c[:12])
        print(f"   {name:8s} score {sc:.3f}  asym {asym:+.2f}  tau63 ~{tau:4d} steps   {head} ...")


if __name__ == "__main__":
    main()
