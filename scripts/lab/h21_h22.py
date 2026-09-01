"""H21: freeze-T-mid at w1' (static heterogeneous T vs dynamic T).
H22: bandpass upper edge (short-period B3 columns)."""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from common import run_closed_loop, run_open_loop

OUT = Path(__file__).resolve().parents[1] / "out"
LAB = OUT / "lab"


def run(t):
    r = run_open_loop(t) if t.get("_mode") == "open" else run_closed_loop(t)
    r["t"] = t["_t"]
    return r


def main():
    cfg = json.loads((OUT / "sweep" / "results.json").read_text())["configs"][236]
    w1p = dict(n_nodes=cfg["n_nodes"], p_link=cfg["p_link"],
               input_weight=cfg["input_weight"], weight_init_mean=0.75,
               weight_init_sd=cfg["weight_init_sd"], leak=cfg["leak"],
               target_lr=cfg["target_lr"], threshold_ratio=cfg["threshold_ratio"])
    trk = dict(gain=cfg["gain"])

    tasks = []
    for arm in ("full", "freeze-T-mid", "freeze-T-only"):
        for s in range(24):
            tasks.append(dict(res=dict(w1p), trk=dict(trk), seed=s, arm=arm,
                              snap_every=7200, _t=f"H21:{arm}"))
    for P in (8, 15, 30):
        for s in range(8):
            tasks.append(dict(res={"weight_lr": 0.1, "target_lr": 0.01}, seed=s,
                              n_steps=4000, recon=True,
                              schedule={"kind": "sine", "amp": 20.0, "period": P},
                              _t=f"H22:P{P}", _mode="open"))

    with ProcessPoolExecutor(10) as pool:
        rows = list(pool.map(run, tasks, chunksize=2))
    slim = [{k: v for k, v in r.items() if k not in ("snaps", "f_t", "law")} for r in rows]
    (LAB / "h21_h22.json").write_text(json.dumps(slim))

    print("H21 (w1', 24 seeds, score_late):")
    for arm in ("full", "freeze-T-mid", "freeze-T-only"):
        v = np.array([r["score_late"] for r in rows if r["t"] == f"H21:{arm}"])
        print(f"   {arm:14s} {v.mean():.3f}±{v.std():.3f}  frac>=0.35 {np.mean(v >= 0.35):.2f}")
    print("H22 (wlr=0.1, recon gain at short periods):")
    for P in (8, 15, 30):
        v = [r["recon_gain"] for r in rows if r["t"] == f"H22:P{P}"]
        print(f"   P={P:3d}  {np.mean(v):.3f}")


if __name__ == "__main__":
    main()
