"""H24: ridge-law out-of-family test at w1' + ingredient attribution."""
from __future__ import annotations
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from common import run_closed_loop

OUT = Path(__file__).resolve().parents[1] / "out"
LAB = OUT / "lab"

def run(t):
    r = run_closed_loop(t)
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
    for wlr in (0.3, 0.65, 1.0):
        for s in range(16):
            tasks.append(dict(res=dict(w1p, weight_lr=wlr), trk=dict(trk), seed=s,
                              arm="full", snap_every=7200, _t=f"w1p:wlr{wlr}"))
    base = dict(leak=0.25, weight_lr=0.1)
    grafts = {"base": {}, "gain28": {}, "rho1.5": {"threshold_ratio": 1.525},
              "win0.83": {"input_weight": 0.828},
              "gain+rho": {"threshold_ratio": 1.525}}
    gtrk = {"base": {}, "gain28": {"gain": 28.35}, "rho1.5": {},
            "win0.83": {}, "gain+rho": {"gain": 28.35}}
    for name, res_extra in grafts.items():
        for s in range(12):
            tasks.append(dict(res=dict(base, **res_extra), trk=dict(gtrk[name]),
                              seed=s, arm="full", snap_every=7200, _t=f"graft:{name}"))
    with ProcessPoolExecutor(10) as pool:
        rows = list(pool.map(run, tasks, chunksize=2))
    (LAB / "h24_ridge_oof.json").write_text(json.dumps(
        [{k: v for k, v in r.items() if k != "snaps"} for r in rows]))
    print("H24a — w1' x wlr (16 seeds):")
    for wlr in (0.3, 0.65, 1.0):
        v = np.array([r["score_late"] for r in rows if r["t"] == f"w1p:wlr{wlr}"])
        print(f"   wlr={wlr:<5} {v.mean():.3f}±{v.std():.3f}  frac>=0.35 {np.mean(v>=0.35):.2f}  min {v.min():.2f}")
    print("H24b — grafts onto ridge25 (12 seeds):")
    for name in grafts:
        v = np.array([r["score_late"] for r in rows if r["t"] == f"graft:{name}"])
        print(f"   {name:9s} {v.mean():.3f}±{v.std():.3f}  frac>=0.35 {np.mean(v>=0.35):.2f}  min {v.min():.2f}")

if __name__ == "__main__":
    main()
