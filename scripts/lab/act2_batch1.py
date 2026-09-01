"""Act II batch 1: five experiments in one pool (see LEDGER H8-H12).

A1 closed-loop wlr x tlr plane (channel competition in behavior)
A2 open-loop darkness (endogenous churn test)
A3 closed-loop leak x wlr plane
A4 K3 autopsy arms at the w1' config (best-known medium; config 236 with
   weight_init_mean=0.75, loaded from the sweep JSON)
A5 open-loop fine slip-speed curve at wlr=0.1
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from common import run_closed_loop, run_open_loop

OUT = Path(__file__).resolve().parents[1] / "out"
LAB = OUT / "lab"
SEEDS = list(range(12))


def dispatch(task):
    r = run_open_loop(task) if task["mode"] == "open" else run_closed_loop(task)
    r["tag"] = task["_tag"]
    for k in ("wlr", "tlr", "leak", "speed"):
        if k in task:
            r[k] = task[k]
    return r


def w1prime_overrides():
    cfg = json.loads((OUT / "sweep" / "results.json").read_text())["configs"][236]
    res = dict(n_nodes=cfg["n_nodes"], p_link=cfg["p_link"],
               input_weight=cfg["input_weight"], weight_init_mean=0.75,
               weight_init_sd=cfg["weight_init_sd"], leak=cfg["leak"],
               target_lr=cfg["target_lr"], threshold_ratio=cfg["threshold_ratio"])
    trk = dict(gain=cfg["gain"])
    return res, trk


def main():
    tasks = []
    # A1: wlr x tlr closed loop
    for wlr in (0.0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0):
        for tlr in (0.001, 0.01, 0.1):
            for s in SEEDS:
                tasks.append(dict(mode="closed", res={"weight_lr": wlr, "target_lr": tlr},
                                  trk={}, seed=s, arm="full", _tag="A1",
                                  wlr=wlr, tlr=tlr, snap_every=720))
    # A2: darkness open loop
    for wlr in (0.1, 1.0, 3.0):
        for tlr in (0.001, 0.01, 0.1):
            for s in SEEDS[:6]:
                tasks.append(dict(mode="open", res={"weight_lr": wlr, "target_lr": tlr},
                                  seed=s, schedule={"kind": "dark"}, n_steps=3000,
                                  _tag="A2", wlr=wlr, tlr=tlr))
    # A3: leak x wlr closed loop
    for leak in (0.05, 0.1, 0.25, 0.5, 0.75, 0.9):
        for wlr in (0.03, 0.1, 0.3, 1.0):
            for s in SEEDS:
                tasks.append(dict(mode="closed", res={"leak": leak, "weight_lr": wlr},
                                  trk={}, seed=s, arm="full", _tag="A3",
                                  leak=leak, wlr=wlr, snap_every=720))
    # A4: autopsy at w1'
    res_w1p, trk_w1p = w1prime_overrides()
    for arm in ("full", "no-learn", "lesion", "freeze-mid", "freeze-mid-resetT",
                "shuffle-mid", "freeze-W-only", "freeze-T-only"):
        for s in SEEDS:
            tasks.append(dict(mode="closed", res=dict(res_w1p), trk=dict(trk_w1p),
                              seed=s, arm=arm, _tag=f"A4:{arm}", snap_every=720))
    # A5: fine slip curve, wlr=0.1
    for speed in np.geomspace(0.05, 8.0, 12):
        for s in SEEDS:
            tasks.append(dict(mode="open", res={"weight_lr": 0.1, "target_lr": 0.01},
                              seed=s, schedule={"kind": "slip", "speed": float(speed)},
                              n_steps=3000, _tag="A5", speed=round(float(speed), 3)))

    print(f"{len(tasks)} runs...")
    with ProcessPoolExecutor(10) as pool:
        rows = list(pool.map(dispatch, tasks, chunksize=4))
    LAB.mkdir(exist_ok=True)
    slim = [{k: v for k, v in r.items() if k not in ("f_t", "law", "snaps")} for r in rows]
    (LAB / "act2_batch1.json").write_text(json.dumps(slim))
    snapful = [r for r in rows if r["tag"] in ("A1", "A3") and r["seed"] < 3 and "snaps" in r]
    (LAB / "act2_batch1_snaps.json").write_text(json.dumps(snapful))

    def sel(tag, **kv):
        out = []
        for r in rows:
            if r["tag"] != tag:
                continue
            if all(abs(r.get(k, np.nan) - v) < 1e-9 for k, v in kv.items()):
                out.append(r)
        return out

    print("\n══ A1: closed-loop score_late (wlr rows x tlr cols); frac(seeds ≥0.35) in parens")
    print("          tlr=0.001        tlr=0.01         tlr=0.1")
    for wlr in (0.0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0):
        cells = []
        for tlr in (0.001, 0.01, 0.1):
            v = np.array([r["score_late"] for r in sel("A1", wlr=wlr, tlr=tlr)])
            cells.append(f"{v.mean():.3f}({np.mean(v >= 0.35):.2f})")
        print(f"   wlr={wlr:<5} " + "   ".join(f"{c:>14s}" for c in cells))
    print("   … prop_spiked means:")
    for wlr in (0.0, 0.1, 1.0):
        cells = [np.mean([r["prop_spiked"] for r in sel("A1", wlr=wlr, tlr=tlr)])
                 for tlr in (0.001, 0.01, 0.1)]
        print(f"   wlr={wlr:<5} " + "   ".join(f"{c:14.3f}" for c in cells))

    print("\n══ A2: darkness f_late (endogenous churn)")
    for wlr in (0.1, 1.0, 3.0):
        cells = [np.mean([r["f_mean_late"] for r in sel("A2", wlr=wlr, tlr=tlr)])
                 for tlr in (0.001, 0.01, 0.1)]
        print(f"   wlr={wlr:<4} " + "  ".join(f"{c:.3f}" for c in cells))

    print("\n══ A3: score_late leak x wlr")
    print("            wlr=0.03  wlr=0.1  wlr=0.3  wlr=1.0")
    for leak in (0.05, 0.1, 0.25, 0.5, 0.75, 0.9):
        cells = [np.mean([r["score_late"] for r in sel("A3", leak=leak, wlr=wlr)])
                 for wlr in (0.03, 0.1, 0.3, 1.0)]
        print(f"   leak={leak:<5} " + "  ".join(f"{c:7.3f}" for c in cells))

    print("\n══ A4: autopsy at w1' (score_late mean±sd)")
    for arm in ("full", "no-learn", "lesion", "freeze-mid", "freeze-mid-resetT",
                "shuffle-mid", "freeze-W-only", "freeze-T-only"):
        v = np.array([r["score_late"] for r in rows if r["tag"] == f"A4:{arm}"])
        print(f"   {arm:20s} {v.mean():.3f}±{v.std():.3f}")

    print("\n══ A5: f_late vs slip speed (wlr=0.1)")
    speeds = sorted({r["speed"] for r in rows if r["tag"] == "A5"})
    for sp in speeds:
        v = np.array([r["f_mean_late"] for r in sel("A5", speed=sp)])
        print(f"   speed {sp:6.3f}  f {v.mean():.4f}±{v.std():.4f}")


if __name__ == "__main__":
    main()
