"""K1: open-loop adaptation factorial — which channel absorbs drive, and when
does spiking survive?

Full N=200 default network driven by a SCRIPTED retina (no motor loop):
weight_lr {0, 0.01, 0.1, 1.0} x target_lr {0.001, 0.01, 0.1} x schedule
{stationary theta=0, slip 0.25, slip 1, slip 4 deg/step, jump 0->30 at 1500}
x 12 CRN seeds, 3000 steps.

Preregistered (ledger H5/H6):
- Stationary drive is silenced (f_late ~ 0) for every lr combo with any
  learning; time-to-silence shrinks as the DOMINANT channel's lr grows.
- f_late increases with slip speed (fluctuation-driven spiking).
- At weight_lr=1.0 (default), target_lr barely matters while spiking is dense
  (weight channel dominates); at weight_lr=0, target_lr sets everything.
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from common import run_open_loop

LAB = Path(__file__).resolve().parents[1] / "out" / "lab"

W_LRS = [0.0, 0.01, 0.1, 1.0]
T_LRS = [0.001, 0.01, 0.1]
SCHEDULES = [
    ("stat", {"kind": "stationary", "theta": 0.0}),
    ("slip.25", {"kind": "slip", "speed": 0.25}),
    ("slip1", {"kind": "slip", "speed": 1.0}),
    ("slip4", {"kind": "slip", "speed": 4.0}),
    ("jump", {"kind": "jump", "theta0": 0.0, "theta1": 30.0, "t_jump": 1500}),
]
SEEDS = list(range(12))


def main():
    tasks = []
    for wlr in W_LRS:
        for tlr in T_LRS:
            for sname, sched in SCHEDULES:
                for seed in SEEDS:
                    tasks.append(dict(
                        res={"weight_lr": wlr, "target_lr": tlr},
                        seed=seed, schedule=sched, n_steps=3000,
                        per_node=(seed < 2 and sname in ("slip1", "jump")
                                  and wlr == 1.0 and tlr == 0.01),
                        _name=sname,
                    ))
    with ProcessPoolExecutor(10) as pool:
        rows = list(pool.map(run_open_loop, tasks, chunksize=4))
    for t, r in zip(tasks, rows):
        r["name"] = t["_name"]
        r["wlr"] = t["res"]["weight_lr"]
        r["tlr"] = t["res"]["target_lr"]

    LAB.mkdir(exist_ok=True)
    slim = [{k: v for k, v in r.items() if k not in ("f_t", "law")} for r in rows]
    (LAB / "k1_openloop.json").write_text(json.dumps(slim))
    law_rows = [r for r in rows if "law" in r]
    (LAB / "k1_law_data.json").write_text(json.dumps(law_rows))

    def cell(wlr, tlr, name, field="f_mean_late"):
        vals = [r[field] for r in rows if r["wlr"] == wlr and r["tlr"] == tlr and r["name"] == name]
        return float(np.mean(vals))

    print(f"K1: {len(rows)} open-loop runs (12 seeds each cell)\n")
    for name, _ in SCHEDULES:
        print(f"── {name}: mean f_late by (weight_lr x target_lr)")
        print("        tlr=0.001  tlr=0.01  tlr=0.1")
        for wlr in W_LRS:
            vals = [cell(wlr, tlr, name) for tlr in T_LRS]
            print(f"   wlr={wlr:<5} " + "  ".join(f"{v:8.4f}" for v in vals))
        print()

    print("── time-to-silence (median silence_step, stationary only; -1 = never):")
    print("        tlr=0.001  tlr=0.01  tlr=0.1")
    for wlr in W_LRS:
        med = []
        for tlr in T_LRS:
            v = [r["silence_step"] for r in rows
                 if r["wlr"] == wlr and r["tlr"] == tlr and r["name"] == "stat"]
            med.append(int(np.median(v)))
        print(f"   wlr={wlr:<5} " + "  ".join(f"{v:8d}" for v in med))

    print(f"\nwrote {LAB/'k1_openloop.json'} (+ k1_law_data.json, "
          f"{len(law_rows)} per-node law runs)")


if __name__ == "__main__":
    main()
