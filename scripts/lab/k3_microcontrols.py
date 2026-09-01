"""K3: closed-loop micro-controls before any phase mapping.

1. Gain sensitivity: gain {2.5, 10, 40} x 12 CRN seeds at defaults — how much
   of "score" is actuation scale? (If large: phases live on internal coords.)
2. Freeze-mid transfer: does the Pong hysteresis (freeze-from-init harmless,
   freeze-mid collapses) hold in tracking? arms {full, no-learn, freeze-mid}
   + decompositions {freeze-mid-resetT, freeze-mid-resetW, shuffle-mid,
   freeze-W-only, freeze-T-only} x 12 seeds.
3. Segment stationarity at baseline (from the full arm's seg_scores).

Preregistered (ledger H7): freeze-mid collapses only if the network is a
"medium"; at paper defaults (a weak baseline ~0.40) the prior reflex-vs-medium
work suggests partial collapse; resetT distinguishes threshold-inflation
artifact from computation-in-plasticity.
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from common import run_closed_loop

LAB = Path(__file__).resolve().parents[1] / "out" / "lab"
SEEDS = list(range(12))
ARMS = ["full", "no-learn", "freeze-mid", "freeze-mid-resetT", "freeze-mid-resetW",
        "shuffle-mid", "freeze-W-only", "freeze-T-only", "lesion"]


def main():
    tasks = []
    for g in (2.5, 10.0, 40.0):
        for s in SEEDS:
            tasks.append(dict(res={}, trk={"gain": g}, seed=s, arm="full", _tag=f"gain{g}"))
    for arm in ARMS:
        for s in SEEDS:
            tasks.append(dict(res={}, trk={}, seed=s, arm=arm, _tag=f"arm:{arm}"))

    with ProcessPoolExecutor(10) as pool:
        rows = list(pool.map(run_closed_loop, tasks, chunksize=2))
    for t, r in zip(tasks, rows):
        r["tag"] = t["_tag"]

    LAB.mkdir(exist_ok=True)
    (LAB / "k3_microcontrols.json").write_text(json.dumps(rows))

    def agg(tag, field="score_late"):
        v = np.array([r[field] for r in rows if r["tag"] == tag])
        return f"{v.mean():.3f}±{v.std():.3f} (frac≥0.35: {np.mean(v >= 0.35):.2f})"

    print(f"K3: {len(rows)} runs\n")
    print("── gain sensitivity (score_late, 12 seeds):")
    for g in (2.5, 10.0, 40.0):
        tag = f"gain{g}"
        v = np.array([r["score_late"] for r in rows if r["tag"] == tag])
        f = np.array([r["prop_spiked"] for r in rows if r["tag"] == tag])
        d = np.array([r["dir_agree"] for r in rows if r["tag"] == tag])
        print(f"   gain={g:<5} score {v.mean():.3f}±{v.std():.3f}  "
              f"dir-agree {d.mean():.3f}  prop_spiked {f.mean():.3f}")

    print("\n── autopsy arms at defaults (score_late; first/second-half score for mid arms):")
    for arm in ARMS:
        tag = f"arm:{arm}"
        sel = [r for r in rows if r["tag"] == tag]
        v = np.array([r["score_late"] for r in sel])
        segs = np.array([r["seg_scores"] for r in sel])
        h1 = segs[:, :5].mean() if segs.shape[1] >= 10 else float("nan")
        h2 = segs[:, 5:].mean() if segs.shape[1] >= 10 else float("nan")
        print(f"   {arm:20s} late {v.mean():.3f}±{v.std():.3f}   "
              f"half1 {h1:.3f}  half2 {h2:.3f}")

    print("\n── segment stationarity (full arm, per-segment mean over 12 seeds):")
    segs = np.array([r["seg_scores"] for r in rows if r["tag"] == "arm:full"])
    print("   " + "  ".join(f"{v:.3f}" for v in segs.mean(axis=0)))
    print(f"\nwrote {LAB/'k3_microcontrols.json'}")


if __name__ == "__main__":
    main()
