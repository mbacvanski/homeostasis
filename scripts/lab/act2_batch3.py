"""Act II batch 3: the flow-ratchet swap test (H18) + target_init at w1' (H16b)
+ the ratchet-pawl check on already-recorded policy statistics.

B4: effector swap at t=3600 (arm "swap-mid") vs full, at ridge25 / w1' /
default, 24 seeds. Segment scores around the swap tell the recovery story.
B5: w1' with tlr=0 x target_init {1, 2, 3, 4} x 12 seeds — is w1's target
benefit just gain normalization?
Pawl: from act2_batch1.json A1 rows — mean |dH| in out-of-view heading-error
bins vs in-view bins (prediction: out-of-view turning collapses).
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from common import ERR_EDGES, run_closed_loop

OUT = Path(__file__).resolve().parents[1] / "out"
LAB = OUT / "lab"


def w1p():
    cfg = json.loads((OUT / "sweep" / "results.json").read_text())["configs"][236]
    res = dict(n_nodes=cfg["n_nodes"], p_link=cfg["p_link"],
               input_weight=cfg["input_weight"], weight_init_mean=0.75,
               weight_init_sd=cfg["weight_init_sd"], leak=cfg["leak"],
               target_lr=cfg["target_lr"], threshold_ratio=cfg["threshold_ratio"])
    return res, dict(gain=cfg["gain"])


def dispatch(task):
    r = run_closed_loop(task)
    r["tag"] = task["_tag"]
    return r


def main():
    w1p_res, w1p_trk = w1p()
    variants = {
        "ridge25": ({"leak": 0.25, "weight_lr": 0.1}, {}),
        "w1prime": (w1p_res, w1p_trk),
        "default": ({}, {}),
    }
    tasks = []
    for name, (res, trk) in variants.items():
        for arm in ("full", "swap-mid"):
            for s in range(24):
                tasks.append(dict(res=dict(res), trk=dict(trk), seed=s, arm=arm,
                                  snap_every=2400, _tag=f"B4:{name}:{arm}"))
    for ti in (1.0, 2.0, 3.0, 4.0):
        res = dict(w1p_res, target_lr=0.0, target_init=ti)
        for s in range(12):
            tasks.append(dict(res=res, trk=dict(w1p_trk), seed=s, arm="full",
                              snap_every=7200, _tag=f"B5:ti{ti}"))

    print(f"{len(tasks)} runs...")
    with ProcessPoolExecutor(10) as pool:
        rows = list(pool.map(dispatch, tasks, chunksize=2))
    LAB.mkdir(exist_ok=True)
    slim = [{k: v for k, v in r.items() if k not in ("snaps",)} for r in rows]
    (LAB / "act2_batch3.json").write_text(json.dumps(slim))

    print("\n══ B4 swap test: per-segment scores (segments 1-10; swap at segment 6)")
    for name in variants:
        for arm in ("full", "swap-mid"):
            segs = np.array([r["seg_scores"] for r in rows
                             if r["tag"] == f"B4:{name}:{arm}"])
            m = segs.mean(axis=0)
            print(f"   {name:8s} {arm:9s} " + " ".join(f"{v:.2f}" for v in m)
                  + f"   segs9-10 {segs[:, 8:].mean():.3f}")
        # recovery metric: swap segs 9-10 vs full segs 9-10, and swap seg 6 (the dip)
        sw = np.array([r["seg_scores"] for r in rows if r["tag"] == f"B4:{name}:swap-mid"])
        fu = np.array([r["seg_scores"] for r in rows if r["tag"] == f"B4:{name}:full"])
        print(f"   {name:8s} dip seg6 {sw[:, 5].mean():.3f} -> recovery ratio "
              f"(swap segs9-10)/(full segs9-10) = {sw[:, 8:].mean() / max(fu[:, 8:].mean(), 1e-9):.2f}")

    print("\n══ B5 w1' tlr=0 x target_init (score_late, 12 seeds; full-w1' anchor 0.85):")
    for ti in (1.0, 2.0, 3.0, 4.0):
        v = np.array([r["score_late"] for r in rows if r["tag"] == f"B5:ti{ti}"])
        print(f"   target_init={ti}: {v.mean():.3f}±{v.std():.3f} (frac≥0.35 {np.mean(v>=0.35):.2f})")

    print("\n══ Ratchet pawl (from act2_batch1 A1 rows): |mean dH| by heading-error zone")
    b1 = json.loads((LAB / "act2_batch1.json").read_text())
    centers = (ERR_EDGES[:-1] + ERR_EDGES[1:]) / 2
    out_view = np.abs(centers) > 95
    in_view = np.abs(centers) <= 45
    for wlr in (0.1, 1.0):
        rowsA = [r for r in b1 if r.get("tag") == "A1" and r.get("wlr") == wlr
                 and r.get("tlr") == 0.01 and "policy" in r]
        cnt = np.sum([r["policy"]["count"] for r in rowsA], axis=0)
        s = np.sum([r["policy"]["sum"] for r in rowsA], axis=0)
        ss = np.sum([r["policy"]["sumsq"] for r in rowsA], axis=0)
        mean_dh = np.where(cnt > 0, s / np.maximum(cnt, 1), 0.0)
        var_dh = np.where(cnt > 1, ss / np.maximum(cnt, 1) - mean_dh**2, 0.0)
        rms = np.sqrt(np.maximum(var_dh + mean_dh**2, 0))
        def zone(mask):
            w = cnt[mask]
            return float(np.average(rms[mask], weights=np.maximum(w, 1)))
        print(f"   wlr={wlr}: RMS dH in-view {zone(in_view):.3f}  out-of-view {zone(out_view):.3f}"
              f"   occupancy out-of-view {cnt[out_view].sum() / cnt.sum():.2f}")


if __name__ == "__main__":
    main()
