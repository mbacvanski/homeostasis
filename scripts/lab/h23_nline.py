"""H23: N-line at leak=0.25 x wlr {0.05,0.1,0.2}, N {50,100,200,400}, 12 seeds.
Plus a crude local pre-fit of the ridge law wlr*(leak) from the A3 plane."""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from common import run_closed_loop

LAB = Path(__file__).resolve().parents[1] / "out" / "lab"


def run(t):
    r = run_closed_loop(t)
    r["n"] = t["res"]["n_nodes"]
    r["wlr"] = t["res"]["weight_lr"]
    return r


def main():
    tasks = [dict(res={"n_nodes": n, "weight_lr": w}, trk={}, seed=s, arm="full",
                  snap_every=7200)
             for n in (50, 100, 200, 400) for w in (0.05, 0.1, 0.2)
             for s in range(12)]
    with ProcessPoolExecutor(10) as pool:
        rows = list(pool.map(run, tasks, chunksize=2))
    slim = [{k: v for k, v in r.items() if k != "snaps"} for r in rows]
    (LAB / "h23_nline.json").write_text(json.dumps(slim))

    print("H23 N-line (score_late, 12 seeds; frac>=0.35):")
    print("          wlr=0.05        wlr=0.1         wlr=0.2")
    for n in (50, 100, 200, 400):
        cells = []
        for w in (0.05, 0.1, 0.2):
            v = np.array([r["score_late"] for r in rows if r["n"] == n and r["wlr"] == w])
            cells.append(f"{v.mean():.3f} ({np.mean(v>=0.35):.2f})")
        print(f"   N={n:<4} " + "   ".join(f"{c:>13s}" for c in cells))

    # crude ridge-law pre-fit from local A3
    b1 = json.loads((LAB / "act2_batch1.json").read_text())
    a3 = [r for r in b1 if r.get("tag") == "A3"]
    leaks = (0.05, 0.1, 0.25, 0.5, 0.75)
    wlrs = (0.03, 0.1, 0.3, 1.0)
    pts = []
    for lk in leaks:
        best_w, best_s = None, -1
        for w in wlrs:
            v = np.mean([r["score_late"] for r in a3
                         if abs(r["leak"] - lk) < 1e-9 and abs(r["wlr"] - w) < 1e-9])
            if v > best_s:
                best_s, best_w = v, w
        if best_s >= 0.3:
            pts.append((lk, best_w))
    if len(pts) >= 3:
        L = np.log([p[0] for p in pts])
        W = np.log([p[1] for p in pts])
        b, logc = np.polyfit(L, W, 1)
        print(f"\nridge pre-fit (coarse local A3, argmax per leak): "
              f"wlr* ≈ {np.exp(logc):.3f} · leak^{b:.2f}   points: {pts}")


if __name__ == "__main__":
    main()
