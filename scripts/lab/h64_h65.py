"""H64: dense-fast policy flatness. H65: info-per-spike vs sparsity."""
from __future__ import annotations
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from common import run_closed_loop, run_open_loop, ERR_EDGES

LAB = Path(__file__).resolve().parents[1] / "out" / "lab"

def closed(task):
    r = run_closed_loop(task)
    pol = r["policy"]
    cnt = np.maximum(np.array(pol["count"]), 1)
    mean_dh = np.array(pol["sum"]) / cnt
    centers = 0.5 * (np.array(ERR_EDGES[:-1]) + np.array(ERR_EDGES[1:]))
    sel = np.abs(centers) <= 90
    amp = float(np.abs(mean_dh[sel]).mean())
    return dict(p=task["res"]["p_link"], seed=task["seed"], amp=amp,
                f=r["prop_spiked"], w=r["w_mean_final"], score=r["score_late"])

def opened(task):
    r = run_open_loop(task)
    return dict(p=task["res"]["p_link"], wlr=task["res"]["weight_lr"],
                seed=task["seed"], gain=r.get("recon_gain", 0.0),
                f=r["f_mean_late"])

def main():
    ctasks = [dict(res={"weight_lr": 0.3, "target_lr": 0.01, "p_link": p,
                        "input_p_link": 0.1}, seed=s, pin_output_p=0.1)
              for p in (0.1, 0.8) for s in range(8)]
    otasks = [dict(res={"weight_lr": w, "target_lr": 0.01, "p_link": p,
                        "input_p_link": 0.1}, seed=s, n_steps=4000, recon=True,
                   schedule={"kind": "sine", "amp": 20.0, "period": 120})
              for p in (0.02, 0.1, 0.4) for w in (0.03, 0.1) for s in range(8)]
    with ProcessPoolExecutor(10) as pool:
        crows = list(pool.map(closed, ctasks, chunksize=2))
        orows = list(pool.map(opened, otasks, chunksize=2))
    (LAB / "h64_policy.json").write_text(json.dumps(crows))
    (LAB / "h65_infospike.json").write_text(json.dumps(orows))
    print("H64 policy amplitude (wlr=0.3):")
    for p in (0.1, 0.8):
        sel = [r for r in crows if r["p"] == p]
        print(f"  p={p}: amp {np.mean([r['amp'] for r in sel]):.4f}"
              f"  f {np.mean([r['f'] for r in sel]):.3f}"
              f"  wpN {np.mean([r['w'] for r in sel]) * p * 200:.2f}"
              f"  score {np.mean([r['score'] for r in sel]):.3f}")
    print("H65 gain/f (info per spike proxy):")
    for w in (0.03, 0.1):
        line = f"  wlr={w}: "
        for p in (0.02, 0.1, 0.4):
            sel = [r for r in orows if r["p"] == p and r["wlr"] == w]
            g = np.mean([r["gain"] for r in sel])
            f = np.mean([r["f"] for r in sel])
            line += f"p={p}: {g:.3f}/{f:.3f}={g / max(f, 1e-6):5.2f}   "
        print(line)

if __name__ == "__main__":
    main()
