"""H54: recurrent sparsity via the drive-variance channel."""
from __future__ import annotations
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from common import run_closed_loop

LAB = Path(__file__).resolve().parents[1] / "out" / "lab"
PS = (0.02, 0.05, 0.1, 0.2, 0.4, 0.8)

def run(t):
    r = run_closed_loop(t)
    return dict(p=t["res"]["p_link"], wlr=t["res"]["weight_lr"], seed=t["seed"],
                pinned=bool(t.get("pin_output_p")), score=r["score_late"],
                f=r["prop_spiked"], w=r["w_mean_final"], g=r["g_final"],
                duty=r["input_duty"], T=r["T_final"])

def main():
    tasks = [dict(res={"weight_lr": w, "target_lr": 0.01, "p_link": p,
                       "input_p_link": 0.1}, seed=s, pin_output_p=0.1)
             for w in (0.1, 0.03) for p in PS for s in range(16)]
    tasks += [dict(res={"weight_lr": 0.1, "target_lr": 0.01, "p_link": p,
                        "input_p_link": 0.1}, seed=s)
              for p in (0.02, 0.8) for s in range(16)]
    with ProcessPoolExecutor(10) as pool:
        rows = list(pool.map(run, tasks, chunksize=4))
    (LAB / "h54_sparsity.json").write_text(json.dumps(rows))
    N = 200
    for w in (0.1, 0.03):
        print(f"wlr={w} (output pinned): p -> score(frac>=.35) | f | w*p*N | g | duty")
        for p in PS:
            sel = [r for r in rows if r["pinned"] and r["wlr"] == w and r["p"] == p]
            sc = [r["score"] for r in sel]
            print(f"  p={p:<4} {np.mean(sc):.3f}({np.mean([s>=.35 for s in sc]):.2f})"
                  f" | {np.mean([r['f'] for r in sel]):.3f}"
                  f" | {np.mean([r['w'] for r in sel])*p*N:6.2f}"
                  f" | {np.mean([r['g'] for r in sel]):5.2f}"
                  f" | {np.mean([r['duty'] for r in sel]):.2f}")
    print("unpinned output (wlr=0.1):")
    for p in (0.02, 0.8):
        sel = [r for r in rows if not r["pinned"] and r["p"] == p]
        print(f"  p={p}: score {np.mean([r['score'] for r in sel]):.3f}")

if __name__ == "__main__":
    main()
