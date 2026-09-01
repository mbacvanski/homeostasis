"""H53: does homeostatic plasticity repair competence after mid-run node death?"""
from __future__ import annotations
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from common import run_closed_loop

LAB = Path(__file__).resolve().parents[1] / "out" / "lab"
RES = {"weight_lr": 0.1, "target_lr": 0.01}

def run(t):
    r = run_closed_loop(t)
    ss = r["seg_scores"]
    return dict(arm=t["arm"], k=t.get("kill_frac", 0.0), seed=t["seed"],
                pre=float(np.mean(ss[5:10])), drop=float(np.mean(ss[10:12])),
                rec=float(np.mean(ss[15:20])), segs=ss,
                f_win=r["snaps"]["f_win"], T=r["snaps"]["T_mean"],
                w=r["snaps"]["w_mean"])

def main():
    tasks = [dict(res=RES, seed=s, n_steps=14400, arm=arm, kill_frac=k)
             for k in (0.1, 0.3, 0.5)
             for arm in ("kill-mid", "kill-mid-frozen") for s in range(16)]
    tasks += [dict(res=RES, seed=s, n_steps=14400, arm="full") for s in range(16)]
    with ProcessPoolExecutor(10) as pool:
        rows = list(pool.map(run, tasks, chunksize=2))
    (LAB / "h53_selfrepair.json").write_text(json.dumps(rows))
    base = [r for r in rows if r["arm"] == "full"]
    print(f"no-kill baseline: pre {np.mean([r['pre'] for r in base]):.3f}"
          f" late {np.mean([r['rec'] for r in base]):.3f}")
    print("k    arm              pre -> drop -> recovered   (rec/pre)")
    for k in (0.1, 0.3, 0.5):
        for arm in ("kill-mid", "kill-mid-frozen"):
            sel = [r for r in rows if r["arm"] == arm and r["k"] == k]
            pre = np.mean([r["pre"] for r in sel])
            dr = np.mean([r["drop"] for r in sel])
            rec = np.mean([r["rec"] for r in sel])
            rat = np.mean([r["rec"] / max(r["pre"], 1e-9) for r in sel])
            print(f"{k:.1f}  {arm:<16} {pre:.3f} -> {dr:.3f} -> {rec:.3f}   ({rat:.2f})")

if __name__ == "__main__":
    main()
