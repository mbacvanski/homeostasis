"""H54b: optimal wlr increases with wiring density."""
from __future__ import annotations
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from common import run_closed_loop

LAB = Path(__file__).resolve().parents[1] / "out" / "lab"

def run(t):
    r = run_closed_loop(t)
    return dict(p=t["res"]["p_link"], wlr=t["res"]["weight_lr"], seed=t["seed"],
                score=r["score_late"], f=r["prop_spiked"], w=r["w_mean_final"])

def main():
    tasks = [dict(res={"weight_lr": w, "target_lr": 0.01, "p_link": p,
                       "input_p_link": 0.1}, seed=s, pin_output_p=0.1)
             for p in (0.02, 0.8) for w in (0.3, 1.0) for s in range(16)]
    with ProcessPoolExecutor(10) as pool:
        rows = list(pool.map(run, tasks, chunksize=4))
    (LAB / "h54b_matched.json").write_text(json.dumps(rows))
    old = json.load(open(LAB / "h54_sparsity.json"))
    for p in (0.02, 0.8):
        line = f"p={p}: "
        for w in (0.03, 0.1, 0.3, 1.0):
            sel = [r for r in rows if r["p"] == p and r["wlr"] == w]
            if not sel:
                sel = [r for r in old if r.get("pinned") and r["p"] == p and r["wlr"] == w]
            sc = [r["score"] for r in sel]
            line += f"wlr={w}: {np.mean(sc):.3f}({np.mean([x>=.35 for x in sc]):.2f})  "
        print(line)

if __name__ == "__main__":
    main()
