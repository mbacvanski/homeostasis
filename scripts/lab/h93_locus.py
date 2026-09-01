"""H93: bias-carrier locus vs leak along the ridge."""
from __future__ import annotations
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from b8_bias_carrier import evaluate

LAB = Path(__file__).resolve().parents[1] / "out" / "lab"

def main():
    cells = [(lk, round(1.04 * lk ** 1.41, 3)) for lk in (0.15, 0.25, 0.4, 0.55, 0.7)]
    tasks = [(f"leak{lk}", {"leak": lk, "weight_lr": w, "target_lr": 0.01}, {}, s)
             for lk, w in cells for s in range(8)]
    with ProcessPoolExecutor(10) as pool:
        rows = list(pool.map(evaluate, tasks, chunksize=2))
    (LAB / "h93_locus.json").write_text(json.dumps(rows))
    purities = []
    for lk, w in cells:
        sel = [r for r in rows if r["name"] == f"leak{lk}"]
        rec = np.mean([abs(r["corr_drec_dir"]) for r in sel])
        inn = np.mean([abs(r["corr_din_dir"]) for r in sel])
        pur = rec / max(rec + inn, 1e-9)
        purities.append(pur)
        print(f"leak {lk} (wlr {w}): score {np.mean([r['score'] for r in sel]):.3f}"
              f" | |corr_rec| {rec:.3f} |corr_in| {inn:.3f} | W-purity {pur:.2f}")
    from scipy.stats import spearmanr
    print(f"rank corr(leak, purity): {spearmanr([c[0] for c in cells], purities).statistic:+.2f}")

if __name__ == "__main__":
    main()
