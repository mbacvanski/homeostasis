"""H98 analysis: formation/destruction rates and the predicted crest."""
from __future__ import annotations
import json
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np

LAB = Path(__file__).resolve().parents[1] / "out" / "lab"
THR = 0.5

def rates(seg_lists):
    f_n = f_d = d_n = d_d = 0
    for ss in seg_lists:
        s = np.array(ss[3:])  # skip warmup segments
        lo = s[:-1] < THR
        hi = ~lo
        up = s[1:] >= THR
        f_n += int(np.sum(lo & up)); f_d += int(np.sum(lo))
        d_n += int(np.sum(hi & ~up)); d_d += int(np.sum(hi))
    p_f = f_n / max(f_d, 1)
    p_d = d_n / max(d_d, 1)
    return p_f, p_d

def main():
    rows = [json.loads(l) for l in open(LAB / "ridge98_results.jsonl")]
    cells = defaultdict(list)
    for r in rows:
        cells[r["_cell"]].append(r["seg_scores"])
    by_leak = defaultdict(dict)
    for cell, segs in cells.items():
        lk, w = cell[1:].split("_w")
        by_leak[float(lk)][float(w)] = segs
    print("leak   wlr    p_form  p_destroy  pred locked  measured score")
    for lk in sorted(by_leak):
        best_pred, best_meas = None, None
        for w in sorted(by_leak[lk]):
            segs = by_leak[lk][w]
            p_f, p_d = rates(segs)
            locked = p_f / max(p_f + p_d, 1e-9)
            meas = np.mean([np.mean(s[5:]) for s in segs])
            print(f"{lk:<6} {w:<6} {p_f:.3f}   {p_d:.3f}      {locked:.3f}        {meas:.3f}")
            if best_pred is None or locked > best_pred[1]:
                best_pred = (w, locked)
            if best_meas is None or meas > best_meas[1]:
                best_meas = (w, meas)
        ratio = best_pred[0] / best_meas[0]
        print(f"  -> predicted crest wlr {best_pred[0]}, measured {best_meas[0]}"
              f"  (ratio {ratio:.2f} {'OK' if 1/1.5 <= ratio <= 1.5 else 'MISS'})")

if __name__ == "__main__":
    main()
