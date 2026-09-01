"""H45 analysis: the ridge across N (cluster3 results)."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

LAB = Path(__file__).resolve().parents[1] / "out" / "lab"

def main():
    rows = json.loads((LAB / "cluster3_results.json").read_text())
    print(f"{len(rows)} rows")
    print("score_late (24 seeds; frac>=0.35):")
    print("          wlr=0.05        wlr=0.1         wlr=0.2")
    for n in (200, 500, 1000, 2000):
        cells = []
        for w in (0.05, 0.1, 0.2):
            v = np.array([r["score_late"] for r in rows if r["_n"] == n and r["_wlr"] == w])
            cells.append(f"{v.mean():.3f} ({np.mean(v>=0.35):.2f}) n={len(v)}" if len(v) else "--")
        print(f"   N={n:<5} " + "  ".join(f"{c:>18s}" for c in cells))
    print("\nprop_spiked and |E| at the wlr=0.1 column:")
    for n in (200, 500, 1000, 2000):
        sel = [r for r in rows if r["_n"] == n and r["_wlr"] == 0.1]
        if sel:
            print(f"   N={n:<5} f {np.mean([r['prop_spiked'] for r in sel]):.3f}  "
                  f"|E| {np.mean([r['mean_abs_E'] for r in sel]):.3f}  "
                  f"flow {np.mean([r['input_flow'] for r in sel]):.2f}")

if __name__ == "__main__":
    main()
