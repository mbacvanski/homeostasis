"""K1b: does the single-node duty law survive inside the full network?

Fits f_i,w = clip((mu_i,w / T_i,w - leak) / rho, 0, 1) per (node, 120-step
window) on the per-node data recorded by k1_openloop.py (wlr=1.0, tlr=0.01,
slip1 & jump, seeds 0-1) — total drive mu includes the recurrent term, so
this asks whether the churn regime still satisfies the balance law on window
averages, with no free parameters.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

LAB = Path(__file__).resolve().parents[1] / "out" / "lab"


def main():
    rows = json.loads((LAB / "k1_law_data.json").read_text())
    print(f"K1b duty-law fit on {len(rows)} per-node runs\n")
    for r in rows:
        law = r["law"]
        drive = np.array(law["drive"])   # (n_win, N) window-mean total drive
        f = np.array(law["f"])           # window-mean spike rate
        T = np.array(law["T"])
        leak, rho = law["leak"], law["rho"]
        pred = np.clip((drive / np.maximum(T, 1e-9) - leak) / rho, 0.0, 1.0)
        # drop the first two windows (transient)
        P, F = pred[2:].ravel(), f[2:].ravel()
        resid = F - P
        # correlation and calibration
        rho_s = np.corrcoef(P, F)[0, 1]
        print(f"── {r['schedule']['kind']:6s} seed {r['seed']}: "
              f"n={len(F)}  corr {rho_s:+.3f}  median|resid| {np.median(np.abs(resid)):.4f}  "
              f"bias {resid.mean():+.4f}  f range {F.min():.2f}-{F.max():.2f}")
        # binned calibration curve
        bins = np.linspace(0, 1, 11)
        digit = np.digitize(P, bins) - 1
        cal = ["   pred-bin → mean actual: "]
        for b in range(10):
            m = digit == b
            if m.sum() >= 20:
                cal.append(f"{bins[b]:.1f}:{F[m].mean():.2f}")
        print("  " + " ".join(cal))


if __name__ == "__main__":
    main()
