"""Analyze cluster batch 1 (48-seed fine ridge R1, A1 replication R2, N-line R3).

Expects scripts/out/lab/cluster1_results.json (list of run rows, harvested
from the cluster's lab_out/*.jsonl). H14: fit wlr* = c * leak^b on the
non-held cells (checkerboard _held flag), validate on held cells.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

LAB = Path(__file__).resolve().parents[1] / "out" / "lab"


def parab_peak(ws, ss):
    """Argmax over log-wlr with parabolic refinement."""
    i = int(np.argmax(ss))
    if i in (0, len(ws) - 1):
        return ws[i]
    x = np.log([ws[i - 1], ws[i], ws[i + 1]])
    y = np.array([ss[i - 1], ss[i], ss[i + 1]])
    denom = (x[0] - x[1]) * (x[0] - x[2]) * (x[1] - x[2])
    a = (x[2] * (y[1] - y[0]) + x[1] * (y[0] - y[2]) + x[0] * (y[2] - y[1])) / denom
    b = (x[2]**2 * (y[0] - y[1]) + x[1]**2 * (y[2] - y[0]) + x[0]**2 * (y[1] - y[2])) / denom
    if a >= 0:
        return ws[i]
    return float(np.exp(np.clip(-b / (2 * a), x[0], x[2])))


def main():
    rows = json.loads((LAB / "cluster1_results.json").read_text())
    r1 = [r for r in rows if r.get("_tag") == "R1"]
    r2 = [r for r in rows if r.get("_tag") == "R2"]
    r3 = [r for r in rows if r.get("_tag") == "R3"]
    print(f"rows: R1 {len(r1)}  R2 {len(r2)}  R3 {len(r3)}")

    leaks = sorted({r["_leak"] for r in r1})
    wlrs = sorted({r["_wlr"] for r in r1})

    def cell(leak, wlr):
        v = [r["score_late"] for r in r1
             if r["_leak"] == leak and r["_wlr"] == wlr]
        held = any(r["_held"] for r in r1 if r["_leak"] == leak and r["_wlr"] == wlr)
        return (np.mean(v) if v else np.nan, len(v), held)

    print("\n══ R1 fine ridge (48 seeds/cell; * = held-out):")
    print("        " + "  ".join(f"w={w:<5}" for w in wlrs))
    peaks_fit, peaks_held = [], []
    for lk in leaks:
        vals, helds = [], []
        for w in wlrs:
            m, n, held = cell(lk, w)
            vals.append(m)
            helds.append(held)
        print(f"  leak={lk:<5}" + "  ".join(
            f"{v:.2f}{'*' if h else ' '}" for v, h in zip(vals, helds)))
        # peak using only NON-held cells for the fit set; all cells for held eval
        fit_ws = [w for w, h in zip(wlrs, helds) if not h]
        fit_vs = [v for v, h in zip(vals, helds) if not h]
        if max(vals) >= 0.35:
            peaks_held.append((lk, parab_peak(wlrs, vals)))
            if len(fit_ws) >= 3 and max(fit_vs) >= 0.35:
                peaks_fit.append((lk, parab_peak(fit_ws, fit_vs)))

    if len(peaks_fit) >= 3:
        L = np.log([p[0] for p in peaks_fit])
        W = np.log([p[1] for p in peaks_fit])
        b, logc = np.polyfit(L, W, 1)
        c = float(np.exp(logc))
        print(f"\n  H14 fit on non-held cells: wlr* = {c:.3f} · leak^{b:.2f}")
        print("  held-out check (peak from ALL cells vs law prediction):")
        for lk, pk in peaks_held:
            pred = c * lk ** b
            ratio = pk / pred
            print(f"    leak={lk:<5} peak {pk:.3f}  pred {pred:.3f}  ratio {ratio:.2f}")

    print("\n══ R2: A1 plane at 48 seeds (score / frac≥0.35):")
    for wlr in sorted({r["_wlr"] for r in r2}):
        cells = []
        for tlr in (0.001, 0.01, 0.1):
            v = np.array([r["score_late"] for r in r2
                          if r["_wlr"] == wlr and r["_tlr"] == tlr])
            cells.append(f"{v.mean():.3f}({np.mean(v >= 0.35):.2f})" if len(v) else "--")
        print(f"   wlr={wlr:<5} " + "  ".join(f"{c:>12s}" for c in cells))

    print("\n══ R3: N-line at fixed p=0.1 (the confound demo; 24 seeds):")
    for n in sorted({r["_n"] for r in r3}):
        line = f"   N={n:<4}"
        for wlr in (0.05, 0.1, 0.2):
            v = np.array([r["score_late"] for r in r3
                          if r["_n"] == n and r["_wlr"] == wlr])
            line += f"  wlr{wlr}: {v.mean():.3f}" if len(v) else ""
        print(line)


if __name__ == "__main__":
    main()
