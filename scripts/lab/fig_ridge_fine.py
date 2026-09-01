"""Fine-ridge figure from cluster1 results, with the fitted law overlaid."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LAB = Path(__file__).resolve().parents[1] / "out" / "lab"
rows = [r for r in json.loads((LAB / "cluster1_results.json").read_text())
        if r.get("_tag") == "R1"]
leaks = sorted({r["_leak"] for r in rows})
wlrs = sorted({r["_wlr"] for r in rows})
M = np.full((len(leaks), len(wlrs)), np.nan)
for i, lk in enumerate(leaks):
    for j, w in enumerate(wlrs):
        v = [r["score_late"] for r in rows if r["_leak"] == lk and r["_wlr"] == w]
        if v:
            M[i, j] = np.mean(v)
fig, ax = plt.subplots(figsize=(9, 5.6))
im = ax.imshow(M, origin="lower", aspect="auto", cmap="viridis", vmin=0.25, vmax=0.56)
ax.set_xticks(range(len(wlrs)), [str(w) for w in wlrs], fontsize=8)
ax.set_yticks(range(len(leaks)), [str(l) for l in leaks], fontsize=8)
ax.set_xlabel("weight_lr")
ax.set_ylabel("leak")
# law overlay: wlr* = 1.04 * leak^1.41 in index coordinates (log-spaced grids)
lw = np.log(wlrs); ll = np.log(leaks)
xs = []
for lk in leaks:
    w_star = 1.04 * lk ** 1.41
    xs.append(np.interp(np.log(w_star), lw, np.arange(len(wlrs))))
ax.plot(xs, range(len(leaks)), "w--", lw=2, label="wlr* = 1.04·leak$^{1.41}$ (fit on 75% of cells)")
for i in range(len(leaks)):
    for j in range(len(wlrs)):
        if not np.isnan(M[i, j]):
            ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", fontsize=6.5,
                    color="w" if M[i, j] < 0.45 else "k")
ax.legend(loc="upper left", fontsize=9)
ax.set_title("The matched-timescale ridge at 48 seeds/cell (cluster; 4800 runs)\n"
             "0.25 = dead statue; crest broad at low leak, narrow at high leak")
fig.colorbar(im, label="score (segments 6-10)")
fig.tight_layout()
fig.savefig(LAB / "fig_ridge_fine.png", dpi=140)
print("wrote fig_ridge_fine.png")
