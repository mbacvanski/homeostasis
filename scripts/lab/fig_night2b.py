"""Composite for the second-wind arcs: band, T-liability, bits/spike."""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LAB = Path(__file__).resolve().parents[1] / "out" / "lab"
fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))

# (a) speed-band psychometric
band = json.load(open(LAB / "h75_band.json"))
xs = sorted(float(k) for k in band)
ys = [band[str(x)] if str(x) in band else band[f"{x}"] for x in xs]
ax = axes[0]
ax.plot(xs, ys, "o-", c="tab:purple")
ax.axhline(0.6, ls=":", c="gray", lw=0.8)
ax.axvspan(0.10, 0.26, color="tab:purple", alpha=0.08)
for v, lbl in ((0.086, "link D"), (0.255, "pacemaker")):
    ax.axvline(v, ls="--", lw=0.8, c="k")
    ax.text(v, 0.06, lbl, rotation=90, fontsize=7, va="bottom", ha="right")
ax.set_xlabel("target speed (units/step)"); ax.set_ylabel("best evolvable near4")
ax.set_title("(a) the followable speed band\n(plateau 0.10–0.26, H75)")

# (b) T-window bars (48-seed cluster)
rows = [json.loads(l) for l in open(LAB / "twin48_results.jsonl")]
cells = {}
for r in rows:
    cells.setdefault(r["_cell"], []).append(r["seg_scores"])
ax = axes[1]
names = [("full", "target adaptation on"), ("freezeT3600", "T frozen at 3600"),
         ("freezeT0", "T frozen at birth")]
for i, (c, lbl) in enumerate(names):
    E = np.mean([np.mean(v[5:10]) for v in cells[c]])
    L = np.mean([np.mean(v[25:30]) for v in cells[c]])
    ax.bar(i - 0.18, E, width=0.34, color="#b8c4d8")
    ax.bar(i + 0.18, L, width=0.34, color="#3d5a80")
ax.set_xticks(range(3)); ax.set_xticklabels([lbl for _, lbl in names], fontsize=7.5)
ax.bar(0, 0, color="#b8c4d8", label="early (segs 5–9)")
ax.bar(0, 0, color="#3d5a80", label="late (segs 25–29)")
ax.legend(fontsize=8); ax.set_ylabel("tracking score")
ax.set_title("(b) any target adaptation harms\n(48 seeds, t=+3.24, H76)")

# (c) bits per spike
rows = json.load(open(LAB / "h78_bits.json"))
ax = axes[2]
for P, c in ((60, "tab:blue"), (120, "tab:green"), (240, "tab:red")):
    ys = [np.mean([r["eff"] for r in rows if r["p"] == p and r["P"] == P]) * 1e3
          for p in (0.02, 0.1, 0.4)]
    ax.semilogx([0.02, 0.1, 0.4], ys, "o-", color=c, label=f"period {P}")
ax.set_xticks([0.02, 0.1, 0.4]); ax.set_xticklabels(["0.02", "0.1", "0.4"])
ax.tick_params(which="minor", bottom=False, labelbottom=False)
ax.set_xlabel("recurrent p_link"); ax.set_ylabel("mbits per spike")
ax.set_title("(c) density buys information efficiency\n(~11× bits/spike, H78)")
ax.legend(fontsize=8)

fig.tight_layout()
fig.savefig(LAB / "fig_night2b.png", dpi=150)
print("saved")
