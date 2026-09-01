"""Composite figure for the night-2 arcs: noise, self-repair, sparsity, horizon."""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LAB = Path(__file__).resolve().parents[1] / "out" / "lab"
fig, axes = plt.subplots(1, 4, figsize=(16, 3.6))

# (a) noise inverted-U per wlr
rows = json.load(open(LAB / "h51_noise.json"))
ax = axes[0]
for wlr, c in ((0.03, "tab:blue"), (0.1, "tab:green"), (1.0, "tab:red")):
    xs, ys = [], []
    for sig in (0.0, 0.1, 0.2):
        sel = [r["score"] for r in rows if r["wlr"] == wlr and r["sig"] == sig]
        xs.append(sig); ys.append(np.mean(sel))
    ax.plot(xs, ys, "o-", color=c, label=f"wlr={wlr}")
ax.set_xlabel("sensor noise σ"); ax.set_ylabel("tracking score")
ax.set_title("(a) noise rescues the under-plastic\n(dark-trap escape, H51)")
ax.legend(fontsize=8); ax.axhline(0.25, ls=":", c="gray", lw=0.8)

# (b) self-repair f trajectories
rows = json.load(open(LAB / "h53_selfrepair.json"))
ax = axes[1]
for arm, c, lbl in (("kill-mid", "tab:green", "learning on"),
                    ("kill-mid-frozen", "tab:red", "frozen at kill")):
    F = np.array([r["f_win"] for r in rows if r["arm"] == arm and abs(r["k"] - 0.3) < 1e-9])
    t = (np.arange(F.shape[1]) + 1) * 240
    ax.plot(t, F.mean(0), color=c, label=lbl)
base = np.array([r["f_win"] for r in rows if r["arm"] == "full"])
ax.plot((np.arange(base.shape[1]) + 1) * 240, base.mean(0), c="gray", lw=0.8, label="no kill")
ax.axvline(7200, ls="--", c="k", lw=0.8)
ax.set_xlabel("step (kill 30% at 7200)"); ax.set_ylabel("spike rate f")
ax.set_title("(b) synaptic scaling repairs f\n(Law 3 as deafferentation response, H53)")
ax.legend(fontsize=8)

# (c) sparsity: score vs p at two wlr + conservation inset
rows = json.load(open(LAB / "h54_sparsity.json"))
ax = axes[2]
PS = (0.02, 0.05, 0.1, 0.2, 0.4, 0.8)
for wlr, c in ((0.1, "tab:green"), (0.03, "tab:blue")):
    ys = [np.mean([r["score"] for r in rows if r.get("pinned") and r["wlr"] == wlr and r["p"] == p])
          for p in PS]
    ax.semilogx(PS, ys, "o-", color=c, label=f"wlr={wlr}")
ax.set_xlabel("recurrent p_link"); ax.set_ylabel("tracking score")
ax.set_title("(c) sparse is pre-adapted\n(best cell p=.02, wlr=.03: 0.566, H54)")
ax.legend(fontsize=8, loc="upper center")
ax.set_xticks(PS); ax.set_xticklabels([str(p) for p in PS], fontsize=8)
ax.tick_params(which="minor", bottom=False, labelbottom=False)
axi = ax.inset_axes([0.56, 0.13, 0.4, 0.3])
wpn = [np.mean([r["w"] for r in rows if r.get("pinned") and r["wlr"] == 0.1 and r["p"] == p]) * p * 200
       for p in PS]
axi.semilogx(PS, wpn, "s-", c="k", ms=3)
axi.set_ylim(0, 3); axi.set_title("Σw_in conserved (wlr=.1)", fontsize=7)
axi.set_xticks([]); axi.tick_params(labelsize=6)
axi.tick_params(which="minor", bottom=False, labelbottom=False)

# (d) horizon: skill gap vs crossing duration
ax = axes[3]
dur = [70, 140, 275]
h55b = json.load(open(LAB / "h55b_horizon.json"))
gaps = [np.mean(h55b[f"h34-champ@{s}"]) - np.mean(h55b[f"blind@{s}"]) for s in (0.15, 0.08, 0.04)]
ax.plot(dur, gaps, "o-", c="tab:purple")
ax.axvspan(90, 225, color="orange", alpha=0.15, label="re-lock horizon (H31)")
ax.axhline(0, c="gray", lw=0.8)
ax.set_xlabel("ballistic crossing duration (steps)")
ax.set_ylabel("catch gap: champion − blind")
ax.set_title("(d) skill appears past the lock horizon\n(third clause, H55b)")
ax.legend(fontsize=8)

fig.tight_layout()
fig.savefig(LAB / "fig_night2.png", dpi=150)
print("saved", LAB / "fig_night2.png")
