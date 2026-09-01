"""Campaign figures -> scripts/out/lab/fig_*.png (all from committed JSONs)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

LAB = Path(__file__).resolve().parents[1] / "out" / "lab"


def load(name):
    return json.loads((LAB / name).read_text())


def heat(ax, rows, xs, ys, xk, yk, title, xlab, ylab):
    M = np.full((len(ys), len(xs)), np.nan)
    F = np.full((len(ys), len(xs)), np.nan)
    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            v = [r["score_late"] for r in rows
                 if abs(r.get(xk, 9e9) - x) < 1e-9 and abs(r.get(yk, 9e9) - y) < 1e-9]
            if v:
                M[i, j] = np.mean(v)
                F[i, j] = np.mean(np.array(v) >= 0.35)
    im = ax.imshow(M, origin="lower", aspect="auto", cmap="viridis", vmin=0.25, vmax=0.7)
    for i in range(len(ys)):
        for j in range(len(xs)):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i,j]:.2f}\n{F[i,j]:.0%}", ha="center", va="center",
                        fontsize=7, color="w" if M[i, j] < 0.5 else "k")
    ax.set_xticks(range(len(xs)), [str(x) for x in xs])
    ax.set_yticks(range(len(ys)), [str(y) for y in ys])
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_title(title)
    return im


def main():
    b1 = load("act2_batch1.json")
    b2 = load("act2_batch2.json")
    b6b = load("b6b_reentrainment.json")
    k2 = load("k2_single_node.json")
    b3rows = [r for r in b2 if r.get("tag") == "B3"]

    # fig 1: the two planes
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    a1 = [r for r in b1 if r.get("tag") == "A1"]
    im = heat(axes[0], a1, [0.001, 0.01, 0.1], [0.0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0],
              "tlr", "wlr", "A1: channel competition (score, 12 seeds)",
              "target_lr", "weight_lr")
    a3 = [r for r in b1 if r.get("tag") == "A3"]
    heat(axes[1], a3, [0.03, 0.1, 0.3, 1.0], [0.05, 0.1, 0.25, 0.5, 0.75, 0.9],
         "wlr", "leak", "A3: the matched-timescale ridge", "weight_lr", "leak")
    fig.colorbar(im, ax=axes, shrink=0.8, label="score (segments 6-10)")
    fig.suptitle("Cell text: mean score / fraction of seeds ≥ 0.35 · paper default = (wlr 1.0, tlr 0.01)")
    fig.savefig(LAB / "fig_planes.png", dpi=140, bbox_inches="tight")

    # fig 2: transfer function
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    periods = sorted({r["period"] for r in b3rows})
    for wlr, c in zip((0.03, 0.1, 0.3, 1.0), ("#888", "#0E6E63", "#c98f00", "#a33")):
        g = [np.mean([r["recon_gain"] for r in b3rows
                      if r["wlr"] == wlr and r["period"] == p]) for p in periods]
        ax.plot(periods, g, "o-", color=c, label=f"wlr={wlr}")
    ax.set_xscale("log")
    ax.set_xlabel("slip period P (steps)")
    ax.set_ylabel("stimulus-position gain in spikes")
    ax.set_title("The ridge is a signal-to-noise optimum\n(spike-readout reconstruction of a 20° sinusoidal slip)")
    ax.axvline(420, ls=":", c="k", lw=1)
    ax.text(430, 0.21, "P* = 2π/(wlr·f̄)\npredicted for wlr=0.1", fontsize=8)
    ax.legend()
    fig.tight_layout()
    fig.savefig(LAB / "fig_transfer.png", dpi=140)

    # fig 3: re-entrainment
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for name, c in (("wlr0.03", "#888"), ("wlr0.1", "#0E6E63"), ("wlr0.3", "#c98f00"),
                    ("wlr1.0", "#a33"), ("w1prime", "#333")):
        cur = np.mean([r["curve"] for r in b6b if r["name"] == name], axis=0)
        t = (np.arange(len(cur)) + 0.5) * 30
        ax.plot(t, cur, "-", color=c, lw=2 if name in ("w1prime", "wlr0.1") else 1.2,
                label=name)
    ax.axhline(1.0, ls=":", c="k", lw=1)
    ax.axhline(0.0, ls="-", c="k", lw=0.5)
    ax.set_xlabel("steps after stimulus reversal")
    ax.set_ylabel("mean dH toward NEW direction (deg/step)")
    ax.set_title("Re-entrainment after reversal: velocity re-locking\n(1.0 = perfect follow; note the negative momentum lobe at high wlr)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(LAB / "fig_reentrainment.png", dpi=140)

    # fig 4: single-node phase map (cold start, rho=2)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    states = ["dead-floor", "silent-comf", "spiking", "frozen-cycle"]
    cmap = ListedColormap(["#777", "#4a90d9", "#2f9e44", "#e8890c"])
    for ax, hot in zip(axes, (False, True)):
        sel = [r for r in k2 if r["rho"] == 2.0 and r["hot"] == hot]
        mus = sorted({r["mu"] for r in sel})
        leaks = sorted({r["leak"] for r in sel})
        M = np.full((len(leaks), len(mus)), np.nan)
        for r in sel:
            M[leaks.index(r["leak"]), mus.index(r["mu"])] = states.index(r["state"])
        ax.imshow(M, origin="lower", aspect="auto", cmap=cmap, vmin=-0.5, vmax=3.5)
        ax.set_xticks(range(0, len(mus), 4), [f"{m:.2f}" for m in mus[::4]], fontsize=7)
        ax.set_yticks(range(len(leaks)), [str(v) for v in leaks])
        ax.set_xlabel("drive μ")
        ax.set_title(("hot start" if hot else "cold start") + " (ρ=2)")
        # law lines: mu = leak (comfort split), mu = 2*leak (cold spike boundary)
        for k, ls, lab in ((1.0, ":", "μ=leak·T_floor"), (2.0, "--", "μ=ρ·leak·T₀")):
            xs = [np.searchsorted(mus, k * l) - 0.5 for l in leaks]
            ax.plot(xs, range(len(leaks)), ls, c="k", lw=1.2, label=lab)
    axes[0].set_ylabel("leak")
    axes[0].legend(fontsize=7, loc="lower right")
    handles = [plt.Rectangle((0, 0), 1, 1, color=cmap(i)) for i in range(4)]
    axes[1].legend(handles, states, fontsize=7, loc="lower right")
    fig.suptitle("Single-node phase map: the four end states and the two analytic boundaries")
    fig.tight_layout()
    fig.savefig(LAB / "fig_single_node_phases.png", dpi=140)

    print("wrote fig_planes, fig_transfer, fig_reentrainment, fig_single_node_phases")


if __name__ == "__main__":
    main()
