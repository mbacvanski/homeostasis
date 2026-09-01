"""K0: does initial recurrent gain organize the existing 241-config sweep?

Free reanalysis (zero new runs) of scripts/out/sweep/results.json and
scripts/out/metric_screen.json. The mean-field skeleton predicts an initial
recurrent gain

    g_init = n_nodes * p_link * weight_init_mean / (threshold_ratio * target_init)

(target_init = 1.0 throughout the sweep) should organize spiking level and the
dead / entrained / autarkic / saturated classes, with score structured (possibly
nonmonotone) in g. Also computes the input-drive proxy per node,
mu_in = n_sensors * p_link * input_weight (per unit sensor activation), tests the
dead-zone prediction (dead iff drive small relative to leak), and re-examines the
input_flow~score correlation controlling for prop_spiked (is flow more than
"not dead"?).

Gate A: if g_init shows no organization, the g-centric plan reframes as a search
for the true collapse coordinate.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parents[1] / "out"
LAB = OUT / "lab"
N_SENSORS = 62


def spearman(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = np.sqrt((ra**2).sum() * (rb**2).sum())
    return float((ra * rb).sum() / denom) if denom else 0.0


def partial_spearman(x, y, z):
    """Spearman(x, y) after rank-regressing z out of both."""
    def resid(v, c):
        r = np.argsort(np.argsort(v)).astype(float)
        rc = np.argsort(np.argsort(c)).astype(float)
        rc = (rc - rc.mean()) / (rc.std() or 1.0)
        beta = np.dot(r - r.mean(), rc) / len(r)
        return (r - r.mean()) - beta * rc

    rx, ry = resid(x, z), resid(y, z)
    denom = np.sqrt((rx**2).sum() * (ry**2).sum())
    return float((rx * ry).sum() / denom) if denom else 0.0


def g_init(cfg):
    return cfg["n_nodes"] * cfg["p_link"] * cfg["weight_init_mean"] / cfg["threshold_ratio"]


def mu_in(cfg):
    return N_SENSORS * cfg["p_link"] * cfg["input_weight"]


def main():
    sweep = json.loads((OUT / "sweep" / "results.json").read_text())
    screen = json.loads((OUT / "metric_screen.json").read_text())

    configs = sweep["configs"]
    rows = []
    for cid_str, agg in sweep["search"].items():
        cid = int(cid_str)
        cfg = configs[cid]
        rows.append(
            dict(
                cid=cid,
                g=g_init(cfg),
                mu=mu_in(cfg),
                score=agg["score"],
                spiked=agg["prop_spiked"],
                dir_agree=agg["dir_agree"],
                **{k: cfg[k] for k in cfg},
            )
        )
    R = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    n = len(rows)

    print(f"K0 reanalysis over {n} sweep configs (search aggregates, 6 seeds each)")
    print()

    # 1. Does g_init organize anything?
    print("── Spearman vs g_init:")
    for target in ("score", "spiked", "dir_agree"):
        print(f"   {target:10s} {spearman(R['g'], R[target]):+.3f}")
    print("── Spearman vs mu_in (input-drive proxy):")
    for target in ("score", "spiked"):
        print(f"   {target:10s} {spearman(R['mu'], R[target]):+.3f}")

    # deciles of g
    order = np.argsort(R["g"])
    print("\n── g_init deciles (median per bin):")
    print("   decile      g range        score  spiked  n_work")
    for d in range(10):
        idx = order[d * n // 10 : (d + 1) * n // 10]
        g_lo, g_hi = R["g"][idx].min(), R["g"][idx].max()
        works = int((R["score"][idx] >= 0.35).sum())
        print(
            f"   {d}   {g_lo:8.2f}-{g_hi:8.2f}   "
            f"{np.median(R['score'][idx]):.3f}   {np.median(R['spiked'][idx]):.3f}   {works}/{len(idx)}"
        )

    # 2. Class geography
    dead = R["spiked"] < 0.02
    saturated = R["spiked"] > 0.95
    working = (R["score"] >= 0.35) & ~dead & ~saturated
    other = ~(dead | saturated | working)
    print("\n── class geography (g_init quartiles [q25, median, q75]):")
    for name, mask in [("dead", dead), ("saturated", saturated), ("working", working), ("other", other)]:
        if mask.sum() == 0:
            print(f"   {name:10s} n=0")
            continue
        q = np.percentile(R["g"][mask], [25, 50, 75])
        qm = np.percentile(R["mu"][mask], [25, 50, 75])
        print(
            f"   {name:10s} n={int(mask.sum()):3d}  g=[{q[0]:7.2f} {q[1]:7.2f} {q[2]:7.2f}]"
            f"  mu_in=[{qm[0]:6.2f} {qm[1]:6.2f} {qm[2]:6.2f}]  "
            f"score_med={np.median(R['score'][mask]):.3f}"
        )

    # 3. Dead-zone prediction: dead iff drive can't reach threshold.
    #    Crude criterion: mu_in * s_bar / leak < threshold_ratio * target_init.
    #    We don't know s_bar (mean sensor activation); report the empirical
    #    separating value of mu_in/(leak*threshold_ratio) instead.
    ratio = R["mu"] / (R["leak"] * R["threshold_ratio"])
    if dead.sum() and (~dead).sum():
        print("\n── dead-zone coordinate mu_in/(leak*rho):")
        print(f"   dead:  q10-q90 = {np.percentile(ratio[dead], 10):.2f} .. {np.percentile(ratio[dead], 90):.2f}  (n={int(dead.sum())})")
        print(f"   alive: q10-q90 = {np.percentile(ratio[~dead], 10):.2f} .. {np.percentile(ratio[~dead], 90):.2f}")
        # best single threshold separating dead from alive on this coordinate
        thr_grid = np.percentile(ratio, np.linspace(1, 99, 197))
        acc = [((ratio < t) == dead).mean() for t in thr_grid]
        best = int(np.argmax(acc))
        print(f"   best split at {thr_grid[best]:.2f}: accuracy {acc[best]:.3f}")

    # 4. input_flow vs score controlling prop_spiked (screen rows, sweep kind only)
    srows = [r for r in screen if r.get("name", "").startswith("sweep")]
    flow = np.array([r["input_flow"] for r in srows])
    sscore = np.array([r["score"] for r in srows])
    sspiked = np.array([r["prop_spiked"] for r in srows])
    print(f"\n── metric_screen ({len(srows)} sweep rows):")
    print(f"   Spearman(input_flow, score)                    {spearman(flow, sscore):+.3f}")
    print(f"   partial, controlling prop_spiked               {partial_spearman(flow, sscore, sspiked):+.3f}")
    print(f"   Spearman(prop_spiked, score)                   {spearman(sspiked, sscore):+.3f}")

    # 5. every param vs score, for reference
    print("\n── param Spearman vs score (reference):")
    for k in ("n_nodes", "p_link", "input_weight", "weight_init_mean", "weight_init_sd",
              "leak", "target_lr", "threshold_ratio", "gain", "g", "mu"):
        print(f"   {k:18s} {spearman(R[k], R['score']):+.3f}")

    # 6. where do the known cheaters sit?
    cheats = [r for r in screen if r.get("kind") in ("hum", "statue")]
    if cheats:
        print("\n── degenerate champions (from metric_screen params where available): kinds",
              sorted({r["kind"] for r in cheats}), f"n={len(cheats)}")

    LAB.mkdir(exist_ok=True)
    (LAB / "k0.json").write_text(json.dumps(
        dict(g=R["g"].tolist(), mu=R["mu"].tolist(), score=R["score"].tolist(),
             spiked=R["spiked"].tolist(),
             classes=dict(dead=int(dead.sum()), saturated=int(saturated.sum()),
                          working=int(working.sum()), other=int(other.sum()))), indent=1))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, y, lab in [(axes[0], R["score"], "score (within-45)"),
                       (axes[1], R["spiked"], "prop_spiked"),
                       (axes[2], R["dir_agree"], "dir agree")]:
        colors = np.where(dead, "k", np.where(saturated, "crimson", np.where(working, "tab:green", "tab:gray")))
        ax.scatter(R["g"], y, c=colors, s=14, alpha=0.75)
        ax.set_xscale("log")
        ax.set_xlabel("g_init = N·p·w̄ / ρ")
        ax.set_ylabel(lab)
    axes[0].set_title("black=dead, red=saturated, green=working (score≥0.35)")
    fig.tight_layout()
    fig.savefig(LAB / "k0_ginit.png", dpi=130)
    print(f"\nwrote {LAB / 'k0.json'} and k0_ginit.png")


if __name__ == "__main__":
    main()
