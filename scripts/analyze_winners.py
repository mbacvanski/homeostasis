"""Mechanistic analysis of the sweep winners: WHY do they track better?

Compares the paper baseline against the two best sweep configurations
(id 236, the dense/near-saturated regime; id 234, the sparse/near-silent
regime) with four experiment families, all on fresh seeds:

1. attribution — revert each winner parameter to the paper value one at a
   time (necessity), and graft gain / leak / gain+leak onto the baseline
   (sufficiency);
2. ablations — learning frozen, recurrent connectivity lesioned (intrinsic
   target homeostasis intact), and both, for each config;
3. policy curves — mean applied turn dH as a function of the egocentric
   error angle at sensing time, plus R^2 of that mapping (how deterministic
   the sensorimotor loop is);
4. regime stats + speed generalization (stimulus at 1, 2, 3 deg/step).

Reads winner configs from scripts/out/sweep/results.json; writes figures and
a JSON summary to scripts/out/analysis/.

Usage: python scripts/analyze_winners.py [--workers 10]
"""

from __future__ import annotations

import os

for var in ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(var, "1")

import argparse
import json
import pathlib
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from homeostasis import ReservoirConfig, TrackingConfig, TrackingSimulation
from homeostasis.tracking import angular_difference

N_STEPS = 7200
SEEDS = list(range(200, 212))  # fresh seeds, disjoint from search AND eval
BAND = 45.0
ERR_EDGES = np.arange(-180.0, 181.0, 10.0)  # 36 policy bins

RESERVOIR_KEYS = (
    "n_nodes", "p_link", "input_weight", "weight_init_mean",
    "weight_init_sd", "leak", "target_lr", "threshold_ratio",
)

PAPER = {
    "n_nodes": 200, "p_link": 0.1, "input_weight": 0.75,
    "weight_init_mean": 0.75, "weight_init_sd": 0.1, "leak": 0.25,
    "target_lr": 0.01, "threshold_ratio": 2.0, "gain": 10.0,
}


def evaluate(task: dict) -> dict:
    """Run one condition; returns scores, policy bins, and regime stats."""
    cfg = task["cfg"]
    r_cfg = ReservoirConfig(**{k: cfg[k] for k in RESERVOIR_KEYS})
    t_cfg = TrackingConfig(gain=cfg["gain"], stimulus_speed=task.get("speed", 1.0))
    sim = TrackingSimulation(r_cfg, t_cfg, seed=task["seed"])
    net = sim.network
    if task.get("lesion"):
        net.adjacency[:] = False
        net.weights[:] = 0.0
        net._rebuild_structure_caches()
    net.learning_enabled = task.get("learning", True)

    record = task.get("record", False)
    h = sim.run(N_STEPS, record_spikes=record)

    in_band = np.abs(angular_difference(h.stimulus_angle, h.heading)) <= BAND
    kernel = np.ones(50) / 50.0
    smoothed = np.convolve(h.d_heading, kernel, mode="same")
    agree = float(np.mean(np.sign(smoothed[720:]) == h.stimulus_direction[720:]))

    out = {
        "label": task["label"],
        "seed": task["seed"],
        "score": float(in_band.mean()),
        "dir_agree": agree,
        "prop_spiked": float(h.prop_spiked.mean()),
    }

    if record:
        # policy bins: error at sensing time -> applied turn that step
        idx = np.digitize(h.error, ERR_EDGES) - 1
        out["bin_count"] = np.bincount(idx, minlength=36).tolist()
        out["bin_sum"] = np.bincount(idx, weights=h.d_heading, minlength=36).tolist()
        out["bin_sumsq"] = np.bincount(idx, weights=h.d_heading**2, minlength=36).tolist()
        # regime stats: where do spikes live?
        has_input = net.input_adjacency.sum(axis=0) > 0
        total_spikes = h.spikes.sum()
        out["frac_spikes_input_nodes"] = (
            float(h.spikes[:, has_input].sum() / total_spikes) if total_spikes else 0.0
        )
        out["frac_nodes_with_input"] = float(has_input.mean())
        out["mean_abs_effector_diff"] = float(np.abs(h.outputs[:, 0] - h.outputs[:, 1]).mean())
        out["mean_effector_sum"] = float((h.outputs[:, 0] + h.outputs[:, 1]).mean())
    return out


def run_all(tasks, workers):
    results = []
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(evaluate, t) for t in tasks]
        for i, fut in enumerate(as_completed(futures), 1):
            results.append(fut.result())
            if i % 100 == 0 or i == len(tasks):
                rate = i / (time.perf_counter() - t0)
                print(f"  {i}/{len(tasks)} runs ({rate:.0f}/s, ~{(len(tasks) - i) / rate:.0f}s left)")
    by_label: dict[str, list[dict]] = {}
    for r in results:
        by_label.setdefault(r["label"], []).append(r)
    return by_label


def agg_scores(runs):
    s = np.array([r["score"] for r in runs])
    return float(s.mean()), float(s.std())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    parser.add_argument("--out", type=str, default="scripts/out/analysis")
    args = parser.parse_args()
    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    sweep = json.loads(pathlib.Path("scripts/out/sweep/results.json").read_text())
    w1 = sweep["configs"][236]  # dense regime
    w2 = sweep["configs"][234]  # sparse regime
    named = {"base": dict(PAPER), "w1": dict(w1), "w2": dict(w2)}

    tasks: list[dict] = []

    # -- full runs (policy + regime stats recorded) -------------------------
    for label, cfg in named.items():
        tasks += [{"label": label, "cfg": cfg, "seed": s, "record": True} for s in SEEDS]

    # -- attribution: winner -> paper single-parameter reverts --------------
    for wname in ("w1", "w2"):
        for p in PAPER:
            if named[wname][p] == PAPER[p]:
                continue
            cfg = dict(named[wname]); cfg[p] = PAPER[p]
            tasks += [{"label": f"{wname} revert {p}", "cfg": cfg, "seed": s} for s in SEEDS]

    # -- attribution: paper + winner-1 grafts (sufficiency) ------------------
    grafts = {
        "base + gain(w1)": {"gain": w1["gain"]},
        "base + leak(w1)": {"leak": w1["leak"]},
        "base + gain+leak(w1)": {"gain": w1["gain"], "leak": w1["leak"]},
    }
    for label, over in grafts.items():
        cfg = dict(PAPER); cfg.update(over)
        tasks += [{"label": label, "cfg": cfg, "seed": s} for s in SEEDS]

    # -- ablations ----------------------------------------------------------
    for label, cfg in named.items():
        for cond, kw in (
            ("no-learn", {"learning": False}),
            ("lesioned", {"lesion": True}),
            ("lesioned no-learn", {"lesion": True, "learning": False}),
        ):
            tasks += [{"label": f"{label} {cond}", "cfg": cfg, "seed": s, **kw} for s in SEEDS]

    # -- speed generalization ------------------------------------------------
    for label, cfg in named.items():
        for speed in (2.0, 3.0):
            tasks += [{"label": f"{label} speed{speed:.0f}", "cfg": cfg, "seed": s, "speed": speed}
                      for s in SEEDS]

    print(f"{len(tasks)} runs on {args.workers} workers")
    by_label = run_all(tasks, args.workers)

    # ---- report -----------------------------------------------------------
    def show(label):
        m, sd = agg_scores(by_label[label])
        return f"{m:.3f} ± {sd:.3f}"

    print("\n=== full configs (12 fresh seeds) ===")
    for label in named:
        runs = by_label[label]
        print(f"{label:>6}: score {show(label)} | dir "
              f"{np.mean([r['dir_agree'] for r in runs]):.2f} | spike "
              f"{np.mean([r['prop_spiked'] for r in runs]):.3f} | "
              f"spikes on input-nodes {np.mean([r['frac_spikes_input_nodes'] for r in runs]):.2f} "
              f"(structural {np.mean([r['frac_nodes_with_input'] for r in runs]):.2f}) | "
              f"|L-R| {np.mean([r['mean_abs_effector_diff'] for r in runs]):.3f} | "
              f"L+R {np.mean([r['mean_effector_sum'] for r in runs]):.2f}")

    print("\n=== attribution: revert one parameter to paper value ===")
    for wname in ("w1", "w2"):
        full, _ = agg_scores(by_label[wname])
        print(f"{wname} full: {full:.3f}")
        rows = []
        for p in PAPER:
            label = f"{wname} revert {p}"
            if label in by_label:
                m, sd = agg_scores(by_label[label])
                rows.append((m - full, p, m, sd))
        for delta, p, m, sd in sorted(rows):
            print(f"  revert {p:>17} ({named[wname][p]:.3g} -> {PAPER[p]:.3g}): "
                  f"{m:.3f} ± {sd:.3f}  (Δ {delta:+.3f})")

    print("\n=== grafts onto the paper baseline ===")
    for label in grafts:
        print(f"  {label:>22}: {show(label)}")

    print("\n=== ablations (score) ===")
    print(f"{'':>6} {'full':>15} {'no-learn':>15} {'lesioned':>15} {'les+no-learn':>15}")
    for label in named:
        cells = [show(label)]
        for cond in ("no-learn", "lesioned", "lesioned no-learn"):
            cells.append(show(f"{label} {cond}"))
        print(f"{label:>6} " + " ".join(f"{c:>15}" for c in cells))

    print("\n=== stimulus-speed generalization (score) ===")
    print(f"{'':>6} {'1 deg/step':>15} {'2 deg/step':>15} {'3 deg/step':>15}")
    for label in named:
        print(f"{label:>6} {show(label):>15} {show(label + ' speed2'):>15} "
              f"{show(label + ' speed3'):>15}")

    # policy curves + R^2
    policy = {}
    for label in named:
        runs = by_label[label]
        count = np.sum([r["bin_count"] for r in runs], axis=0)
        s1 = np.sum([r["bin_sum"] for r in runs], axis=0)
        s2 = np.sum([r["bin_sumsq"] for r in runs], axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            mean = np.where(count > 0, s1 / np.maximum(count, 1), np.nan)
            var_in_bin = np.where(count > 0, s2 / np.maximum(count, 1) - mean**2, np.nan)
        total_n = count.sum()
        g_mean = s1.sum() / total_n
        total_var = s2.sum() / total_n - g_mean**2
        sse = np.nansum(var_in_bin * count)
        r2 = float(1.0 - sse / (total_var * total_n)) if total_var > 0 else 0.0
        policy[label] = {"mean": mean, "sd": np.sqrt(np.maximum(var_in_bin, 0)),
                         "count": count, "r2": r2}
        print(f"policy R^2({label}) = {r2:.3f}")

    make_plots(named, by_label, grafts, policy, out_dir)

    payload = {
        "seeds": SEEDS,
        "labels": {label: agg_scores(by_label[label]) for label in by_label},
        "policy_r2": {label: policy[label]["r2"] for label in policy},
        "configs": named,
    }
    (out_dir / "analysis.json").write_text(json.dumps(payload))
    print(f"\nsaved {out_dir}/analysis.json and figures")


def make_plots(named, by_label, grafts, policy, out_dir) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"base": "black", "w1": "tab:blue", "w2": "tab:orange"}
    names = {"base": "paper baseline", "w1": "#1 dense (id 236)", "w2": "#2 sparse (id 234)"}

    def agg(label):
        s = np.array([r["score"] for r in by_label[label]])
        return s.mean(), s.std()

    # ---- attribution figure ----
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), width_ratios=[3, 3, 2])
    for ax, wname in zip(axes[:2], ("w1", "w2")):
        full, _ = agg(wname)
        params = [p for p in named["base"] if f"{wname} revert {p}" in by_label]
        deltas = [agg(f"{wname} revert {p}")[0] - full for p in params]
        sds = [agg(f"{wname} revert {p}")[1] for p in params]
        order = np.argsort(deltas)
        ax.barh(np.arange(len(params)), [deltas[i] for i in order],
                xerr=[sds[i] / np.sqrt(12) for i in order],
                color=[colors[wname]] * len(params), alpha=0.8)
        ax.set_yticks(np.arange(len(params)))
        ax.set_yticklabels([f"{params[i]} → {named['base'][params[i]]:.3g}" for i in order],
                           fontsize=8.5)
        ax.axvline(0, color="gray", lw=1)
        ax.set_xlabel("Δ score when reverted to paper value")
        ax.set_title(f"{names[wname]} (full {full:.2f})", fontsize=10)
    base_full, _ = agg("base")
    labels = list(grafts)
    vals = [agg(lbl)[0] for lbl in labels]
    sds = [agg(lbl)[1] / np.sqrt(12) for lbl in labels]
    axes[2].barh(np.arange(len(labels)), [v - base_full for v in vals], xerr=sds,
                 color="dimgray", alpha=0.85)
    axes[2].set_yticks(np.arange(len(labels)))
    axes[2].set_yticklabels([lbl.replace("base + ", "+") for lbl in labels], fontsize=8.5)
    axes[2].axvline(0, color="gray", lw=1)
    axes[2].set_xlabel("Δ score vs baseline")
    axes[2].set_title(f"grafts onto baseline ({base_full:.2f})", fontsize=10)
    fig.suptitle("Parameter attribution (12 fresh seeds; bars = mean Δ, whiskers = s.e.)")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_attribution.png", dpi=150)

    # ---- ablation figure ----
    fig, ax = plt.subplots(figsize=(9, 4.4))
    conds = ["full", "no-learn", "lesioned", "lesioned no-learn"]
    width = 0.25
    for k, label in enumerate(named):
        means, errs = [], []
        for cond in conds:
            lbl = label if cond == "full" else f"{label} {cond}"
            m, sd = agg(lbl)
            means.append(m)
            errs.append(sd / np.sqrt(12))
        ax.bar(np.arange(len(conds)) + (k - 1) * width, means, width, yerr=errs,
               color=colors[label], alpha=0.85, label=names[label], capsize=3)
    ax.axhline(0.25, color="gray", ls=":", lw=1, label="chance")
    ax.set_xticks(np.arange(len(conds)))
    ax.set_xticklabels(["full model", "learning frozen", "recurrence lesioned",
                        "lesioned + frozen"])
    ax.set_ylabel("within-45° fraction")
    ax.set_title("What each mechanism contributes (12 fresh seeds)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_ablations.png", dpi=150)

    # ---- policy curves ----
    fig, ax = plt.subplots(figsize=(9, 4.6))
    centers = (ERR_EDGES[:-1] + ERR_EDGES[1:]) / 2
    ax.axvspan(-180, -90, color="gray", alpha=0.10)
    ax.axvspan(90, 180, color="gray", alpha=0.10)
    ax.text(-135, 0.02, "outside FOV", ha="center", fontsize=8, color="gray")
    ax.text(135, 0.02, "outside FOV", ha="center", fontsize=8, color="gray")
    for label in named:
        p = policy[label]
        ax.plot(centers, p["mean"], color=colors[label], lw=1.8,
                label=f"{names[label]}  (R²={p['r2']:.2f})")
        ax.fill_between(centers, p["mean"] - p["sd"], p["mean"] + p["sd"],
                        color=colors[label], alpha=0.12)
    ax.axhline(0, color="gray", lw=1)
    ax.axvline(0, color="gray", lw=0.6, ls=":")
    ax.set_xlabel("egocentric error at sensing time (deg; + = stimulus to the left)")
    ax.set_ylabel("applied turn ΔH that step (deg)")
    ax.set_title("The emergent control law: mean turn vs. error (shading = per-bin s.d.)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_policy.png", dpi=150)

    # ---- speed generalization ----
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for label in named:
        xs, ys, es = [1, 2, 3], [], []
        for cond in ("", " speed2", " speed3"):
            m, sd = agg(label + cond)
            ys.append(m)
            es.append(sd / np.sqrt(12))
        ax.errorbar(xs, ys, yerr=es, marker="o", color=colors[label],
                    label=names[label], capsize=3)
    ax.axhline(0.25, color="gray", ls=":", lw=1)
    ax.set_xticks([1, 2, 3])
    ax.set_xlabel("stimulus speed (deg/step)")
    ax.set_ylabel("within-45° fraction")
    ax.set_title("Generalization to faster stimuli")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_speed.png", dpi=150)


if __name__ == "__main__":
    main()
