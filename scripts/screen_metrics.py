"""Screen candidate internal metrics for correlation with tracking score.

Evaluates ~250 configurations (the 240-config random sweep, named trackers,
and the known degenerate champions - statues and autarkic hums) on a battery
of internally-computable metrics, then reports for each metric:

  - Spearman correlation with within-45 score across the random population
    (degenerates excluded from the correlation, reported separately);
  - the Goodhart check: mean score of the top-10 configs BY that metric
    (what selection on the metric would actually have dragged in);
  - where the known cheaters rank under the metric (percentile; a usable
    fitness metric must place them low).

All metrics are computable from the organism's own state (sensors, spikes,
errors, weights, efference); no world-frame quantities.

Usage: python scripts/screen_metrics.py [--workers 10] [--seeds 3]
"""

from __future__ import annotations

import os

for _v in ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import pathlib
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from homeostasis import ReservoirConfig, TrackingConfig, TrackingSimulation

ROOT = pathlib.Path(__file__).resolve().parent.parent
GENOME_KEYS = ("n_nodes", "p_link", "input_weight", "weight_init_mean",
               "weight_init_sd", "leak", "target_lr", "threshold_ratio", "gain")
STEPS = 3600
SETTLE = 720
WIN = 120
RATE_LO, RATE_HI = 0.05, 0.90


def evaluate(task):
    cfg, seed = task
    r_kwargs = {k: cfg[k] for k in GENOME_KEYS if k != "gain"}
    sim = TrackingSimulation(ReservoirConfig(**r_kwargs),
                             TrackingConfig(gain=cfg["gain"]), seed=seed)
    n = sim.network.config.n_nodes
    n_sens = 62
    idx = np.arange(n_sens)

    flow = np.zeros(STEPS - SETTLE)
    prop = np.zeros(STEPS - SETTLE)
    dh_arr = np.zeros(STEPS - SETTLE)
    centroid = np.full(STEPS - SETTLE, np.nan)
    err_mean = np.zeros(STEPS - SETTLE)
    in45 = 0
    spike_hist = []           # spike vectors every 10 steps for turnover
    w_snaps = []              # weight snapshots every 240 steps
    spike_counts = np.zeros(n)
    sensor_sums = np.zeros(n_sens)
    wl = 0
    band_i = band_s = 0.0
    n_win = 0

    for t in range(STEPS):
        e_deg = sim.env.heading_error()
        state, dh = sim.step()
        if t < SETTLE:
            continue
        i = t - SETTLE
        s = state.inputs
        tot = float(s.sum())
        flow[i] = tot
        prop[i] = state.prop_spiked
        dh_arr[i] = dh
        err_mean[i] = float(np.mean(np.abs(state.error)))
        if tot > 0.05:
            centroid[i] = float((s * idx).sum() / tot)
        in45 += 1 if abs(e_deg) <= 45.0 else 0
        if i % 10 == 0:
            spike_hist.append(state.spiked.copy())
        if i % 240 == 0:
            w_snaps.append(sim.network.weights.copy())
        spike_counts += state.spiked
        sensor_sums += s
        wl += 1
        if wl == WIN:
            r = spike_counts / WIN
            band_i += float(np.mean((r > RATE_LO) & (r < RATE_HI)))
            sv = sensor_sums / WIN
            band_s += float(np.mean((sv > RATE_LO) & (sv < RATE_HI)))
            spike_counts[:] = 0.0
            sensor_sums[:] = 0.0
            wl = 0
            n_win += 1

    T = len(flow)
    # pattern turnover: 1 - mean corr between spike vectors 120 steps apart
    lag = 12  # spike_hist stride 10 -> 120 steps
    cors = []
    for a in range(lag, len(spike_hist)):
        x, y = spike_hist[a - lag].astype(float), spike_hist[a].astype(float)
        if x.std() > 0 and y.std() > 0:
            cors.append(float(np.corrcoef(x, y)[0, 1]))
    turnover = 1.0 - float(np.mean(cors)) if cors else 0.0
    # homeostatic work: mean |dW| per link per step across snapshots
    works = []
    n_links = max(int(np.count_nonzero(sim.network.adjacency)), 1)
    for a in range(1, len(w_snaps)):
        works.append(float(np.abs(w_snaps[a] - w_snaps[a - 1]).sum()) / (240 * n_links))
    work = float(np.mean(works)) if works else 0.0
    # sensorimotor contingency: corr(my turn at t, centroid shift at t+1)
    dc = np.diff(centroid)
    valid = ~np.isnan(dc)
    x = dh_arr[:-1][valid]
    y = dc[valid]
    if valid.sum() > 30 and x.std() > 1e-9 and y.std() > 1e-9:
        contingency = abs(float(np.corrcoef(x, y)[0, 1]))
    else:
        contingency = 0.0

    return {
        "score": in45 / T,
        "input_flow": float(flow.mean()),
        "input_duty": float((flow > 0.05).mean()),
        "input_std": float(flow.std()),
        "input_roughness": float(np.abs(np.diff(flow)).mean()),
        "sensor_band": band_s / max(n_win, 1),
        "rate_band": band_i / max(n_win, 1),
        "mean_abs_E": float(err_mean.mean()),
        "prop_spiked": float(prop.mean()),
        "act_dynamism": float(prop.std()),
        "pattern_turnover": turnover,
        "homeo_work": work,
        "contingency": contingency,
    }


METRICS = ["input_flow", "input_duty", "input_std", "input_roughness",
           "sensor_band", "rate_band", "mean_abs_E", "prop_spiked",
           "act_dynamism", "pattern_turnover", "homeo_work", "contingency"]


def spearman(a, b):
    def ranks(v):
        o = np.argsort(v)
        r = np.empty(len(v))
        r[o] = np.arange(len(v))
        return r
    return float(np.corrcoef(ranks(np.asarray(a)), ranks(np.asarray(b)))[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()

    sweep = json.loads((ROOT / "scripts/out/sweep/results.json").read_text())
    zoo = {}       # name -> (config, kind)
    for i, c in enumerate(sweep["configs"]):
        zoo[f"sweep{i}"] = (c, "random" if i else "paper")
    w1 = dict(sweep["configs"][236])
    zoo["w1prime"] = (dict(w1, weight_init_mean=0.75), "tracker")
    for f, kind in [("scripts/out/evolution/champions.json", "hum"),
                    ("scripts/out/evolution_boundary/champions.json", "statue")]:
        p = ROOT / f
        if p.exists():
            for e in json.loads(p.read_text()):
                zoo[e["id"]] = ({k: e["params"][k] for k in GENOME_KEYS}, kind)

    names = list(zoo)
    seeds = list(range(400, 400 + args.seeds))
    tasks = [(zoo[nm][0], s) for nm in names for s in seeds]
    print(f"{len(tasks)} runs ({len(names)} configs x {len(seeds)} seeds)", flush=True)
    t0 = time.perf_counter()
    with ProcessPoolExecutor(args.workers) as pool:
        res = list(pool.map(evaluate, tasks, chunksize=4))
    print(f"evaluated in {time.perf_counter() - t0:.0f}s", flush=True)

    rows = []
    for i, nm in enumerate(names):
        rs = res[i * len(seeds):(i + 1) * len(seeds)]
        row = {k: float(np.mean([r[k] for r in rs])) for k in rs[0]}
        row["name"] = nm
        row["kind"] = zoo[nm][1]
        rows.append(row)
    (ROOT / "scripts/out/metric_screen.json").write_text(json.dumps(rows))

    pop = [r for r in rows if r["kind"] in ("random", "paper")]
    cheats = [r for r in rows if r["kind"] in ("hum", "statue")]
    scores = [r["score"] for r in pop]

    print(f"\n=== metric screen over {len(pop)} ordinary configs "
          f"(+{len(cheats)} known cheaters held out) ===")
    print(f"{'metric':>17} {'rho(score)':>10} {'top10 score':>12} {'cheater pctile':>15}")
    results = []
    for m in METRICS:
        vals = [r[m] for r in pop]
        rho = spearman(vals, scores)
        # Goodhart: what score does selecting the top decile BY the metric buy?
        top = sorted(pop, key=lambda r: -r[m])[:10]
        top_score = float(np.mean([r["score"] for r in top]))
        # where do the cheaters land in this metric's ranking?
        pct = [float(np.mean([v <= c[m] for v in vals])) for c in cheats]
        cheat_pct = float(np.mean(pct)) if pct else float("nan")
        results.append((m, rho, top_score, cheat_pct))
        print(f"{m:>17} {rho:>10.2f} {top_score:>12.3f} {cheat_pct:>15.2f}")
    print("\n(top10 score: mean within-45 of the 10 configs ranked highest by the metric;")
    print(" chance 0.25, paper 0.44. cheater pctile: mean rank of statues+hums under the")
    print(" metric, 1.0 = cheaters top the ranking, 0.0 = correctly at the bottom.)")

    # scatter panel for the six strongest |rho| metrics
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    top6 = sorted(results, key=lambda r: -abs(r[1]))[:6]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
    colors = {"random": "tab:blue", "paper": "black", "tracker": "tab:green",
              "hum": "tab:red", "statue": "tab:orange"}
    for ax, (m, rho, tops, cpct) in zip(axes.ravel(), top6):
        for kind, c in colors.items():
            sel = [r for r in rows if r["kind"] == kind]
            ax.scatter([r[m] for r in sel], [r["score"] for r in sel], s=22 if kind != "random" else 9,
                       color=c, alpha=0.8 if kind != "random" else 0.45,
                       label=kind if kind != "random" else None, zorder=3 if kind != "random" else 2)
        ax.set_title(f"{m}  (rho={rho:+.2f}, top10={tops:.2f})", fontsize=10)
        ax.set_xlabel(m, fontsize=8)
        ax.set_ylabel("score", fontsize=8)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Internal metrics vs tracking score (blue = random configs; markers = named/cheaters)")
    fig.tight_layout()
    fig.savefig(ROOT / "scripts/out/metric_screen.png", dpi=150)
    print("saved scripts/out/metric_screen.json and metric_screen.png", flush=True)


if __name__ == "__main__":
    main()
