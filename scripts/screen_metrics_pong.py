"""Cross-task metric screen: do internal metrics correlate with performance in
BOTH tracking and Pong, with the same formulas and no per-task tuning?

Tracking population: the 241-config random sweep + known degenerates.
Pong population: 70 random configurations around the published Pong network
(plus the published config), 20k steps each.

Metrics (all from the organism's own state):
  input_flow        mean total sensor activation
  input_duty        fraction of steps with any input
  smooth_presence   mean over step-pairs of present*present*exp(-|d centroid|/6)
                    ("sustained sensory continuity" - the unified candidate)
  mean_jump         mean |centroid change| between present steps (degrees)
  rate_band, mean_abs_E, prop_spiked, act_dynamism   (interior references)

Usage: python scripts/screen_metrics_pong.py [--workers 10]
"""

from __future__ import annotations

import os

for _v in ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import dataclasses
import json
import pathlib
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from homeostasis import (
    PONG_RESERVOIR_CONFIG,
    PongConfig,
    PongSimulation,
    ReservoirConfig,
    TrackingConfig,
    TrackingSimulation,
)

ROOT = pathlib.Path(__file__).resolve().parent.parent
TRACK_KEYS = ("n_nodes", "p_link", "input_weight", "weight_init_mean",
              "weight_init_sd", "leak", "target_lr", "threshold_ratio", "gain")
JUMP_SCALE = 6.0  # degrees of centroid motion per step at which continuity halves-ish


class MetricStream:
    """Online internal-metric accumulator over (inputs, spiked, error) steps."""

    def __init__(self, sensor_values, n_nodes, window=120):
        self.vals = np.asarray(sensor_values, dtype=float)
        self.window = window
        self.n = n_nodes
        self.flow_sum = 0.0
        self.present_sum = 0
        self.smooth_sum = 0.0
        self.jump_sum = 0.0
        self.jump_n = 0
        self.pairs = 0
        self.prev_c = None
        self.prev_present = False
        self.abs_e = 0.0
        self.props = []
        self.spike_counts = np.zeros(n_nodes)
        self.wl = 0
        self.band = 0.0
        self.wins = 0
        self.count = 0

    def push(self, inputs, spiked, error, prop):
        tot = float(inputs.sum())
        present = tot > 0.05
        c = float((inputs * self.vals).sum() / tot) if present else None
        self.flow_sum += tot
        self.present_sum += 1 if present else 0
        if self.count > 0:
            self.pairs += 1
            if present and self.prev_present:
                jump = abs(c - self.prev_c)
                self.smooth_sum += float(np.exp(-jump / JUMP_SCALE))
                self.jump_sum += jump
                self.jump_n += 1
        self.prev_c, self.prev_present = c, present
        self.abs_e += float(np.mean(np.abs(error)))
        self.props.append(prop)
        self.spike_counts += spiked
        self.wl += 1
        if self.wl == self.window:
            r = self.spike_counts / self.window
            self.band += float(np.mean((r > 0.05) & (r < 0.9)))
            self.spike_counts[:] = 0.0
            self.wl = 0
            self.wins += 1
        self.count += 1

    def metrics(self):
        p = np.array(self.props)
        return {
            "input_flow": self.flow_sum / self.count,
            "input_duty": self.present_sum / self.count,
            "smooth_presence": self.smooth_sum / max(self.pairs, 1),
            "mean_jump": self.jump_sum / max(self.jump_n, 1),
            "rate_band": self.band / max(self.wins, 1),
            "mean_abs_E": self.abs_e / self.count,
            "prop_spiked": float(p.mean()),
            "act_dynamism": float(p.std()),
        }


def eval_tracking(task):
    cfg, seed = task
    sim = TrackingSimulation(
        ReservoirConfig(**{k: cfg[k] for k in TRACK_KEYS if k != "gain"}),
        TrackingConfig(gain=cfg["gain"]), seed=seed)
    ms = MetricStream(sim.env.config.sensor_offsets, sim.network.config.n_nodes)
    in45 = 0
    for t in range(3600):
        e = sim.env.heading_error()
        state, _ = sim.step()
        if t >= 720:
            ms.push(state.inputs, state.spiked, state.error, state.prop_spiked)
            in45 += 1 if abs(e) <= 45 else 0
    out = ms.metrics()
    out["score"] = in45 / 2880
    return out


PONG_KEYS = ("n_nodes", "p_link", "input_weight", "weight_init_mean",
             "weight_init_sd", "inhibitory_fraction", "leak", "target_lr",
             "threshold_ratio")


def eval_pong(task):
    cfg, seed = task
    pong_cfg = PongConfig(gain=cfg["gain"])
    r_cfg = dataclasses.replace(
        PONG_RESERVOIR_CONFIG,
        **{k: cfg[k] for k in PONG_KEYS},
    )
    sim = PongSimulation(r_cfg, pong_cfg, seed=seed)
    ms = MetricStream(pong_cfg.sensor_values, r_cfg.n_nodes)
    settle = 1000
    for t in range(20000):
        state, _, _ = sim.step()
        if t >= settle:
            ms.push(state.inputs, state.spiked, state.error, state.prop_spiked)
    out = ms.metrics()
    hits = np.asarray(sim.env.hits, dtype=float)
    out["score"] = float(hits.mean()) if hits.size else 0.0
    out["opps"] = int(hits.size)
    return out


def sample_pong_config(rng):
    log_u = lambda lo, hi: float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
    return {
        "n_nodes": int(rng.integers(200, 701)),
        "p_link": log_u(0.03, 0.3),
        "input_weight": log_u(0.5, 8.0),
        "weight_init_mean": float(rng.uniform(-0.5, 0.5)),
        "weight_init_sd": log_u(0.05, 0.6),
        "inhibitory_fraction": float(rng.uniform(0.0, 0.5)),
        "leak": float(rng.uniform(0.05, 0.6)),
        "target_lr": log_u(0.01, 0.4),
        "threshold_ratio": float(rng.uniform(1.2, 4.0)),
        "gain": log_u(20.0, 400.0),
    }


PUBLISHED_PONG = {
    "n_nodes": 500, "p_link": 0.1, "input_weight": 2.75,
    "weight_init_mean": 0.0, "weight_init_sd": 0.2, "inhibitory_fraction": 0.25,
    "leak": 0.25, "target_lr": 0.1, "threshold_ratio": 2.0, "gain": 100.0,
}

METRICS = ["input_flow", "input_duty", "smooth_presence", "mean_jump",
           "rate_band", "mean_abs_E", "prop_spiked", "act_dynamism"]


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
    ap.add_argument("--pong-configs", type=int, default=70)
    args = ap.parse_args()

    # ---- tracking population (reuse sweep) --------------------------------
    sweep = json.loads((ROOT / "scripts/out/sweep/results.json").read_text())
    track_cfgs = list(sweep["configs"])
    track_tasks = [(c, s) for c in track_cfgs for s in (500, 501)]

    # ---- pong population ---------------------------------------------------
    rng = np.random.default_rng(7)
    pong_cfgs = [dict(PUBLISHED_PONG)] + [sample_pong_config(rng)
                                          for _ in range(args.pong_configs)]
    pong_tasks = [(c, s) for c in pong_cfgs for s in (500, 501)]

    t0 = time.perf_counter()
    with ProcessPoolExecutor(args.workers) as pool:
        print(f"tracking: {len(track_tasks)} runs", flush=True)
        track_res = list(pool.map(eval_tracking, track_tasks, chunksize=4))
        print(f"  done {time.perf_counter()-t0:.0f}s; pong: {len(pong_tasks)} runs",
              flush=True)
        pong_res = list(pool.map(eval_pong, pong_tasks, chunksize=2))
    print(f"all evaluated {time.perf_counter()-t0:.0f}s", flush=True)

    def collapse(cfgs, res):
        rows = []
        for i in range(len(cfgs)):
            rs = res[i * 2:(i + 1) * 2]
            rows.append({k: float(np.mean([r[k] for r in rs])) for k in rs[0]})
        return rows

    track_rows = collapse(track_cfgs, track_res)
    pong_rows = collapse(pong_cfgs, pong_res)
    # drop pong rows with too few opportunities to score meaningfully
    pong_rows = [r for r in pong_rows if r.get("opps", 99) >= 15]
    json.dump({"tracking": track_rows, "pong": pong_rows},
              open(ROOT / "scripts/out/cross_task_screen.json", "w"))

    ts = [r["score"] for r in track_rows]
    ps = [r["score"] for r in pong_rows]
    print(f"\ntracking n={len(track_rows)} (score range {min(ts):.2f}-{max(ts):.2f}); "
          f"pong n={len(pong_rows)} (hit rate range {min(ps):.2f}-{max(ps):.2f}, chance 0.20)")
    print(f"\n{'metric':>16} {'rho track':>10} {'rho pong':>9} {'top10 trk':>10} {'top10 pong':>11}")
    for m in METRICS:
        tv = [r[m] for r in track_rows]
        pv = [r[m] for r in pong_rows]
        rt = spearman(tv, ts)
        rp = spearman(pv, ps)
        top_t = float(np.mean([r["score"] for r in
                               sorted(track_rows, key=lambda r: -r[m])[:10]]))
        top_p = float(np.mean([r["score"] for r in
                               sorted(pong_rows, key=lambda r: -r[m])[:10]]))
        print(f"{m:>16} {rt:>10.2f} {rp:>9.2f} {top_t:>10.3f} {top_p:>11.3f}")
    print("\n(top10 = mean task score of the 10 configs ranked highest by the metric;")
    print(" tracking chance 0.25 / paper 0.44; pong chance 0.20 / paper 0.58)")

    # figure: the two headline metrics on both tasks
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for row, (rows, sc, task, chance) in enumerate(
            [(track_rows, ts, "tracking (within-45)", 0.25),
             (pong_rows, ps, "Pong (hit rate)", 0.20)]):
        for col, m in enumerate(["input_flow", "smooth_presence"]):
            ax = axes[row][col]
            ax.scatter([r[m] for r in rows], sc, s=14, alpha=0.6, color="tab:blue")
            ax.axhline(chance, color="gray", ls=":", lw=1)
            rho = spearman([r[m] for r in rows], sc)
            ax.set_title(f"{task}: {m} (rho={rho:+.2f})", fontsize=10)
            ax.set_xlabel(m, fontsize=9)
            ax.set_ylabel("task score", fontsize=9)
    fig.suptitle("Does the metric generalize? input flow vs sustained sensory continuity")
    fig.tight_layout()
    fig.savefig(ROOT / "scripts/out/cross_task_screen.png", dpi=150)
    print("saved scripts/out/cross_task_screen.{json,png}", flush=True)


if __name__ == "__main__":
    main()
