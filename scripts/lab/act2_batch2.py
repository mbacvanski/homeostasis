"""Act II batch 2: nulls (H15), tlr=0 (H16), transfer function (H17)."""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from common import TrackingConfig, TrackingEnv, run_closed_loop, run_open_loop

LAB = Path(__file__).resolve().parents[1] / "out" / "lab"
SEG = 720


def run_null(task):
    policy = task["policy"]
    tcfg = TrackingConfig()
    env = TrackingEnv(tcfg)
    rng = np.random.default_rng(task["seed"])
    offs = env._sensor_offsets
    gain = tcfg.gain
    n_steps = 7200
    in45 = flow_sum = 0.0
    cands = np.linspace(-gain, gain, 9)
    for i in range(n_steps):
        herr = env.heading_error()
        acts = env.sense()
        flow_sum += acts.sum()
        if abs(herr) <= 45:
            in45 += 1
        if policy == "random":
            dh = rng.uniform(-gain, gain)
        elif policy == "pcontrol":
            s = acts.sum()
            cen = float((acts @ offs) / s) if s > 0.05 else 0.0
            dh = float(np.clip(cen, -gain, gain))
        elif policy == "greedy":
            best, dh = -1.0, 0.0
            for c in cands:
                h = (env.heading + c) % 360.0
                d = np.abs((env.stimulus_angle - (h + offs) + 180.0) % 360.0 - 180.0)
                a = np.exp(-(d ** 2) / tcfg.tuning_width)
                a[d <= tcfg.plateau_width] = 1.0
                tot = a.sum()
                if tot > best:
                    best, dh = tot, float(c)
        env.heading = (env.heading + dh) % 360.0
        env.advance_stimulus()
    return dict(policy=policy, seed=task["seed"], score=in45 / n_steps,
                flow=flow_sum / n_steps, tag="B1")


def dispatch(task):
    kind = task.pop("_kind")
    if kind == "null":
        return run_null(task)
    if kind == "closed":
        r = run_closed_loop(task)
        r["tag"] = task["_tag"]
        return r
    r = run_open_loop(task)
    r["tag"] = task["_tag"]
    r["wlr"] = task["res"]["weight_lr"]
    r["period"] = task["schedule"]["period"]
    return r


def main():
    tasks = []
    # B1 nulls
    for pol in ("random", "pcontrol", "greedy"):
        for s in range(12 if pol == "random" else 4):
            tasks.append(dict(_kind="null", policy=pol, seed=s))
    # B2 tlr=0 vs 0.01 at ridge cells + defaults
    for name, res in [("ridge25", {"leak": 0.25, "weight_lr": 0.1}),
                      ("ridge05", {"leak": 0.05, "weight_lr": 0.03}),
                      ("default", {})]:
        for tlr in (0.0, 0.01):
            for s in range(24):
                r = dict(res, target_lr=tlr)
                tasks.append(dict(_kind="closed", res=r, trk={}, seed=s,
                                  arm="full", snap_every=2400,
                                  _tag=f"B2:{name}:tlr{tlr}"))
    # B3 transfer function
    for wlr in (0.03, 0.1, 0.3, 1.0):
        for period in (30, 60, 120, 240, 480, 960, 1920):
            for s in range(8):
                n = int(max(4000, 4 * period))
                tasks.append(dict(_kind="open",
                                  res={"weight_lr": wlr, "target_lr": 0.01},
                                  seed=s, n_steps=n, recon=True,
                                  schedule={"kind": "sine", "amp": 20.0,
                                            "period": period},
                                  _tag="B3"))

    print(f"{len(tasks)} runs...")
    with ProcessPoolExecutor(10) as pool:
        rows = list(pool.map(dispatch, tasks, chunksize=2))
    LAB.mkdir(exist_ok=True)
    slim = [{k: v for k, v in r.items() if k not in ("f_t", "snaps", "law")} for r in rows]
    (LAB / "act2_batch2.json").write_text(json.dumps(slim))

    print("\n══ B1 nulls (score / flow):")
    for pol in ("random", "pcontrol", "greedy"):
        v = [r for r in rows if r.get("policy") == pol]
        print(f"   {pol:9s} score {np.mean([r['score'] for r in v]):.3f}  "
              f"flow {np.mean([r['flow'] for r in v]):.2f}")

    print("\n══ B2 tlr=0 vs 0.01 (score_late, 24 seeds, frac≥0.35):")
    for name in ("ridge25", "ridge05", "default"):
        line = f"   {name:8s}"
        for tlr in (0.0, 0.01):
            v = np.array([r["score_late"] for r in rows
                          if r.get("tag") == f"B2:{name}:tlr{tlr}"])
            line += f"  tlr={tlr}: {v.mean():.3f} ({np.mean(v >= 0.35):.2f})"
        print(line)

    print("\n══ B3 transfer function: recon_gain (rows=wlr, cols=period)")
    periods = (30, 60, 120, 240, 480, 960, 1920)
    print("          " + "  ".join(f"P={p:<5d}" for p in periods))
    for wlr in (0.03, 0.1, 0.3, 1.0):
        cells = []
        for p in periods:
            v = [r["recon_gain"] for r in rows
                 if r.get("tag") == "B3" and r.get("wlr") == wlr and r.get("period") == p
                 and "recon_gain" in r]
            cells.append(np.mean(v) if v else float("nan"))
        print(f"   wlr={wlr:<5}" + "  ".join(f"{c:7.3f}" for c in cells))
    print("   … recon_valid_frac (fraction of steps with any spike):")
    for wlr in (0.03, 0.1, 0.3, 1.0):
        cells = []
        for p in periods:
            v = [r["recon_valid_frac"] for r in rows
                 if r.get("tag") == "B3" and r.get("wlr") == wlr and r.get("period") == p
                 and "recon_valid_frac" in r]
            cells.append(np.mean(v) if v else float("nan"))
        print(f"   wlr={wlr:<5}" + "  ".join(f"{c:7.3f}" for c in cells))


if __name__ == "__main__":
    main()
