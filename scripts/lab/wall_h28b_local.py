"""H28b: event-triggered analysis with TIME-LOCAL baselines (fixes the
epoch confound: hits cluster early when everything is elevated)."""
from __future__ import annotations
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from common import make_configs  # noqa: F401
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from homeostasis.simulation import WALL_RESERVOIR_CONFIG, WallSimulation  # noqa: E402
from homeostasis.wall import WallConfig  # noqa: E402

LAB = Path(__file__).resolve().parents[1] / "out" / "lab"
L = 150  # local-baseline half-width


def evaluate(seed):
    sim = WallSimulation(WALL_RESERVOIR_CONFIG, WallConfig(), seed=seed)
    n = 3600
    absE = np.empty(n); dw = np.empty(n); flow = np.empty(n); hits = np.zeros(n, bool)
    prev_w = sim.network.weights.copy()
    for i in range(n):
        state, dh, h = sim.step()
        absE[i] = float(np.mean(np.abs(state.error)))
        dw[i] = float(np.abs(sim.network.weights - prev_w).sum())
        flow[i] = float(state.inputs.sum())
        prev_w = sim.network.weights.copy()
        hits[i] = h
    ev = [i for i in np.flatnonzero(hits) if L <= i < n - L]
    ratios = {"E": [], "W": [], "F": []}
    lag_curve = np.zeros(9)
    n_used = 0
    for i in ev:
        loc = slice(i - L, i + L)
        mask = ~hits[loc]
        if mask.sum() < 50:
            continue
        bE = absE[loc][mask].mean(); bW = dw[loc][mask].mean(); bF = flow[loc][mask].mean()
        ratios["E"].append(absE[i:i+3].mean() / max(bE, 1e-12))
        ratios["W"].append(dw[i:i+3].mean() / max(bW, 1e-12))
        ratios["F"].append(flow[i:i+3].mean() / max(bF, 1e-12))
        for k in range(-2, 7):
            lag_curve[k+2] += dw[i+k] / max(bW, 1e-12)
        n_used += 1
    out = dict(seed=seed, n_events=n_used)
    for k in ratios:
        out[k] = float(np.mean(ratios[k])) if ratios[k] else float("nan")
    out["lag"] = (lag_curve / max(n_used, 1)).tolist()
    return out


def main():
    with ProcessPoolExecutor(10) as pool:
        rows = list(pool.map(evaluate, range(16)))
    (LAB / "wall_h28b.json").write_text(json.dumps(rows))
    rows = [r for r in rows if r["n_events"] > 0]
    print(f"{len(rows)} runs, {sum(r['n_events'] for r in rows)} events, local baseline ±{L} steps")
    for k, name in (("E", "|E|"), ("W", "|dW|"), ("F", "flow")):
        v = [r[k] for r in rows]
        print(f"   {name:5s} at event (0..+2) vs local non-hit baseline: {np.mean(v):.2f}x ± {np.std(v):.2f}")
    lag = np.mean([r["lag"] for r in rows], axis=0)
    print("   |dW| lag curve (-2..+6): " + " ".join(f"{v:.2f}" for v in lag))


if __name__ == "__main__":
    main()
