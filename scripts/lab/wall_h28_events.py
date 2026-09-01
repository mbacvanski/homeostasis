"""H28: event-triggered plasticity bursts around wall hits."""
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
W = 30  # window half-width around events


def evaluate(task):
    seed, learn = task
    sim = WallSimulation(WALL_RESERVOIR_CONFIG, WallConfig(), seed=seed)
    sim.network.learning_enabled = learn
    n = 3600
    absE = np.empty(n); dw = np.empty(n); hits = np.zeros(n, bool)
    prev_w = sim.network.weights.copy()
    for i in range(n):
        state, dh, h = sim.step()
        absE[i] = float(np.mean(np.abs(state.error)))
        dw[i] = float(np.abs(sim.network.weights - prev_w).sum())
        prev_w = sim.network.weights.copy()
        hits[i] = h
    # event-triggered averages (hits after step 100, with clean pre-window)
    ev = [i for i in np.flatnonzero(hits) if W <= i < n - W]
    if ev:
        segE = np.mean([absE[i-W:i+W] for i in ev], axis=0)
        segW = np.mean([dw[i-W:i+W] for i in ev], axis=0)
    else:
        segE = np.zeros(2*W); segW = np.zeros(2*W)
    return dict(seed=seed, learn=learn, n_hits=int(hits.sum()),
                segE=segE.tolist(), segW=segW.tolist(),
                base_E=float(absE[~hits].mean()), base_W=float(dw[~hits].mean()),
                cum_hits=np.cumsum(hits)[::120].tolist(),
                cum_dw=np.cumsum(dw)[::120].tolist())


def main():
    tasks = [(s, True) for s in range(16)] + [(s, False) for s in range(8)]
    with ProcessPoolExecutor(10) as pool:
        rows = list(pool.map(evaluate, tasks))
    (LAB / "wall_h28.json").write_text(json.dumps(rows))
    for learn in (True, False):
        sel = [r for r in rows if r["learn"] == learn and r["n_hits"] > 0]
        segE = np.mean([r["segE"] for r in sel], axis=0)
        segW = np.mean([r["segW"] for r in sel], axis=0)
        bE = np.mean([r["base_E"] for r in sel]); bW = np.mean([r["base_W"] for r in sel])
        print(f"\n══ learning={learn} ({len(sel)} runs with hits)")
        print(f"   |E| at event 0..+5 vs baseline: "
              + " ".join(f"{segE[W+k]/max(bE,1e-9):.2f}x" for k in range(6)))
        print(f"   |dW| at event 0..+5 vs baseline: "
              + " ".join(f"{segW[W+k]/max(bW,1e-9):.2f}x" for k in range(6)))
        print(f"   pre-event mean (−30..−5): E {np.mean(segE[:W-5])/max(bE,1e-9):.2f}x  "
              f"W {np.mean(segW[:W-5])/max(bW,1e-9):.2f}x")
    # (b) cumulative coupling, learning on
    sel = [r for r in rows if r["learn"] and r["n_hits"] > 2]
    cors = []
    for r in sel:
        ch = np.diff(r["cum_hits"]); cw = np.diff(r["cum_dw"])
        if ch.std() > 0 and cw.std() > 0:
            cors.append(float(np.corrcoef(ch, cw)[0, 1]))
    print(f"\n   per-window corr(hit count, |dW|): mean {np.mean(cors):+.2f} over {len(cors)} runs")


if __name__ == "__main__":
    main()
