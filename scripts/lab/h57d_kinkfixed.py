"""H57d: randomized kink at flight age 100 — causal test of in-view kink poison."""
from __future__ import annotations
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from homeostasis.reservoir import ReservoirConfig  # noqa: E402
from homeostasis.simulation import run_pursuit  # noqa: E402
from homeostasis.pursuit import PursuitConfig  # noqa: E402
from h55_intercept import H34_CHAMP  # noqa: E402

LAB = Path(__file__).resolve().parents[1] / "out" / "lab"
RES_KEYS = ("n_nodes", "p_link", "input_weight", "weight_init_mean",
            "leak", "target_lr", "threshold_ratio", "weight_lr")
CATCH_R = 1.5

def crossing_rows(h):
    jump = np.hypot(np.diff(h.sx), np.diff(h.sy)) > 1.0
    bounds = [0] + (np.flatnonzero(jump) + 1).tolist() + [len(h.sx)]
    rows = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        if b - a < 30:
            continue
        rows.append(dict(dur=b - a, caught=bool(h.dist[a:b].min() < CATCH_R),
                         caught_pre100=bool(h.dist[a:min(a + 100, b)].min() < CATCH_R)))
    return rows

def evaluate(task):
    genome, seed, kink_at = task
    pc = PursuitConfig(eye_offsets=(0.0,), sensors_per_eye=91,
                       stimulus_motion="ballistic", stimulus_speed=0.04,
                       ballistic_kink_at=kink_at,
                       wheel_base=genome["wheel_base"],
                       intensity_scale=genome["intensity_scale"])
    res = ReservoirConfig(n_inputs=pc.n_sensors, **{k: genome[k] for k in RES_KEYS})
    h = run_pursuit(n_steps=14400, seed=seed, reservoir_config=res, pursuit_config=pc)
    return crossing_rows(h)

def main():
    out = {}
    with ProcessPoolExecutor(10) as pool:
        for label, g in (("champ", H34_CHAMP),
                         ("blind", {**H34_CHAMP, "input_weight": 1e-6})):
            for kink_at in (0, 100):
                rowsets = list(pool.map(evaluate, [(g, s, kink_at) for s in range(41, 49)]))
                flat = [x for rs in rowsets for x in rs]
                treated = [r for r in flat if r["dur"] > 100]
                out[f"{label}@{kink_at}"] = flat
                print(f"{label} kink_at={kink_at}: crossings>100 steps n={len(treated)}"
                      f"  catch {np.mean([r['caught'] for r in treated]):.3f}"
                      f"  pre100 catch {np.mean([r['caught_pre100'] for r in treated]):.3f}")
    (LAB / "h57d_kinkfixed.json").write_text(json.dumps(out))
    for label in ("champ", "blind"):
        a = [r for r in out[f"{label}@0"] if r["dur"] > 100]
        b = [r for r in out[f"{label}@100"] if r["dur"] > 100]
        print(f"{label}: kink effect {np.mean([r['caught'] for r in b]) - np.mean([r['caught'] for r in a]):+.3f}"
              f"  (placebo pre100: {np.mean([r['caught_pre100'] for r in b]) - np.mean([r['caught_pre100'] for r in a]):+.3f})")

if __name__ == "__main__":
    main()
