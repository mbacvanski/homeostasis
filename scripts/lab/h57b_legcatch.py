"""H57b: per-leg catch on waypoint — is transient interception intact?"""
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

def leg_catch(h, catch_r=1.5):
    phi = np.arctan2(np.diff(h.sy), np.diff(h.sx))
    kink = np.abs((np.diff(phi) + np.pi) % (2 * np.pi) - np.pi) > 0.3
    bounds = [0] + (np.flatnonzero(kink) + 1).tolist() + [len(h.sx)]
    catches, n = 0, 0
    for a, b in zip(bounds[:-1], bounds[1:]):
        if b - a < 30:
            continue
        n += 1
        if float(h.dist[a:b].min()) < catch_r:
            catches += 1
    return catches / max(n, 1), n

def evaluate(task):
    genome, seed, speed = task
    pc = PursuitConfig(eye_offsets=(0.0,), sensors_per_eye=91,
                       stimulus_motion="waypoint", stimulus_speed=speed,
                       wheel_base=genome["wheel_base"],
                       intensity_scale=genome["intensity_scale"])
    res = ReservoirConfig(n_inputs=pc.n_sensors, **{k: genome[k] for k in RES_KEYS})
    h = run_pursuit(n_steps=7200, seed=seed, reservoir_config=res, pursuit_config=pc)
    c, n = leg_catch(h)
    return dict(catch=c, n_legs=n, near3=float((h.dist[3600:] < 3).mean()),
                agent_speed=float(np.hypot(np.diff(h.x), np.diff(h.y)).mean()))

def main():
    out = {}
    with ProcessPoolExecutor(10) as pool:
        for speed in (0.08, 0.04):
            for label, g in (("champ", H34_CHAMP),
                             ("blind", {**H34_CHAMP, "input_weight": 1e-6})):
                rows = list(pool.map(evaluate, [(g, s, speed) for s in range(41, 49)]))
                out[f"{label}@{speed}"] = rows
                print(f"speed {speed} {label}: leg-catch {np.mean([r['catch'] for r in rows]):.3f}"
                      f" (~{np.mean([r['n_legs'] for r in rows]):.0f} legs)"
                      f"  near3 {np.mean([r['near3'] for r in rows]):.3f}"
                      f"  agent speed {np.mean([r['agent_speed'] for r in rows]):.3f}")
    (LAB / "h57b_legcatch.json").write_text(json.dumps(out))
    for speed in (0.08, 0.04):
        gap = (np.mean([r["catch"] for r in out[f"champ@{speed}"]])
               - np.mean([r["catch"] for r in out[f"blind@{speed}"]]))
        print(f"speed {speed}: per-leg catch gap {gap:+.3f}")

if __name__ == "__main__":
    main()
