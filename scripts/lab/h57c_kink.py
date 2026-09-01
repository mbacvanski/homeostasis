"""H57c: do in-view kinks poison ballistic interception? Self-controlled."""
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

def crossings_kinked(h):
    """Split at respawn jumps; kinks = heading changes without jumps."""
    dx, dy = np.diff(h.sx), np.diff(h.sy)
    jump = np.hypot(dx, dy) > 1.0
    phi = np.arctan2(dy, dx)
    turn = np.abs((np.diff(phi) + np.pi) % (2 * np.pi) - np.pi) > 0.3
    kink = turn & ~jump[1:] & ~jump[:-1]
    bounds = [0] + (np.flatnonzero(jump) + 1).tolist() + [len(h.sx)]
    res = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        if b - a < 30:
            continue
        has_kink = bool(kink[max(a - 1, 0):b - 1].any())
        res.append((has_kink, float(h.dist[a:b].min()) < CATCH_R))
    return res

def evaluate(task):
    genome, seed = task
    pc = PursuitConfig(eye_offsets=(0.0,), sensors_per_eye=91,
                       stimulus_motion="ballistic", stimulus_speed=0.04,
                       ballistic_kink_hazard=1 / 150,
                       wheel_base=genome["wheel_base"],
                       intensity_scale=genome["intensity_scale"])
    res = ReservoirConfig(n_inputs=pc.n_sensors, **{k: genome[k] for k in RES_KEYS})
    h = run_pursuit(n_steps=14400, seed=seed, reservoir_config=res, pursuit_config=pc)
    return crossings_kinked(h)

def main():
    out = {}
    with ProcessPoolExecutor(10) as pool:
        for label, g in (("champ", H34_CHAMP),
                         ("blind", {**H34_CHAMP, "input_weight": 1e-6})):
            rowsets = list(pool.map(evaluate, [(g, s) for s in range(41, 49)]))
            flat = [x for rs in rowsets for x in rs]
            kf = [c for k, c in flat if not k]
            kk = [c for k, c in flat if k]
            out[label] = dict(kink_free=[int(c) for c in kf], kinked=[int(c) for c in kk])
            print(f"{label}: kink-free catch {np.mean(kf):.3f} (n={len(kf)})"
                  f" | kinked catch {np.mean(kk):.3f} (n={len(kk)})"
                  f" | diff {np.mean(kf)-np.mean(kk):+.3f}")
    (LAB / "h57c_kink.json").write_text(json.dumps(out))

if __name__ == "__main__":
    main()
