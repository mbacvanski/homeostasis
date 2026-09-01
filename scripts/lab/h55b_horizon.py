"""H55b: does slowing ballistic targets past the re-lock horizon open a skill gap?"""
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
from h55_intercept import H34_CHAMP, crossing_stats  # noqa: E402

LAB = Path(__file__).resolve().parents[1] / "out" / "lab"
GA_CHAMP = json.loads((LAB / "h55_champion.json").read_text())["champion"]

def evaluate(task):
    genome, seed, speed = task
    pc = PursuitConfig(eye_offsets=(0.0,), sensors_per_eye=91,
                       stimulus_motion="ballistic", stimulus_speed=speed,
                       wheel_base=genome["wheel_base"],
                       intensity_scale=genome["intensity_scale"])
    res_keys = ("n_nodes", "p_link", "input_weight", "weight_init_mean",
                "leak", "target_lr", "threshold_ratio", "weight_lr")
    res = ReservoirConfig(n_inputs=pc.n_sensors,
                          **{k: genome[k] for k in res_keys})
    h = run_pursuit(n_steps=7200, seed=seed, reservoir_config=res, pursuit_config=pc)
    catch, n = crossing_stats(h)
    return dict(catch=catch, n=n)

def main():
    variants = {"ga-champ": GA_CHAMP, "h34-champ": H34_CHAMP,
                "blind": {**GA_CHAMP, "input_weight": 1e-6}}
    with ProcessPoolExecutor(10) as pool:
        out = {}
        for speed in (0.15, 0.08, 0.04):
            for label, g in variants.items():
                rows = list(pool.map(evaluate, [(g, s, speed) for s in range(41, 49)]))
                out[f"{label}@{speed}"] = [r["catch"] for r in rows]
                print(f"speed {speed} {label:<10} catch {np.mean([r['catch'] for r in rows]):.3f}"
                      f" (crossings ~{np.mean([r['n'] for r in rows]):.0f})")
    (LAB / "h55b_horizon.json").write_text(json.dumps(out))
    for speed in (0.15, 0.08, 0.04):
        gap = np.mean(out[f"ga-champ@{speed}"]) - np.mean(out[f"blind@{speed}"])
        gap2 = np.mean(out[f"h34-champ@{speed}"]) - np.mean(out[f"blind@{speed}"])
        print(f"speed {speed}: skill gap GA {gap:+.3f} | H34 {gap2:+.3f}")

if __name__ == "__main__":
    main()
