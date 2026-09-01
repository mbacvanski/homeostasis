"""H72: does target adaptation selectively relabel input-driven nodes?"""
from __future__ import annotations
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from common import make_configs
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from homeostasis.reservoir import HomeostaticReservoir  # noqa: E402
from homeostasis.tracking import TrackingEnv  # noqa: E402

LAB = Path(__file__).resolve().parents[1] / "out" / "lab"

def stationary_acts(tcfg):
    d = np.abs((0.0 - tcfg.sensor_offsets + 180.0) % 360.0 - 180.0)
    acts = np.exp(-(d ** 2) / tcfg.tuning_width)
    acts[d <= tcfg.plateau_width] = 1.0
    return acts

def run(seed):
    rcfg, tcfg = make_configs({"weight_lr": 0.1, "target_lr": 0.01,
                               "input_p_link": 0.1}, {})
    net = HomeostaticReservoir(rcfg, seed=seed)
    env = TrackingEnv(tcfg)
    T0 = net.targets.copy()
    mu_in = stationary_acts(tcfg) @ net.input_weights
    for _ in range(21600):
        state = net.step(env.sense())
        env.apply_action(*state.outputs)
        env.advance_stimulus()
    dT = net.targets - T0
    r = float(np.corrcoef(dT, mu_in)[0, 1])
    hi = float(dT[mu_in >= np.quantile(mu_in, 0.9)].mean())
    lo = float(dT[mu_in <= np.quantile(mu_in, 0.1)].mean())
    return dict(seed=seed, r=r, hi=hi, lo=lo)

def main():
    with ProcessPoolExecutor(10) as pool:
        rows = list(pool.map(run, range(12)))
    (LAB / "h72_relabel.json").write_text(json.dumps(rows))
    print(f"corr(dT, mu_in): mean {np.mean([x['r'] for x in rows]):+.3f} "
          f"(range {min(x['r'] for x in rows):+.2f}..{max(x['r'] for x in rows):+.2f})")
    print(f"dT top-decile {np.mean([x['hi'] for x in rows]):+.3f} vs "
          f"bottom {np.mean([x['lo'] for x in rows]):+.3f}")

if __name__ == "__main__":
    main()
