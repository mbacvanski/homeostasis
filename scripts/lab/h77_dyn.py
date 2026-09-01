"""H77: per-node target drift vs lifetime activity."""
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

def run(seed):
    rcfg, tcfg = make_configs({"weight_lr": 0.1, "target_lr": 0.01,
                               "input_p_link": 0.1}, {})
    net = HomeostaticReservoir(rcfg, seed=seed)
    env = TrackingEnv(tcfg)
    T0 = net.targets.copy()
    spikes = np.zeros(rcfg.n_nodes)
    for _ in range(21600):
        state = net.step(env.sense())
        spikes += net.spiked
        env.apply_action(*state.outputs)
        env.advance_stimulus()
    dT = net.targets - T0
    r = float(np.corrcoef(dT, spikes)[0, 1]) if spikes.std() > 0 and dT.std() > 0 else 0.0
    return dict(seed=seed, r=r, dT_mean=float(dT.mean()),
                dT_spikers=float(dT[spikes > np.median(spikes)].mean()),
                dT_quiet=float(dT[spikes <= np.median(spikes)].mean()))

def main():
    with ProcessPoolExecutor(10) as pool:
        rows = list(pool.map(run, range(12)))
    (LAB / "h77_dyn.json").write_text(json.dumps(rows))
    print(f"corr(dT, spikes): mean {np.mean([x['r'] for x in rows]):+.3f} "
          f"(range {min(x['r'] for x in rows):+.2f}..{max(x['r'] for x in rows):+.2f})")
    print(f"dT active half {np.mean([x['dT_spikers'] for x in rows]):+.3f} vs "
          f"quiet half {np.mean([x['dT_quiet'] for x in rows]):+.3f}")

if __name__ == "__main__":
    main()
