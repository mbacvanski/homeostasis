"""H88: acquisition basin around the pacemaker ring."""
from __future__ import annotations
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from homeostasis.reservoir import HomeostaticReservoir, ReservoirConfig  # noqa: E402
from homeostasis.simulation import WallSimulation  # noqa: E402
from homeostasis.pursuit import PursuitConfig, PursuitEnv  # noqa: E402
from h50_depth import PACE_CFG, PACE_SEED, LAB, RES_KEYS  # noqa: E402

CHAMP = json.loads((LAB / "h48e_warm.json").read_text())
CX, CY, R0 = 19.7, 19.7, 7.8

def run(task):
    off, ang = task
    r = R0 + off
    sx = float(np.clip(CX + r * np.cos(ang), 1.0, 29.0))
    sy = float(np.clip(CY + r * np.sin(ang), 1.0, 29.0))
    g = CHAMP["champion"]
    pc = PursuitConfig(eye_offsets=(0.0,), sensors_per_eye=91, box_size=30.0,
                       initial_agent_x=sx, initial_agent_y=sy,
                       wheel_base=g["wheel_base"],
                       intensity_scale=g["intensity_scale"])
    res = ReservoirConfig(n_inputs=pc.n_sensors, **{k: g[k] for k in RES_KEYS})
    net = HomeostaticReservoir(res, seed=CHAMP["champ_seed"])
    env = PursuitEnv(pc, rng=net.rng)
    A = WallSimulation(wall_config=PACE_CFG, seed=PACE_SEED)
    n = 7200
    d = np.empty(n)
    for i in range(n):
        A.step()
        env.sx, env.sy = A.env.x, A.env.y
        d[i] = env.distance()
        st = net.step(env.sense())
        env.apply_action(*map(float, st.outputs))
        env.steps += 1
    return dict(off=off, ang=float(ang), lock=float((d[n // 2:] < 4.8).mean()))

def main():
    tasks = [(off, ang) for off in (-6, -3, 0, 3, 6, 9, 12)
             for ang in np.linspace(0, 2 * np.pi, 4, endpoint=False)]
    with ProcessPoolExecutor(10) as pool:
        rows = list(pool.map(run, tasks, chunksize=1))
    (LAB / "h88_basin.json").write_text(json.dumps(rows))
    for off in (-6, -3, 0, 3, 6, 9, 12):
        sel = [r["lock"] for r in rows if r["off"] == off]
        print(f"offset {off:+3d}: acquire {np.mean([x >= 0.8 for x in sel]):.2f}"
              f"  (locks {[round(x, 2) for x in sel]})")

if __name__ == "__main__":
    main()
