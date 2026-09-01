"""H90: slow-start curriculum for acquisition."""
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

def record_A(n=10800):
    A = WallSimulation(wall_config=PACE_CFG, seed=PACE_SEED)
    xy = np.zeros((n, 2))
    for i in range(n):
        A.step()
        xy[i] = (A.env.x, A.env.y)
    return xy

TRAJ = record_A()

def traj_at(t_virtual):
    i0 = int(np.floor(t_virtual))
    fr = t_virtual - i0
    i1 = min(i0 + 1, len(TRAJ) - 1)
    return TRAJ[i0] * (1 - fr) + TRAJ[i1] * fr

def run(task):
    off, ang, curriculum = task
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
    n = 7200
    tv = 0.0
    d = np.empty(n)
    for i in range(n):
        rate = (0.5 if (curriculum and i < 3600) else 1.0)
        tv += rate
        px, py = traj_at(min(tv, len(TRAJ) - 1.001))
        env.sx, env.sy = float(px), float(py)
        d[i] = env.distance()
        st = net.step(env.sense())
        env.apply_action(*map(float, st.outputs))
        env.steps += 1
    return dict(off=off, ang=float(ang), curriculum=bool(curriculum),
                lock=float((d[3600:] < 4.8).mean()))

def main():
    tasks = [(off, ang, c) for c in (False, True)
             for off in (-6, -3, 0, 3, 6, 9, 12)
             for ang in np.linspace(0, 2 * np.pi, 4, endpoint=False)]
    with ProcessPoolExecutor(10) as pool:
        rows = list(pool.map(run, tasks, chunksize=1))
    (LAB / "h90_scaffold.json").write_text(json.dumps(rows))
    for c in (False, True):
        sel = [r for r in rows if r["curriculum"] == c]
        acq = np.mean([r["lock"] >= 0.8 for r in sel])
        print(f"{'curriculum' if c else 'constant  '}: acquire {acq:.2f}"
              f" ({sum(r['lock'] >= 0.8 for r in sel)}/28)")

if __name__ == "__main__":
    main()
