"""H102: coupled follower or memorized cycle?"""
from __future__ import annotations
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np

from h67_jitter import LOOP, CHAMP, RES_KEYS
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from homeostasis.reservoir import HomeostaticReservoir, ReservoirConfig  # noqa: E402
from homeostasis.pursuit import PursuitConfig, PursuitEnv  # noqa: E402

LAB = Path(__file__).resolve().parents[1] / "out" / "lab"

def run(task):
    mode, val = task
    g = CHAMP["champion"]
    pc = PursuitConfig(eye_offsets=(0.0,), sensors_per_eye=91, box_size=30.0,
                       initial_agent_x=15.0, initial_agent_y=10.0,
                       wheel_base=g["wheel_base"],
                       intensity_scale=g["intensity_scale"])
    res = ReservoirConfig(n_inputs=pc.n_sensors, **{k: g[k] for k in RES_KEYS})
    net = HomeostaticReservoir(res, seed=CHAMP["champ_seed"])
    env = PursuitEnv(pc, rng=net.rng)
    n = 7200
    dist = np.empty(n)
    for i in range(n):
        off = 0.0
        if mode == "const":
            off = val
        elif mode == "ramp3600":
            off = val * min(max(i - 3600, 0) / 1000.0, 1.0)
        if mode == "phase":
            px, py = LOOP[(i + int(val)) % len(LOOP)]
        else:
            px, py = LOOP[i]
        env.sx = float(px) + off
        env.sy = float(py)
        dist[i] = env.distance()
        st = net.step(env.sense())
        env.apply_action(*map(float, st.outputs))
        env.steps += 1
    late = float((dist[n // 2:] < 4).mean())
    seg5 = float((dist[4600:5800] < 4).mean())
    return dict(mode=mode, val=val, near4_late=late, near4_mid=seg5)

def main():
    tasks = [("const", 0.1), ("const", 0.3), ("const", 1.0),
             ("ramp3600", 0.3), ("ramp3600", 1.0),
             ("phase", 1800)]
    with ProcessPoolExecutor(6) as pool:
        rows = list(pool.map(run, tasks))
    (LAB / "h102_whatlock.json").write_text(json.dumps(rows))
    for r in rows:
        print(f"{r['mode']:<9} {r['val']:<6} near4 late {r['near4_late']:.3f}"
              f"  (mid-window {r['near4_mid']:.3f})")

if __name__ == "__main__":
    main()
