"""H101: does slow smooth irregularity become followable?"""
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
    sd, tau, jseed = task
    g = CHAMP["champion"]
    pc = PursuitConfig(eye_offsets=(0.0,), sensors_per_eye=91, box_size=30.0,
                       initial_agent_x=15.0, initial_agent_y=10.0,
                       wheel_base=g["wheel_base"],
                       intensity_scale=g["intensity_scale"])
    res = ReservoirConfig(n_inputs=pc.n_sensors, **{k: g[k] for k in RES_KEYS})
    net = HomeostaticReservoir(res, seed=CHAMP["champ_seed"])
    env = PursuitEnv(pc, rng=net.rng)
    jrng = np.random.default_rng(10100 + jseed)
    n = 7200
    jx = jy = 0.0
    kick = sd * np.sqrt(2.0 / tau)
    dist = np.empty(n)
    for i in range(n):
        jx += -jx / tau + float(jrng.normal(0.0, kick))
        jy += -jy / tau + float(jrng.normal(0.0, kick))
        px, py = LOOP[i]
        env.sx = float(px) + jx
        env.sy = float(py) + jy
        dist[i] = env.distance()
        st = net.step(env.sense())
        env.apply_action(*map(float, st.outputs))
        env.steps += 1
    return dict(sd=sd, tau=tau, jseed=jseed,
                near4=float((dist[n // 2:] < 4).mean()))

def main():
    cells = [(0.5, t) for t in (60, 240, 960, 3840)] + [(1.0, 960), (1.0, 3840)]
    tasks = [(sd, tau, j) for sd, tau in cells for j in range(8)]
    with ProcessPoolExecutor(10) as pool:
        rows = list(pool.map(run, tasks, chunksize=2))
    (LAB / "h101_slowjitter.json").write_text(json.dumps(rows))
    for sd, tau in cells:
        sel = [r["near4"] for r in rows if r["sd"] == sd and r["tau"] == tau]
        print(f"sd={sd} tau={tau:<5} near4 {np.mean(sel):.3f} (SD {np.std(sel):.3f})"
              f"  locked>=0.8: {sum(x >= 0.8 for x in sel)}/8")

if __name__ == "__main__":
    main()
