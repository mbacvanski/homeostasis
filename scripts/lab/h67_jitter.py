"""H67: psychometric curve of entrainment vs pacemaker positional jitter."""
from __future__ import annotations
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from homeostasis.reservoir import HomeostaticReservoir, ReservoirConfig  # noqa: E402
from homeostasis.pursuit import PursuitConfig, PursuitEnv  # noqa: E402

LAB = Path(__file__).resolve().parents[1] / "out" / "lab"
RES_KEYS = ("n_nodes", "p_link", "input_weight", "weight_init_mean",
            "leak", "target_lr", "threshold_ratio", "weight_lr")
CHAMP = json.loads((LAB / "h48e_warm.json").read_text())

def _recorded_full(n=7200):
    """The live pacemaker's own trajectory from step 0 — no tiling."""
    from h50_depth import PACE_CFG, PACE_SEED
    from homeostasis.simulation import WallSimulation
    A = WallSimulation(wall_config=PACE_CFG, seed=PACE_SEED)
    xs, ys = [], []
    for _ in range(n):
        A.step(); xs.append(A.env.x); ys.append(A.env.y)
    return np.stack([xs, ys], axis=1)

LOOP = _recorded_full()

def run(task):
    sd, jseed = task
    g = CHAMP["champion"]
    pc = PursuitConfig(eye_offsets=(0.0,), sensors_per_eye=91, box_size=30.0,
                       initial_agent_x=15.0, initial_agent_y=10.0,
                       wheel_base=g["wheel_base"],
                       intensity_scale=g["intensity_scale"])
    res = ReservoirConfig(n_inputs=pc.n_sensors, **{k: g[k] for k in RES_KEYS})
    net = HomeostaticReservoir(res, seed=CHAMP["champ_seed"])
    env = PursuitEnv(pc, rng=net.rng)
    jrng = np.random.default_rng(9000 + jseed)
    n = 7200
    tau = 60.0
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
    late = slice(3600, None)
    return dict(sd=sd, jseed=jseed, near4=float((dist[late] < 4).mean()),
                dist=float(dist[late].mean()))

def main():
    tasks = [(sd, j) for sd in (0.0, 0.1, 0.25, 0.5, 1.0, 2.0) for j in range(8)]
    with ProcessPoolExecutor(10) as pool:
        rows = list(pool.map(run, tasks, chunksize=2))
    (LAB / "h67_jitter.json").write_text(json.dumps(rows))
    for sd in (0.0, 0.1, 0.25, 0.5, 1.0, 2.0):
        sel = [r for r in rows if r["sd"] == sd]
        print(f"sd={sd:<5} near4 {np.mean([r['near4'] for r in sel]):.3f}"
              f"  dist {np.mean([r['dist'] for r in sel]):.2f}")

if __name__ == "__main__":
    main()
