"""H75: GA-attainability psychometric over target speed."""
from __future__ import annotations
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from homeostasis.reservoir import HomeostaticReservoir, ReservoirConfig  # noqa: E402
from homeostasis.pursuit import PursuitConfig, PursuitEnv  # noqa: E402
from h33_evolve_pursuit import mutate, crossover, random_genome  # noqa: E402
from h50_depth import CHAIN_FILE, START_Y, LAB, RES_KEYS  # noqa: E402

R = 7.0

def evaluate(task):
    speed, genome, seed = task
    om = -speed / R
    pc = PursuitConfig(eye_offsets=(0.0,), sensors_per_eye=91, box_size=30.0,
                       initial_agent_x=15.0, initial_agent_y=START_Y[3],
                       wheel_base=genome["wheel_base"],
                       intensity_scale=genome["intensity_scale"])
    res = ReservoirConfig(n_inputs=pc.n_sensors, **{k: genome[k] for k in RES_KEYS})
    net = HomeostaticReservoir(res, seed=seed)
    env = PursuitEnv(pc, rng=net.rng)
    n = 7200
    dist = np.empty(n)
    for i in range(n):
        th = om * i
        env.sx = 15.0 + R * np.cos(th)
        env.sy = 15.0 + R * np.sin(th)
        dist[i] = env.distance()
        st = net.step(env.sense())
        env.apply_action(*map(float, st.outputs))
        env.steps += 1
    late = slice(n // 2, None)
    near = float((dist[late] < 4).mean())
    return dict(fit=near - float(dist[late].mean()) / 30.0, near4=near)

def ga(speed, warm, rng, gens=6, pop_n=16):
    ws = 41414
    pop = [(dict(warm), ws)] + \
          [(mutate(dict(warm), rng), ws) for _ in range(4)] + \
          [(mutate(dict(warm), rng), int(rng.integers(0, 100000))) for _ in range(5)] + \
          [(random_genome(rng), int(rng.integers(0, 100000))) for _ in range(pop_n - 10)]
    best = 0.0
    with ProcessPoolExecutor(10) as pool:
        for gen in range(gens):
            rows = list(pool.map(evaluate, [(speed, g, s) for g, s in pop], chunksize=1))
            fits = [r["fit"] for r in rows]
            bi = int(np.argmax(fits))
            best = max(best, rows[bi]["near4"])
            elite = pop[bi]
            new = [elite]
            while len(new) < len(pop):
                idx = rng.choice(pop_n, 3, replace=False)
                a = pop[idx[int(np.argmax([fits[i] for i in idx]))]]
                idx = rng.choice(pop_n, 3, replace=False)
                b = pop[idx[int(np.argmax([fits[i] for i in idx]))]]
                g = mutate(crossover(a[0], b[0], rng), rng)
                s = a[1] if rng.random() < 0.5 else int(rng.integers(0, 100000))
                new.append((g, s))
            pop = new
    return best

def main():
    warm = json.loads(CHAIN_FILE.read_text())["chain"][-1][0]
    out = {}
    for speed in (0.06, 0.10, 0.14, 0.18, 0.22, 0.26, 0.30):
        rng = np.random.default_rng(751)
        b = ga(speed, warm, rng)
        out[str(speed)] = b
        print(f"speed {speed}: best near4 {b:.2f}", flush=True)
    (LAB / "h75_band.json").write_text(json.dumps(out))

if __name__ == "__main__":
    main()
