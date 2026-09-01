"""H68: GA followers onto full replays of D's vs A's recorded trajectories."""
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
from h33_evolve_pursuit import mutate, crossover, tournament, random_genome  # noqa: E402
from h50_depth import CHAIN_FILE, PACE_CFG, PACE_SEED, START_Y, make_follower, LAB, RES_KEYS  # noqa: E402

def record_trajs(n=7200):
    chain = [(g, s) for g, s in json.loads(CHAIN_FILE.read_text())["chain"]]
    A = WallSimulation(wall_config=PACE_CFG, seed=PACE_SEED)
    links = [make_follower(g, s, START_Y[i]) for i, (g, s) in enumerate(chain)]
    TA = np.zeros((n, 2)); TD = np.zeros((n, 2))
    for i in range(n):
        A.step()
        tx, ty = A.env.x, A.env.y
        TA[i] = (tx, ty)
        for j, (net, env) in enumerate(links):
            env.sx, env.sy = tx, ty
            st = net.step(env.sense())
            env.apply_action(*map(float, st.outputs)); env.steps += 1
            tx, ty = env.x, env.y
        TD[i] = (tx, ty)
    return TA, TD, chain[-1][0]

TA, TD, D_GENOME = record_trajs()
TRAJS = {"A": TA, "D": TD}

def evaluate(task):
    which, genome, seed = task
    traj = TRAJS[which]
    pc = PursuitConfig(eye_offsets=(0.0,), sensors_per_eye=91, box_size=30.0,
                       initial_agent_x=15.0, initial_agent_y=START_Y[3],
                       wheel_base=genome["wheel_base"],
                       intensity_scale=genome["intensity_scale"])
    res = ReservoirConfig(n_inputs=pc.n_sensors, **{k: genome[k] for k in RES_KEYS})
    net = HomeostaticReservoir(res, seed=seed)
    env = PursuitEnv(pc, rng=net.rng)
    n = len(traj)
    dist = np.empty(n)
    for i in range(n):
        env.sx, env.sy = float(traj[i, 0]), float(traj[i, 1])
        dist[i] = env.distance()
        st = net.step(env.sense())
        env.apply_action(*map(float, st.outputs))
        env.steps += 1
    late = slice(n // 2, None)
    near = float((dist[late] < 4).mean())
    return dict(fit=near - float(dist[late].mean()) / 30.0, near4=near,
                dist=float(dist[late].mean()))

def ga(which, rng, gens=8, pop_n=16):
    wg = dict(D_GENOME)
    ws = 41414
    pop = [(wg, ws)] + \
          [(mutate(dict(wg), rng), ws) for _ in range(4)] + \
          [(mutate(dict(wg), rng), int(rng.integers(0, 100000))) for _ in range(5)] + \
          [(random_genome(rng), int(rng.integers(0, 100000))) for _ in range(pop_n - 10)]
    best = None
    with ProcessPoolExecutor(10) as pool:
        for gen in range(gens):
            rows = list(pool.map(evaluate, [(which, g, s) for g, s in pop], chunksize=1))
            fits = [r["fit"] for r in rows]
            bi = int(np.argmax(fits))
            if best is None or rows[bi]["fit"] > best["fit"]:
                best = dict(rows[bi], champion=pop[bi][0], champ_seed=pop[bi][1])
            elite = pop[bi]
            new = [elite]
            while len(new) < len(pop):
                pa = tournament(pop, fits, rng); pb = tournament(pop, fits, rng)
                g = mutate(crossover(pa[0], pb[0], rng), rng)
                s = pa[1] if rng.random() < 0.5 else pb[1]
                if rng.random() < 0.2:
                    s = int(rng.integers(0, 100000))
                new.append((g, int(s)))
            pop = new
    return best

def main():
    out = {}
    for which in ("A", "D"):
        rng = np.random.default_rng(681)
        best = ga(which, rng)
        out[which] = dict(near4=best["near4"], dist=best["dist"])
        print(f"replay-{which}: near4 {best['near4']:.2f} dist {best['dist']:.2f}")
    (LAB / "h68_replay_d.json").write_text(json.dumps(out))

if __name__ == "__main__":
    main()
