"""H82: WTA control + GA for emergent figure-ground."""
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
from h33_evolve_pursuit import mutate, crossover, random_genome  # noqa: E402
from h50_depth import PACE_CFG, LAB, RES_KEYS  # noqa: E402

CHAMP = json.loads((LAB / "h48e_warm.json").read_text())
A1_SEED, A2_SEED = 3, 33

def cosim(genome, net_seed, wta=False, n=7200):
    pc = PursuitConfig(eye_offsets=(0.0,), sensors_per_eye=91, box_size=30.0,
                       initial_agent_x=15.0, initial_agent_y=10.0,
                       wheel_base=genome["wheel_base"],
                       intensity_scale=genome["intensity_scale"])
    res = ReservoirConfig(n_inputs=pc.n_sensors, **{k: genome[k] for k in RES_KEYS})
    net = HomeostaticReservoir(res, seed=net_seed)
    env = PursuitEnv(pc, rng=net.rng)
    sims = [WallSimulation(wall_config=PACE_CFG, seed=s) for s in (A1_SEED, A2_SEED)]
    D = np.zeros((n, 2))
    for i in range(n):
        bumps = []
        for j, A in enumerate(sims):
            A.step()
            env.sx, env.sy = A.env.x, A.env.y
            D[i, j] = env.distance()
            bumps.append(env.sense())
        if wta:
            acts = bumps[0] if bumps[0].sum() >= bumps[1].sum() else bumps[1]
        else:
            acts = bumps[0] + bumps[1]
        st = net.step(acts)
        env.apply_action(*map(float, st.outputs))
        env.steps += 1
    late = D[n // 2:]
    lockA = float((late[:, 0] < 4).mean()); lockB = float((late[:, 1] < 4).mean())
    return lockA, lockB, float(late.min(axis=1).mean())

def evaluate(task):
    genome, seed = task
    a, b, dmin = cosim(genome, seed)
    return dict(fit=max(a, b) - dmin / 60.0, lockA=a, lockB=b)

def main():
    a, b, _ = cosim(CHAMP["champion"], CHAMP["champ_seed"], wta=True)
    print(f"WTA control: lock A1 {a:.3f} | A2 {b:.3f}", flush=True)
    rng = np.random.default_rng(82)
    wg, ws = CHAMP["champion"], CHAMP["champ_seed"]
    pop = [(dict(wg), ws)] + \
          [(mutate(dict(wg), rng), ws) for _ in range(4)] + \
          [(mutate(dict(wg), rng), int(rng.integers(0, 100000))) for _ in range(5)] + \
          [(random_genome(rng), int(rng.integers(0, 100000))) for _ in range(6)]
    best = None
    with ProcessPoolExecutor(10) as pool:
        for gen in range(8):
            rows = list(pool.map(evaluate, [(g, s) for g, s in pop], chunksize=1))
            fits = [r["fit"] for r in rows]
            bi = int(np.argmax(fits))
            if best is None or rows[bi]["fit"] > best["fit"]:
                best = dict(rows[bi], champion=pop[bi][0], champ_seed=pop[bi][1])
            print(f"gen {gen}: best lock {max(rows[bi]['lockA'], rows[bi]['lockB']):.3f}"
                  f" (A {rows[bi]['lockA']:.2f} / B {rows[bi]['lockB']:.2f})", flush=True)
            elite = pop[bi]
            new = [elite]
            while len(new) < len(pop):
                idx = rng.choice(len(pop), 3, replace=False)
                pa = pop[idx[int(np.argmax([fits[i] for i in idx]))]]
                idx = rng.choice(len(pop), 3, replace=False)
                pb = pop[idx[int(np.argmax([fits[i] for i in idx]))]]
                g = mutate(crossover(pa[0], pb[0], rng), rng)
                s = pa[1] if rng.random() < 0.5 else int(rng.integers(0, 100000))
                new.append((g, s))
            pop = new
    (LAB / "h82_attention.json").write_text(json.dumps(
        dict(wta=[a, b], best=dict(lockA=best["lockA"], lockB=best["lockB"],
                                   champion=best["champion"],
                                   champ_seed=best["champ_seed"]))))
    print(f"GA best: lock A {best['lockA']:.3f} / B {best['lockB']:.3f}")

if __name__ == "__main__" and "--sticky" not in sys.argv:
    main()

def cosim_sticky(genome, net_seed, n=10800, ratio=2.0, patience=100):
    pc = PursuitConfig(eye_offsets=(0.0,), sensors_per_eye=91, box_size=30.0,
                       initial_agent_x=15.0, initial_agent_y=10.0,
                       wheel_base=genome["wheel_base"],
                       intensity_scale=genome["intensity_scale"])
    res = ReservoirConfig(n_inputs=pc.n_sensors, **{k: genome[k] for k in RES_KEYS})
    net = HomeostaticReservoir(res, seed=net_seed)
    env = PursuitEnv(pc, rng=net.rng)
    sims = [WallSimulation(wall_config=PACE_CFG, seed=s) for s in (A1_SEED, A2_SEED)]
    D = np.zeros((n, 2))
    sel, streak, switches = 0, 0, 0
    for i in range(n):
        bumps = []
        for j, A in enumerate(sims):
            A.step()
            env.sx, env.sy = A.env.x, A.env.y
            D[i, j] = env.distance()
            bumps.append(env.sense())
        s0, s1 = bumps[0].sum(), bumps[1].sum()
        riv = 1 - sel
        riv_sum = (s0, s1)[riv]
        cur_sum = (s0, s1)[sel]
        streak = streak + 1 if riv_sum >= ratio * cur_sum else 0
        if streak >= patience:
            sel, streak, switches = riv, 0, switches + 1
        st = net.step(bumps[sel])
        env.apply_action(*map(float, st.outputs))
        env.steps += 1
    late = D[n // 2:]
    return (float((late[:, 0] < 4).mean()), float((late[:, 1] < 4).mean()), switches)

def main_sticky():
    a, b, sw = cosim_sticky(CHAMP["champion"], CHAMP["champ_seed"])
    print(f"sticky WTA: lock A1 {a:.3f} | A2 {b:.3f} | switches {sw}")
    (LAB / "h84_sticky.json").write_text(json.dumps(dict(lockA=a, lockB=b, switches=sw)))

if __name__ == "__main__" and "--sticky" in sys.argv:
    main_sticky()
