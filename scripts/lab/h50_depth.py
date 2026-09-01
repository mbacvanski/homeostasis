"""H50: extend the entrainment chain link by link."""
from __future__ import annotations
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from h33_evolve_pursuit import mutate, crossover, tournament, random_genome
from h48c_live_chain import PACE_CFG, PACE_SEED, LAB, RES_KEYS
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from homeostasis.reservoir import HomeostaticReservoir, ReservoirConfig  # noqa: E402
from homeostasis.simulation import WallSimulation  # noqa: E402
from homeostasis.pursuit import PursuitConfig, PursuitEnv  # noqa: E402

# chain[i] = (genome, seed) for follower i (0 = B). Loaded per depth stage.
CHAIN_FILE = LAB / "h50_chain.json"


def make_follower(genome, seed, start_y):
    pc = PursuitConfig(eye_offsets=(0.0,), sensors_per_eye=91, box_size=30.0,
                       initial_agent_x=15.0, initial_agent_y=start_y,
                       wheel_base=genome["wheel_base"],
                       intensity_scale=genome["intensity_scale"])
    res = ReservoirConfig(n_inputs=pc.n_sensors, **{k: genome[k] for k in RES_KEYS})
    net = HomeostaticReservoir(res, seed=seed)
    return net, PursuitEnv(pc, rng=net.rng)


START_Y = [10.0, 5.0, 3.0, 26.0, 24.0]  # per-link native/assigned starts

def cosim_chain(chain, cand, cand_seed, n=3600, jitter=False):
    """Run pacemaker + established chain + candidate follower of the tail."""
    A = WallSimulation(wall_config=PACE_CFG, seed=PACE_SEED)
    links = [make_follower(g, s, START_Y[i]) for i, (g, s) in enumerate(chain)]
    netX, envX = make_follower(cand, cand_seed, START_Y[len(chain)])
    dX = np.empty(n)
    dists = [np.empty(n) for _ in links]
    for i in range(n):
        A.step()
        tx, ty = A.env.x, A.env.y
        for j, (net, env) in enumerate(links):
            env.sx, env.sy = tx, ty
            dists[j][i] = env.distance()
            st = net.step(env.sense())
            env.apply_action(*map(float, st.outputs)); env.steps += 1
            tx, ty = env.x, env.y
        envX.sx, envX.sy = tx, ty
        dX[i] = envX.distance()
        st = netX.step(envX.sense())
        envX.apply_action(*map(float, st.outputs)); envX.steps += 1
    late = slice(n // 2, None)
    out = dict(near4=float((dX[late] < 4).mean()), dist=float(dX[late].mean()),
               sd=float(dX[late].std()))
    if jitter:
        out["link_sd"] = [float(d[late].std()) for d in dists] + [out["sd"]]
        out["link_dist"] = [float(d[late].mean()) for d in dists] + [out["dist"]]
    return out


def evaluate(task):
    chain, g, s = task
    r = cosim_chain(chain, g, s)
    return dict(fit=r["near4"] - r["dist"] / 30.0, **r)


def ga_link(chain, warm, rng, gens=8, pop_n=16):
    wg, ws = warm
    pop = [(dict(wg), ws)] + \
          [(mutate(dict(wg), rng), ws) for _ in range(4)] + \
          [(mutate(dict(wg), rng), int(rng.integers(0, 100000))) for _ in range(5)] + \
          [(random_genome(rng), int(rng.integers(0, 100000))) for _ in range(pop_n - 10)]
    best = None
    with ProcessPoolExecutor(10) as pool:
        for gen in range(gens):
            rows = list(pool.map(evaluate, [(chain, g, s) for g, s in pop], chunksize=1))
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
    b = json.loads((LAB / "h48e_warm.json").read_text())
    c = json.loads((LAB / "h49_chain3.json").read_text())
    chain = [(b["champion"], b["champ_seed"]), (c["champion"], c["champ_seed"])]
    rng = np.random.default_rng(113)
    names = ["D", "E"]
    for name in names:
        warm = chain[-1]
        best = ga_link(chain, warm, rng)
        print(f"link {name}: near4 {best['near4']:.2f} dist {best['dist']:.2f} "
              f"sd {best['sd']:.2f}", flush=True)
        if best["near4"] < 0.6:
            print(f"chain BREAKS at link {name}")
            break
        chain.append((best["champion"], best["champ_seed"]))
    # jitter profile of the final chain
    r = cosim_chain(chain[:-1], chain[-1][0], chain[-1][1], n=7200, jitter=True)
    print("per-link dist:", [round(v, 2) for v in r["link_dist"]])
    print("per-link sd  :", [round(v, 2) for v in r["link_sd"]])
    (CHAIN_FILE).write_text(json.dumps(dict(
        depth=len(chain) + 1, link_dist=r["link_dist"], link_sd=r["link_sd"],
        chain=[[g, s] for g, s in chain])))


if __name__ == "__main__":
    main()
