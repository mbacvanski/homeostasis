"""H49: three-agent chain — C follows B follows A."""
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

BWARM = json.loads((LAB / "h48e_warm.json").read_text())


def make_follower(genome, seed, start_y):
    pc = PursuitConfig(eye_offsets=(0.0,), sensors_per_eye=91, box_size=30.0,
                       initial_agent_x=15.0, initial_agent_y=start_y,
                       wheel_base=genome["wheel_base"],
                       intensity_scale=genome["intensity_scale"])
    res = ReservoirConfig(n_inputs=pc.n_sensors, **{k: genome[k] for k in RES_KEYS})
    net = HomeostaticReservoir(res, seed=seed)
    return net, PursuitEnv(pc, rng=net.rng)


def cosim3(genomeC, seedC, n=3600, record=False):
    A = WallSimulation(wall_config=PACE_CFG, seed=PACE_SEED)
    netB, envB = make_follower(BWARM["champion"], BWARM["champ_seed"], 10.0)
    netC, envC = make_follower(genomeC, seedC, 5.0)
    dBC = np.empty(n)
    tr = {"ax": [], "ay": [], "bx": [], "by": [], "cx": [], "cy": []} if record else None
    for i in range(n):
        A.step()
        envB.sx, envB.sy = A.env.x, A.env.y
        sB = netB.step(envB.sense())
        envB.apply_action(*map(float, sB.outputs)); envB.steps += 1
        envC.sx, envC.sy = envB.x, envB.y
        dBC[i] = envC.distance()
        sC = netC.step(envC.sense())
        envC.apply_action(*map(float, sC.outputs)); envC.steps += 1
        if record and i % 3 == 0:
            tr["ax"].append(A.env.x); tr["ay"].append(A.env.y)
            tr["bx"].append(envB.x); tr["by"].append(envB.y)
            tr["cx"].append(envC.x); tr["cy"].append(envC.y)
    late = slice(n // 2, None)
    out = dict(near4=float((dBC[late] < 4).mean()), dist=float(dBC[late].mean()))
    if record:
        out["traj"] = tr
    return out


def evaluate(task):
    g, s = task
    r = cosim3(g, s)
    return dict(fit=r["near4"] - r["dist"] / 30.0, **r)


def main():
    rng = np.random.default_rng(103)
    wg, ws = BWARM["champion"], BWARM["champ_seed"]
    pop = [(dict(wg), ws)] + \
          [(mutate(dict(wg), rng), ws) for _ in range(5)] + \
          [(mutate(dict(wg), rng), int(rng.integers(0, 100000))) for _ in range(6)] + \
          [(random_genome(rng), int(rng.integers(0, 100000))) for _ in range(8)]
    best = None
    with ProcessPoolExecutor(10) as pool:
        for gen in range(10):
            rows = list(pool.map(evaluate, pop, chunksize=1))
            fits = [r["fit"] for r in rows]
            bi = int(np.argmax(fits))
            if best is None or rows[bi]["fit"] > best["fit"]:
                best = dict(rows[bi], gen=gen, champion=pop[bi][0], champ_seed=pop[bi][1])
            print(f"gen {gen:02d}  C-near4 {rows[bi]['near4']:.2f} dist {rows[bi]['dist']:.2f}",
                  flush=True)
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
    long = cosim3(best["champion"], best["champ_seed"], n=10800, record=True)
    best["long_near4"] = long["near4"]; best["long_dist"] = long["dist"]
    (LAB / "h49_chain3.json").write_text(json.dumps(
        {k: v for k, v in best.items() if k != "traj"}))
    print(f"BEST: C-near4 {best['near4']:.2f} | LONG: {long['near4']:.2f} dist {long['dist']:.2f}")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tr = long["traj"]
    half = len(tr["ax"]) // 2
    fig, ax = plt.subplots(figsize=(6.8, 6.8))
    ax.plot(tr["ax"][half:], tr["ay"][half:], "-", lw=2.0, color="tab:red", label="A pacemaker (blind)")
    ax.plot(tr["bx"][half:], tr["by"][half:], "-", lw=1.1, color="tab:blue", label="B follows A")
    ax.plot(tr["cx"][half:], tr["cy"][half:], "-", lw=0.8, color="tab:green", label="C follows B")
    ax.set_xlim(0, 30); ax.set_ylim(0, 30); ax.set_aspect("equal")
    ax.legend(fontsize=8)
    ax.set_title(f"Three-agent chain (late half): C-B dist {long['dist']:.2f}")
    fig.tight_layout(); fig.savefig(LAB / "fig_chain3.png", dpi=130)
    print("wrote fig_chain3.png")


if __name__ == "__main__":
    main()
