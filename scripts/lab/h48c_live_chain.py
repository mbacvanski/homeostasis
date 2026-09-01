"""H48c: the live homeostatic ecology — evolve a follower of a LIVE wall circler."""
from __future__ import annotations
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from common import make_configs  # noqa: F401
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from homeostasis.reservoir import HomeostaticReservoir, ReservoirConfig  # noqa: E402
from homeostasis.simulation import WallSimulation  # noqa: E402
from homeostasis.wall import WallConfig  # noqa: E402
from homeostasis.pursuit import PursuitConfig, PursuitEnv  # noqa: E402
from h33_evolve_pursuit import random_genome, mutate, crossover, tournament  # noqa: E402

LAB = Path(__file__).resolve().parents[1] / "out" / "lab"
RES_KEYS = ("n_nodes", "p_link", "input_weight", "weight_init_mean",
            "leak", "target_lr", "threshold_ratio", "weight_lr")
PACE_CFG = WallConfig(box_size=30.0, initial_x=15.0, initial_y=15.0, wheel_base=2.5)
PACE_SEED = 3


def cosim(genome, bseed, n=3600, record=False):
    A = WallSimulation(wall_config=PACE_CFG, seed=PACE_SEED)
    pc = PursuitConfig(eye_offsets=(0.0,), sensors_per_eye=91, box_size=30.0,
                       initial_agent_x=15.0, initial_agent_y=10.0,
                       wheel_base=genome["wheel_base"],
                       intensity_scale=genome["intensity_scale"])
    res = ReservoirConfig(n_inputs=pc.n_sensors, **{k: genome[k] for k in RES_KEYS})
    net = HomeostaticReservoir(res, seed=bseed)
    envB = PursuitEnv(pc, rng=net.rng)
    dist = np.empty(n)
    tr = {"ax": [], "ay": [], "bx": [], "by": []} if record else None
    for i in range(n):
        A.step()
        envB.sx, envB.sy = A.env.x, A.env.y
        dist[i] = envB.distance()
        state = net.step(envB.sense())
        envB.apply_action(*map(float, state.outputs))
        envB.steps += 1
        if record and i % 3 == 0:
            tr["ax"].append(A.env.x); tr["ay"].append(A.env.y)
            tr["bx"].append(envB.x); tr["by"].append(envB.y)
    late = slice(n // 2, None)
    out = dict(near4=float((dist[late] < 4).mean()), dist=float(dist[late].mean()))
    if record:
        out["traj"] = tr
    return out


def evaluate(task):
    genome, seed = task
    r = cosim(genome, seed)
    return dict(fit=r["near4"] - r["dist"] / 30.0, **r)


def main():
    rng = np.random.default_rng(83)
    pop = [(random_genome(rng), int(rng.integers(0, 100000))) for _ in range(20)]
    best = None
    with ProcessPoolExecutor(10) as pool:
        for gen in range(10):
            rows = list(pool.map(evaluate, pop, chunksize=1))
            fits = [r["fit"] for r in rows]
            bi = int(np.argmax(fits))
            if best is None or rows[bi]["fit"] > best["fit"]:
                best = dict(rows[bi], gen=gen, champion=pop[bi][0], champ_seed=pop[bi][1])
            print(f"gen {gen:02d}  near4 {rows[bi]['near4']:.2f} dist {rows[bi]['dist']:.2f}",
                  flush=True)
            elite = pop[bi]
            new = [elite]
            while len(new) < len(pop):
                pa = tournament(pop, fits, rng); pb = tournament(pop, fits, rng)
                g = mutate(crossover(pa[0], pb[0], rng), rng)
                s = pa[1] if rng.random() < 0.5 else pb[1]
                if rng.random() < 0.25:
                    s = int(rng.integers(0, 100000))
                new.append((g, int(s)))
            pop = new
    long = cosim(best["champion"], best["champ_seed"], n=10800, record=True)
    best["long_near4"] = long["near4"]; best["long_dist"] = long["dist"]
    (LAB / "h48c_chain.json").write_text(json.dumps(best))
    print(f"BEST gen {best['gen']}: near4 {best['near4']:.2f} dist {best['dist']:.2f} | "
          f"LONG 10800: near4 {long['near4']:.2f} dist {long['dist']:.2f}")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tr = long["traj"]
    half = len(tr["ax"]) // 2
    fig, ax = plt.subplots(figsize=(6.6, 6.6))
    ax.plot(tr["ax"][half:], tr["ay"][half:], "-", lw=1.6, color="tab:red",
            label="pacemaker (wall circler, blind)")
    ax.plot(tr["bx"][half:], tr["by"][half:], "-", lw=0.9, color="tab:blue",
            label="follower (senses pacemaker)")
    ax.set_xlim(0, 30); ax.set_ylim(0, 30); ax.set_aspect("equal")
    ax.legend(fontsize=8)
    ax.set_title(f"The live homeostatic ecology (late half): dist {long['dist']:.2f}")
    fig.tight_layout(); fig.savefig(LAB / "fig_live_chain.png", dpi=130)
    print("wrote fig_live_chain.png")


if __name__ == "__main__":
    main()
