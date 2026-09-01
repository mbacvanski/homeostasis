"""H40: early-dynamics prediction of the wiring lottery.
H41: discomfort-triggered annealing (shuffle-when-nonstationary)."""
from __future__ import annotations
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from common import make_configs  # noqa: F401
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from homeostasis.reservoir import HomeostaticReservoir, ReservoirConfig  # noqa: E402
from homeostasis.pursuit import PursuitConfig, PursuitEnv  # noqa: E402

LAB = Path(__file__).resolve().parents[1] / "out" / "lab"
RES_KEYS = ("n_nodes", "p_link", "input_weight", "weight_init_mean",
            "leak", "target_lr", "threshold_ratio", "weight_lr")


def build(champ, seed):
    pc = PursuitConfig(eye_offsets=(0.0,), sensors_per_eye=91,
                       wheel_base=champ["wheel_base"],
                       intensity_scale=champ["intensity_scale"])
    res = ReservoirConfig(n_inputs=pc.n_sensors, **{k: champ[k] for k in RES_KEYS})
    net = HomeostaticReservoir(res, seed=seed)
    env = PursuitEnv(pc, rng=net.rng)
    return net, env


def h40_run(task):
    champ, seed = task
    net, env = build(champ, seed)
    n = 3600
    flow = np.empty(n); dist = np.empty(n)
    ang = np.empty(n)
    for i in range(n):
        dist[i] = env.distance()
        ang[i] = np.arctan2(env.y - env.sy, env.x - env.sx)
        state = net.step(env.sense())
        flow[i] = float(state.inputs.sum())
        env.apply_action(*map(float, state.outputs))
        env.advance_stimulus()
    near_final = float((dist[1800:] < 3).mean())
    rev = np.rad2deg(np.diff(np.unwrap(ang[:200]))).mean()
    return dict(seed=seed, near_final=near_final,
                early_flow=float(flow[:200].mean()),
                early_flow_trend=float(flow[100:200].mean() - flow[:100].mean()),
                early_rev=float(rev),
                early_dist_trend=float(dist[100:200].mean() - dist[:100].mean()))


def h41_run(task):
    champ, seed, anneal = task
    net, env = build(champ, seed)
    n = 14400
    dist = np.empty(n); flow_buf = []
    shuffles = 0
    for i in range(n):
        dist[i] = env.distance()
        state = net.step(env.sense())
        flow_buf.append(float(state.inputs.sum()))
        env.apply_action(*map(float, state.outputs))
        env.advance_stimulus()
        if anneal and (i + 1) % 600 == 0 and i < n - 1800:
            fb = np.array(flow_buf[-600:])
            if fb.mean() < 2.2:   # insufficient engagement (flow-seeking anneal)
                mask = net.adjacency
                vals = net.weights[mask]
                net.weights[mask] = net.rng.permutation(vals)
                shuffles += 1
    late = slice(n - 3600, None)
    return dict(seed=seed, anneal=bool(anneal),
                near_late=float((dist[late] < 3).mean()),
                dist_late=float(dist[late].mean()), shuffles=shuffles)


def main():
    champ = json.loads((LAB / "h34_joint.json").read_text())[-1]["champion"]
    seeds = [3000 + s for s in range(32)]
    with ProcessPoolExecutor(10) as pool:
        r40 = list(pool.map(h40_run, [(champ, s) for s in seeds], chunksize=2))
        r41 = list(pool.map(h41_run,
                            [(champ, 3000 + s, a) for a in (False, True) for s in range(16)],
                            chunksize=1))
    (LAB / "h40_h41.json").write_text(json.dumps(dict(h40=r40, h41=r41)))

    lock = np.array([r["near_final"] >= 0.8 for r in r40])
    print(f"H40: {lock.sum()}/32 wirings lock (near_final >= 0.8)")
    for k in ("early_flow", "early_flow_trend", "early_rev", "early_dist_trend"):
        v = np.array([r[k] for r in r40])
        # best single threshold accuracy
        accs = [max(((v < t) == lock).mean(), ((v >= t) == lock).mean())
                for t in np.percentile(v, np.linspace(5, 95, 37))]
        print(f"   {k:18s} lockers {v[lock].mean():+.3f}  failers {v[~lock].mean():+.3f}  "
              f"best split acc {max(accs):.2f}")

    for a in (False, True):
        sel = [r for r in r41 if r["anneal"] == a]
        locked = sum(r["near_late"] >= 0.8 for r in sel)
        print(f"H41 anneal={a}: locked {locked}/16  "
              f"near_late mean {np.mean([r['near_late'] for r in sel]):.2f}  "
              f"median {np.median([r['near_late'] for r in sel]):.2f}  "
              f"shuffles mean {np.mean([r['shuffles'] for r in sel]):.1f}")


if __name__ == "__main__":
    main()
