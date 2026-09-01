"""H42: structural homeostasis (grow/prune) on the pursuit lottery."""
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
WIN = 120
E_GROW = -0.05
E_PRUNE = 0.05


def run_structural(task):
    champ, seed, structural = task
    pc = PursuitConfig(eye_offsets=(0.0,), sensors_per_eye=91,
                       wheel_base=champ["wheel_base"],
                       intensity_scale=champ["intensity_scale"])
    res = ReservoirConfig(n_inputs=pc.n_sensors, **{k: champ[k] for k in RES_KEYS})
    net = HomeostaticReservoir(res, seed=seed)
    env = PursuitEnv(pc, rng=net.rng)
    N = res.n_nodes
    deg_max = max(4, int(3 * res.p_link * N))
    n = 14400
    dist = np.empty(n)
    e_acc = np.zeros(N)
    grown = pruned = 0
    for i in range(n):
        dist[i] = env.distance()
        state = net.step(env.sense())
        e_acc += state.error
        env.apply_action(*map(float, state.outputs))
        env.advance_stimulus()
        if structural and (i + 1) % WIN == 0:
            e_mean = e_acc / WIN
            e_acc[:] = 0.0
            deg = net.adjacency.sum(axis=0)
            changed = False
            for node in range(N):
                if e_mean[node] < E_GROW and deg[node] < deg_max:
                    cands = np.flatnonzero(~net.adjacency[:, node])
                    cands = cands[cands != node]
                    if cands.size:
                        src = int(net.rng.choice(cands))
                        net.adjacency[src, node] = True
                        net.weights[src, node] = float(
                            net.rng.normal(res.weight_init_mean, res.weight_init_sd))
                        grown += 1
                        changed = True
                elif e_mean[node] > E_PRUNE and deg[node] > 2:
                    ins = np.flatnonzero(net.adjacency[:, node])
                    w = np.abs(net.weights[ins, node])
                    src = int(ins[np.argmin(w)])
                    net.adjacency[src, node] = False
                    net.weights[src, node] = 0.0
                    pruned += 1
                    changed = True
            if changed:
                net._rebuild_structure_caches()
    late = slice(n - 3600, None)
    return dict(seed=seed, structural=bool(structural),
                near_late=float((dist[late] < 3).mean()),
                dist_late=float(dist[late].mean()),
                grown=grown, pruned=pruned,
                final_mean_deg=float(net.adjacency.sum(axis=0).mean()))


def main():
    champ = json.loads((LAB / "h34_joint.json").read_text())[-1]["champion"]
    tasks = [(champ, 3000 + s, st) for st in (False, True) for s in range(16)]
    with ProcessPoolExecutor(10) as pool:
        rows = list(pool.map(run_structural, tasks, chunksize=1))
    (LAB / "h42_structural.json").write_text(json.dumps(rows))
    for st in (False, True):
        sel = [r for r in rows if r["structural"] == st]
        locked = sum(r["near_late"] >= 0.8 for r in sel)
        print(f"structural={st}: locked {locked}/16  "
              f"near mean {np.mean([r['near_late'] for r in sel]):.2f} "
              f"median {np.median([r['near_late'] for r in sel]):.2f}  "
              f"dist {np.mean([r['dist_late'] for r in sel]):.2f}"
              + (f"  grown {np.mean([r['grown'] for r in sel]):.0f} "
                 f"pruned {np.mean([r['pruned'] for r in sel]):.0f} "
                 f"deg {np.mean([r['final_mean_deg'] for r in sel]):.1f}" if st else ""))


if __name__ == "__main__":
    main()
