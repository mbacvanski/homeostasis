"""H42b: structural-rule variants on pursuit. H43: structural rule on the
tracking wander floor."""
from __future__ import annotations
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from common import ERR_EDGES, make_configs  # noqa: F401
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from homeostasis.reservoir import HomeostaticReservoir, ReservoirConfig  # noqa: E402
from homeostasis.pursuit import PursuitConfig, PursuitEnv  # noqa: E402
from homeostasis.tracking import TrackingConfig, TrackingEnv  # noqa: E402

LAB = Path(__file__).resolve().parents[1] / "out" / "lab"
RES_KEYS = ("n_nodes", "p_link", "input_weight", "weight_init_mean",
            "leak", "target_lr", "threshold_ratio", "weight_lr")


def structural_step(net, res, e_mean, deg_max, sensor_bias):
    deg = net.adjacency.sum(axis=0)
    changed = False
    grown = pruned = 0
    if sensor_bias:
        in_deg_input = net.input_adjacency.sum(axis=0)
        pool_mask = in_deg_input > np.median(in_deg_input)
    for node in range(net.config.n_nodes):
        if e_mean[node] < -0.03 if sensor_bias else e_mean[node] < -0.05:
            pass
    return changed, grown, pruned


def rewire(net, res, e_mean, deg_max, th, sensor_bias):
    changed = False
    grown = pruned = 0
    deg = net.adjacency.sum(axis=0)
    if sensor_bias:
        idi = net.input_adjacency.sum(axis=0)
        good_src = idi > np.median(idi)
    for node in range(net.config.n_nodes):
        if e_mean[node] < -th and deg[node] < deg_max:
            cands = np.flatnonzero(~net.adjacency[:, node])
            cands = cands[cands != node]
            if sensor_bias:
                pref = cands[good_src[cands]]
                cands = pref if pref.size else cands
            if cands.size:
                src = int(net.rng.choice(cands))
                net.adjacency[src, node] = True
                net.weights[src, node] = float(
                    net.rng.normal(res.weight_init_mean, res.weight_init_sd))
                grown += 1
                changed = True
        elif e_mean[node] > th and deg[node] > 2:
            ins = np.flatnonzero(net.adjacency[:, node])
            src = int(ins[np.argmin(np.abs(net.weights[ins, node]))])
            net.adjacency[src, node] = False
            net.weights[src, node] = 0.0
            pruned += 1
            changed = True
    return changed, grown, pruned


def pursuit_variant(task):
    champ, seed, name, win, th, bias = task
    dev_end = 3600 if name == "devwindow" else 10**9
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
    for i in range(n):
        dist[i] = env.distance()
        state = net.step(env.sense())
        e_acc += state.error
        env.apply_action(*map(float, state.outputs))
        env.advance_stimulus()
        if (i + 1) % win == 0 and i < dev_end:
            changed, _, _ = rewire(net, res, e_acc / win, deg_max, th, bias)
            e_acc[:] = 0.0
            if changed:
                net._rebuild_structure_caches()
    late = slice(n - 3600, None)
    return dict(seed=seed, name=name, near_late=float((dist[late] < 3).mean()))


def tracking_arm(task):
    seed, structural = task
    dev_end = 3600 if structural == "dev" else (10**9 if structural else -1)
    rcfg, tcfg = make_configs({"leak": 0.25, "weight_lr": 0.1}, {})
    net = HomeostaticReservoir(rcfg, seed=seed)
    env = TrackingEnv(tcfg)
    N = rcfg.n_nodes
    deg_max = max(4, int(3 * rcfg.p_link * N))
    n = 14400
    e_acc = np.zeros(N)
    in45 = np.zeros(n, bool)
    for i in range(n):
        in45[i] = abs(env.heading_error()) <= 45
        state = net.step(env.sense())
        e_acc += state.error
        e1, e2 = state.outputs
        env.apply_action(float(e1), float(e2))
        env.advance_stimulus()
        if (i + 1) % 120 == 0 and i < dev_end:
            changed, _, _ = rewire(net, rcfg, e_acc / 120, deg_max, 0.05, False)
            e_acc[:] = 0.0
            if changed:
                net._rebuild_structure_caches()
    segs = in45.reshape(-1, 720).mean(axis=1)
    return dict(seed=seed, structural=bool(structural),
                score_late=float(segs[10:].mean()), segs=segs.tolist())


def main():
    champ = json.loads((LAB / "h34_joint.json").read_text())[-1]["champion"]
    variants = [("gentle", 240, 0.03, False), ("sensor-bias", 120, 0.05, True),
                ("both", 240, 0.03, True)]
    tasks = [(champ, 3000 + s, n_, w, t, b) for (n_, w, t, b) in variants for s in range(16)]
    with ProcessPoolExecutor(10) as pool:
        rows = list(pool.map(pursuit_variant, tasks, chunksize=1))
        trows = list(pool.map(tracking_arm,
                              [(s, st) for st in (False, True) for s in range(24)],
                              chunksize=2))
    (LAB / "h42b_h43.json").write_text(json.dumps(dict(pursuit=rows, tracking=trows)))
    print("H42b pursuit variants (locked/16; baseline grow-prune was 3/16):")
    for (n_, *_ ) in variants:
        sel = [r for r in rows if r["name"] == n_]
        print(f"   {n_:12s} locked {sum(r['near_late'] >= 0.8 for r in sel)}/16  "
              f"mean {np.mean([r['near_late'] for r in sel]):.2f}")
    print("H43 tracking ridge25 x structural (24 seeds, segments 11-20):")
    for st in (False, True):
        sel = [r for r in trows if r["structural"] == st]
        v = np.array([r["score_late"] for r in sel])
        segs = np.array([r["segs"] for r in sel])[:, 10:]
        lag1 = np.corrcoef(segs[:, :-1].ravel(), segs[:, 1:].ravel())[0, 1]
        print(f"   structural={st}: score {v.mean():.3f}  frac>=0.35 {np.mean(v >= 0.35):.2f}  "
              f"seg lag-1 corr {lag1:+.2f}")


if __name__ == "__main__":
    main()
