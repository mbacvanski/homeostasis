"""K4: is the seed lottery just the sign/slope of the wired-in reflex kernel?

For each wiring seed, compute — from the adjacency matrices alone, zero
simulation — the direct sensor→effector kernel

    D(s) = sum_n in_adj[s,n] * w_in * (out_adj[n,L]/deg_L - out_adj[n,R]/deg_R)

i.e. how much a spike wave from sensor s pushes the turn signal (L-R).
Tracking wants D increasing with the sensor's offset angle (stimulus to the
left → turn left), so the kernel SLOPE (regression of D on sensor offset) is
a structural predictor of per-seed tracking. Also a one-hop recurrent
correction D2 (through W_init) and the same at the w1' config.

Then run the actual task (48 seeds, defaults + w1') and correlate.

Preregistered (H13): slope sign predicts above/below-chance tracking;
|slope| correlates with |score - 0.25|; prediction is stronger at defaults
(reflex regime per K3) than at w1' (medium regime).
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from common import HomeostaticReservoir, make_configs, run_closed_loop

OUT = Path(__file__).resolve().parents[1] / "out"
LAB = OUT / "lab"
SEEDS = list(range(48))


def kernel(res_over, trk_over, seed):
    rcfg, tcfg = make_configs(dict(res_over), dict(trk_over))
    net = HomeostaticReservoir(rcfg, seed=seed)
    degs = np.maximum(net.output_adjacency.sum(axis=0), 1)
    eff = net.output_adjacency[:, 0] / degs[0] - net.output_adjacency[:, 1] / degs[1]
    D1 = net.input_weights @ eff                     # (n_sensors,)
    # one-hop recurrent correction: sensor -> node -> node -> effector
    gain2 = net.weights / np.maximum(net.config.threshold_ratio * net.targets, 1e-9)
    D2 = net.input_weights @ (gain2 @ eff)
    offs = tcfg.sensor_offsets
    o = (offs - offs.mean()) / offs.std()
    slope1 = float(np.dot(o, D1) / len(o))
    slope2 = float(np.dot(o, D1 + D2) / len(o))
    return slope1, slope2


def eval_task(task):
    r = run_closed_loop(task)
    return dict(seed=task["seed"], tag=task["_tag"], score=r["score_late"],
                dir_agree=r["dir_agree"], flow=r["input_flow"])


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    d = np.sqrt((ra**2).sum() * (rb**2).sum())
    return float((ra * rb).sum() / d) if d else 0.0


def main():
    w1p_cfg = json.loads((OUT / "sweep" / "results.json").read_text())["configs"][236]
    w1p_res = dict(n_nodes=w1p_cfg["n_nodes"], p_link=w1p_cfg["p_link"],
                   input_weight=w1p_cfg["input_weight"], weight_init_mean=0.75,
                   weight_init_sd=w1p_cfg["weight_init_sd"], leak=w1p_cfg["leak"],
                   target_lr=w1p_cfg["target_lr"], threshold_ratio=w1p_cfg["threshold_ratio"])
    w1p_trk = dict(gain=w1p_cfg["gain"])

    variants = {"default": ({}, {}), "wlr0.1": ({"weight_lr": 0.1}, {}),
                "w1prime": (w1p_res, w1p_trk)}
    tasks = []
    for name, (res, trk) in variants.items():
        for s in SEEDS:
            tasks.append(dict(res=dict(res), trk=dict(trk), seed=s, arm="full",
                              _tag=name, snap_every=7200))
    with ProcessPoolExecutor(10) as pool:
        rows = list(pool.map(eval_task, tasks, chunksize=4))

    report = {}
    for name, (res, trk) in variants.items():
        ks = np.array([kernel(res, trk, s) for s in SEEDS])
        sc = np.array([r["score"] for r in rows if r["tag"] == name])
        da = np.array([r["dir_agree"] for r in rows if r["tag"] == name])
        s1, s2 = ks[:, 0], ks[:, 1]
        excess = sc - 0.25
        print(f"\n══ {name}: 48 seeds, score {sc.mean():.3f}±{sc.std():.3f}, "
              f"frac≥0.35 {np.mean(sc >= 0.35):.2f}, frac<0.20 {np.mean(sc < 0.20):.2f}")
        print(f"   sign(slope1) vs sign(excess score): agree "
              f"{np.mean(np.sign(s1) == np.sign(excess)):.2f}")
        print(f"   Spearman(slope1, score)        {spearman(s1, sc):+.3f}")
        print(f"   Spearman(slope2, score)        {spearman(s2, sc):+.3f}")
        print(f"   Spearman(|slope1|, |excess|)   {spearman(np.abs(s1), np.abs(excess)):+.3f}")
        print(f"   Spearman(slope1, dir_agree)    {spearman(s1, da):+.3f}")
        report[name] = dict(slope1=s1.tolist(), slope2=s2.tolist(),
                            score=sc.tolist(), dir_agree=da.tolist())

    LAB.mkdir(exist_ok=True)
    (LAB / "k4_reflex_kernel.json").write_text(json.dumps(report))
    print(f"\nwrote {LAB/'k4_reflex_kernel.json'}")


if __name__ == "__main__":
    main()
