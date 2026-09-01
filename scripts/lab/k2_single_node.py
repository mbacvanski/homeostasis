"""K2: single-node bifurcation scan — the basins the mean-field predicts.

A single node (no recurrence possible), constant drive mu per step, swept over
mu x leak x threshold_ratio from two initial conditions (cold x0=0; hot
x0 = 3*mu/leak). End states classified over the last 500 of 3000 steps:

  dead-floor      f=0, T at floor, E<0 forever (uncomfortable)
  silent-comf     f=0, |E| -> 0 (T adapted to x* = mu/leak)
  spiking         f>0, cycle-mean E ~ 0 (marginal branch; T free/slow)
  frozen-cycle    f>0, |E| large but T immobile (mode-locked discomfort)

Preregistered (ledger H4): cold start silent iff mu < rho*leak*T0 (no initial
threshold crossing); silent-comfortable iff also mu >= leak*T_floor; sustained
spiking above; hot/cold disagreement (bistability) somewhere in the
mu in (leak*T_floor*rho_regionish) band.
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from common import HomeostaticReservoir, ReservoirConfig  # noqa: F401

LAB = Path(__file__).resolve().parents[1] / "out" / "lab"
STEPS = 3000
TAIL = 500


def evaluate(task):
    mu, leak, rho, hot = task
    cfg = ReservoirConfig(
        n_nodes=1, n_inputs=1, n_outputs=1, p_link=0.0, input_p_link=1.0,
        input_weight=mu, leak=leak, threshold_ratio=rho,
    )
    net = HomeostaticReservoir(cfg, seed=0)
    if hot:
        net.x[:] = 3.0 * mu / max(leak, 1e-9)
    inp = np.ones(1)
    f = np.empty(STEPS)
    E = np.empty(STEPS)
    T = np.empty(STEPS)
    for i in range(STEPS):
        s = net.step(inp)
        f[i] = s.prop_spiked
        E[i] = float(s.error[0])
        T[i] = float(s.targets[0])
    f_l, E_l, T_l = f[-TAIL:], E[-TAIL:], T[-TAIL:]
    f_late = float(f_l.mean())
    mean_E = float(E_l.mean())
    mean_absE = float(np.abs(E_l).mean())
    T_drift = float(abs(T_l[-1] - T_l[0]))
    at_floor = bool(T_l[-1] <= cfg.target_floor + 1e-9)
    if f_late == 0.0:
        state = "dead-floor" if (at_floor and mean_E < -0.02) else "silent-comf"
    else:
        state = "spiking" if abs(mean_E) < 0.02 else "frozen-cycle"
    return dict(mu=mu, leak=leak, rho=rho, hot=bool(hot), state=state,
                f=round(f_late, 4), meanE=round(mean_E, 4), absE=round(mean_absE, 4),
                T=round(float(T_l[-1]), 4), T_drift=round(T_drift, 5))


def main():
    mus = np.geomspace(0.05, 20.0, 25)
    leaks = [0.05, 0.1, 0.25, 0.5, 0.75]
    rhos = [1.2, 1.5, 2.0, 3.0, 4.0]
    tasks = [(float(m), l, r, h) for m in mus for l in leaks for r in rhos for h in (False, True)]
    with ProcessPoolExecutor(10) as pool:
        rows = list(pool.map(evaluate, tasks, chunksize=16))

    LAB.mkdir(exist_ok=True)
    (LAB / "k2_single_node.json").write_text(json.dumps(rows))

    # Test the preregistered boundaries at rho=2, both starts
    print(f"K2: {len(rows)} runs. State counts:")
    from collections import Counter
    print("  ", Counter(r["state"] for r in rows))

    print("\n── cold-start boundary check (prediction: spikes iff mu > rho*leak*T0):")
    ok = tot = 0
    for r in rows:
        if r["hot"]:
            continue
        predicted_spiking = r["mu"] > r["rho"] * r["leak"] * 1.0
        actually = r["state"] in ("spiking", "frozen-cycle")
        ok += predicted_spiking == actually
        tot += 1
    print(f"   accuracy {ok}/{tot} = {ok/tot:.3f}")

    print("\n── silent-comfortable vs dead-floor (prediction: comf iff mu >= leak*T_floor):")
    ok = tot = 0
    for r in rows:
        if r["state"] not in ("silent-comf", "dead-floor"):
            continue
        pred_comf = r["mu"] >= r["leak"] * 1.0
        ok += pred_comf == (r["state"] == "silent-comf")
        tot += 1
    print(f"   accuracy {ok}/{tot} = {ok/tot:.3f}" if tot else "   (no silent cells)")

    print("\n── duty law on spiking cells: f vs (mu/T - leak)/rho:")
    sp = [r for r in rows if r["state"] == "spiking"]
    if sp:
        pred = np.array([(r["mu"] / r["T"] - r["leak"]) / r["rho"] for r in sp])
        act = np.array([r["f"] for r in sp])
        resid = act - pred
        print(f"   n={len(sp)}  median |f - pred| = {np.median(np.abs(resid)):.4f}  "
              f"(f range {act.min():.3f}-{act.max():.3f})")

    print("\n── bistability (hot vs cold end-state disagreement):")
    by_key = {}
    for r in rows:
        by_key.setdefault((r["mu"], r["leak"], r["rho"]), {})[r["hot"]] = r["state"]
    dis = [(k, v[False], v[True]) for k, v in by_key.items() if v[False] != v[True]]
    print(f"   {len(dis)}/{len(by_key)} cells disagree")
    for k, c, h in dis[:8]:
        print(f"     mu={k[0]:.3f} leak={k[1]} rho={k[2]}: cold={c} hot={h}")

    print("\n── frozen-cycle census (the period-locked discomfort states):")
    fr = [r for r in rows if r["state"] == "frozen-cycle"]
    if fr:
        print(f"   n={len(fr)}; leak values: {sorted(set(r['leak'] for r in fr))}; "
              f"|E| median {np.median([r['absE'] for r in fr]):.3f}")
    print(f"\nwrote {LAB/'k2_single_node.json'}")


if __name__ == "__main__":
    main()
