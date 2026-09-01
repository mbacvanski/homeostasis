"""H31a/b: open-loop bias-flip time vs closed-loop excursion accounting."""
from __future__ import annotations
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from common import HomeostaticReservoir, TrackingEnv, make_configs, scripted_theta

LAB = Path(__file__).resolve().parents[1] / "out" / "lab"
WIN = 30


def openloop_flip(task):
    wlr, seed = task
    rcfg, tcfg = make_configs({"weight_lr": wlr}, {})
    net = HomeostaticReservoir(rcfg, seed=seed)
    env = TrackingEnv(tcfg)
    offs = env._sensor_offsets
    L = net.output_adjacency[:, 0]; R = net.output_adjacency[:, 1]
    n1, n2 = 1500, 2500
    # slip +1 deg/step for n1, then -1 for n2, continuous
    th = np.empty(n1 + n2)
    cur = 0.0
    for i in range(n1 + n2):
        th[i] = (cur + 180.0) % 360.0 - 180.0
        cur += 1.0 if i < n1 else -1.0
    n = n1 + n2
    nw = n // WIN
    dfp = np.zeros(nw)
    for i in range(nw * WIN):
        d = np.abs((th[i] - offs + 180.0) % 360.0 - 180.0)
        acts = np.exp(-(d ** 2) / tcfg.tuning_width)
        acts[d <= tcfg.plateau_width] = 1.0
        state = net.step(acts)
        w = i // WIN
        dfp[w] += (float(state.spiked[L].mean()) - float(state.spiked[R].mean())) / WIN
    pre = dfp[(n1 // WIN) - 10:(n1 // WIN)]
    post = dfp[(n1 // WIN):]
    sgn = np.sign(pre.mean()) if abs(pre.mean()) > 1e-6 else 1.0
    flip = next((k * WIN for k, v in enumerate(post) if np.sign(v) == -sgn and abs(v) > 0.2 * abs(pre.mean())), -1)
    return dict(wlr=wlr, seed=seed, flip_steps=int(flip),
                pre_asym=float(pre.mean()), post_asym=float(post[-10:].mean()))


def main():
    tasks = [(wlr, s) for wlr in (0.03, 0.1, 0.3, 1.0) for s in range(8)]
    with ProcessPoolExecutor(10) as pool:
        rows = list(pool.map(openloop_flip, tasks))
    (LAB / "h31_openloop.json").write_text(json.dumps(rows))
    print("H31a open-loop pool-asymmetry flip after slip reversal:")
    for wlr in (0.03, 0.1, 0.3, 1.0):
        v = [r["flip_steps"] for r in rows if r["wlr"] == wlr and r["flip_steps"] >= 0]
        nn = [r for r in rows if r["wlr"] == wlr]
        pre = np.mean([abs(r["pre_asym"]) for r in nn])
        print(f"   wlr={wlr:<5} flip median {np.median(v) if v else float('nan'):6.0f} steps "
          f"(n={len(v)}/8)  |pre asym| {pre:.4f}")

    # H31b: excursion accounting from b6b
    b6b = json.loads((LAB / "b6b_reentrainment.json").read_text())
    print("\nH31b closed-loop: tau63 vs excursion depth (per config):")
    for name in ("wlr0.03", "wlr0.1", "wlr0.3", "wlr1.0", "w1prime"):
        taus, excs = [], []
        for r in [x for x in b6b if x["name"] == name]:
            c = np.array(r["curve"])
            asym = c[-8:].mean()
            if asym <= 0:
                continue
            tau = next((k * 30 for k, v in enumerate(c) if v >= 0.63 * asym), -1)
            neg = c[c < 0]
            exc = float(-(c[:np.argmax(c >= 0)] * 30).sum()) if (c < 0).any() and c[0] < 0 else 0.0
            taus.append(tau); excs.append(exc)
        print(f"   {name:8s} tau median {np.median(taus):5.0f}  excursion median {np.median(excs):6.1f} deg")
    # cross-seed correlation pooled
    taus, excs = [], []
    for r in b6b:
        c = np.array(r["curve"]); asym = c[-8:].mean()
        if asym <= 0.05:
            continue
        tau = next((k * 30 for k, v in enumerate(c) if v >= 0.63 * asym), -1)
        exc = float(-(c[:np.argmax(c >= 0)] * 30).sum()) if c[0] < 0 else 0.0
        if tau >= 0:
            taus.append(tau); excs.append(exc)
    ra = np.argsort(np.argsort(taus)).astype(float); rb = np.argsort(np.argsort(excs)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    rho = float((ra*rb).sum()/np.sqrt((ra**2).sum()*(rb**2).sum()))
    print(f"   pooled Spearman(tau63, excursion) over {len(taus)} runs: {rho:+.3f}")


if __name__ == "__main__":
    main()
