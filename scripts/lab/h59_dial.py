"""H59: predict network spike rate from the wiring file alone; then design to a prescribed rate."""
from __future__ import annotations
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from homeostasis.reservoir import HomeostaticReservoir  # noqa: E402
from common import make_configs  # noqa: E402

LAB = Path(__file__).resolve().parents[1] / "out" / "lab"

def stationary_acts(tcfg):
    offs = np.concatenate([eye + np.arange(-(tcfg.sensors_per_eye - 1) / 2.0,
                                           (tcfg.sensors_per_eye + 1) / 2.0) * tcfg.sensor_spacing
                           for eye in tcfg.eye_offsets]) if hasattr(tcfg, "eye_offsets") else None
    d = np.abs((0.0 - tcfg.sensor_offsets + 180.0) % 360.0 - 180.0)
    acts = np.exp(-(d ** 2) / tcfg.tuning_width)
    acts[d <= tcfg.plateau_width] = 1.0
    return acts

def predict_f(net, acts, leak, rho):
    """Per-node vector fixed point, iterated from the cold-start basin f=0."""
    mu_in = acts @ net.input_weights
    W = net.weights
    T = net.targets
    f = np.zeros_like(T)
    for _ in range(600):
        drive = mu_in + f @ W
        # Law 2 (duty) gated by Law 1 (reachability: x* = drive/leak must
        # reach the threshold rho*T from a cold start), capped at 1/step.
        f_new = np.where(drive >= leak * rho * T,
                         np.clip((drive / T - leak) / rho, 0.0, 1.0), 0.0)
        f = f + 0.3 * (f_new - f)
    return float(f.mean())

def run_cell(task):
    res_over, seed, n_steps = task["res"], task["seed"], 3000
    rcfg, tcfg = make_configs({**res_over, "weight_lr": 0.0, "target_lr": 0.0}, {})
    net = HomeostaticReservoir(rcfg, seed=seed)
    net.learning_enabled = False
    acts = stationary_acts(tcfg)
    fp = predict_f(net, acts, rcfg.leak, rcfg.threshold_ratio)
    fs = []
    for i in range(n_steps):
        st = net.step(acts)
        if i >= n_steps // 2:
            fs.append(st.prop_spiked)
    return dict(seed=seed, res=res_over, f_pred=fp, f_meas=float(np.mean(fs)))

def main():
    designs = [
        {"target_init": 1.0}, {"target_init": 2.0}, {"target_init": 4.0},
        {"input_weight": 1.0}, {"input_weight": 4.0},
        {"leak": 0.5, "target_init": 2.0},
    ]
    tasks = [dict(res=d, seed=s) for d in designs for s in range(12)]
    with ProcessPoolExecutor(10) as pool:
        rows = list(pool.map(run_cell, tasks, chunksize=2))
    (LAB / "h59_dial.json").write_text(json.dumps(rows))
    P = np.array([r["f_pred"] for r in rows])
    M = np.array([r["f_meas"] for r in rows])
    ok = M > 1e-4
    rel = np.abs(P[ok] - M[ok]) / M[ok]
    print(f"prediction: r = {np.corrcoef(P, M)[0,1]:.4f}, median rel err {np.median(rel)*100:.1f}%"
          f"  (n={len(rows)}, silent cells {np.sum(~ok)})")
    for d in designs:
        sel = [r for r in rows if r["res"] == d]
        print(f"  {str(d):<42} pred {np.mean([r['f_pred'] for r in sel]):.3f}"
              f"  meas {np.mean([r['f_meas'] for r in sel]):.3f}")

if __name__ == "__main__":
    main()

# ---- part (b): design to prescribed f* by inverting the model per seed ----

def _predict_for(res_over, seed):
    rcfg, tcfg = make_configs({**res_over, "weight_lr": 0.0, "target_lr": 0.0}, {})
    net = HomeostaticReservoir(rcfg, seed=seed)
    return predict_f(net, stationary_acts(tcfg), rcfg.leak, rcfg.threshold_ratio)

def design(task):
    fstar, seed, route = task
    lo, hi = (1.0, 8.0)
    for _ in range(28):
        mid = 0.5 * (lo + hi)
        over = {"target_init": mid} if route == "T" else {"input_weight": mid, "target_init": 3.0}
        fp = _predict_for(over, seed)
        # f decreases in T, increases in input weight
        if route == "T":
            lo, hi = (lo, mid) if fp < fstar else (mid, hi)
        else:
            lo, hi = (mid, hi) if fp < fstar else (lo, mid)
    knob = 0.5 * (lo + hi)
    over = {"target_init": knob} if route == "T" else {"input_weight": knob, "target_init": 3.0}
    row = run_cell(dict(res=over, seed=seed))
    row.update(fstar=fstar, route=route, knob=knob)
    return row

def main_b():
    tasks = [(fs, s, r) for fs in (0.05, 0.2, 0.4) for r in ("T", "IW")
             for s in range(12)]
    with ProcessPoolExecutor(10) as pool:
        rows = list(pool.map(design, tasks, chunksize=2))
    (LAB / "h59_design.json").write_text(json.dumps(rows))
    print("design-to-rate: f* | route | model-pred | measured (per-seed mean+-sd) | hit+-20%")
    for fs in (0.05, 0.2, 0.4):
        for r in ("T", "IW"):
            sel = [x for x in rows if x["fstar"] == fs and x["route"] == r]
            m = np.array([x["f_meas"] for x in sel])
            p = np.array([x["f_pred"] for x in sel])
            hit = np.mean(np.abs(m - fs) <= 0.2 * fs)
            print(f"  f*={fs:<5} {r:<3} pred {p.mean():.3f}  meas {m.mean():.3f}+-{m.std():.3f}"
                  f"  hit {hit:.2f}  knob~{np.mean([x['knob'] for x in sel]):.2f}")

if __name__ == "__main__":
    import sys as _s
    if "--design" in _s.argv:
        main_b()
