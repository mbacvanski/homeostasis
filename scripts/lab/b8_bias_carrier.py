"""B8: which pathway carries the turn bias — learned recurrence or retinal lag?

During closed-loop tracking, the turn command is gain*(f_L - f_R) where f_o
is pool o's spike fraction. By Law 2, pool duty differences are slaved to
pool drive/target differences. Decompose, per 30-step window and per pool:

    duty_in  = mean over pool nodes of (input drive)/(rho*T)
    duty_rec = mean over pool nodes of (recurrent drive)/(rho*T)

and ask which difference (L-R) tracks the stimulus direction:
  - W-stored bias hypothesis: Delta duty_rec flips with direction over the
    re-entrainment timescale (the bias lives in the weights);
  - lag-servo hypothesis: Delta duty_in flips with direction (the agent
    trails the stimulus, the off-center retina drives the pools directly),
    and the mean heading error (lag) has direction-dependent sign.

Preregistered (H20): at w1' the recurrent component dominates (the follower
is genuinely W-entrained); the input component contributes with the sign of
the lag; at defaults (churn) neither correlates strongly (score is low).
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from common import HomeostaticReservoir, TrackingEnv, make_configs

OUT = Path(__file__).resolve().parents[1] / "out"
LAB = OUT / "lab"
WIN = 30


def w1p():
    cfg = json.loads((OUT / "sweep" / "results.json").read_text())["configs"][236]
    return (dict(n_nodes=cfg["n_nodes"], p_link=cfg["p_link"],
                 input_weight=cfg["input_weight"], weight_init_mean=0.75,
                 weight_init_sd=cfg["weight_init_sd"], leak=cfg["leak"],
                 target_lr=cfg["target_lr"], threshold_ratio=cfg["threshold_ratio"]),
            dict(gain=cfg["gain"]))


def evaluate(task):
    name, res_over, trk_over, seed = task
    rcfg, tcfg = make_configs(res_over, trk_over)
    net = HomeostaticReservoir(rcfg, seed=seed)
    env = TrackingEnv(tcfg)
    L = net.output_adjacency[:, 0]
    R = net.output_adjacency[:, 1]
    rho = rcfg.threshold_ratio
    n_steps = 7200
    n_win = n_steps // WIN
    rec = {k: np.zeros(n_win) for k in
           ("din_L", "din_R", "drec_L", "drec_R", "f_L", "f_R", "dh",
            "dir", "herr")}
    for i in range(n_steps):
        w = i // WIN
        rec["dir"][w] += env.stimulus_direction / WIN
        rec["herr"][w] += env.heading_error() / WIN
        inputs = env.sense()
        din = inputs @ net.input_weights
        drec = net._spiked_f @ net.weights
        T = net.targets
        state = net.step(inputs)
        rec["din_L"][w] += (din[L] / (rho * T[L])).mean() / WIN
        rec["din_R"][w] += (din[R] / (rho * T[R])).mean() / WIN
        rec["drec_L"][w] += (drec[L] / (rho * T[L])).mean() / WIN
        rec["drec_R"][w] += (drec[R] / (rho * T[R])).mean() / WIN
        rec["f_L"][w] += float(state.spiked[L].mean()) / WIN
        rec["f_R"][w] += float(state.spiked[R].mean()) / WIN
        e_left, e_right = state.outputs
        dh = env.apply_action(e_left, e_right)
        rec["dh"][w] += dh / WIN
        env.advance_stimulus()

    d_in = rec["din_L"] - rec["din_R"]
    d_rec = rec["drec_L"] - rec["drec_R"]
    d_f = rec["f_L"] - rec["f_R"]
    direc = np.sign(rec["dir"])
    keep = np.abs(rec["dir"]) > 0.9  # windows without a reversal inside
    def corr(a, b):
        a, b = a[keep], b[keep]
        if a.std() < 1e-12 or b.std() < 1e-12:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])
    score = float(np.mean(np.abs(rec["herr"]) <= 45))
    return dict(
        name=name, seed=seed, score=score,
        corr_df_dh=corr(d_f, rec["dh"]),
        corr_din_dir=corr(d_in, direc),
        corr_drec_dir=corr(d_rec, direc),
        corr_df_dir=corr(d_f, direc),
        corr_herr_dir=corr(rec["herr"], direc),
        mean_lag_ccw=float(rec["herr"][keep & (direc > 0)].mean()),
        mean_lag_cw=float(rec["herr"][keep & (direc < 0)].mean()),
        sd_din=float(d_in[keep].std()), sd_drec=float(d_rec[keep].std()),
        beta_in=corr(d_f, d_in), beta_rec=corr(d_f, d_rec),
    )


def main():
    w1p_res, w1p_trk = w1p()
    variants = {"w1prime": (w1p_res, w1p_trk),
                "ridge25": ({"leak": 0.25, "weight_lr": 0.1}, {}),
                "default": ({}, {})}
    tasks = [(n, r, t, s) for n, (r, t) in variants.items() for s in range(6)]
    with ProcessPoolExecutor(10) as pool:
        rows = list(pool.map(evaluate, tasks))
    (LAB / "b8_bias_carrier.json").write_text(json.dumps(rows))

    for name in variants:
        sel = [r for r in rows if r["name"] == name]
        def m(k):
            return np.mean([r[k] for r in sel])
        print(f"\n══ {name} (6 seeds, score {m('score'):.3f})")
        print(f"   sanity: corr(Δf, dH) = {m('corr_df_dh'):+.3f}   corr(Δf, direction) = {m('corr_df_dir'):+.3f}")
        print(f"   carrier: corr(Δduty_REC, dir) = {m('corr_drec_dir'):+.3f}   "
              f"corr(Δduty_IN, dir) = {m('corr_din_dir'):+.3f}")
        print(f"   couplings to Δf: rec {m('beta_rec'):+.3f}  in {m('beta_in'):+.3f}   "
              f"(component SDs: rec {m('sd_drec'):.4f}, in {m('sd_din'):.4f})")
        print(f"   lag: herr mean CCW {m('mean_lag_ccw'):+.1f}° / CW {m('mean_lag_cw'):+.1f}°  "
              f"corr(herr, dir) = {m('corr_herr_dir'):+.3f}")


if __name__ == "__main__":
    main()
