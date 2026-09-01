"""B6: what IS the control strategy? Directional following vs flow-gated dither.

For ridge25 / w1' / default x 6 seeds, run the standard task via run_tracking
(full History) and compute per run:
  - net rotation per 720-step segment vs the stimulus's net motion that
    segment (a genuine follower matches sign AND magnitude ~720 deg... no:
    720 steps x 1 deg/step = 720 deg per segment, i.e. 2 laps);
  - the follow ratio: (agent net rotation)/(stimulus net rotation) per
    segment — 1.0 = locked follower, ~0 = stationary ditherer;
  - conditional turn stats: mean dH and RMS dH for in-view (|err|<=45),
    edge (45<|err|<=95), dark (|err|>95);
  - signed-response curve: mean dH vs err sign in view (proportionality);
  - dwell structure: fraction of steps |dH| < 0.5 (stalls) by zone.
Also writes heading-vs-stimulus trajectory PNGs for seed 0 of each variant.

Preregistered (H19): agents on the ridge are FOLLOWERS (follow ratio > 0.7
in good segments) — dithering alone cannot reach 0.85 at w1' given reversals;
but the response is flow-gated rather than error-proportional (flat mean dH
vs error magnitude in view, near-zero dH in dark).
"""

from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from common import make_configs  # noqa: F401  (BLAS pins)
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from homeostasis.simulation import run_tracking  # noqa: E402
from homeostasis.reservoir import ReservoirConfig  # noqa: E402
from homeostasis.tracking import TrackingConfig  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "out"
LAB = OUT / "lab"
SEG = 720


def w1p():
    cfg = json.loads((OUT / "sweep" / "results.json").read_text())["configs"][236]
    res = dict(n_nodes=cfg["n_nodes"], p_link=cfg["p_link"],
               input_weight=cfg["input_weight"], weight_init_mean=0.75,
               weight_init_sd=cfg["weight_init_sd"], leak=cfg["leak"],
               target_lr=cfg["target_lr"], threshold_ratio=cfg["threshold_ratio"])
    return res, dict(gain=cfg["gain"])


def evaluate(task):
    name, res_over, trk_over, seed = task
    rcfg, tcfg = make_configs(res_over, trk_over)
    h = run_tracking(n_steps=7200, seed=seed, reservoir_config=rcfg,
                     tracking_config=tcfg, record_spikes=False)
    err = h.error
    dh = h.d_heading
    stim_dir = h.stimulus_direction
    n_seg = 7200 // SEG
    follow = []
    for s in range(n_seg):
        sl = slice(s * SEG, (s + 1) * SEG)
        agent_net = float(dh[sl].sum())
        stim_net = float((stim_dir[sl] * h.stimulus_speed[sl]).sum())
        follow.append(agent_net / stim_net if abs(stim_net) > 1 else 0.0)
    zones = dict(inview=np.abs(err) <= 45,
                 edge=(np.abs(err) > 45) & (np.abs(err) <= 95),
                 dark=np.abs(err) > 95)
    zstats = {z: dict(mean=float(dh[m].mean()) if m.any() else 0.0,
                      rms=float(np.sqrt((dh[m] ** 2).mean())) if m.any() else 0.0,
                      stall=float((np.abs(dh[m]) < 0.5).mean()) if m.any() else 0.0,
                      occ=float(m.mean()))
              for z, m in zones.items()}
    inv = zones["inview"]
    signed = float(np.mean(np.sign(err[inv]) * dh[inv])) if inv.any() else 0.0
    score = float(np.mean(np.abs(err) <= 45))
    return dict(name=name, seed=seed, score=score, follow=follow,
                zstats=zstats, signed_resp=signed,
                heading=h.heading[::6].tolist() if seed == 0 else None,
                stim=h.stimulus_angle[::6].tolist() if seed == 0 else None)


def main():
    w1p_res, w1p_trk = w1p()
    variants = {"ridge25": ({"leak": 0.25, "weight_lr": 0.1}, {}),
                "w1prime": (w1p_res, w1p_trk),
                "default": ({}, {})}
    tasks = [(n, r, t, s) for n, (r, t) in variants.items() for s in range(6)]
    with ProcessPoolExecutor(10) as pool:
        rows = list(pool.map(evaluate, tasks))
    LAB.mkdir(exist_ok=True)
    (LAB / "b6_strategy.json").write_text(json.dumps(rows))

    for name in variants:
        sel = [r for r in rows if r["name"] == name]
        fol = np.array([r["follow"] for r in sel])
        sc = np.mean([r["score"] for r in sel])
        good = fol[np.array([np.array(r["follow"]) for r in sel]) != 0]
        print(f"\n══ {name} (6 seeds, score {sc:.3f})")
        print(f"   follow ratio per segment (mean over seeds): "
              + " ".join(f"{v:+.2f}" for v in fol.mean(axis=0)))
        print(f"   follow ratio overall: {fol.mean():+.3f} (|f|>0.5 in "
              f"{np.mean(np.abs(fol) > 0.5):.2f} of segments)")
        for z in ("inview", "edge", "dark"):
            zs = [r["zstats"][z] for r in sel]
            print(f"   {z:7s} occ {np.mean([q['occ'] for q in zs]):.2f}  "
                  f"meandH {np.mean([q['mean'] for q in zs]):+.3f}  "
                  f"rmsdH {np.mean([q['rms'] for q in zs]):.3f}  "
                  f"stall {np.mean([q['stall'] for q in zs]):.2f}")
        print(f"   signed in-view response (sign(err)*dH mean): "
              f"{np.mean([r['signed_resp'] for r in sel]):+.4f}")

    # trajectory figures
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    for ax, name in zip(axes, variants):
        r = next(r for r in rows if r["name"] == name and r["seed"] == 0)
        t = np.arange(len(r["heading"])) * 6
        ax.plot(t, np.unwrap(np.deg2rad(r["stim"])) * 180 / np.pi, lw=1.2, label="stimulus")
        ax.plot(t, np.unwrap(np.deg2rad(r["heading"])) * 180 / np.pi, lw=1.0, label="heading")
        ax.set_ylabel(f"{name}\nunwrapped deg")
        ax.legend(loc="upper left", fontsize=8)
    axes[-1].set_xlabel("step")
    fig.suptitle("Trajectories, seed 0: follower vs dither is visible by eye")
    fig.tight_layout()
    fig.savefig(LAB / "b6_trajectories.png", dpi=130)
    print(f"\nwrote {LAB/'b6_strategy.json'} and b6_trajectories.png")


if __name__ == "__main__":
    main()
