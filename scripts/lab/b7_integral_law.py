"""B7: verify the integral-controller law for total incoming weight.

Claim (from the update rule's structure): per step, for each node n with at
least one presynaptic spike at t-1, the TOTAL incoming recurrent weight
changes by exactly  d(sum_in W)_n = -weight_lr * E_n ; on steps with no
presynaptic spike it does not change. If exact, the weight channel is a pure
gated integral controller on the node's drive.

Test: one closed-loop tracking run per config (defaults, ridge25, w1'),
recording per-step (sum_in W)_n before/after and E_n; report the max
|deviation| from the law across all (node, step) pairs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from common import make_configs
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from homeostasis.reservoir import HomeostaticReservoir  # noqa: E402
from homeostasis.tracking import TrackingEnv  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "out"


def check(name, res_over, trk_over, seed=0, n_steps=600):
    rcfg, tcfg = make_configs(res_over, trk_over)
    net = HomeostaticReservoir(rcfg, seed=seed)
    env = TrackingEnv(tcfg)
    wlr = rcfg.weight_lr
    max_dev_gated = 0.0
    max_dev_ungated = 0.0
    for i in range(n_steps):
        pre = net.weights.sum(axis=0).copy()
        had_presyn = net.spiked.any()
        gated = (net._spiked_f @ net._adjacency_f) > 0  # nodes with spiking afferents
        inputs = env.sense()
        state = net.step(inputs)
        post = net.weights.sum(axis=0)
        d = post - pre
        pred = np.where(gated, -wlr * state.error, 0.0)
        dev = np.abs(d - pred)
        if had_presyn:
            max_dev_gated = max(max_dev_gated, float(dev[gated].max() if gated.any() else 0.0))
        max_dev_ungated = max(max_dev_ungated, float(dev[~gated].max() if (~gated).any() else 0.0))
        e_left, e_right = state.outputs
        env.apply_action(e_left, e_right)
        env.advance_stimulus()
    print(f"   {name:8s} max |d(sum W) + wlr*E| on gated nodes: {max_dev_gated:.2e}; "
          f"on ungated: {max_dev_ungated:.2e}")


def main():
    cfg = json.loads((OUT / "sweep" / "results.json").read_text())["configs"][236]
    w1p = dict(n_nodes=cfg["n_nodes"], p_link=cfg["p_link"],
               input_weight=cfg["input_weight"], weight_init_mean=0.75,
               weight_init_sd=cfg["weight_init_sd"], leak=cfg["leak"],
               target_lr=cfg["target_lr"], threshold_ratio=cfg["threshold_ratio"])
    print("B7 integral-controller law check (600 steps each):")
    check("default", {}, {})
    check("ridge25", {"leak": 0.25, "weight_lr": 0.1}, {})
    check("w1prime", w1p, {"gain": cfg["gain"]})


if __name__ == "__main__":
    main()
