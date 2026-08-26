"""Verify the blend-comfort counterfactual: after [subject, think step],
which self-heard articulation is homeostatically most comfortable?

Claim under test: uniform mumble < 60/40 blend < expected one-hot <
unexpected one-hot in mean |error|, because sparse +5 input projections
make concentrated (one-hot) drive high-variance across neurons while
blends smooth it. If true, the stage-3 mumble mouth had a comfort gradient
pointing TOWARD blending, not away from it.
"""

from __future__ import annotations

import os

for _v in ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from stage1d_completion import restore, snapshot  # noqa: E402
from stage1e_grammar_replication import EYE, TID, VERB_P, sample_sentence  # noqa: E402
from stage2a_speaking import ZERO, make_net, warm_up  # noqa: E402

N_NETS = 20
N_PROBES = 50


def main():
    rows = {k: [] for k in ("expected one-hot", "unexpected one-hot",
                            "60/40 blend", "uniform mumble")}
    drive_sd = {k: [] for k in rows}
    for seed in range(N_NETS):
        net = make_net(seed)
        warm_up(net, seed, gap=0)
        rng = np.random.default_rng(60000 + seed)
        for _ in range(N_PROBES):
            for tok in sample_sentence(rng):        # keep the stream moving
                net.step(EYE[TID[tok]])
            subj = "man" if rng.random() < 0.5 else "dog"
            net.step(EYE[TID[subj]])
            net.step(ZERO)                          # the think step
            snap = snapshot(net)
            fav = TID[VERB_P[subj][0]]
            unf = TID["bites"] if VERB_P[subj][0] == "walks" else TID["walks"]
            variants = {
                "expected one-hot": EYE[fav],
                "unexpected one-hot": EYE[unf],
                "60/40 blend": 0.6 * EYE[fav] + 0.4 * EYE[unf],
                "uniform mumble": np.full(5, 0.2),
            }
            for name, v in variants.items():
                restore(net, snap)
                state = net.step(v)
                rows[name].append(np.abs(state.error).mean())
                drive_sd[name].append((v @ net.input_weights).std())
    print(f"{N_NETS} nets x {N_PROBES} probes; self-heard input one step "
          f"after [subject, think step]\n")
    print(f"{'articulation':>20s}  {'mean |error|':>12s}  {'input-drive sd':>14s}")
    for name in rows:
        print(f"{name:>20s}  {np.mean(rows[name]):12.4f}  "
              f"{np.mean(drive_sd[name]):14.3f}")


if __name__ == "__main__":
    main()
