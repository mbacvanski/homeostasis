"""Does the grammar world ALSO flip to suppression at deeper silence?

Stage 1e replicated the 2021 completion result at silence step 1 only; the
char-level suppression (stage 1f/1h) only develops at silence steps 2-3,
which the grammar world was never tested at. Same protocol as stage 1e
(exact 2021 setting, 100 nets), but after feeding the subject we record
THREE silent steps and correlate each with:

  - verb-position codes (favored vs unfavored verb) - the step-1 prediction;
  - the just-fed subject's own code at position 0 - the echo;
  - at step 2, object-position codes (chain-favored obj: P(obj|subj)=.625);
  - at step 3, the space code at position 3 (certain, if prediction survives).
"""

from __future__ import annotations

import os

for _v in ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import pathlib
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from stage1d_completion import restore, snapshot  # noqa: E402
from stage1e_grammar_replication import (  # noqa: E402
    EYE, N_SENT, CODE_SENTS, TID, TOKENS, VERB_P, OBJ_P, sample_sentence,
)

from homeostasis import HomeostaticReservoir, ReservoirConfig  # noqa: E402


def run_one(seed):
    cfg = ReservoirConfig(
        n_nodes=100, n_inputs=5, n_outputs=2,
        p_link=0.1, input_p_link=0.1, input_weight=5.0,
        weight_init_mean=0.0, weight_init_sd=1.0,
        leak=0.25, target_lr=0.01, weight_lr=0.1,
        clamp_negative_activations=True,
    )
    net = HomeostaticReservoir(cfg, seed=seed)
    rng = np.random.default_rng(10000 + seed)

    code_sum = np.zeros((5, 4, 100))
    code_n = np.zeros((5, 4))
    for si in range(N_SENT):
        for pos, tok in enumerate(sample_sentence(rng)):
            state = net.step(EYE[TID[tok]])
            if si >= N_SENT - CODE_SENTS:
                code_sum[TID[tok], pos] += state.spiked
                code_n[TID[tok], pos] += 1

    def corr(a, b):
        a = a - a.mean(); b = b - b.mean()
        d = np.linalg.norm(a) * np.linalg.norm(b)
        return float(a @ b / d) if d > 1e-12 else 0.0

    def code(tok, pos):
        n = code_n[TID[tok], pos]
        return code_sum[TID[tok], pos] / n if n else None

    zero = np.zeros(5)
    out = {}
    snap = snapshot(net)
    for subj in ("man", "dog"):
        restore(net, snap)
        net.step(EYE[TID[subj]])
        fav_v = VERB_P[subj][0]
        unf_v = "bites" if fav_v == "walks" else "walks"
        fav_o = OBJ_P[fav_v][0]                     # chain-favored, P=.625
        unf_o = "man" if fav_o == "dog" else "dog"
        for lag in (1, 2, 3):
            state = net.step(zero)
            pat = state.spiked.astype(float)
            def put(name, c):
                if c is not None:
                    out.setdefault(f"lag{lag} {name}", []).append(corr(pat, c))
            put("verb favored", code(fav_v, 1))
            put("verb unfavored", code(unf_v, 1))
            put("echo (subject@0)", code(subj, 0))
            if lag == 2:
                put("object chain-fav", code(fav_o, 2))
                put("object chain-unf", code(unf_o, 2))
            if lag == 3:
                put("space@3", code("space", 3))
    return {k: float(np.mean(v)) for k, v in out.items()}


def main():
    t0 = time.perf_counter()
    with ProcessPoolExecutor(10) as pool:
        results = list(pool.map(run_one, range(100)))
    print(f"100 networks in {time.perf_counter()-t0:.0f}s "
          f"(after feeding the subject; mean over both subjects)\n")
    keys = sorted({k for r in results for k in r})
    for k in keys:
        vals = np.array([r[k] for r in results if k in r])
        print(f"  {k:>24}: {np.nanmean(vals):+.3f} ± {np.nanstd(vals):.3f}")


if __name__ == "__main__":
    main()
