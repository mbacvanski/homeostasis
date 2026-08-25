"""Replicate the 2021 paper's fading-memory completion result in their EXACT
setting: the [subject, verb, object, space] grammar, N=100, their parameters.

Grammar (their Fig. 7 / Table 1): subject uniform over {man, dog};
man->walks .75 / bites .25, dog->walks .25 / bites .75;
walks->dog .75 / man .25, bites->dog .25 / man .75; space always fourth.

Protocol (their section 5.4): train 1000 sentences (4000 steps). Codes =
mean spike pattern per (token, position) over the last 100 sentences. Then,
from the trained state: feed one or two words of a fresh sentence, cut the
input, and correlate the silent-step spike pattern with each code. Their
Table 2 grand means: after ['man'], the silent pattern correlates 0.467 with
'walks'-as-verb codes vs 0.319 with 'bites'-as-verb; after ['man','walks'],
0.452 with 'dog'-as-object. Averaged here over 100 independent networks.
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

from homeostasis import HomeostaticReservoir, ReservoirConfig  # noqa: E402

TOKENS = ["man", "dog", "walks", "bites", "space"]
TID = {t: i for i, t in enumerate(TOKENS)}
VERB_P = {"man": ("walks", 0.75), "dog": ("bites", 0.75)}
OBJ_P = {"walks": ("dog", 0.75), "bites": ("man", 0.75)}
N_SENT = 1000
CODE_SENTS = 100     # codes from the last 100 sentences
EYE = np.eye(5)


def sample_sentence(rng):
    s = "man" if rng.random() < 0.5 else "dog"
    fav_v, p = VERB_P[s]
    v = fav_v if rng.random() < p else ("bites" if fav_v == "walks" else "walks")
    fav_o, p = OBJ_P[v]
    o = fav_o if rng.random() < p else ("man" if fav_o == "dog" else "dog")
    return [s, v, o, "space"]


def run_one(seed):
    cfg = ReservoirConfig(
        n_nodes=100, n_inputs=5, n_outputs=2,
        p_link=0.1, input_p_link=0.1, input_weight=5.0,
        weight_init_mean=0.0, weight_init_sd=1.0,
        leak=0.25, target_lr=0.01, weight_lr=0.1,
        clamp_negative_activations=True,
    )
    net = HomeostaticReservoir(cfg, seed=seed)
    rng = np.random.default_rng(10000 + seed)   # grammar stream, separate

    code_sum = np.zeros((5, 4, 100))            # token x position x nodes
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
    # test A: feed 'man' as subject, then silence; correlate silent step 1
    # with verb-position codes (their: walks 0.467 vs bites 0.319)
    for subj in ("man", "dog"):
        restore(net, snap)
        net.step(EYE[TID[subj]])
        state = net.step(zero)
        pat = state.spiked.astype(float)
        for verb in ("walks", "bites"):
            c = code(verb, 1)
            out[f"{subj}->silence ~ {verb}_verb"] = corr(pat, c) if c is not None else np.nan
    # test B: feed subject+verb, then silence; correlate with object codes
    for subj in ("man", "dog"):
        verb = VERB_P[subj][0]
        restore(net, snap)
        net.step(EYE[TID[subj]])
        net.step(EYE[TID[verb]])
        state = net.step(zero)
        pat = state.spiked.astype(float)
        for obj in ("dog", "man"):
            c = code(obj, 2)
            out[f"{subj},{verb}->silence ~ {obj}_obj"] = corr(pat, c) if c is not None else np.nan
    return out


def main():
    t0 = time.perf_counter()
    with ProcessPoolExecutor(10) as pool:
        results = list(pool.map(run_one, range(100)))
    print(f"100 networks in {time.perf_counter()-t0:.0f}s "
          f"(2021 Table 2 grand means for comparison in parentheses)\n")
    keys = results[0].keys()
    ref = {"man->silence ~ walks_verb": 0.467, "man->silence ~ bites_verb": 0.319,
           "man,walks->silence ~ dog_obj": 0.452, "man,walks->silence ~ man_obj": 0.213}
    for k in keys:
        vals = np.array([r[k] for r in results])
        extra = f"  (2021: {ref[k]:.3f})" if k in ref else ""
        print(f"  {k:>38}: {np.nanmean(vals):+.3f} ± {np.nanstd(vals):.3f}{extra}")


if __name__ == "__main__":
    main()
