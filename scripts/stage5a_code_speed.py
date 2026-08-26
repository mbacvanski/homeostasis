"""How fast does the reservoir's verb code move under grammar reversal?

Sets the mouth timescale for stage 5a: probe the completion preference
every 100 sentences through [400 original + 2000 reversed] sentences,
scoring the silent-state pattern against (a) sliding-window EMA verb codes
(lambda=0.02/arrival, ~50-arrival memory) and (b) frozen end-of-warmup
codes. The (a) curve shows when the internal flip completes; the mouth's
memory should be shorter than that but longer than single-sentence noise.
Gap = corr with ORIGINAL-favored verb code minus original-unfavored,
averaged over both subjects; positive = still old grammar.
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
from stage1e_grammar_replication import EYE, TID  # noqa: E402
from stage2a_speaking import ZERO, make_net, warm_up  # noqa: E402
from stage4b_reversal_passive import FAV_ORIG, NOUNS, corr, rev_sentence  # noqa: E402

N_ORIG = 400
N_REV = 2000
PROBE_EVERY = 100
LAM = 0.02
WALKS, BITES = TID["walks"], TID["bites"]


def orig_sentence(rng):
    s = NOUNS[rng.random() < 0.5]
    fav_v = WALKS if s == TID["man"] else BITES
    v = fav_v if rng.random() < 0.75 else (BITES if fav_v == WALKS else WALKS)
    fav_o = TID["dog"] if v == WALKS else TID["man"]
    o = fav_o if rng.random() < 0.75 else (
        TID["man"] if fav_o == TID["dog"] else TID["dog"])
    return (s, v, o, TID["space"])


def gap(net, codes):
    snap = snapshot(net)
    g = []
    for s in NOUNS:
        restore(net, snap)
        net.step(EYE[s])
        pat = net.step(ZERO).spiked.astype(float)
        fav = FAV_ORIG[s]
        unf = BITES if fav == WALKS else WALKS
        g.append(corr(pat, codes[fav]) - corr(pat, codes[unf]))
    restore(net, snap)
    return float(np.mean(g))


def run_one(seed):
    net = make_net(seed)
    warm_up(net, seed, gap=0)
    rng = np.random.default_rng(70000 + seed)
    ema = {WALKS: np.zeros(100), BITES: np.zeros(100)}
    # prime the EMA codes on 100 original sentences before measuring
    for _ in range(100):
        for pos, tok in enumerate(orig_sentence(rng)):
            st = net.step(EYE[tok])
            if pos == 1:
                ema[tok] = (1 - LAM) * ema[tok] + LAM * st.spiked
    frozen_codes = {t: ema[t].copy() for t in ema}
    curve_ema, curve_old = [], []
    for si in range(N_ORIG + N_REV):
        sent = (orig_sentence if si < N_ORIG else rev_sentence)(rng)
        for pos, tok in enumerate(sent):
            st = net.step(EYE[tok])
            if pos == 1:
                ema[tok] = (1 - LAM) * ema[tok] + LAM * st.spiked
        if (si + 1) % PROBE_EVERY == 0:
            curve_ema.append(gap(net, ema))
            curve_old.append(gap(net, frozen_codes))
    return curve_ema, curve_old


def main():
    t0 = time.perf_counter()
    with ProcessPoolExecutor(10) as pool:
        results = list(pool.map(run_one, range(10)))
    e = np.mean([r[0] for r in results], axis=0)
    o = np.mean([r[1] for r in results], axis=0)
    n_orig_probes = N_ORIG // PROBE_EVERY
    print(f"10 nets in {time.perf_counter()-t0:.0f}s; probes every "
          f"{PROBE_EVERY} sentences; reversal after probe {n_orig_probes}.\n")
    print("sentences  gap(sliding codes)  gap(frozen old codes)")
    for i in range(len(e)):
        mark = "  <- reversal" if i + 1 == n_orig_probes else ""
        print(f"{(i+1)*PROBE_EVERY:9d}  {e[i]:+18.3f}  {o[i]:+20.3f}{mark}")


if __name__ == "__main__":
    main()
