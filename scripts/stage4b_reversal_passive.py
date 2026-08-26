"""Can passive listening even reverse a learned grammar prejudice?

Stage 4's reversal test showed the speaking agent retains the old grammar
with learning ON. Two readings: (a) the internal prediction never flips
(entrenchment) or (b) it flips but the fixed warmup-tuned pools can't see
it. This isolates (a): 800 sentences of the original grammar, then 2000
sentences of pure REVERSED-grammar listening (no dialogue), probing the
completion preference stage-1e-style at start and end - correlate the
silent state after 'man'/'dog' with verb codes. The end probe is scored
with BOTH the original warmup codes and fresh post-reversal codes, so a
flipped prediction cannot hide behind a stale template.
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

N_REV_SENTS = 2000
CODE_SENTS = 100
NOUNS = (TID["man"], TID["dog"])
FAV_ORIG = {TID["man"]: TID["walks"], TID["dog"]: TID["bites"]}


def rev_sentence(rng):
    s = NOUNS[rng.random() < 0.5]
    fav_v = TID["bites"] if s == TID["man"] else TID["walks"]   # reversed
    v = fav_v if rng.random() < 0.75 else (
        TID["walks"] if fav_v == TID["bites"] else TID["bites"])
    fav_o = TID["man"] if v == TID["walks"] else TID["dog"]     # reversed
    o = fav_o if rng.random() < 0.75 else (
        TID["man"] if fav_o == TID["dog"] else TID["dog"])
    return (s, v, o, TID["space"])


def corr(a, b):
    a = a - a.mean(); b = b - b.mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / d) if d > 1e-12 else 0.0


def probe(net, verb_codes):
    """Old-grammar-favored minus old-grammar-unfavored correlation gap,
    averaged over both subjects. Positive = still predicts OLD grammar."""
    snap = snapshot(net)
    gaps = []
    for s in NOUNS:
        restore(net, snap)
        net.step(EYE[s])
        pat = net.step(ZERO).spiked.astype(float)
        fav = FAV_ORIG[s]
        unf = TID["bites"] if fav == TID["walks"] else TID["walks"]
        if verb_codes[fav] is not None and verb_codes[unf] is not None:
            gaps.append(corr(pat, verb_codes[fav]) - corr(pat, verb_codes[unf]))
    restore(net, snap)
    return float(np.mean(gaps))


def run_one(seed):
    net = make_net(seed)
    codes_orig = warm_up(net, seed, gap=0)
    verb_orig = {t: None for t in (TID["walks"], TID["bites"])}
    for t, c in codes_orig:
        if t in verb_orig and verb_orig[t] is None:
            verb_orig[t] = c        # first (= verb-position) code per verb
    gap_before = probe(net, verb_orig)

    rng = np.random.default_rng(70000 + seed)
    csum = {t: np.zeros(100) for t in verb_orig}
    cn = {t: 0 for t in verb_orig}
    for si in range(N_REV_SENTS):
        for pos, tok in enumerate(rev_sentence(rng)):
            state = net.step(EYE[tok])
            if si >= N_REV_SENTS - CODE_SENTS and pos == 1 and tok in csum:
                csum[tok] += state.spiked
                cn[tok] += 1
    verb_new = {}
    for t in csum:
        c = csum[t] / max(cn[t], 1)
        c = c - c.mean()
        n = np.linalg.norm(c)
        verb_new[t] = c / n if n > 1e-12 else None

    return {"seed": seed,
            "before(old codes)": gap_before,
            "after(old codes)": probe(net, verb_orig),
            "after(new codes)": probe(net, verb_new)}


def main():
    t0 = time.perf_counter()
    with ProcessPoolExecutor(10) as pool:
        results = list(pool.map(run_one, range(50)))
    print(f"50 networks in {time.perf_counter()-t0:.0f}s; gap = correlation "
          f"with OLD-grammar-favored verb minus OLD-unfavored, after the "
          f"subject + one silent step.\n  positive = still predicts the old "
          f"grammar; negative = flipped to the new one.\n")
    for k in ("before(old codes)", "after(old codes)", "after(new codes)"):
        vals = np.array([r[k] for r in results])
        n_pos = int((vals > 0).sum())
        print(f"  {k:>18s}: {vals.mean():+.3f} ± {vals.std():.3f}   "
              f"(positive in {n_pos}/50 nets)")


if __name__ == "__main__":
    main()
