"""Stage 5a: online continuous auditory-motor mouth, passive reversal assay.

The mouth is a continuous reservoir-to-mouth weight matrix M (N x 5).
When an externally supplied word w is heard (teacher speech only - the
assay is fully passive), its column tracks the concurrent centered spike
pattern:

    M[:, w] <- (1 - eta) M[:, w] + eta (s - s_bar),   then L2-normalize

with s_bar a slow EMA of per-neuron spike rates (rate 0.01/step) and
eta = 0.05/arrival (~20-arrival memory ~ 40 sentences), chosen from the
measured code-movement curve (stage5a_code_speed.py: the internal flip
completes within ~100-400 sentences of reversal; frozen templates stay
stale at +0.13 indefinitely). At a think step the mouth reads

    r_w = M[:, w] . (s - s_bar),      m = softmax(beta r),  beta = 5.

Assay: 800 warmup sentences (everything plastic), then per arm
400 original + 2000 reversed sentences with the arm's freeze flags:

    reservoir x mouth in {plastic, frozen}^2

Probes every 100 sentences: (a) completion articulation - feed a subject,
one think step, read m; report the ORIGINAL-grammar-favored share among
the two verbs (0.5 = neutral; crossing below 0.5 = the spoken preference
reversed); (b) heard-word decoding accuracy in the trailing window
(argmax_w r at each word's arrival, before that arrival updates M) -
if the mouth cannot decode what is currently being heard, completion
readings are uninterpretable.

Registered expectations (from the stage-4 audit discussion):
  plastic/plastic  share crosses below 0.5, decoding stays high
  plastic/frozen   share stays high (stale mouth), decoding degrades
  frozen/plastic   decoding high; tests context-shift-only retuning
  frozen/frozen    full retention baseline
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
from stage2a_speaking import ZERO, make_net  # noqa: E402
from stage4b_reversal_passive import FAV_ORIG, NOUNS, rev_sentence  # noqa: E402
from stage5a_code_speed import orig_sentence  # noqa: E402

WARMUP = 800
N_ORIG = 400
N_REV = 2000
PROBE_EVERY = 100
ETA = 0.05
SBAR_RATE = 0.01
BETA = 5.0
WALKS, BITES = TID["walks"], TID["bites"]
ARMS = (("plastic", "plastic"), ("plastic", "frozen"),
        ("frozen", "plastic"), ("frozen", "frozen"))


class Mouth:
    def __init__(self, n):
        self.M = np.zeros((n, 5))
        self.sbar = np.zeros(n)

    def observe(self, spiked):
        self.sbar = (1 - SBAR_RATE) * self.sbar + SBAR_RATE * spiked

    def hear(self, w, spiked):
        c = spiked - self.sbar
        self.M[:, w] = (1 - ETA) * self.M[:, w] + ETA * c
        nrm = np.linalg.norm(self.M[:, w])
        if nrm > 1e-12:
            self.M[:, w] /= nrm

    def read(self, spiked):
        r = (spiked - self.sbar) @ self.M
        e = np.exp(BETA * (r - r.max()))
        return e / e.sum()

    def decode(self, spiked):
        return int(np.argmax((spiked - self.sbar) @ self.M))

    def copy(self):
        m = Mouth(len(self.sbar))
        m.M = self.M.copy()
        m.sbar = self.sbar.copy()
        return m


def share_probe(net, mouth, n_lags=3):
    """Original-favored share and verb mass at think-step lags 1..n_lags.
    Context residue decays passively with the leak; maintained prediction
    should hold the share across lags - the lag profile separates them."""
    snap = snapshot(net)
    shares = [[] for _ in range(n_lags)]
    vmass = [[] for _ in range(n_lags)]
    for s in NOUNS:
        restore(net, snap)
        net.step(EYE[s])
        for lag in range(n_lags):
            st = net.step(ZERO)
            m = mouth.read(st.spiked.astype(float))
            fav = FAV_ORIG[s]
            unf = BITES if fav == WALKS else WALKS
            if m[fav] + m[unf] > 1e-12:
                shares[lag].append(m[fav] / (m[fav] + m[unf]))
            vmass[lag].append(m[WALKS] + m[BITES])
    restore(net, snap)
    return ([float(np.mean(x)) for x in shares],
            [float(np.mean(x)) for x in vmass])


def run_one(seed):
    net = make_net(seed)
    mouth = Mouth(100)
    rng = np.random.default_rng(80000 + seed)
    for _ in range(WARMUP):
        for tok in orig_sentence(rng):
            st = net.step(EYE[tok])
            sp = st.spiked.astype(float)
            mouth.observe(sp)
            mouth.hear(tok, sp)
    snap = snapshot(net)
    mouth0 = mouth.copy()
    rng_state = rng.bit_generator.state

    out = {"seed": seed}
    for res_arm, mouth_arm in ARMS:
        restore(net, snap)
        net.learning_enabled = res_arm == "plastic"
        mo = mouth0.copy()
        rng.bit_generator.state = rng_state       # identical word schedule
        shares, vmasses, accs = [], [], []
        acc_win = []
        for si in range(N_ORIG + N_REV):
            sent = (orig_sentence if si < N_ORIG else rev_sentence)(rng)
            for tok in sent:
                st = net.step(EYE[tok])
                sp = st.spiked.astype(float)
                acc_win.append(mo.decode(sp) == tok)
                if mouth_arm == "plastic":
                    mo.observe(sp)
                    mo.hear(tok, sp)
            if (si + 1) % PROBE_EVERY == 0:
                sh, vm = share_probe(net, mo)
                shares.append(sh[0])
                vmasses.append(vm[0])
                accs.append(float(np.mean(acc_win)))
                acc_win = []
        final_sh, _ = share_probe(net, mo)
        net.learning_enabled = True
        out[f"{res_arm}/{mouth_arm}"] = (shares, vmasses, accs, final_sh)
    return out


def main():
    t0 = time.perf_counter()
    with ProcessPoolExecutor(10) as pool:
        results = list(pool.map(run_one, range(50)))
    n_probes = (N_ORIG + N_REV) // PROBE_EVERY
    marks = [4, 6, 8, 10, 14, 18, 24]          # probe indices (x100 sents)
    print(f"50 nets in {time.perf_counter()-t0:.0f}s; reversal at sentence "
          f"{N_ORIG}. share = mass on ORIGINAL-favored verb among verbs at a "
          f"completion probe (0.5 neutral, <0.5 = spoken preference "
          f"reversed); acc = heard-word decoding in trailing 100 sentences.\n")
    hdr = "".join(f"  s{m*PROBE_EVERY:>5d}" for m in marks)
    for metric, idx in (("ORIGINAL-favored share", 0),
                        ("heard-word decode accuracy", 2),
                        ("verb mass at completion probe", 1)):
        print(f"--- {metric} ---")
        print(f"{'reservoir/mouth':>18s}{hdr}")
        for res_arm, mouth_arm in ARMS:
            k = f"{res_arm}/{mouth_arm}"
            rows = np.array([r[k][idx] for r in results])
            mean = rows.mean(axis=0)
            line = "".join(f"  {mean[m-1]:6.3f}" for m in marks)
            print(f"{k:>18s}{line}")
        print()
    print("--- share by think-step lag, end of reversal (context residue "
          "decays with lag; maintained prediction holds) ---")
    print(f"{'reservoir/mouth':>18s}    lag1    lag2    lag3")
    for res_arm, mouth_arm in ARMS:
        k = f"{res_arm}/{mouth_arm}"
        lags = np.array([r[k][3] for r in results]).mean(axis=0)
        print(f"{k:>18s}  " + "  ".join(f"{v:6.3f}" for v in lags))


if __name__ == "__main__":
    main()
