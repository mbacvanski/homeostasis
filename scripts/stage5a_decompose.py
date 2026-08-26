"""Decompose the mouth's verb columns: word component vs context residue.

Stage 5a's frozen-reservoir/plastic-mouth arm reversed fully, which is only
possible if M's verb columns carry the context each verb arrives in (leak
0.75 keeps ~3/4 of the previous word's activation in the arrival pattern).
Quantify it: correlate M[:, verb] with EMA arrival codes of the two
subjects, at the end of the original phase and the end of the reversal.
context_bias(walks) = corr(M walks, dog code) - corr(M walks, man code):
originally walks follows man 75% so the bias should be man-ward (negative);
after reversal it should flip dog-ward (positive). Mirrored for bites.
"""

from __future__ import annotations

import os

for _v in ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from stage1e_grammar_replication import EYE, TID  # noqa: E402
from stage2a_speaking import make_net  # noqa: E402
from stage4b_reversal_passive import corr, rev_sentence  # noqa: E402
from stage5a_code_speed import orig_sentence  # noqa: E402
from stage5a_mouth import ETA, WARMUP, N_ORIG, N_REV, Mouth  # noqa: E402

LAM = 0.02
MAN, DOG, WALKS, BITES = TID["man"], TID["dog"], TID["walks"], TID["bites"]


def run_one(seed):
    net = make_net(seed)
    mouth = Mouth(100)
    rng = np.random.default_rng(80000 + seed)
    ctx = {MAN: np.zeros(100), DOG: np.zeros(100)}

    def stream(n_sents, maker):
        for _ in range(n_sents):
            for pos, tok in enumerate(maker(rng)):
                st = net.step(EYE[tok])
                sp = st.spiked.astype(float)
                mouth.observe(sp)
                mouth.hear(tok, sp)
                if pos == 0:
                    ctx[tok] = (1 - LAM) * ctx[tok] + LAM * sp

    def bias():
        out = {}
        for verb in (WALKS, BITES):
            col = mouth.M[:, verb]
            out[verb] = corr(col, ctx[DOG]) - corr(col, ctx[MAN])
        return out

    stream(WARMUP + N_ORIG, orig_sentence)
    before = bias()
    stream(N_REV, rev_sentence)
    after = bias()
    return before, after


def main():
    results = [run_one(s) for s in range(10)]
    print("context bias of mouth verb columns: corr(column, dog-subject "
          "code) - corr(column, man-subject code)\n(walks follows man 75% "
          "originally -> man-ward/negative; after reversal follows dog "
          "75% -> dog-ward/positive; bites mirrored)\n")
    for name, verb in (("walks", WALKS), ("bites", BITES)):
        b = np.mean([r[0][verb] for r in results])
        a = np.mean([r[1][verb] for r in results])
        print(f"  M[:, {name}]  original phase {b:+.3f}   after reversal {a:+.3f}")


if __name__ == "__main__":
    main()
