"""Stage 3: contingent dialogue with a mumble mouth.

The stage-2 speaking setup broke the tracking/Pong mechanism three ways: the
teacher was indifferent to what the agent said, the winner-take-all word
choice gave drift no slope to climb, and no word was more comfortable than
any other. This redesign restores the sensorimotor loop:

WORLD - one 4-slot sentence at a time, words alternating by turn:
  slot 0 subject   TEACHER (uniform man/dog)
  slot 1 verb      AGENT
  slot 2 object    TEACHER, continuing from the word the agent ACTUALLY said
                   (P(object | agent's verb) if it was a verb, else uniform)
  slot 3 boundary  AGENT
So every agent word immediately shapes the next thing it hears, and the
agent never speaks twice in a row (fights the drift collapse measured in
stage 2: quality died at the second self-spoken word).

MOUTH - continuous articulation instead of a cliff: pool readings r are
contrast-sharpened (r^BETA, normalized) into a mixture m over the 5 words;
the agent HEARS m itself (total loudness fixed at one word's worth). If
max(m) >= THETA the word "stands" and the world advances from it; below
that it was a mumble: the teacher waits one silent step (lost throughput)
and speaks the slot itself (takeover). A blurred mixture is objectively
harder to absorb than a clean expected word, so drift has a comfort
gradient toward clear, grammatical speech.

ARMS
  dialogue        tuned-20 pools, contingent teacher, learning on
  random-pools    same but the paper's random 10-neuron pools
  non-contingent  teacher picks the object from its OWN silently-sampled
                  verb, ignoring the agent - removes only the contingency
  frozen          as dialogue but all learning off after warmup

Reported in 3 bins of 200 sentences to expose trends: does closed-loop
experience improve speech (contingent) where it never did in stage 2?
"""

from __future__ import annotations

import os

for _v in ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import pathlib
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from stage1d_completion import restore, snapshot  # noqa: E402
from stage1e_grammar_replication import (  # noqa: E402
    EYE, TID, TOKENS, VERB_P, OBJ_P,
)
from stage2a_speaking import VERBS, ZERO, make_net, warm_up  # noqa: E402

N_SENTS = int(os.environ.get("SENTS", "600"))
BIN = N_SENTS // 3
BETA = 3.0        # articulation contrast: m ~ r^BETA
THETA = 0.5       # max(m) >= THETA counts as an intelligible word
ARMS = ("dialogue", "random-pools", "non-contingent", "frozen")


def run_one(seed):
    net = make_net(seed)
    codes = warm_up(net, seed, gap=0)
    snap = snapshot(net)

    tok_mean = np.zeros((5, 100))
    for tid in range(5):
        tok_mean[tid] = np.mean([c for t, c in codes if t == tid], axis=0)
    tuned = np.zeros((100, 5), dtype=bool)
    for tid in range(5):
        tuned[np.argsort(tok_mean[tid])[-20:], tid] = True
    rand10 = np.random.default_rng(40000 + seed).random((100, 5)) < 0.1

    def set_pools(adj):
        net.output_adjacency = adj
        net._output_adjacency_f = adj.astype(float)
        net._output_in_degree = adj.sum(axis=0)

    out = {"seed": seed}
    for arm in ARMS:
        restore(net, snap)
        set_pools(rand10 if arm == "random-pools" else tuned)
        net.learning_enabled = arm != "frozen"
        trng = np.random.default_rng(50000 + seed)

        def agent_turn(grammatical_word):
            """One think step, mumble, stand-or-takeover.
            Returns (stood_word_or_None, sharpness)."""
            state = net.step(ZERO)                       # the breath
            r = state.outputs.astype(float)
            if r.sum() > 1e-12:
                m = r ** BETA
                m /= m.sum()
            else:
                m = None
            net.step(m if m is not None else ZERO)       # hears its own mumble
            if m is not None and m.max() >= THETA:
                return TOKENS[int(np.argmax(m))], float(m.max())
            net.step(ZERO)                               # stall: lost step
            net.step(EYE[TID[grammatical_word]])         # teacher takeover
            return None, float(m.max()) if m is not None else 0.0

        rows = []
        for _ in range(N_SENTS):
            subj = "man" if trng.random() < 0.5 else "dog"
            net.step(EYE[TID[subj]])
            fav_v, p = VERB_P[subj]
            gram_v = fav_v if trng.random() < p else (
                "bites" if fav_v == "walks" else "walks")
            w_verb, sharp_v = agent_turn(gram_v)
            heard_verb = w_verb if w_verb is not None else gram_v
            # teacher's object: contingent on what it actually heard,
            # or on its own private sentence (non-contingent control)
            basis = (gram_v if arm == "non-contingent" else
                     heard_verb if heard_verb in OBJ_P else None)
            if basis is not None:
                fav_o, p = OBJ_P[basis]
                obj = fav_o if trng.random() < p else (
                    "man" if fav_o == "dog" else "dog")
            else:
                obj = "man" if trng.random() < 0.5 else "dog"
            net.step(EYE[TID[obj]])
            w_end, sharp_e = agent_turn("space")
            rows.append((subj, w_verb, sharp_v, w_end, sharp_e))

        for b in range(N_SENTS // BIN):
            chunk = rows[b * BIN:(b + 1) * BIN]
            spoken_v = [(s, w) for s, w, *_ in chunk if w is not None]
            verbs = [(s, w) for s, w in spoken_v if w in VERBS]
            spoken_e = [w for *_, w, _ in chunk if w is not None]
            out[f"{arm}|bin{b}"] = {
                "verb_spoken": len(spoken_v) / len(chunk),
                "verb_is_verb": (len(verbs) / len(spoken_v)
                                 if spoken_v else np.nan),
                "verb_fav": (np.mean([w == VERB_P[s][0] for s, w in verbs])
                             if verbs else np.nan),
                "end_spoken": len(spoken_e) / len(chunk),
                "end_is_space": (np.mean([w == "space" for w in spoken_e])
                                 if spoken_e else np.nan),
                "sharp": float(np.mean([x for _, _, sv, _, se in chunk
                                        for x in (sv, se)])),
            }
        net.learning_enabled = True
    return out


def main():
    t0 = time.perf_counter()
    with ProcessPoolExecutor(int(os.environ.get("WORKERS", "10"))) as pool:
        results = list(pool.map(run_one, range(100)))
    n_bins = N_SENTS // BIN
    print(f"100 networks in {time.perf_counter()-t0:.0f}s; {N_SENTS} dialogue "
          f"sentences each (teacher: subject + object; agent: verb + "
          f"boundary; word stands if max(mixture) >= {THETA})\n")
    fields = [
        ("verb_spoken",  "verb slot: agent word stood (not takeover)"),
        ("verb_is_verb", "  ...stood word is actually a verb  (chance 0.40)"),
        ("verb_fav",     "  ...and the grammar-favored one    (teacher 0.75)"),
        ("end_spoken",   "boundary slot: agent word stood"),
        ("end_is_space", "  ...stood word is the boundary     (chance 0.20)"),
        ("sharp",        "mean articulation sharpness (1 = pure word)"),
    ]
    for arm in ARMS:
        print(f"=== {arm} ===")
        print(f"{'':52s}" + "".join(f"  sents {b*BIN}-{(b+1)*BIN}"
                                    for b in range(n_bins)))
        for f, desc in fields:
            row = ""
            for b in range(n_bins):
                vals = np.array([r[f"{arm}|bin{b}"][f] for r in results])
                row += f"   {np.nanmean(vals):.3f}±{np.nanstd(vals):.2f}"
            print(f"{desc:52s}{row}")
        print()


if __name__ == "__main__":
    main()
