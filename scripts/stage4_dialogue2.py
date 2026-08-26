"""Stage 4: dialogue with the mechanics the stage-3 audit demanded.

Changes from stage 3, mapped to the audit's conditions:

1. MIXTURE CONTINGENCY, no cliffs: the teacher's response is drawn from
   P(y | m, slot) = sum_w m_w P(y | w, slot). A 51/49 articulation and a
   100/0 articulation now produce measurably different continuation
   distributions. No intelligibility threshold, no argmax anywhere in the
   environment.
2. NO SELF-HEARING: the agent never hears its own blend, so the measured
   comfort-of-smoothing gradient (verify_blend_comfort.py: mumble |E| 0.300
   vs clean word 0.499) is out of the input path entirely. The only input
   consequence of articulation is the teacher's sampled one-hot response.
3. CORRECT LOCAL CREDIT: the response lands on the step immediately after
   the think step whose spikes selected the mixture. The one-step update
   rule adjusts exactly the outgoing synapses of the action-selecting
   spikers by the comfort of the consequence: comfortable response ->
   coalition persists; surprising response -> coalition rewired.
4. EQUAL DURATION, NO TUTORING: every dialogue sentence is exactly 5 steps
   (subject, think, response, think, close/noise) regardless of success.
   A confused response is a uniform random word - higher entropy, zero
   grammatical information. The correct word is never supplied on failure.
5. BOUNDARY HAS CONSEQUENCES: the sentence-close step samples 'space' with
   probability m(space) (clean, predictable close) else a uniform word.
6. HONEST ARMS: 'pretrained' pools (the supervised tuned-20 decoder, named
   as such) and the paper's fixed random pools, each with learning on and
   frozen: 4 arms. The grounding question lives in random+learning-on.
7. DIRECT HOMEOSTATIC ACCOUNTING: mean |error| recorded separately for
   think steps, clean responses, and confused responses.

Grammar-world surface form is kept alive by interleaving one full teacher
monologue sentence after every dialogue sentence.

THE DECISIVE TEST - grammar reversal at half-time: all transition tables
(monologue and response) flip their 75/25 preferences. A regulated speaker
with learning on should migrate its articulation toward the new grammar;
a frozen speaker should retain the old one. 'fav_share' is always measured
against the ORIGINAL grammar, so adaptation shows as fav_share < 0.5 in
phase 2.

Registered predictions: if the corrected loop can ground a mouth, random+
learning-on verb-mass rises across phase-1 bins while random+frozen stays
flat; pretrained+learning-on fav_share falls below 0.5 after reversal
while pretrained+frozen stays high.
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
from stage1e_grammar_replication import EYE, TID, TOKENS  # noqa: E402
from stage2a_speaking import ZERO, make_net, warm_up  # noqa: E402

N_SENTS = int(os.environ.get("SENTS", "4000"))     # reversal at half-time
N_BINS = 8
BETA = 3.0
VERB_IDS = (TID["walks"], TID["bites"])
NOUN_IDS = (TID["man"], TID["dog"])
SPACE = TID["space"]
UNIFORM5 = np.full(5, 0.2)
ARMS = ("pretrained", "pretrained-frozen", "random", "random-frozen")


def tables(reversed_):
    """Per-phase transition tables as id -> (5,) probability vectors."""
    p = 0.25 if reversed_ else 0.75
    verb = {}          # subject id -> P(verb)
    for s, fav in ((TID["man"], TID["walks"]), (TID["dog"], TID["bites"])):
        v = np.zeros(5)
        v[fav] = p
        v[VERB_IDS[0] if fav == VERB_IDS[1] else VERB_IDS[1]] = 1 - p
        verb[s] = v
    obj = {}           # verb id -> P(object)
    for w, fav in ((TID["walks"], TID["dog"]), (TID["bites"], TID["man"])):
        v = np.zeros(5)
        v[fav] = p
        v[NOUN_IDS[0] if fav == NOUN_IDS[1] else NOUN_IDS[1]] = 1 - p
        obj[w] = v
    return verb, obj


def run_one(seed):
    net = make_net(seed)
    codes = warm_up(net, seed, gap=0)          # 800 sentences, original grammar
    snap = snapshot(net)

    tok_mean = np.zeros((5, 100))
    for tid in range(5):
        tok_mean[tid] = np.mean([c for t, c in codes if t == tid], axis=0)
    pretrained = np.zeros((100, 5), dtype=bool)
    for tid in range(5):
        pretrained[np.argsort(tok_mean[tid])[-20:], tid] = True
    rand10 = np.random.default_rng(40000 + seed).random((100, 5)) < 0.1

    def set_pools(adj):
        net.output_adjacency = adj
        net._output_adjacency_f = adj.astype(float)
        net._output_in_degree = adj.sum(axis=0)

    fav_orig = {TID["man"]: TID["walks"], TID["dog"]: TID["bites"]}

    out = {"seed": seed}
    for arm in ARMS:
        restore(net, snap)
        set_pools(rand10 if arm.startswith("random") else pretrained)
        net.learning_enabled = not arm.endswith("frozen")
        rng = np.random.default_rng(50000 + seed)
        bins = [dict(verb_mass=[], fav_share=[], space_mass=[], sharp=[],
                     e_think=[], e_clean=[], e_conf=[], conf=[], spike=[])
                for _ in range(N_BINS)]
        for si in range(N_SENTS):
            verb_t, obj_t = tables(reversed_=si >= N_SENTS // 2)
            b = bins[si * N_BINS // N_SENTS]

            def mouth(state):
                r = state.outputs.astype(float)
                if r.sum() <= 1e-12:
                    return UNIFORM5.copy()
                m = r ** BETA
                return m / m.sum()

            # --- dialogue sentence: subject, think, response, think, close
            subj = NOUN_IDS[rng.random() < 0.5]
            net.step(EYE[subj])
            st = net.step(ZERO)
            b["e_think"].append(np.abs(st.error).mean())
            b["spike"].append(st.prop_spiked)
            m1 = mouth(st)
            vm = m1[VERB_IDS[0]] + m1[VERB_IDS[1]]
            b["verb_mass"].append(vm)
            if vm > 1e-12:
                b["fav_share"].append(m1[fav_orig[subj]] / vm)
            b["sharp"].append(m1.max())
            w = rng.choice(5, p=m1)                 # what the teacher "heard"
            if w in obj_t:                          # a verb: grammatical reply
                resp = rng.choice(5, p=obj_t[w])
                st = net.step(EYE[resp])
                b["e_clean"].append(np.abs(st.error).mean())
                b["conf"].append(0.0)
            else:                                   # confusion: uniform word
                resp = rng.choice(5)
                st = net.step(EYE[resp])
                b["e_conf"].append(np.abs(st.error).mean())
                b["conf"].append(1.0)
            st = net.step(ZERO)
            m2 = mouth(st)
            b["space_mass"].append(m2[SPACE])
            if rng.random() < m2[SPACE]:
                net.step(EYE[SPACE])                # clean, predictable close
            else:
                net.step(EYE[rng.choice(5)])        # noise close
            # --- one full monologue sentence in the current grammar
            s2 = NOUN_IDS[rng.random() < 0.5]
            v2 = rng.choice(5, p=verb_t[s2])
            o2 = rng.choice(5, p=obj_t[v2])
            for tok in (s2, v2, o2, SPACE):
                net.step(EYE[tok])
        net.learning_enabled = True
        out[arm] = [{k: float(np.nanmean(v)) if v else np.nan
                     for k, v in bb.items()} for bb in bins]
    return out


def main():
    t0 = time.perf_counter()
    with ProcessPoolExecutor(int(os.environ.get("WORKERS", "10"))) as pool:
        results = list(pool.map(run_one, range(100)))
    half = N_BINS // 2
    print(f"100 networks in {time.perf_counter()-t0:.0f}s; {N_SENTS} dialogue "
          f"sentences, GRAMMAR REVERSED from bin {half} on. fav_share is "
          f"measured against the ORIGINAL grammar (0.5 = neutral).\n")
    fields = [
        ("verb_mass",  "articulation mass on the two verbs   (uniform 0.40)"),
        ("fav_share",  "share on originally-favored verb     (neutral 0.50)"),
        ("space_mass", "close-slot mass on the boundary word (uniform 0.20)"),
        ("sharp",      "articulation sharpness max(m)        (uniform 0.20)"),
        ("conf",       "confused (non-verb-driven) responses"),
        ("e_think",    "mean |error| on think steps"),
        ("e_clean",    "mean |error| hearing grammatical replies"),
        ("e_conf",     "mean |error| hearing confused replies"),
    ]
    for arm in ARMS:
        print(f"=== {arm} ===")
        hdr = "".join(f"   bin{b}{'*' if b >= half else ' '}"
                      for b in range(N_BINS))
        print(f"{'':52s}{hdr}   (* = reversed grammar)")
        for f, desc in fields:
            row = ""
            for b in range(N_BINS):
                vals = np.array([r[arm][b][f] for r in results])
                row += f"  {np.nanmean(vals):.3f}"
            print(f"{desc:52s}{row}")
        print()


if __name__ == "__main__":
    main()
