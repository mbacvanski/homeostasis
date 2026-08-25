"""Speaking agent v1: close the loop in the 5-word grammar world.

Setup (per network, 2021-exact config plus 5 effector nodes): listen to 800
teacher sentences; then 400 blocks of [3 teacher sentences + 1 completion
trial]. In a trial the teacher says only a subject; the agent then fills the
3 remaining slots (verb, object, boundary), and each word it speaks is fed
back as its own next input at full strength - it hears itself. Between
trials it hears the teacher again (the interleaving that guards against the
agent drifting into a private statistics of its own babble).

The mouth: the model's built-in output layer (each output node reads a
random ~p_link subset of neurons and reports the fraction that just spiked;
passive, never learned). Output node i is defined to mean token i; the word
spoken each step is the argmax reading. Note n_outputs=5 instead of 2
changes only the (last-drawn) output adjacency, so the reservoir itself is
bit-identical to the stage-1e replication for the same seed.

Timing variants (the v1 lesson: a mouth read at the same step a word
arrives reads the word's own imprint, not the prediction - the prediction
reaches spikes one step later):
  A  words back-to-back, mouth read at arrival        (v1: parrots)
  B  words back-to-back, 1 silent think step before each spoken word
  C  spaced world: 1 silent step between ALL words, listening included
     (fresh warmup in that rhythm); mouth read on the silent step

Arms (same warmed-up state, same teacher schedule, restored between arms):
  speak          argmax of the 5 pool readings; learning stays on
  speak-frozen   same mouth, all learning off after warmup
  speak-redraw   pools re-drawn with a fresh RNG before EVERY trial - blocks
                 the loop from self-organizing around one consistent mouth
  oracle-mouth   speak the token of the best-correlating stored
                 (token, position) spike pattern - an upper bound on how
                 much grammar an ideal readout could extract from the state

Success criteria set in advance: verb slot actually contains a verb (chance
0.40), the grammar-favored verb appears ~0.75 of the time among verbs
(chance 0.50), full verb+object+boundary pattern (chance 0.032), and the
perseveration rate - back-to-back repeats, the predicted failure mode
(chance 0.20; teacher speech has 0.00).
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
    EYE, TID, TOKENS, VERB_P, OBJ_P, sample_sentence,
)

from homeostasis import HomeostaticReservoir, ReservoirConfig  # noqa: E402

WARMUP_SENTS = 800
CODE_SENTS = 100
N_BLOCKS = 400
TEACH_PER_BLOCK = 3
VERBS = {"walks", "bites"}
NOUNS = {"man", "dog"}
ZERO = np.zeros(5)
ARMS = ("speak", "speak-frozen", "speak-redraw", "oracle-mouth")
# (label, gap between heard words, silent think steps before each spoken word)
CONFIGS = [
    ("A back-to-back, read at arrival", 0, 0),
    ("B back-to-back, think 1 step", 0, 1),
    ("C spaced world, think 1 step", 1, 1),
]


def make_net(seed):
    cfg = ReservoirConfig(
        n_nodes=100, n_inputs=5, n_outputs=5,
        p_link=0.1, input_p_link=0.1, input_weight=5.0,
        weight_init_mean=0.0, weight_init_sd=1.0,
        leak=0.25, target_lr=0.01, weight_lr=0.1,
        clamp_negative_activations=True,
    )
    return HomeostaticReservoir(cfg, seed=seed)


def warm_up(net, seed, gap):
    """Listen to the teacher; return stored (token, position) spike codes,
    collected at word-arrival steps of the last CODE_SENTS sentences."""
    rng = np.random.default_rng(10000 + seed)
    code_sum = np.zeros((5, 4, 100))
    code_n = np.zeros((5, 4))
    for si in range(WARMUP_SENTS):
        for pos, tok in enumerate(sample_sentence(rng)):
            state = net.step(EYE[TID[tok]])
            if si >= WARMUP_SENTS - CODE_SENTS:
                code_sum[TID[tok], pos] += state.spiked
                code_n[TID[tok], pos] += 1
            for _ in range(gap):
                net.step(ZERO)
    codes = []
    for tid in range(5):
        for pos in range(4):
            if code_n[tid, pos] > 0:
                c = code_sum[tid, pos] / code_n[tid, pos]
                c = c - c.mean()
                nrm = np.linalg.norm(c)
                if nrm > 1e-12:
                    codes.append((tid, c / nrm))
    return codes


def score_trials(trials):
    n = len(trials)
    m = dict.fromkeys(("verb_ok", "verb_fav", "obj_ok", "obj_fav", "end_ok",
                       "full_ok", "repeat", "stuck", "parrot"), 0.0)
    n_verb = n_objfav = 0
    full_by_trial = np.zeros(n)
    for i, (subj, sp) in enumerate(trials):
        w = [TOKENS[a] for a in sp]
        seq = [subj] + w
        m["parrot"] += w[0] == subj
        v_ok = w[0] in VERBS
        o_ok = w[1] in NOUNS
        e_ok = w[2] == "space"
        m["verb_ok"] += v_ok
        m["obj_ok"] += o_ok
        m["end_ok"] += e_ok
        full = v_ok and o_ok and e_ok
        m["full_ok"] += full
        full_by_trial[i] = full
        if v_ok:
            n_verb += 1
            m["verb_fav"] += w[0] == VERB_P[subj][0]
            if o_ok:
                n_objfav += 1
                m["obj_fav"] += w[1] == OBJ_P[w[0]][0]
        m["repeat"] += np.mean([seq[k] == seq[k + 1] for k in range(3)])
        m["stuck"] += w[0] == w[1] == w[2]
    for k in ("verb_ok", "obj_ok", "end_ok", "full_ok", "repeat", "stuck",
              "parrot"):
        m[k] /= n
    m["verb_fav"] = m["verb_fav"] / n_verb if n_verb else np.nan
    m["obj_fav"] = m["obj_fav"] / n_objfav if n_objfav else np.nan
    m["full_first100"] = float(full_by_trial[:100].mean())
    m["full_last100"] = float(full_by_trial[-100:].mean())
    return m


def run_one(seed):
    # fixed schedule shared by all configs and arms: paired comparison
    sched_rng = np.random.default_rng(20000 + seed)
    schedule = [([sample_sentence(sched_rng) for _ in range(TEACH_PER_BLOCK)],
                 "man" if sched_rng.random() < 0.5 else "dog")
                for _ in range(N_BLOCKS)]
    out = {"seed": seed}
    for label, gap, think in CONFIGS:
        net = make_net(seed)
        codes = warm_up(net, seed, gap)
        snap = snapshot(net)
        orig_out_adj = net.output_adjacency.copy()

        def set_pools(adj):
            net.output_adjacency = adj
            net._output_adjacency_f = adj.astype(float)
            net._output_in_degree = adj.sum(axis=0)

        for arm in ARMS:
            restore(net, snap)
            set_pools(orig_out_adj)
            net.learning_enabled = arm != "speak-frozen"
            redraw_rng = np.random.default_rng(30000 + seed)
            trials = []
            for teach_sents, subj in schedule:
                for sent in teach_sents:
                    for tok in sent:
                        net.step(EYE[TID[tok]])
                        for _ in range(gap):
                            net.step(ZERO)
                if arm == "speak-redraw":
                    set_pools(redraw_rng.random((100, 5)) < 0.1)
                state = net.step(EYE[TID[subj]])
                spoken = []
                for _ in range(3):
                    for _ in range(think):
                        state = net.step(ZERO)
                    if arm == "oracle-mouth":
                        pat = state.spiked.astype(float)
                        pat = pat - pat.mean()
                        a = codes[int(np.argmax([c @ pat
                                                 for _, c in codes]))][0]
                    else:
                        a = int(np.argmax(state.outputs))
                    spoken.append(a)
                    state = net.step(EYE[a])
                for _ in range(gap):
                    net.step(ZERO)
                trials.append((subj, spoken))
            net.learning_enabled = True
            out[f"{label}|{arm}"] = score_trials(trials)
    return out


def main():
    t0 = time.perf_counter()
    with ProcessPoolExecutor(int(os.environ.get("WORKERS", "10"))) as pool:
        results = list(pool.map(run_one, range(100)))
    print(f"100 networks in {time.perf_counter()-t0:.0f}s; 400 completion "
          f"trials per config (teacher says the subject, agent speaks 3 "
          f"words hearing itself; 3 teacher sentences between trials)\n")
    fields = [
        ("verb_ok",   "verb slot contains a verb        (chance 0.40)"),
        ("verb_fav",  "  ...and it's the likely verb    (teacher 0.75)"),
        ("obj_ok",    "object slot contains a noun      (chance 0.40)"),
        ("obj_fav",   "  ...and it's the likely object  (teacher 0.75)"),
        ("end_ok",    "3rd slot is the boundary word    (chance 0.20)"),
        ("full_ok",   "whole pattern verb+noun+boundary (chance 0.032)"),
        ("parrot",    "1st word = the subject just heard (chance 0.20)"),
        ("repeat",    "word repeated back-to-back       (chance 0.20)"),
        ("stuck",     "same word in all 3 slots         (chance 0.04)"),
        ("full_first100", "whole pattern, trials 1-100"),
        ("full_last100",  "whole pattern, trials 301-400"),
    ]
    for label, _, _ in CONFIGS:
        print(f"=== {label} ===")
        print(f"{'':48s}" + "".join(f"{a:>14s}" for a in ARMS))
        for f, desc in fields:
            row = ""
            for a in ARMS:
                vals = np.array([r[f"{label}|{a}"][f] for r in results])
                row += f"  {np.nanmean(vals):.3f}±{np.nanstd(vals):.2f}"
            print(f"{desc:48s}{row}")
        print()


if __name__ == "__main__":
    main()
