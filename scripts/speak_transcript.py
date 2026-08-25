"""Print actual spoken sentences from the stage-2 speaking agents.

Same protocol as stage 2b (back-to-back listening, one silent think step
before each spoken word, learning on): the teacher says a subject, the
agent speaks 3 words hearing itself. Shows the first 8 trials per mouth per
seed verbatim, plus that seed's full-pattern rate over all 400 trials so
the samples can be checked against the aggregate.
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
from stage1e_grammar_replication import EYE, TID, TOKENS, sample_sentence  # noqa: E402
from stage2a_speaking import (  # noqa: E402
    N_BLOCKS, TEACH_PER_BLOCK, VERBS, NOUNS, ZERO, make_net, warm_up,
)

SHOW = 8


def word(a):
    return "·" if TOKENS[a] == "space" else TOKENS[a]


def main():
    for seed in (0, 1, 2):
        sched_rng = np.random.default_rng(20000 + seed)
        schedule = [([sample_sentence(sched_rng)
                      for _ in range(TEACH_PER_BLOCK)],
                     "man" if sched_rng.random() < 0.5 else "dog")
                    for _ in range(N_BLOCKS)]
        net = make_net(seed)
        codes = warm_up(net, seed, gap=0)
        snap = snapshot(net)

        pool_rng = np.random.default_rng(40000 + seed)
        rand10 = pool_rng.random((100, 5)) < 0.1
        tok_mean = np.zeros((5, 100))
        for tid in range(5):
            tok_mean[tid] = np.mean([c for t, c in codes if t == tid], axis=0)
        tuned20 = np.zeros((100, 5), dtype=bool)
        for tid in range(5):
            tuned20[np.argsort(tok_mean[tid])[-20:], tid] = True

        def set_pools(adj):
            net.output_adjacency = adj
            net._output_adjacency_f = adj.astype(float)
            net._output_in_degree = adj.sum(axis=0)

        print(f"### network seed {seed}")
        t3 = [" ".join(word(TID[t]) for t in s) for s in schedule[0][0]]
        print(f"  teacher sounds like:  {' | '.join(t3)}\n")

        for arm, adj in (("random-pool mouth (10 neurons/word)", rand10),
                         ("tuned-pool mouth (20 neurons/word)", tuned20),
                         ("oracle readout (whole population)", None)):
            restore(net, snap)
            if adj is not None:
                set_pools(adj)
            lines, n_full = [], 0
            for bi, (teach_sents, subj) in enumerate(schedule):
                for sent in teach_sents:
                    for tok in sent:
                        net.step(EYE[TID[tok]])
                state = net.step(EYE[TID[subj]])
                spoken = []
                for _ in range(3):
                    state = net.step(ZERO)
                    if adj is None:
                        pat = state.spiked.astype(float)
                        pat = pat - pat.mean()
                        a = codes[int(np.argmax([c @ pat
                                                 for _, c in codes]))][0]
                    else:
                        a = int(np.argmax(state.outputs))
                    spoken.append(a)
                    state = net.step(EYE[a])
                w = [TOKENS[a] for a in spoken]
                full = w[0] in VERBS and w[1] in NOUNS and w[2] == "space"
                n_full += full
                if bi < SHOW:
                    lines.append(f"{subj:>3} → {word(spoken[0])} "
                                 f"{word(spoken[1])} {word(spoken[2])} "
                                 f"{'✓' if full else '✗'}")
            print(f"  {arm}   [{n_full}/{N_BLOCKS} full patterns]")
            for ln in lines:
                print(f"    {ln}")
            print()


if __name__ == "__main__":
    main()
