"""Fatten the mouth: does pool size close the gap to the ideal readout?

Stage 2a found the grammar is in the state (ideal readout speaks the verb
slot correctly on 93% of trials) but the minimal mouth - 5 random pools of
~10 neurons, speak the pool with the highest spiked fraction - stays at
chance. Same protocol as stage 2a's winning timing (words back-to-back,
one silent think step before each spoken word), sweeping the pool density:
each word's pool connects to 10% / 20% / 30% / 50% of the 100 neurons.
Denser pools average out more noise but overlap more (readings correlate);
at 100% all five readings would be identical, so somewhere in between is
the best a random unstructured mouth can do.
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
from stage1e_grammar_replication import EYE, TID, sample_sentence  # noqa: E402
from stage2a_speaking import (  # noqa: E402
    N_BLOCKS, TEACH_PER_BLOCK, ZERO, make_net, score_trials, warm_up,
)

POOL_DENSITIES = (0.1, 0.2, 0.3, 0.5)


def run_one(seed):
    sched_rng = np.random.default_rng(20000 + seed)
    schedule = [([sample_sentence(sched_rng) for _ in range(TEACH_PER_BLOCK)],
                 "man" if sched_rng.random() < 0.5 else "dog")
                for _ in range(N_BLOCKS)]
    net = make_net(seed)
    codes = warm_up(net, seed, gap=0)
    snap = snapshot(net)
    pool_rng = np.random.default_rng(40000 + seed)
    pool_adjs = {f"pool {d:.0%}": pool_rng.random((100, 5)) < d
                 for d in POOL_DENSITIES}
    # tuned pools: word w's pool = the k neurons that fire most distinctively
    # when w is heard (top-k of its mean centered arrival code)
    tok_mean = np.zeros((5, 100))
    for tid in range(5):
        vecs = [c for t, c in codes if t == tid]
        tok_mean[tid] = np.mean(vecs, axis=0)
    for k in (10, 20):
        adj = np.zeros((100, 5), dtype=bool)
        for tid in range(5):
            adj[np.argsort(tok_mean[tid])[-k:], tid] = True
        pool_adjs[f"tuned {k}"] = adj

    def set_pools(adj):
        net.output_adjacency = adj
        net._output_adjacency_f = adj.astype(float)
        net._output_in_degree = adj.sum(axis=0)

    out = {"seed": seed}
    for arm in list(pool_adjs) + ["oracle"]:
        restore(net, snap)
        if arm != "oracle":
            set_pools(pool_adjs[arm])
        trials = []
        for teach_sents, subj in schedule:
            for sent in teach_sents:
                for tok in sent:
                    net.step(EYE[TID[tok]])
            state = net.step(EYE[TID[subj]])
            spoken = []
            for _ in range(3):
                state = net.step(ZERO)          # the one think step
                if arm == "oracle":
                    pat = state.spiked.astype(float)
                    pat = pat - pat.mean()
                    a = codes[int(np.argmax([c @ pat for _, c in codes]))][0]
                else:
                    a = int(np.argmax(state.outputs))
                spoken.append(a)
                state = net.step(EYE[a])
            trials.append((subj, spoken))
        out[arm] = score_trials(trials)
    return out


def main():
    t0 = time.perf_counter()
    with ProcessPoolExecutor(int(os.environ.get("WORKERS", "10"))) as pool:
        results = list(pool.map(run_one, range(100)))
    arms = ([f"pool {d:.0%}" for d in POOL_DENSITIES]
            + ["tuned 10", "tuned 20", "oracle"])
    print(f"100 networks in {time.perf_counter()-t0:.0f}s; back-to-back "
          f"listening, one think step, learning on; pool = random subset of "
          f"neurons per word, speak the pool with the top spiked fraction\n")
    fields = [
        ("verb_ok",   "verb slot contains a verb        (chance 0.40)"),
        ("verb_fav",  "  ...and it's the likely verb    (teacher 0.75)"),
        ("obj_ok",    "object slot contains a noun      (chance 0.40)"),
        ("end_ok",    "3rd slot is the boundary word    (chance 0.20)"),
        ("full_ok",   "whole pattern verb+noun+boundary (chance 0.032)"),
        ("parrot",    "1st word = subject just heard    (chance 0.20)"),
        ("stuck",     "same word in all 3 slots         (chance 0.04)"),
    ]
    print(f"{'':48s}" + "".join(f"{a:>12s}" for a in arms))
    for f, desc in fields:
        row = ""
        for a in arms:
            vals = np.array([r[a][f] for r in results])
            row += f"  {np.nanmean(vals):.3f}±{np.nanstd(vals):.2f}"
        print(f"{desc:48s}{row}")


if __name__ == "__main__":
    main()
