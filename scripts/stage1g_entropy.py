"""Predictability knob in the 2021 world: degrade the grammar's transition
probability from 0.75 toward 0.55 (uniform = 0.5) under the exact 2021
protocol, and watch the completion correlations.
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
import stage1e_grammar_replication as g  # noqa: E402


def run_level(args):
    p, seed = args
    g.VERB_P = {"man": ("walks", p), "dog": ("bites", p)}
    g.OBJ_P = {"walks": ("dog", p), "bites": ("man", p)}
    return g.run_one(seed)


def main():
    for p in (0.75, 0.65, 0.55):
        t0 = time.perf_counter()
        with ProcessPoolExecutor(10) as pool:
            results = list(pool.map(run_level, [(p, s) for s in range(100)]))
        fav = np.mean([r["man->silence ~ walks_verb"] for r in results])
        unf = np.mean([r["man->silence ~ bites_verb"] for r in results])
        fav_o = np.mean([r["man,walks->silence ~ dog_obj"] for r in results])
        unf_o = np.mean([r["man,walks->silence ~ man_obj"] for r in results])
        print(f"P(favored)={p:.2f}: verb slot {fav:+.3f} vs {unf:+.3f} "
              f"(gap {fav-unf:+.3f}) | object slot {fav_o:+.3f} vs {unf_o:+.3f} "
              f"(gap {fav_o-unf_o:+.3f})  [{time.perf_counter()-t0:.0f}s]")


if __name__ == "__main__":
    main()
