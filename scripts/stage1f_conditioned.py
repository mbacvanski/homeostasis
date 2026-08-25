"""Deconfound the char-level completion null: same probes, three scorings.

Trains the frozen sparse network on 300k chars (as stage 1d), then at each
probe (snapshot -> 1 silent step -> restore) scores the silent pattern
against BOTH code libraries:
  - unconditional codes: mean spike pattern per character (stage-1d scoring);
  - conditioned codes: mean spike pattern per (previous char, char) pair
    with >= 30 occurrences - the char-level analog of the 2021 paper's
    position-conditioned codes.
Probes are also split by location: right after whitespace (the analog of
their sentence-boundary anchoring) vs mid-word.
Metrics per scoring: top-1/top-5 for the bigram-most-likely next char among
scoreable candidates, and Spearman(code correlation, P(char | prev)).
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
from stage1b_sparse import build_config, spearman  # noqa: E402
from stage1d_completion import restore, snapshot  # noqa: E402

from homeostasis import HomeostaticReservoir  # noqa: E402
from homeostasis.text import CharStream  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
TRAIN = 300000
SETTLE = 20000
N_PROBES = 600
GAP = 100
MIN_COUNT = 30


def run_one(seed):
    stream = CharStream.from_file(ROOT / "data/tinyshakespeare.txt",
                                  limit=TRAIN + N_PROBES * GAP + 1000)
    v = stream.n_tokens
    net = HomeostaticReservoir(build_config(v, 500, plastic=False), seed=seed)

    big = np.full((v, v), 0.5)
    np.add.at(big, (stream.ids[:-1], stream.ids[1:]), 1.0)
    big /= big.sum(axis=1, keepdims=True)

    uncond_sum = np.zeros((v, 500))
    uncond_n = np.zeros(v)
    cond_sum = np.zeros((v, v, 500))
    cond_n = np.zeros((v, v))
    prev = None
    for t in range(TRAIN):
        cid = stream.ids[t]
        state = net.step(stream.one_hot(cid))
        if t >= SETTLE:
            uncond_sum[cid] += state.spiked
            uncond_n[cid] += 1
            if prev is not None:
                cond_sum[prev, cid] += state.spiked
                cond_n[prev, cid] += 1
        prev = cid
    uncond = uncond_sum / np.maximum(uncond_n[:, None], 1)

    ws_ids = {stream.char_to_id[c] for c in " \n" if c in stream.char_to_id}
    zero = np.zeros(v)

    def score(pattern, p_id, codes, avail_mask):
        avail = np.flatnonzero(avail_mask)
        if len(avail) < 5:
            return None
        C = codes[avail] - codes[avail].mean(axis=0, keepdims=True)
        pat = pattern - pattern.mean()
        num = C @ pat
        den = np.linalg.norm(C, axis=1) * max(np.linalg.norm(pat), 1e-12)
        corrs = num / np.maximum(den, 1e-12)
        probs = big[p_id, avail]
        order = avail[np.argsort(corrs)[::-1]]
        ml = avail[np.argmax(probs)]
        return {"top1": float(order[0] == ml),
                "top5": float(ml in order[:5]),
                "rho": spearman(corrs, probs)}

    buckets = {}   # (scoring, location) -> list of dicts
    t = TRAIN
    for _ in range(N_PROBES):
        for _ in range(GAP):
            net.step(stream.one_hot(stream.ids[t]))
            t += 1
        snap = snapshot(net)
        p_id = stream.ids[t - 1]
        state = net.step(zero)
        restore(net, snap)
        pat = state.spiked.astype(float)
        loc = "after-space" if p_id in ws_ids else "mid-word"
        s_u = score(pat, p_id, uncond, uncond_n >= MIN_COUNT)
        cond_codes = cond_sum[p_id] / np.maximum(cond_n[p_id][:, None], 1)
        s_c = score(pat, p_id, cond_codes, cond_n[p_id] >= MIN_COUNT)
        for name, s in (("unconditional", s_u), ("conditioned", s_c)):
            if s is not None:
                buckets.setdefault((name, loc), []).append(s)
                buckets.setdefault((name, "all"), []).append(s)
    out = {"seed": seed}
    for key, rows in buckets.items():
        out["|".join(key)] = {
            "n": len(rows),
            "top1": float(np.mean([r["top1"] for r in rows])),
            "top5": float(np.mean([r["top5"] for r in rows])),
            "rho": float(np.mean([r["rho"] for r in rows])),
        }
    return out


def main():
    t0 = time.perf_counter()
    with ProcessPoolExecutor(3) as pool:
        results = list(pool.map(run_one, [0, 1, 2]))
    print(f"3 seeds in {time.perf_counter()-t0:.0f}s (silence step 1 only; "
          f"chance top-1 ~ 1/n_candidates)\n")
    keys = sorted({k for r in results for k in r if k != "seed"})
    for k in keys:
        rows = [r[k] for r in results if k in r]
        print(f"  {k:>28}: top1 {np.mean([x['top1'] for x in rows]):.3f} | "
              f"top5 {np.mean([x['top5'] for x in rows]):.3f} | "
              f"rho {np.mean([x['rho'] for x in rows]):+.3f} | "
              f"n/seed ~{int(np.mean([x['n'] for x in rows]))}")


if __name__ == "__main__":
    main()
