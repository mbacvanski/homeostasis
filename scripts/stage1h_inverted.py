"""Decode the suppression: can an INVERTED readout predict the next char?

Stage 1f showed the bigram-modal next char is ~10x below chance as the BEST
match to the silent pattern (steps 2-3) yet elevated in the top-5. Two
questions this script answers with the same probes:

1. Inverted decoder: predict the candidate whose code correlates LEAST with
   the silent pattern. Scored against the char that ACTUALLY came next in
   the stream, vs three baselines on the same probes: chance (1/n), the
   bigram model's own modal prediction (the ceiling for any bigram-informed
   signal), and always-predict-most-frequent.
2. Echo identity: what occupies rank 1? Hypothesis: the just-seen char's own
   code. Measured as the fraction of probes where the previous char's code
   is the top match, plus mean normalized ranks (0 = top, 1 = bottom) for
   the previous char, the bigram-modal char, and the actual next char.
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
from stage1b_sparse import build_config  # noqa: E402
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

    zero = np.zeros(v)

    def score(pattern, p_id, nxt, codes, counts, avail_mask):
        avail = np.flatnonzero(avail_mask)
        if len(avail) < 5:
            return None
        C = codes[avail] - codes[avail].mean(axis=0, keepdims=True)
        pat = pattern - pattern.mean()
        num = C @ pat
        den = np.linalg.norm(C, axis=1) * max(np.linalg.norm(pat), 1e-12)
        corrs = num / np.maximum(den, 1e-12)
        order = avail[np.argsort(corrs)[::-1]]        # best match first
        n_cand = len(avail)
        pos = {c: i for i, c in enumerate(order)}
        ml = avail[np.argmax(big[p_id, avail])]       # bigram-modal char
        mf = avail[np.argmax(counts[avail])]          # most frequent char
        row = {"n_cand": n_cand, "chance1": 1.0 / n_cand,
               "chance5": min(5.0 / n_cand, 1.0),
               "nrank_modal": pos[ml] / (n_cand - 1),
               "top_char": int(order[0]),
               "echo_avail": float(p_id in pos)}
        if p_id in pos:
            row["echo_top1"] = float(order[0] == p_id)
            row["nrank_echo"] = pos[p_id] / (n_cand - 1)
        if nxt in pos:                                # decoder metrics need the
            row["covered"] = 1.0                      # true next char scoreable
            row["inv1"] = float(order[-1] == nxt)
            row["inv5"] = float(nxt in order[-5:])
            row["fwd1"] = float(order[0] == nxt)
            row["fwd5"] = float(nxt in order[:5])
            row["bigram1"] = float(ml == nxt)
            row["freq1"] = float(mf == nxt)
            row["nrank_next"] = pos[nxt] / (n_cand - 1)
        else:
            row["covered"] = 0.0
        return row

    buckets = {}   # (scoring, lag) -> list of dicts
    t = TRAIN
    for _ in range(N_PROBES):
        for _ in range(GAP):
            net.step(stream.one_hot(stream.ids[t]))
            t += 1
        snap = snapshot(net)
        p_id = stream.ids[t - 1]
        nxt = stream.ids[t]                            # actual next char
        cond_codes = cond_sum[p_id] / np.maximum(cond_n[p_id][:, None], 1)
        for lag in (1, 2, 3):
            state = net.step(zero)
            pat = state.spiked.astype(float)
            s_u = score(pat, p_id, nxt, uncond, uncond_n, uncond_n >= MIN_COUNT)
            s_c = score(pat, p_id, nxt, cond_codes, cond_n[p_id],
                        cond_n[p_id] >= MIN_COUNT)
            for name, sc in (("unconditional", s_u), ("conditioned", s_c)):
                if sc is not None:
                    buckets.setdefault((name, lag), []).append(sc)
        restore(net, snap)

    from collections import Counter

    out = {"seed": seed}
    for key, rows in buckets.items():
        agg = {"n": len(rows)}
        cnt = Counter(r["top_char"] for r in rows)
        agg["rank1_ids"] = [(stream.vocab[i], round(c / len(rows), 3))
                            for i, c in cnt.most_common(3)]
        for f in ("n_cand", "chance1", "chance5", "covered", "echo_avail",
                  "nrank_modal"):
            agg[f] = float(np.mean([r[f] for r in rows]))
        for f in ("echo_top1", "nrank_echo"):
            vals = [r[f] for r in rows if f in r]
            agg[f] = float(np.mean(vals)) if vals else float("nan")
        for f in ("inv1", "inv5", "fwd1", "fwd5", "bigram1", "freq1",
                  "nrank_next"):
            vals = [r[f] for r in rows if r["covered"]]
            agg[f] = float(np.mean([r[f] for r in rows if f in r]))
        out["|".join(str(x) for x in key)] = agg
    return out


def main():
    t0 = time.perf_counter()
    with ProcessPoolExecutor(3) as pool:
        results = list(pool.map(run_one, [0, 1, 2]))
    print(f"3 seeds in {time.perf_counter()-t0:.0f}s. Decoder target = the char "
          f"that actually followed the probe context in the stream.\n"
          f"inv1/inv5 = predict LEAST-correlated candidate(s); "
          f"bigram1 = bigram model's own accuracy (ceiling); "
          f"nrank: 0 = best match, 1 = worst.\n")
    keys = sorted({k for r in results for k in r if k != "seed"})
    for k in keys:
        rows = [r[k] for r in results if k in r]
        m = {f: np.nanmean([x[f] for x in rows]) for f in rows[0]
             if f != "rank1_ids"}
        print(f"  {k:>16}: inv1 {m['inv1']:.3f} (chance {m['chance1']:.3f}) | "
              f"inv5 {m['inv5']:.3f} (chance {m['chance5']:.3f}) | "
              f"fwd1 {m['fwd1']:.3f} | bigram1 {m['bigram1']:.3f} | "
              f"freq1 {m['freq1']:.3f}")
        print(f"  {'':>16}  nrank: next {m['nrank_next']:.2f} "
              f"modal {m['nrank_modal']:.2f} echo {m['nrank_echo']:.2f} | "
              f"echo top1 {m['echo_top1']:.3f} (echo avail {m['echo_avail']:.2f}) | "
              f"covered {m['covered']:.2f} of n ~{int(m['n'])}")
        ids = "; ".join(
            "seed{}: {}".format(r["seed"], ", ".join(
                f"{ch!r} {fr:.2f}" for ch, fr in r[k]["rank1_ids"]))
            for r in results if k in r)
        print(f"  {'':>16}  rank-1 identity: {ids}")


if __name__ == "__main__":
    main()
