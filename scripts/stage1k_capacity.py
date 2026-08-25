"""What flips silence from suppression to rollout: overlap or model size?

The grammar world (0.5 tokens/neuron, vocab/N = 0.05) rolls the sequence
forward during silence; the char world (6.5 chars/neuron, vocab/N = 0.13)
suppresses expected continuations (stages 1f-1j). Those two knobs
dissociate:

  arm        vocab  N     input_p  chars/neuron  vocab/N   tests
  base        65    500   0.10     6.5           0.130     (known suppressor)
  reduced27   27    500   0.10     2.7           0.054     both knobs down
  thin-in     65    500   0.04     2.6           0.130     overlap only down
  n2000       65    2000  0.10     6.5           0.033     model size only up

chars/neuron = vocab x input_p_link (how many characters' input territory
each neuron sits in) is INDEPENDENT of N. If reduced27+thin-in improve and
n2000 doesn't, the knob is per-neuron input overlap; if n2000 improves and
thin-in doesn't, it's vocabulary relative to model capacity.

Per arm: the full stage 1h+1j battery (suppression rho with density
partialled out, modal top1/top5 vs analytic chance, forward decoding vs the
bigram ceiling, echo rate, rank-1 identity concentration).

Usage: stage1k_capacity.py --set light   (base, reduced27, thin-in; ~2 min)
       stage1k_capacity.py --set heavy   (n2000; ~13 min, 3 procs)
"""

from __future__ import annotations

import os

for _v in ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import pathlib
import re
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from stage1b_sparse import spearman  # noqa: E402
from stage1d_completion import restore, snapshot  # noqa: E402
from stage1j_density import partial_spearman  # noqa: E402

from homeostasis import HomeostaticReservoir, ReservoirConfig  # noqa: E402
from homeostasis.text import CharStream  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
TRAIN = 300000
SETTLE = 20000
N_PROBES = 600
GAP = 100
MIN_COUNT = 30
LIMIT = TRAIN + N_PROBES * GAP + 1000

ARMS = {
    "base":      dict(vocab="full", n=500, p_in=0.10),
    "reduced27": dict(vocab="reduced", n=500, p_in=0.10),
    "thin-in":   dict(vocab="full", n=500, p_in=0.04),
    "n2000":     dict(vocab="full", n=2000, p_in=0.10),
}
SETS = {"light": ["base", "reduced27", "thin-in"], "heavy": ["n2000"]}


def make_stream(vocab_mode):
    text = (ROOT / "data/tinyshakespeare.txt").read_text()
    if vocab_mode == "reduced":
        text = text.lower().replace("\n", " ")
        text = re.sub(r"[^a-z ]", "", text)
        text = re.sub(r" +", " ", text)
    return CharStream(text[:LIMIT])


def run_one(task):
    arm_name, seed = task
    arm = ARMS[arm_name]
    stream = make_stream(arm["vocab"])
    v = stream.n_tokens
    n = arm["n"]
    cfg = ReservoirConfig(
        n_nodes=n, n_inputs=v, n_outputs=2,
        p_link=0.1, input_p_link=arm["p_in"], input_weight=5.0,
        weight_init_mean=0.0, weight_init_sd=1.0,
        leak=0.25, target_lr=0.01, weight_lr=0.1,
        clamp_negative_activations=True,
    )
    net = HomeostaticReservoir(cfg, seed=seed)

    big = np.full((v, v), 0.5)
    np.add.at(big, (stream.ids[:-1], stream.ids[1:]), 1.0)
    big /= big.sum(axis=1, keepdims=True)

    uncond_sum = np.zeros((v, n))
    uncond_n = np.zeros(v)
    cond_sum = np.zeros((v, v, n))
    cond_n = np.zeros((v, v))
    spike_acc = 0.0
    prev = None
    for t in range(TRAIN):
        cid = stream.ids[t]
        state = net.step(stream.one_hot(cid))
        if t >= TRAIN - 50000:
            spike_acc += state.spiked.mean()
        if t >= SETTLE:
            uncond_sum[cid] += state.spiked
            uncond_n[cid] += 1
            if prev is not None:
                cond_sum[prev, cid] += state.spiked
                cond_n[prev, cid] += 1
        prev = cid
    uncond = uncond_sum / np.maximum(uncond_n[:, None], 1)
    mask_u = uncond_n >= MIN_COUNT
    zero = np.zeros(v)

    def score(pattern, p_id, nxt, codes, avail_mask):
        avail = np.flatnonzero(avail_mask)
        if len(avail) < 8:
            return None
        C = codes[avail] - codes[avail].mean(axis=0, keepdims=True)
        pat = pattern - pattern.mean()
        num = C @ pat
        den = np.linalg.norm(C, axis=1) * max(np.linalg.norm(pat), 1e-12)
        corrs = num / np.maximum(den, 1e-12)
        probs = big[p_id, avail]
        dens = codes[avail].mean(axis=1)
        order = avail[np.argsort(corrs)[::-1]]
        pos = {c: i for i, c in enumerate(order)}
        ml = avail[np.argmax(probs)]
        n_cand = len(avail)
        row = {"n_cand": n_cand, "chance1": 1.0 / n_cand,
               "chance5": min(5.0 / n_cand, 1.0),
               "top1": float(order[0] == ml), "top5": float(ml in order[:5]),
               "rho_p": spearman(corrs, probs),
               "partial": partial_spearman(corrs, probs, dens),
               "top_char": int(order[0]),
               "echo_avail": float(p_id in pos)}
        if p_id in pos:
            row["echo1"] = float(order[0] == p_id)
        if nxt in pos:
            row["fwd1"] = float(order[0] == nxt)
            row["fwd5"] = float(nxt in order[:5])
            row["bigram1"] = float(ml == nxt)
        return row

    buckets = {}
    t = TRAIN
    for _ in range(N_PROBES):
        for _ in range(GAP):
            net.step(stream.one_hot(stream.ids[t]))
            t += 1
        snap = snapshot(net)
        p_id = stream.ids[t - 1]
        nxt = stream.ids[t]
        cond_codes = cond_sum[p_id] / np.maximum(cond_n[p_id][:, None], 1)
        for lag in (1, 2, 3):
            state = net.step(zero)
            pat = state.spiked.astype(float)
            for name, sc in (("uncond", score(pat, p_id, nxt, uncond, mask_u)),
                             ("cond", score(pat, p_id, nxt, cond_codes,
                                            cond_n[p_id] >= MIN_COUNT))):
                if sc is not None:
                    buckets.setdefault((name, lag), []).append(sc)
        restore(net, snap)

    out = {"arm": arm_name, "seed": seed, "vocab": v,
           "spike": spike_acc / 50000}
    for key, rows in buckets.items():
        agg = {"n": len(rows)}
        for f in ("n_cand", "chance1", "chance5", "top1", "top5",
                  "rho_p", "partial", "echo_avail"):
            agg[f] = float(np.mean([r[f] for r in rows]))
        for f in ("echo1", "fwd1", "fwd5", "bigram1"):
            vals = [r[f] for r in rows if f in r]
            agg[f] = float(np.mean(vals)) if vals else float("nan")
        cnt = Counter(r["top_char"] for r in rows)
        agg["rank1"] = [(stream.vocab[i], round(c / len(rows), 2))
                        for i, c in cnt.most_common(2)]
        out["|".join(str(x) for x in key)] = agg
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=list(SETS), default="light")
    args = ap.parse_args()
    arms = SETS[args.set]
    tasks = [(a, s) for a in arms for s in (0, 1, 2)]
    t0 = time.perf_counter()
    with ProcessPoolExecutor(len(tasks)) as pool:
        results = list(pool.map(run_one, tasks))
    print(f"{len(tasks)} runs in {time.perf_counter()-t0:.0f}s "
          f"(3 seeds per arm, means below; rank-1 ids pooled per seed)\n")
    for a in arms:
        rs = [r for r in results if r["arm"] == a]
        arm = ARMS[a]
        v = rs[0]["vocab"]
        print(f"=== {a}: vocab {v}, N={arm['n']}, input_p {arm['p_in']:.2f} "
              f"-> chars/neuron {v*arm['p_in']:.1f}, vocab/N {v/arm['n']:.3f} "
              f"| train spike {np.mean([r['spike'] for r in rs]):.3f} ===")
        for key in sorted(k for k in rs[0] if "|" in k):
            rows = [r[key] for r in rs]
            m = {f: np.nanmean([x[f] for x in rows]) for f in rows[0]
                 if f != "rank1"}
            ids = "; ".join(f"s{r['seed']}:" + ",".join(
                f"{ch!r} {fr:.2f}" for ch, fr in r[key]["rank1"]) for r in rs)
            print(f"  {key:>8}: rho_p {m['rho_p']:+.3f} partial {m['partial']:+.3f}"
                  f" | modal top1 {m['top1']:.3f}/ch {m['chance1']:.3f}"
                  f" top5 {m['top5']:.3f}/ch {m['chance5']:.3f}"
                  f" | fwd1 {m['fwd1']:.3f} fwd5 {m['fwd5']:.3f}"
                  f" bigram1 {m['bigram1']:.3f}"
                  f" | echo1 {m['echo1']:.3f} | cands~{m['n_cand']:.0f}")
            print(f"  {'':>8}  rank-1: {ids}")
        print()


if __name__ == "__main__":
    main()
