"""Stage 1d: fading-memory completion (the 2021 paper's Table 2, at character
level on tiny shakespeare).

Train a frozen-input sparse network on the stream while building each
character's "code" (mean evoked spike pattern). Then, at probe points: freeze
a snapshot of the full network state, feed SILENCE (all-zero input) for 3
steps recording the endogenous spike pattern, restore the snapshot, and
continue. If the network pattern-completes, the silent activity should
resemble the code of the character that WOULD have come next.

Metrics per silence step, over ~400 probes x 3 seeds:
  - top-1 / top-5: is the bigram-most-likely next character's code the best
    correlate of the silent pattern, among all 65? (chance ~1.5% / ~7.7%)
  - same for the character that ACTUALLY came next in the text;
  - graded tracking: Spearman between per-character code correlation and
    per-character bigram probability given the last seen character;
  - shuffle control: same top-1 computed with mismatched probe/context pairs.
"""

from __future__ import annotations

import os

for _v in ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
import pathlib
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from stage1b_sparse import build_config, spearman  # noqa: E402

from homeostasis import HomeostaticReservoir  # noqa: E402
from homeostasis.text import CharStream  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
TRAIN = 300000
SETTLE = 20000
N_PROBES = 400
GAP = 100          # steps of normal streaming between probes
LAGS = 3           # silent steps recorded per probe


def snapshot(net):
    return (net.x.copy(), net.spiked.copy(), net._spiked_f.copy(),
            net.targets.copy(), net.weights.copy(), net.t)


def restore(net, snap):
    net.x, net.spiked, net._spiked_f, net.targets, net.weights, net.t = (
        snap[0].copy(), snap[1].copy(), snap[2].copy(), snap[3].copy(),
        snap[4].copy(), snap[5])


def run_one(seed):
    stream = CharStream.from_file(ROOT / "data/tinyshakespeare.txt",
                                  limit=TRAIN + N_PROBES * GAP + 1000)
    v = stream.n_tokens
    net = HomeostaticReservoir(build_config(v, 500, plastic=False), seed=seed)

    # bigram model over the whole corpus slice (row-normalized counts)
    big = np.full((v, v), 0.5)
    np.add.at(big, (stream.ids[:-1], stream.ids[1:]), 1.0)
    big /= big.sum(axis=1, keepdims=True)

    codes_sum = np.zeros((v, net.config.n_nodes))
    codes_n = np.zeros(v)
    for t in range(TRAIN):
        cid = stream.ids[t]
        state = net.step(stream.one_hot(cid))
        if t >= SETTLE:
            codes_sum[cid] += state.spiked
            codes_n[cid] += 1
    codes = codes_sum / np.maximum(codes_n[:, None], 1)
    codes_c = codes - codes.mean(axis=0, keepdims=True)
    zero = np.zeros(v)

    def corr_all(pattern):
        p = pattern - pattern.mean()
        num = codes_c @ p
        den = np.linalg.norm(codes_c, axis=1) * max(np.linalg.norm(p), 1e-12)
        return num / np.maximum(den, 1e-12)

    probes = []   # dicts: last (char id), actual_next, corrs per lag
    t = TRAIN
    for _ in range(N_PROBES):
        for _ in range(GAP):
            net.step(stream.one_hot(stream.ids[t]))
            t += 1
        snap = snapshot(net)
        last, nxt = stream.ids[t - 1], stream.ids[t]
        lag_corrs = []
        for _k in range(LAGS):
            state = net.step(zero)
            lag_corrs.append(corr_all(state.spiked.astype(float)))
        restore(net, snap)
        probes.append({"last": int(last), "next": int(nxt),
                       "corrs": np.array(lag_corrs)})

    out = {"seed": seed}
    rng = np.random.default_rng(seed)
    for k in range(LAGS):
        top1_ml = top5_ml = top1_true = rho_sum = 0.0
        top1_shuf = 0.0
        perm = rng.permutation(len(probes))
        for i, p in enumerate(probes):
            c = p["corrs"][k]
            order = np.argsort(c)[::-1]
            ml = int(np.argmax(big[p["last"]]))
            top1_ml += order[0] == ml
            top5_ml += ml in order[:5]
            top1_true += order[0] == p["next"]
            rho_sum += spearman(c, big[p["last"]])
            # shuffle control: this probe's pattern vs another probe's context
            other_ml = int(np.argmax(big[probes[perm[i]]["last"]]))
            top1_shuf += order[0] == other_ml
        n = len(probes)
        out[f"lag{k+1}"] = {
            "top1_mostlikely": top1_ml / n, "top5_mostlikely": top5_ml / n,
            "top1_actual": top1_true / n, "rho_prob": rho_sum / n,
            "top1_shuffled": top1_shuf / n,
        }
    return out


def main():
    t0 = time.perf_counter()
    with ProcessPoolExecutor(3) as pool:
        results = list(pool.map(run_one, [0, 1, 2]))
    print(f"3 seeds in {time.perf_counter()-t0:.0f}s "
          f"({N_PROBES} probes each; chance top-1 = {1/65:.3f}, top-5 = {5/65:.3f})\n")
    for r in results:
        print(f"seed {r['seed']}:")
        for k in range(1, LAGS + 1):
            d = r[f"lag{k}"]
            print(f"  silence step {k}: top-1(most-likely) {d['top1_mostlikely']:.3f} "
                  f"| top-5 {d['top5_mostlikely']:.3f} | top-1(actual next) "
                  f"{d['top1_actual']:.3f} | rho(corr, P(char|ctx)) {d['rho_prob']:+.3f} "
                  f"| shuffled-context top-1 {d['top1_shuffled']:.3f}")
    out = ROOT / "scripts/out/stage1b/completion_results.json"
    out.write_text(json.dumps(results))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
