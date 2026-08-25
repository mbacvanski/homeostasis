"""Language stage 1b: SPARSE input projections (the 2021 wiring), characters
represented by their evoked activity.

Wiring: each of the 65 character input nodes connects to a random ~10% of the
500 reservoir neurons at weight +5 (zeros elsewhere) - the 2021 paper's input
design at character granularity. Two arms: inputs frozen (faithful 2021) and
inputs plastic (our extension).

A character's representation is its mean evoked spike pattern (500-vector,
averaged over all its post-settle occurrences), centered across characters
before cosine comparison. Tests, per arm and seed:
  1. surprisal: Spearman(mean activation, bigram surprisal) and same for
     spike rate - the 2021 Fig. 11 effect, expected to appear now;
  2. class structure of evoked responses (within vs between cosine, z vs
     2000 label permutations);
  3. frequency structure: rho(response similarity, -|d log freq|), plus
     rho(response magnitude, log freq) - compensation predicts frequent
     characters evoke SMALLER responses.

Usage: python scripts/stage1b_sparse.py [--steps 400000] [--seeds 3]
"""

from __future__ import annotations

import os

for _v in ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import pathlib
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from homeostasis import HomeostaticReservoir, ReservoirConfig
from homeostasis.text import CharStream, char_class

ROOT = pathlib.Path(__file__).resolve().parent.parent
SETTLE = 20000


def build_config(n_tokens: int, n_nodes: int, plastic: bool) -> ReservoirConfig:
    return ReservoirConfig(
        n_nodes=n_nodes,
        n_inputs=n_tokens,
        n_outputs=2,
        p_link=0.1,
        input_p_link=0.1,          # SPARSE: each char -> ~10% of neurons
        input_weight=5.0,          # 2021: +5 on existing links
        weight_init_mean=0.0,
        weight_init_sd=1.0,        # 2021: recurrent ~ Normal(0, 1)
        leak=0.25,
        target_lr=0.01,
        weight_lr=0.1,             # 2021: L_Wx = 0.1
        input_plastic=plastic,
        clamp_negative_activations=True,
    )


def spearman(a, b):
    def ranks(x):
        o = np.argsort(x)
        r = np.empty(len(x))
        r[o] = np.arange(len(x))
        return r
    return float(np.corrcoef(ranks(np.asarray(a)), ranks(np.asarray(b)))[0, 1])


def run_one(task):
    seed, steps, n_nodes, plastic = task
    stream = CharStream.from_file(ROOT / "data/tinyshakespeare.txt", limit=steps)
    v = stream.n_tokens
    cfg = build_config(v, n_nodes, plastic)
    net = HomeostaticReservoir(cfg, seed=seed)
    w0 = net.input_weights.copy()

    mean_act = np.empty(len(stream))
    prop_spiked = np.empty(len(stream))
    resp_sum = np.zeros((v, n_nodes))
    resp_n = np.zeros(v)
    for t in range(len(stream)):
        cid = stream.ids[t]
        state = net.step(stream.one_hot(cid))
        mean_act[t] = float(np.mean(state.x))
        prop_spiked[t] = state.prop_spiked
        if t >= SETTLE:
            resp_sum[cid] += state.spiked
            resp_n[cid] += 1

    seen = resp_n > 0
    R = resp_sum[seen] / resp_n[seen, None]          # evoked spike patterns
    vocab = [stream.vocab[i] for i in np.flatnonzero(seen)]
    classes = np.array([char_class(ch) for ch in vocab])
    freq = np.log(resp_n[seen] + 1.0)

    Rc = R - R.mean(axis=0, keepdims=True)
    Rn = Rc / np.maximum(np.linalg.norm(Rc, axis=1, keepdims=True), 1e-12)
    sim = Rn @ Rn.T
    k = len(vocab)
    iu = np.triu_indices(k, 1)
    same = classes[iu[0]] == classes[iu[1]]
    within, between = float(sim[iu][same].mean()), float(sim[iu][~same].mean())
    rng = np.random.default_rng(seed)
    gaps = []
    for _ in range(2000):
        perm = rng.permutation(classes)
        sp = perm[iu[0]] == perm[iu[1]]
        gaps.append(sim[iu][sp].mean() - sim[iu][~sp].mean())
    gaps = np.array(gaps)
    z = float((within - between - gaps.mean()) / max(gaps.std(), 1e-12))

    rho_freq_sim = spearman(-np.abs(freq[iu[0]] - freq[iu[1]]), sim[iu])
    rho_freq_mag = spearman(R.sum(axis=1), freq)      # evoked size vs frequency

    surp = stream.bigram_surprisal()[SETTLE:]
    rho_act = spearman(mean_act[SETTLE:], surp)
    rho_spk = spearman(prop_spiked[SETTLE:], surp)

    pairs = sorted(((float(sim[i, j]), vocab[i], vocab[j])
                    for i, j in zip(*iu)), reverse=True)[:10]
    return {
        "seed": seed, "plastic": plastic,
        "gap_z": z, "within": within, "between": between,
        "rho_freq_sim": rho_freq_sim, "rho_freq_mag": rho_freq_mag,
        "rho_act_surprisal": rho_act, "rho_spike_surprisal": rho_spk,
        "prop_spiked": float(prop_spiked[SETTLE:].mean()),
        "input_drift": float(np.abs(net.input_weights - w0).mean()),
        "top_pairs": [(f"{a!r}~{b!r}", round(s, 3)) for s, a, b in pairs],
        "sim": sim.tolist(), "vocab": vocab, "classes": classes.tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=400000)
    ap.add_argument("--n-nodes", type=int, default=500)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", type=str, default="scripts/out/stage1b")
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    tasks = [(s, args.steps, args.n_nodes, plastic)
             for plastic in (False, True) for s in range(args.seeds)]
    t0 = time.perf_counter()
    with ProcessPoolExecutor(min(len(tasks), 10)) as pool:
        results = list(pool.map(run_one, tasks))
    print(f"{len(results)} runs of {args.steps} steps in {time.perf_counter()-t0:.0f}s\n")

    for arm in (False, True):
        label = "plastic inputs" if arm else "frozen inputs (2021-faithful)"
        print(f"--- {label} ---")
        for r in [x for x in results if x["plastic"] == arm]:
            print(f"seed {r['seed']}: class z = {r['gap_z']:+.1f} "
                  f"(within {r['within']:.3f} / between {r['between']:.3f}) | "
                  f"freq: sim rho {r['rho_freq_sim']:+.2f}, magnitude rho "
                  f"{r['rho_freq_mag']:+.2f} | surprisal rho: act "
                  f"{r['rho_act_surprisal']:+.3f} spk {r['rho_spike_surprisal']:+.3f} | "
                  f"spike {r['prop_spiked']:.2f} | drift {r['input_drift']:.3f}")
        r0 = [x for x in results if x["plastic"] == arm][0]
        print(f"  top pairs: {', '.join(f'{p}({s})' for p, s in r0['top_pairs'][:6])}\n")

    (out / "stage1b_results.json").write_text(json.dumps(results))
    print(f"saved {out}/stage1b_results.json")


if __name__ == "__main__":
    main()
