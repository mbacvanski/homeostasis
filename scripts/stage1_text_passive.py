"""Language stage 1: passive entrainment on tiny shakespeare with PLASTIC
input weights (learned character embeddings).

The reservoir listens to a character stream (one char per step, one-hot into
dense plastic input projections) with no effectors and no environment
response — the 2021 setup at corpus scale. Two preregistered checkpoints:

1. EMBEDDINGS: after exposure, each character's learned input-weight vector
   is its embedding. Test: within-class cosine similarity (vowel, consonant,
   uppercase, whitespace, punctuation) exceeds between-class, against a
   permutation null. Input wiring is dense and identically initialized, so
   any structure is learned from context, not drawn.
2. SURPRISAL: mean network activation per step correlates positively with
   bigram surprisal of the current character (the 2021 Fig. 11 effect,
   graded, at character level).

Usage: python scripts/stage1_text_passive.py [--steps 150000] [--seeds 3]
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
SETTLE = 20000  # steps excluded from the surprisal analysis (halved if run is short)


def build_config(n_tokens: int, n_nodes: int) -> ReservoirConfig:
    # 2021-flavored settings: clamp on, L_Wx = 0.1, target_lr = 0.01. Input
    # wiring dense with a small constant weight chosen to deliver the same
    # total drive per active token as the 2021 model (5.0 into ~10% of nodes).
    return ReservoirConfig(
        n_nodes=n_nodes,
        n_inputs=n_tokens,
        n_outputs=2,               # unused; readout ignored in stage 1
        p_link=0.1,
        input_p_link=1.0,
        input_weight=1.0,          # per-node drive 1.0/step -> x* = 4 > threshold
        input_weight_sd=0.1,       # class-free symmetry breaking
        weight_init_mean=0.0,
        weight_init_sd=1.0,        # 2021: recurrent weights ~ Normal(0, 1)
        leak=0.25,
        target_lr=0.01,
        weight_lr=0.1,             # 2021: L_Wx = 0.1
        input_plastic=True,
        clamp_negative_activations=True,  # 2021: activation floor at 0
    )


def run_one(task):
    seed, steps, n_nodes = task
    stream = CharStream.from_file(ROOT / "data/tinyshakespeare.txt", limit=steps)
    cfg = build_config(stream.n_tokens, n_nodes)
    net = HomeostaticReservoir(cfg, seed=seed)
    w0 = net.input_weights.copy()

    mean_act = np.empty(len(stream))
    prop_spiked = np.empty(len(stream))
    for t in range(len(stream)):
        state = net.step(stream.one_hot(stream.ids[t]))
        mean_act[t] = float(np.mean(state.x))
        prop_spiked[t] = state.prop_spiked

    # -- checkpoint 1: embedding structure ---------------------------------
    # Center rows before cosine: the shared mean component (every char drives
    # every node positively) compresses raw cosines toward 1 and hides the
    # differential structure, as with word-frequency components in word2vec.
    W = net.input_weights  # (n_tokens, N), dense
    classes = [char_class(ch) for ch in stream.vocab]
    Wc = W - W.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(Wc, axis=1, keepdims=True)
    Wn = Wc / np.maximum(norms, 1e-12)
    sim = Wn @ Wn.T
    v = stream.n_tokens
    iu = np.triu_indices(v, k=1)
    same = np.array([classes[i] == classes[j] for i, j in zip(*iu)])
    within = float(sim[iu][same].mean())
    between = float(sim[iu][~same].mean())
    # permutation null for the within-between gap
    rng = np.random.default_rng(seed)
    gaps = []
    labels = np.array(classes)
    for _ in range(2000):
        perm = rng.permutation(labels)
        s_perm = np.array([perm[i] == perm[j] for i, j in zip(*iu)])
        gaps.append(sim[iu][s_perm].mean() - sim[iu][~s_perm].mean())
    gaps = np.array(gaps)
    z = float((within - between - gaps.mean()) / max(gaps.std(), 1e-12))

    # -- checkpoint 2: surprisal tracking ----------------------------------
    settle = SETTLE if len(stream) > 2 * SETTLE else len(stream) // 2
    surp = stream.bigram_surprisal()[settle:]
    act = mean_act[settle:]
    def spearman(a, b):
        def ranks(x):
            o = np.argsort(x); r = np.empty(len(x)); r[o] = np.arange(len(x)); return r
        return float(np.corrcoef(ranks(a), ranks(b))[0, 1])
    rho = spearman(act, surp)
    rho_spike = spearman(prop_spiked[settle:], surp)

    # frequency hypothesis: are similar-frequency characters more similar?
    freq = np.bincount(stream.ids, minlength=v).astype(float)
    freq = np.log(freq + 1.0)
    freq_dist = np.abs(freq[iu[0]] - freq[iu[1]])
    rho_freq = spearman(-freq_dist, sim[iu])

    # most-similar pairs, for the qualitative readout
    pairs = sorted(
        ((float(sim[i, j]), stream.vocab[i], stream.vocab[j]) for i, j in zip(*iu)),
        reverse=True)[:12]

    return {
        "seed": seed,
        "within": within, "between": between, "gap_z": z,
        "rho_act_surprisal": rho,
        "rho_spike_surprisal": rho_spike,
        "rho_freq_similarity": rho_freq,
        "prop_spiked": float(prop_spiked[settle:].mean()),
        "embed_drift": float(np.abs(W - w0).mean()),
        "top_pairs": [(f"{a!r}~{b!r}", round(s, 3)) for s, a, b in pairs],
        "sim": sim.tolist(), "vocab": stream.vocab, "classes": classes,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=150000)
    ap.add_argument("--n-nodes", type=int, default=500)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", type=str, default="scripts/out/stage1")
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    tasks = [(s, args.steps, args.n_nodes) for s in range(args.seeds)]
    with ProcessPoolExecutor(min(args.seeds, 10)) as pool:
        results = list(pool.map(run_one, tasks))
    print(f"{len(results)} runs of {args.steps} steps in {time.perf_counter()-t0:.0f}s\n")

    for r in results:
        print(f"seed {r['seed']}: class gap z = {r['gap_z']:+.1f} "
              f"(within {r['within']:.3f} vs between {r['between']:.3f}) | "
              f"freq-similarity rho = {r['rho_freq_similarity']:+.2f} | "
              f"surprisal rho: act {r['rho_act_surprisal']:+.3f} "
              f"spikes {r['rho_spike_surprisal']:+.3f} | "
              f"spike {r['prop_spiked']:.2f} | drift {r['embed_drift']:.3f}")
    print(f"\nseed 0 most-similar embedding pairs: "
          f"{', '.join(f'{p}({s})' for p, s in results[0]['top_pairs'][:8])}")

    (out / "stage1_results.json").write_text(json.dumps(results))

    # similarity-matrix heatmap ordered by class, seed 0
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    r = results[0]
    order = sorted(range(len(r["vocab"])), key=lambda i: (r["classes"][i], r["vocab"][i]))
    sim = np.array(r["sim"])[np.ix_(order, order)]
    labels = [r["vocab"][i] for i in order]
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(sim, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels([repr(c)[1:-1] for c in labels], fontsize=5, rotation=90)
    ax.set_yticklabels([repr(c)[1:-1] for c in labels], fontsize=5)
    fig.colorbar(im, label="cosine similarity of learned embeddings")
    boundaries = [i for i in range(1, len(order))
                  if r["classes"][order[i]] != r["classes"][order[i - 1]]]
    for b in boundaries:
        ax.axhline(b - 0.5, color="black", lw=0.6)
        ax.axvline(b - 0.5, color="black", lw=0.6)
    ax.set_title("Learned character embeddings (input weights), grouped by class")
    fig.tight_layout()
    fig.savefig(out / "stage1_embeddings.png", dpi=150)
    print(f"saved {out}/stage1_results.json and stage1_embeddings.png")


if __name__ == "__main__":
    main()
