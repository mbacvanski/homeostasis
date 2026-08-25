"""Is the deep-silence 'suppression' just a code-density gradient?

Stage 1h found rank 1 at silence steps 2-3 is 'z' on ~97% of probes (the
sparsest code), suggesting: expected chars spike densely at arrival (stage
1c), silence decays toward sparseness, and centered cosine then favors
sparse codes - so the negative rho(corr, P) could be density confound, not
targeted suppression. Test: per probe, Spearman of silence-code correlation
against bigram P, against code density, and the PARTIAL rho(corr, P)
controlling for density (rank-regression residuals). If the partial rho is
~0, density explains the suppression; if still negative, per-context
suppression survives the control.
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


def _ranks(v):
    r = np.empty(len(v))
    r[np.argsort(v)] = np.arange(len(v), dtype=float)
    return r


def partial_spearman(a, b, ctrl):
    """Spearman(a, b) with ctrl partialled out of both (rank residuals)."""
    ra, rb, rc = _ranks(a), _ranks(b), _ranks(ctrl)
    rc = rc - rc.mean()
    den = float(rc @ rc)
    if den < 1e-12:
        return spearman(a, b)
    res_a = (ra - ra.mean()) - ((ra - ra.mean()) @ rc / den) * rc
    res_b = (rb - rb.mean()) - ((rb - rb.mean()) @ rc / den) * rc
    d = np.linalg.norm(res_a) * np.linalg.norm(res_b)
    return float(res_a @ res_b / d) if d > 1e-12 else 0.0


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

    # confound check: are frequent/probable chars' codes denser?
    dens_u = uncond.mean(axis=1)
    mask_u = uncond_n >= MIN_COUNT
    rho_dens_freq = spearman(dens_u[mask_u], np.log(uncond_n[mask_u]))

    zero = np.zeros(v)

    def probe_rhos(pattern, p_id, codes, avail_mask):
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
        return {"rho_p": spearman(corrs, probs),
                "rho_d": spearman(corrs, dens),
                "rho_pd": spearman(probs, dens),
                "partial": partial_spearman(corrs, probs, dens)}

    buckets = {}
    t = TRAIN
    for _ in range(N_PROBES):
        for _ in range(GAP):
            net.step(stream.one_hot(stream.ids[t]))
            t += 1
        snap = snapshot(net)
        p_id = stream.ids[t - 1]
        cond_codes = cond_sum[p_id] / np.maximum(cond_n[p_id][:, None], 1)
        for lag in (1, 2, 3):
            state = net.step(zero)
            pat = state.spiked.astype(float)
            for name, r in (("unconditional",
                             probe_rhos(pat, p_id, uncond, mask_u)),
                            ("conditioned",
                             probe_rhos(pat, p_id, cond_codes,
                                        cond_n[p_id] >= MIN_COUNT))):
                if r is not None:
                    buckets.setdefault((name, lag), []).append(r)
        restore(net, snap)

    out = {"seed": seed, "rho_dens_freq": rho_dens_freq}
    for key, rows in buckets.items():
        out["|".join(str(x) for x in key)] = {
            f: float(np.mean([r[f] for r in rows])) for f in rows[0]
        } | {"n": len(rows)}
    return out


def main():
    t0 = time.perf_counter()
    with ProcessPoolExecutor(3) as pool:
        results = list(pool.map(run_one, [0, 1, 2]))
    print(f"3 seeds in {time.perf_counter()-t0:.0f}s. Per-probe Spearman, "
          f"averaged.\n  rho_p = (silence corr, bigram P)   "
          f"rho_d = (silence corr, code density)\n  rho_pd = (P, density)   "
          f"partial = rho_p with density partialled out\n")
    print(f"  rho(code density, log char count) across chars: "
          + ", ".join(f"{r['rho_dens_freq']:+.2f}" for r in results) + "\n")
    keys = sorted({k for r in results for k in r
                   if k not in ("seed", "rho_dens_freq")})
    for k in keys:
        rows = [r[k] for r in results if k in r]
        m = {f: np.mean([x[f] for x in rows]) for f in rows[0]}
        print(f"  {k:>16}: rho_p {m['rho_p']:+.3f} | rho_d {m['rho_d']:+.3f} | "
              f"rho_pd {m['rho_pd']:+.3f} | partial {m['partial']:+.3f} | "
              f"n ~{int(m['n'])}")


if __name__ == "__main__":
    main()
