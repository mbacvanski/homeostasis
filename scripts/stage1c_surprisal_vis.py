"""Visualize the surprisal effect and test whether it reaches spikes at a lag.

One frozen-input sparse run on tiny shakespeare. Outputs:
  1. a text-aligned panel: ~110 actual characters with the network's mean
     activation and the bigram surprisal of each character;
  2. an event-triggered average: mean activation in a window around
     high-surprisal characters (top decile) vs low (bottom decile);
  3. lagged correlations: Spearman(spike fraction at t+k, surprisal at t)
     for k = 0..6 - does subthreshold surprise convert to spikes later?
"""

from __future__ import annotations

import os

for _v in ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from stage1b_sparse import build_config, spearman  # noqa: E402

from homeostasis import HomeostaticReservoir  # noqa: E402
from homeostasis.text import CharStream  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
SETTLE = 20000
STEPS = 200000

stream = CharStream.from_file(ROOT / "data/tinyshakespeare.txt", limit=STEPS)
net = HomeostaticReservoir(build_config(stream.n_tokens, 500, plastic=False), seed=0)
mean_act = np.empty(len(stream))
prop_spk = np.empty(len(stream))
for t in range(len(stream)):
    state = net.step(stream.one_hot(stream.ids[t]))
    mean_act[t] = float(np.mean(state.x))
    prop_spk[t] = state.prop_spiked

surp = stream.bigram_surprisal()
act, spk, sp = mean_act[SETTLE:], prop_spk[SETTLE:], surp[SETTLE:]

print("lagged Spearman(spike fraction at t+k, surprisal at t):")
for k in range(7):
    r = spearman(spk[k:], sp[: len(sp) - k if k else None])
    print(f"  lag {k}: {r:+.3f}")
print("lagged Spearman(mean activation at t+k, surprisal at t):")
for k in range(4):
    r = spearman(act[k:], sp[: len(sp) - k if k else None])
    print(f"  lag {k}: {r:+.3f}")

# event-triggered averages
hi = np.quantile(sp, 0.9)
lo = np.quantile(sp, 0.1)
win = np.arange(-5, 11)


def eta(mask):
    idx = np.flatnonzero(mask)
    idx = idx[(idx > SETTLE + 5) & (idx < STEPS - 11)]
    return np.array([mean_act[i + win].astype(float) for i in idx]).mean(axis=0), len(idx)


eta_hi, n_hi = eta(surp >= hi)
eta_lo, n_lo = eta(surp <= lo)

# a text window containing a strongly surprising character
cand = np.flatnonzero(surp >= np.quantile(sp, 0.995))
cand = cand[(cand > SETTLE + 200) & (cand < STEPS - 200)]
c0 = int(cand[len(cand) // 2]) - 55
seg = slice(c0, c0 + 110)
chars = [stream.vocab[i] for i in stream.ids[seg]]

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(13, 9.5), height_ratios=[1.3, 1, 1])
xs = np.arange(len(chars))
ax1.plot(xs, mean_act[seg], color="tab:blue", lw=1.4, label="network mean activation")
ax1b = ax1.twinx()
ax1b.bar(xs, surp[seg], color="tab:red", alpha=0.35, label="character surprisal (bits)")
ax1.set_xticks(xs)
ax1.set_xticklabels([c if c != "\n" else "⏎" for c in chars], fontsize=6, family="monospace")
ax1.set_ylabel("mean activation", color="tab:blue")
ax1b.set_ylabel("surprisal, −log2 P(char | prev)", color="tab:red")
ax1.set_title("110 characters of the actual stream: activation (blue line) vs how improbable each character is (red bars)")

ax2.plot(win, eta_hi, color="tab:red", lw=2, label=f"surprising characters (top 10%, n={n_hi})")
ax2.plot(win, eta_lo, color="tab:gray", lw=2, label=f"unsurprising characters (bottom 10%, n={n_lo})")
ax2.axvline(0, color="black", lw=0.8, ls=":")
ax2.set_xlabel("steps relative to the character's arrival (0 = it arrives)")
ax2.set_ylabel("mean activation")
ax2.set_title("Average network response around a character, split by its improbability")
ax2.legend(fontsize=9)

lags = np.arange(7)
r_spk = [spearman(spk[k:], sp[: len(sp) - k if k else None]) for k in lags]
r_act = [spearman(act[k:], sp[: len(sp) - k if k else None]) for k in lags]
ax3.plot(lags, r_act, "o-", color="tab:blue", label="mean activation (subthreshold)")
ax3.plot(lags, r_spk, "s-", color="tab:orange", label="spike fraction")
ax3.axhline(0, color="gray", lw=0.8)
ax3.set_xlabel("lag k: correlate surprisal at t with response at t+k")
ax3.set_ylabel("Spearman correlation")
ax3.set_title("Does the surprisal signal reach the spikes at a delay?")
ax3.legend(fontsize=9)
fig.tight_layout()
out = ROOT / "scripts/out/stage1b/surprisal_vis.png"
fig.savefig(out, dpi=150)
print(f"saved {out}")
