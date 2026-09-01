# The language track: listening, predicting, speaking

*Distilled 2026-08-31 from the exploratory sprint of 2026-08-25 (20 scripts,
`scripts/stage1*` – `scripts/stage5a_*`). All scripts are seed-deterministic;
every number quoted here either comes from a result file in `scripts/out/`, or
was re-verified by re-running the script on 2026-08-31 (marked ✓), or comes
from the original session's recorded output (marked ○ — the cheap re-runs all
reproduced exactly, so these are trustworthy but not re-executed). The
listening half (stages 1–1k) also has a detailed verification report with
per-experiment protocols and a corrections ledger: the "Language Track Audit"
artifact.*

## Why this track exists

Falandays, Nguyen & Spivey (2021), *"Is prediction nothing more than
multi-scale pattern completion of the future?"* (Brain Research 1768:147578,
PDF in the repo root) is the predecessor of the 2024 paper this repo
replicates: the same homeostatic-reservoir idea, applied to **passive
listening** instead of sensorimotor control. Their claim: a network that only
tries to keep each unit comfortable exhibits prediction-like behavior "for
free" — entrained dynamics pattern-complete the future without any predictive
machinery.

The track asked three questions in order:

1. **Replication** — does our implementation reproduce their language results
   (the surprisal response, fading-memory completion)?
2. **Mechanism** — what physically carries the "prediction"?
3. **Closing the loop** — the 2024 paper closes the loop through effectors
   (tracking, Pong). Can the same network close a loop through a *mouth* —
   speak, and eventually hold a contingent dialogue?

Answers, in one line each: (1) yes, fully; (2) prediction is *absorption* —
expected input is canceled at arrival, which means expectation is
**anti-expressed** in spikes; (3) no — four progressively more careful
dialogue designs all failed to ground a mouth through local homeostatic
credit, and the failures were more informative than the successes.

## The two worlds

| | text world | grammar world (2021 exact) |
|---|---|---|
| stimulus | tiny shakespeare, 1 char/step, one-hot over 65 chars | 5 tokens: [subject verb object space], man/dog × walks/bites, 75/25 transitions |
| reservoir | N=500 | N=100 |
| input wiring | each char → random ~10% of nodes at +5, **frozen** | same |
| params | 2021 values: leak .25, weight_lr .1, target_lr .01, clamp on | same |

"Surprisal" of a character = −log₂ P(c_t \| c_{t−1}) under an add-0.5-smoothed
bigram model fit to the same text ([src/homeostasis/text.py](../src/homeostasis/text.py)).
A token's "code" = its mean evoked spike pattern (centered across tokens).

---

## Part I — Listening (stages 1–1k)

### What replicated ✓

**Fading-memory completion, exact 2021 setting** (stage1e, 100 nets, ~2 s):
feed 'man', cut the input, one silent step; the silent spike pattern
correlates +0.544 ± 0.108 with the *walks*-as-verb code vs +0.365 ± 0.132 for
*bites* (2021 Table 2: 0.467 / 0.319). Object completion after 'man walks':
+0.547 vs +0.250 (theirs 0.452 / 0.213). Every cell has the grammar-predicted
ordering; swapping the subject flips every preference. **The completion
machinery is verified end-to-end.**

**The surprisal response, at character level** (stage1b, on disk):
ρ(mean activation, bigram surprisal) = +0.23…+0.32 across seeds with the
2021 sparse frozen input wiring — their Fig. 11 effect. Predictability scales
it smoothly and never inverts it (stage1g ✓: favored-vs-unfavored completion
gap +0.179/+0.297 (verb/object) at P=.75 → +0.036/+0.058 at P=.55).

### The mechanism: prediction is absorption (stage1c ○)

Expected characters are *absorbed*: network activation dips to its minimum at
their arrival and the input is canceled subthreshold. Surprising characters
fail to cancel and discharge **one step later**: ρ(spike fraction, surprisal)
= −0.227 at lag 0, +0.213 at lag 1, ~0 by lag 3. Two consequences run through
everything downstream:

- The surprisal signal lives in *subthreshold activation*, not spike rate, at
  arrival — and reaches spikes only as a dip-then-burst transient one step
  late. That's what any spike-reading effector actually sees.
- **The most-expected token is the *least* expressed in spikes.** Prediction
  by cancellation means a mouth that reads spikes is fighting the network's
  own expectation mechanism (this resurfaces in stages 2–4).

### Two design choices turned out to be load-bearing

**Frozen input weights** (stage1b, on disk): letting input weights learn under
the same local rule *destroys* the surprisal effect (+0.02…+0.04 vs
+0.23…+0.32 frozen). Each node attenuates its own inputs at the synapse —
sensor-level plasticity substitutes for temporal compensation, so the network
never builds any. The 2021 paper's frozen-input choice is not incidental.

**Sparse input wiring** (stage1, on disk): the dense-wiring extension (every
char → all 500 nodes) reduced the plastic-input rule to a frequency counter —
learned "embeddings" organized purely by character frequency
(ρ(row drift, log freq) = +0.80) because all characters were near-identical
stimuli. No class structure, no surprisal. Superseded by the sparse design;
kept as a cautionary result about dense input geometry.

### The char-level completion saga: suppression, echo, and a density artifact

Stage 1d asked the 2021 completion question in the text world and got the
*opposite* sign: at silence steps 2–3 the bigram-expected character's code
**anti-correlates** with the silent state (ρ −0.23…−0.31, on disk). Unpacking
that took five experiments and two retractions (both preserved in the audit's
corrections ledger — each of two one-line stories was wrong in a different
direction):

- **Calibrate chance first** (stage1f ○): conditioned candidate sets have ~20
  members, so top-5 chance is 0.29, not 5/65. Calibrated: the plausible *set*
  is genuinely elevated (top-5 0.444–0.451 vs 0.290) while the single
  most-likely char is specifically suppressed (top-1 0.004–0.006 vs 0.058 —
  10× below chance). Both earlier one-liners ("cancellation regime",
  "scoring artifact") were wrong; the truth is this composite.
- **What occupies rank 1** (stage1h ✓): at silence step 1, the *echo* — the
  just-seen character's own code wins 45% of probes (chance 1.7%). At steps
  2–3, rank 1 is **'z' on 79–98% of probes**: frequent characters have
  *sparser* evoked codes (homeostatic compensation, ρ(density, log count) =
  −0.07…−0.21 by seed), rare ones denser, and decaying silence matches dense
  codes.
- **Density controlled, suppression survives** (stage1j ✓): partialling code
  density out of the probability correlation barely moves it (−0.271→−0.247,
  −0.290→−0.261 unconditional steps 2–3). The anti-correlation is genuine,
  per-context suppression — but **no decoder recovers the next character**,
  inverted readout included (0.035–0.037 vs chance 0.052 vs bigram ceiling
  0.229). Suppression is a reliable signature, not an invertible code.
- **The grammar world does the opposite: covert rollout** (stage1i ✓, same
  probe protocol): after feeding a subject, silent step 1 matches
  verb-position codes (+0.544 fav / +0.367 unfav), step 2 matches
  *object*-position codes with chain ordering (+0.329 / +0.273 — verb codes
  collapse to +0.074/+0.055), step 3 matches space-at-position-4 (+0.215).
  The silent network **advances through the sentence** in order.
  **Rollout vs suppression is a property of the world, not the model.**
- **What controls the sign** (stage1k ✓ light set, ○ N=2000 arm): a factorial
  over alphabet size and wiring density. A 27-letter alphabet removes the
  deep-silence anti-correlation (ρ ≈ 0); thinning input wiring to the same
  chars-per-neuron does **not** (−0.20/−0.29), and neither does N=2000 at the
  same density (−0.32/−0.28) despite vocab/N dropping 4×. All three variants
  also gain genuine step-1 prediction the base lacks (uncond ρ +0.10/+0.18/
  +0.15; N=2000 also cuts the echo 45%→16%). **Only the alphabet's own
  statistics control the deep-silence regime; the mechanism is still open.**
  (Untested knob: rhythm — a period-4 synthetic character world would tell
  whether the grammar world's rollout rides on its fixed 4-step frame.)

---

## Part II — Speaking and dialogue (stages 2a–5a)

The 2024 paper's thesis is that behavior emerges when the loop closes — so
these stages gave the grammar-world listener a mouth and asked whether
homeostasis would organize speech the way it organizes tracking. Setup common
to all: N=100 grammar world; the agent's spoken word is (in stage 2) fed back
as its own next input; "mouth" = some readout of reservoir spikes over the 5
tokens.

### Stage 2a ✓ — timing is decisive; the grammar is in the state, not the mouth

- A mouth read **at word arrival** parrots: an ideal readout (correlate spikes
  against stored (token, position) codes) repeats the just-heard subject on
  100% of trials — and gets the verb slot right on 0%. With **one silent
  think-step** before each spoken word, the same readout speaks grammar: verb
  slot correct 93%, grammar-favored verb 94% (teacher's own rate: 75% — argmax
  overshoots the statistics), favored object 98%, full 3-word pattern 14% vs
  3.2% chance. The full-pattern rate is strongly bimodal across wiring seeds
  (±0.23 SD; seed 0 speaks 364/400 complete sentences, most seeds near zero).
- Quality decays down the sentence (boundary word: 93% → 70% → 18% by slot) —
  compounding drift as the agent hears its own output.
- The paper-style minimal mouth (5 random ~10-neuron pools, argmax spike
  fraction) stays at **chance in every configuration**, with mild
  perseveration (same word ×3 on 11–22% of trials vs 4% chance).
- Learning-on vs frozen: identical over 400 trials (~20% self-hearing) —
  interleaved teacher speech keeps the loop from corrupting the network.

### Stage 2b ✓ — random pools fail at any size; then drift binds

Pool-density sweep (10–50% of neurons per pool): random pools are at chance at
*every* size — they read overall activity level, but token identity lives in
*which* neurons spike, not how many. Hebbian pools (each word's pool = its 20
most-distinctive neurons from listening, still plain spike-fraction argmax)
speak the first word well — verb slot 85%, favored verb 85%, parrot rate 6%
(oracle: 93/94/4) — but **every mouth except the oracle collapses by the
second self-spoken word** (tuned-20 object slot 0.386 ≈ 0.40 chance). First
readout was the binding constraint; once solved, compounding drift is.

`scripts/speak_transcript.py` prints verbatim sentences, which say it better
than the aggregates (seed 0; teacher speaks the subject, agent the rest):

```
random-pool mouth:   dog → dog dog dog ✗      man → · · · ✗
tuned-pool mouth:    man → walks dog · ✓      dog → bites bites bites ✗
oracle readout:      dog → bites man · ✓      man → walks dog · ✓
```

### Stage 3 ✓ — contingent dialogue: a clean null

Redesign to restore the tracking-style sensorimotor loop: word-alternating
turns (teacher: subject → agent: verb → teacher: object *conditioned on what
the agent actually said* → agent: boundary), a continuous "mumble" mouth
(sharpened mixture r^β, θ=0.5 intelligibility threshold, agent hears its own
blend), teacher stall+takeover on mumbles.

Result: the loop is **stable but not adaptive**. The tuned mouth speaks
structured words (62% stand as words; of those 73% real verbs, 88%
grammar-favored), but closed-loop experience changes *nothing*: contingent =
non-contingent = frozen-learning, all bins flat (100 nets × 600 sentences ✓,
and 3000 sentences ○). Random pools babble *confidently* — they stand words
more often (78–83%) at higher articulation sharpness with content exactly at
chance. The tracking mechanism — drift until comfortable — never engaged.

**The audit that followed found six design flaws, all confirmed:**
the environment was too forgiving (takeover repairs every failure in 1 step —
tracking punishes with *persistent* starvation until behavior fixes it); most
input flow was behavior-independent (teacher speaks half the words plus all
repairs — in tracking, 100% of input is behavior-contingent); contingency
passed through a thresholded argmax, so the mixture had no gradient; the
"tuned pools" were a supervised decoder in disguise; consequences landed on
the wrong step's spikes; and — measured directly (verify_blend_comfort ✓) —
**the comfort gradient pointed toward mumbling**: self-heard mean |E| =
0.2996 for a uniform mumble < 0.4262 for a 60/40 blend < 0.4988 for the
expected clean word (sparse +5 projections make concentrated input
high-variance across neurons; blends smooth it). The mouth was being paid to
slur.

Also surfaced here, and structural: **prediction-by-absorption means the
most-expected word is the least expressed in spikes** — the strongest
predictions are precisely the hardest for any spike-reading mouth (the
boundary token, most predictable, stuck at ~30% vs 20% chance in every arm).

### Stage 4 ✓ — corrected mechanics; the null sharpens into a result

Stage 4 rebuilt the world to the audit's specification: mixture contingency
with no thresholds (teacher samples from P(y\|m) = Σ_w m_w P(y\|w)), no
self-hearing, consequence lands on the step after the action-selecting spikes
(one-step rule adjusts exactly the acting coalition's outgoing synapses),
equal-duration uniform-word repair (no tutoring), boundary has consequences,
honest arm naming. Findings:

- The pressure channel is real, and graded: with learning on, mean |E| =
  0.26 on think steps < 0.34 hearing a grammatical reply < 0.41 hearing a
  confused one (+0.065 for confusion). The environment *can* punish bad
  speech.
- **Random pools still do not ground** (verb mass 0.39–0.41 ≈ 0.40 uniform,
  flat over 4000 sentences; 59–61% of their turns draw confused replies):
  local homeostatic credit assignment does not organize a symbolic mouth de
  novo, even with correct credit timing and real consequences.
- Learning-on keeps |E| ~37% below frozen (0.26 vs 0.41 on think steps) and
  mildly helps articulation (verb mass 0.75 vs 0.69) — homeostasis maintains
  *comfort*, just not *grounding*.
- Articulation retained the old grammar after a mid-run 75/25→25/75 reversal
  with learning on AND frozen — which motivated the decisive follow-up:

### Stage 4b ✓ — staleness, not entrenchment (the track's cleanest finding)

Passive listening version of the reversal: 800 original + 2000 reversed
sentences, probe the completion preference against old vs fresh codes.
The internal prediction **flips completely**: favored-gap +0.185 ± 0.065
before (positive in 50/50 nets) → **−0.184 ± 0.063 scored with fresh codes
(0/50 nets still old)** — while the *same states* scored with the stale
warmup templates still read +0.172 (50/50). The network reversed; the old
measuring stick couldn't see it.

**Representations drift with learning, so any fixed decoder becomes stably
wrong.** Not because the network entrenches — because it keeps moving. A
mouth must therefore be plastic: continuously re-tuned from heard-word /
spike co-activity (ecologically legitimate — the word IS the input).

### Stage 5a ✓ — the plastic mouth works, and exposes a track-wide confound

The online mouth: M[:, w] ← (1−η)M + η(s−s̄) on each heard word w
(η=0.05/arrival, L2-normalized; softmax read at think steps). η was set from
the measured code-movement curve (stage5a_code_speed ✓: scored with sliding
codes, the completion gap crosses zero within 100 reversed sentences and
saturates at −0.14 by ~400; scored with frozen warmup templates it reads
+0.13 *forever* — the staleness result as a time series).

2×2 (reservoir × mouth, each plastic/frozen, 50 nets, reversal at sentence
400; "share" = mass on the originally-favored verb, 0.5 = neutral):

| arm | share s400 | s600 | s2400 | heard-word decode |
|---|---|---|---|---|
| plastic / plastic | 0.818 | **0.169** | 0.174 | 0.999 |
| plastic / frozen | 0.816 | 0.822 | 0.821 | 0.998 |
| frozen / plastic | 0.834 | **0.161** | 0.167 | 0.999 |
| frozen / frozen | 0.826 | 0.822 | 0.826 | 0.998 |

The plastic mouth reverses within ≤200 sentences with perfect decoding — the
engineering goal is met. But the **frozen-reservoir arm reverses
identically**, so the flip owes nothing to reservoir re-learning: it is
entirely **mouth-side context tracking**. With leak 0.75 of the previous
word's activation still in each arrival pattern, a verb's mouth column
absorbs the *subject context* that verb arrives in (stage5a_decompose ✓:
walks-column context bias −0.057 (man-ward) → +0.031 (dog-ward) across the
reversal; bites −0.002 → −0.047 — each column moves ~0.05–0.09 toward its
*new* dominant subject). Two further results reconcile the earlier stages:

- **Arrival geometry is stable under weight drift** (plastic/frozen decode
  stays 0.998): input-territory dominated codes don't move. What drifts is
  the *completion/context* geometry — that is what stage 4b's flip was.
- **The reinterpretation cascade**: in a bigram grammar, P(next\|current) *is*
  the context mixture of next's arrivals — so every bigram completion
  correlation in this track (4b's flip, the stage-1e/2021 Table 2
  replication itself, 1f/1g) **conflates prediction with context-residue
  matching**. Surviving pure-prediction evidence: the stage-1i rollout (the
  probe state *advances* through positions — residue can't switch which
  position matches) and the stage-1c absorption timing. Lag-persistence is
  NOT a valid discriminator (genuine rollout moves on rather than holding).
  The clean discriminator, designed but unrun: a **filler grammar** with a
  dependency spanning >1 step (subject residue at verb-think = leak² = 0.56
  with one filler, 0.32 with two — graded separation of residue from
  prediction).

For *speaking engineering*, context-association is legitimate grammar
production. For *mechanism claims about prediction*, all completion-
correlation evidence is downgraded pending the filler grammar.

---

## What stands, in six lines

1. **The 2021 language results replicate exactly** (surprisal + completion,
   their setting), on the same verified core the 2024 replication uses.
2. **Prediction is absorption**: expectation cancels input at arrival;
   surprise discharges one step late; the most-expected token is the least
   spike-expressed. This is the track's central mechanistic finding, and it
   structurally handicaps any spike-reading actuator.
3. **Frozen inputs are load-bearing**: plasticity at the sensor substitutes
   for temporal prediction and destroys it.
4. **Local homeostatic credit does not ground a symbolic mouth de novo** —
   four increasingly careful designs (2a, 2b, 3, 4), all null on arbitrary
   readouts. Grounding random pools remains the track's open hard problem.
5. **Staleness, not entrenchment**: representations keep moving under
   learning; fixed decoders become stably wrong; readouts must co-adapt
   (and a simple online Hebbian mouth suffices — but see 6).
6. **The confound**: in bigram worlds, completion correlations conflate
   prediction with context residue. Pure-prediction evidence survives only
   where residue can't produce it (rollout ordering, absorption timing).

**Connection to the main track**: the dialogue null and the tracking/Pong
successes are two sides of the input-flow thesis. In tracking, 100% of input
flow is behavior-contingent and failure starves the agent *persistently*; in
the dialogue worlds, the teacher supplied most of the flow and repaired every
failure within a step. The loop drifts-to-comfort only when comfort is
reachable *only* through competent behavior. That is the same
boundary-anchored-engagement principle the evolution screen found
(`scripts/screen_metrics.py`), showing up as a negative result in language.

## Open threads, ranked

1. **Stage 5b — contingent vs yoked dialogue with the plastic mouth**
   (designed, unrun): does closed-loop contingency now matter, with a mouth
   that can actually track the reservoir?
2. **Filler grammar** (designed, unrun): the clean prediction-vs-residue
   discriminator; also directly tests whether "prediction" here is more than
   one-step context carryover.
3. **Rhythm knob** (period-4 synthetic char world): is grammar-world rollout
   frame-timing, or content?
4. **Deep-silence mechanism** (stage 1k left it open): why do the alphabet's
   own statistics, and nothing structural, control the suppression regime?
5. **Timescale separation** (untested): if learned embeddings are wanted
   alongside prediction, input lr ≪ recurrent lr.

## Script index

| script | question | one-line answer |
|---|---|---|
| stage1_text_passive | do dense plastic embeddings organize? | yes — by frequency only; artifact of dense wiring |
| stage1b_sparse | does sparse wiring recover surprisal? | yes (+0.23..+0.32); input plasticity destroys it |
| stage1c_surprisal_vis | what carries surprisal? | absorption; subthreshold lag 0, spikes lag 1 |
| stage1d_completion | char-level completion? | no — deep-silence anti-correlation (see 1f–1j) |
| stage1e_grammar_replication | 2021 Table 2? | full replication, all cells |
| stage1f_conditioned | calibrated scoring | set elevated + modal suppressed, composite |
| stage1g_entropy | predictability knob | gap scales smoothly, never inverts |
| stage1h_inverted | what's rank 1? does inversion decode? | echo then 'z'; no decoder works |
| stage1j_density | density artifact? | partly ('z'); suppression survives control |
| stage1i_grammar_depth | grammar world at depth? | covert rollout: verb→object→space |
| stage1k_capacity | what controls the regime? | alphabet statistics; not wiring, not N |
| stage2a_speaking | can it speak? | with a think step + ideal readout, yes (93%) |
| stage2b_mouth | can a random mouth work? | no, at any size; Hebbian speaks 1 word |
| speak_transcript | verbatim samples | (see output) |
| stage3_dialogue | does contingent dialogue teach? | no — stable, not adaptive (audited: 6 flaws) |
| verify_blend_comfort | which speech is comfiest? | mumbling (0.30 < 0.43 < 0.50 \|E\|) |
| stage4_dialogue2 | corrected loop — grounds now? | no; pressure real, grounding absent |
| stage4b_reversal_passive | entrenchment or staleness? | staleness: flips −0.184 fresh, +0.172 stale |
| stage5a_code_speed | how fast do codes move? | flip in ~100–400 sentences |
| stage5a_mouth | plastic mouth 2×2 | works; flip = context, not prediction |
| stage5a_decompose | what's in the columns? | subject-context bias, sign-flips on reversal |
