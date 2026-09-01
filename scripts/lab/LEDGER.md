# Ledger — design-space campaign, 2026-08-31/09-01

Rules: every hypothesis gets a preregistered prediction BEFORE its confirming
experiment; verdicts at stated resolution; "refuted" reserved for powered
open-loop tests. Entries append-only; corrections stay visible.

## H1: initial recurrent gain g_init = N·p·w̄/(ρ·T₀) organizes the sweep

**Predicted** (before K0): g organizes spiking level and phases; score ridge
somewhere in g; dead at low drive, autarkic/saturated at high g.

**K0 verdict (241 existing configs, 0 new runs)**: MIXED, score part unsupported.
- g_init → prop_spiked: rho +0.38; deciles rise 0.004 → 0.75. SUPPORTED for
  activity level.
- Saturated class at high g (median 16.6), dead at low drive: geography as
  predicted qualitatively.
- **g_init → score: rho +0.02 — NOT organized.** Working configs (n=24/241,
  score ≥ 0.35) sit at mid-g (median 6.3) but are distinguished by HIGH input
  drive μ_in = 62·p·w_in (median 8.65 vs 2.33 for dead). Score's best single
  params: gain +0.27, input_weight +0.26, leak +0.21, ρ −0.18, N −0.18.
- Consequence (Gate A): the performance story is NOT g_init. Candidate
  replacements to test: g_final(t) (self-organized, needs w̄(t) logging),
  input-dominance μ_in/recurrent-drive, or behavioral (flow) coordinates only.

## H2: dead-zone law — dead iff drive cannot reach threshold

**Predicted**: aliveness needs steady drive ≳ leak·ρ·T (mean-field; sensor
activation scale s̄ unknown in K0's proxy).

**K0 evidence**: single threshold on μ_in/(leak·ρ) separates dead (n=91) from
alive at **84.6% accuracy** (split at 4.10). SUPPORTED as a first-order law;
residual misclassification plausibly from s̄ variation and recurrent
amplification. Next: compute exact retinal S₀ from sensor geometry and retest;
then open-loop confirmation (K1/K2).

## H3: input flow is the effective objective (prior finding, sharpened)

**Question**: is flow~score (rho 0.77) just "not dead"?

**K0 evidence**: partial Spearman(input_flow, score | prop_spiked) = **+0.803 —
higher than raw**. Flow's link to score is NOT mediated by activity level.
STRENGTHENED. (prop_spiked → score alone: +0.12.)

## Open questions raised

- What coordinate DOES organize score? (g_final? flow is behavioral — need a
  design-space predictor.)
- 91/241 random configs are dead, 9 saturated → the sweep's sampling box mostly
  probes the dead zone; new grids should be placed using H2's boundary.

## H4 (preregistered before K2): single-node basins

Predictions: (a) cold start (x=0, T=1) spikes at least transiently iff
mu > rho*leak*T0; below that, silent forever — comfortable (E→0) iff
mu >= leak*T_floor, else dead-floor. (b) On spiking cells the duty law
f = (mu/T - leak)/rho holds with no free parameters (T measured). (c) Hot and
cold starts disagree (bistability) somewhere; (d) frozen-|E| mode-locked
cycles appear at low leak (the known period-3 at leak 0.25).

## H5 (preregistered before K1): homeostasis silences stationary drive

Stationary retina → f_late ≈ 0 for every lr combination with learning on;
time-to-silence decreases in the dominant channel's lr.

## H6 (preregistered before K1): adaptation-channel dominance

At weight_lr=1.0 (2024 default), f_late is nearly insensitive to target_lr
while spiking is dense (weight channel absorbs ~full error when gated on);
at weight_lr=0, target_lr controls everything. f_late increases with slip
speed (fluctuation-driven spiking).

## H7 (preregistered before K3): freeze decomposition in tracking

Predictions: (a) freeze-from-init ≈ full at paper defaults (prior: no-learn
0.250 exactly = both effectors saturate; actually prior says no-learn collapses
to 0.25 chance — so freeze-from-init HARMS at defaults; the Pong "harmless"
was Pong-specific). (b) freeze-mid < full (self-entrenchment transfers).
(c) If freeze-mid-resetT recovers most of the loss, the collapse is a
threshold-inflation artifact, not computation-in-plasticity. (d) shuffle-mid
(learning on) recovers over time if plasticity is a regulator; stays broken if
learned W structure is the computation. (e) freeze-T-only ≈ full more than
freeze-W-only ≈ full iff the working channel is W (H6 logic). (f) gain
changes score materially (confound confirmed) with dir-agree more stable.

## H4 verdict (K2, 1250 runs): two exact laws, one race, one surprise

- (a) cold-start spike boundary mu > rho*leak*T0: 86.2% — misses concentrate
  near the boundary, where the true condition is a RACE between x rising
  (timescale ~1/leak) and T chasing (target_lr); boundary shifts up when T
  gains a head start. SUPPORTED with stated refinement.
- Comfort split: silent-comfortable iff mu >= leak*T_floor — **100.0%
  (674/674). LAW.**
- (b) duty law f = (mu/T - leak)/rho on spiking cells: **median |f-pred| =
  0.0000, n=419, no free parameters. LAW.**
- (c) bistability: 47/625 cells, low-leak band. Direction is the surprise:
  HOT start → T inflates → silent-comfortable; COLD start → sustained
  marginal spiking. Target inflation is the silencing mechanism (the statue
  cheat is this basin, reached by overdrive).
- (d) frozen-|E| mode-locked cycles are GENERIC (157 cells, every leak value
  tested, median |E| 0.43) — not a leak=0.25 curiosity. Prediction wrong in
  scope. Open: what sets locking vs marginal-spiking?

Consequence for the network story: each node has a local phase set by its
total drive mu_i; recurrence couples the mu_i. "Entrained" networks are
plausibly mixtures of nodes shuttling between local states as drive moves.

## H5 verdict (K1): REFUTED — and the correction is the finding

Stationary drive is NOT generically silenced. f_late under a stationary
retina (N=200 defaults otherwise): wlr=0/0.01 → **1.000 at every target_lr**
(the saturated statue: T inflates until E≈0 while f=1 — comfort without
silence); wlr=0.1 → 0.000-0.026 (true absorption; the 2021 value silences);
wlr=1.0 (2024 default) → 0.39-0.41 at tlr≤0.01, 0.000 at tlr=0.1.
**Corrected law: only the weight channel changes effective drive/gain; the
target channel merely relabels whatever x' is as comfortable.** Silence-by-
adaptation requires weight plasticity in the right range.

## H6 verdict (K1): REFUTED at 2024 defaults — activity is endogenous churn

At wlr=1.0, f_late ≈ 0.28-0.41 for EVERY schedule (stationary, slip 0.25/1/4,
jump) and is insensitive to target_lr (≤0.01) — spiking is dominated by
self-generated weight-churn fluctuations, not stimulus non-stationarity. The
tracking model's canonical prop_spiked ≈ 0.34 is plausibly churn-set, not
stimulus-set (open-loop stationary at defaults gives 0.39). Channel
competition discovered: at wlr=0.1, tlr=0.1 PROTECTS the gain (fast T absorbs
E before weights erode; f stays ~0.99 under slip) while tlr≤0.01 lets weights
absorb (f 0.03-0.31). Also: f vs slip speed is NONMONOTONE at wlr=0.1
(peaks at slow slip) — open thread; note total retinal drive S is
position-invariant, so slip moves only the PATTERN of drive, not its sum.

## K1b verdict: the duty law is a NETWORK law, not just a node law

Per-(node, 120-step window) fit of f = clip((mu/T - leak)/rho) on wlr=1.0
churn-regime open-loop runs (slip + jump, N=200): corr +0.999/+1.000, median
|resid| 0.0025/0.0000, calibration curve on the diagonal, no free parameters.
**Activity level is fully slaved to (total drive, target): all remaining
dynamics live in how W moves mu and how T moves.** (LAW, network-scale.)

## H7 verdict (K3, 144 runs, defaults, seeds 0-11): the channel story inverts the paper's emphasis

score_late (segments 6-10): full 0.325±0.122 | no-learn 0.250 (exact, dead)
| freeze-mid 0.273±0.245 | +resetT 0.263 (NOT a threshold artifact) |
+resetW 0.250 | shuffle-mid 0.394 (recovers ABOVE full; learning re-adapts
from scrambled W) | **freeze-W-only 0.250 (death)** | **freeze-T-only 0.404
(BETTER than full)** | **lesion 0.478 (best of all)**.

Reading: (1) weight plasticity is necessary and sufficient among the two
channels; (2) target adaptation is mildly HARMFUL in-task at defaults —
consistent with K1's channel competition (T absorbs error that W needs) and
with the statue mechanism (T relabels rather than regulates); (3) the
specific learned W is not the computation — the ongoing churn is
(shuffle > full; freeze-mid < full); (4) at PAPER DEFAULTS the recurrent
medium is a net liability (lesion 0.478 > full 0.325 on seeds 0-11) — the
reflex/medium dichotomy is a REGION of design space, not a property of the
model. (5) gain confound confirmed: score 0.268/0.325/0.208 at gain
2.5/10/40; dir-agree tracks engagement better.

## Gate B decisions

- Act II experiment #1: CLOSED-LOOP wlr x tlr plane (channel competition in
  behavior) — where does tracking live between statue (wlr→0), absorption
  (wlr≈0.1), and churn (wlr=1)?
- #2: dark/zero-input open-loop churn test — is f≈0.34 truly endogenous?
- #3: leak x wlr plane (dissipation vs regulation).
- #4: repeat the K3 autopsy at the w1' config (the best-known medium) — does
  the lesion/freeze ordering invert where the medium is functional?
- #5: fine slip-speed curve at wlr=0.1 (the nonmonotone anomaly).
