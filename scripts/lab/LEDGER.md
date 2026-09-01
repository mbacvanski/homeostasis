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

## H8-H12 (preregistered before Act II batch 1)

- **H8 (A1 wlr x tlr closed-loop)**: tracking dead at wlr=0 (statue, 0.25);
  a working band at intermediate wlr; channel-competition signature in
  behavior: score(wlr=0.1, tlr=0.1) << score(wlr=0.1, tlr=0.001) because
  fast targets starve the weight channel.
- **H9 (A2 darkness)**: at wlr=1.0 the network sustains f > 0.2 in TOTAL
  darkness (endogenous churn); at wlr=0.1 darkness → silence. If confirmed,
  the tracking network is a self-exciting medium that input MODULATES.
- **H10 (A3 leak x wlr)**: working band broad in leak at wlr=1.0; low-leak
  cells show frozen-cycle signatures (high |E|, low score).
- **H11 (A4 autopsy at w1')**: the K3 ordering INVERTS where the medium is
  functional: lesion collapses (~0.1, prior w1 evidence), freeze-mid < full,
  and freeze-W-only dead everywhere; if shuffle-mid recovers above full even
  here, "ongoing process > learned structure" is general, not regional.
- **H12 (A5 slip curve at wlr=0.1)**: f(speed) is nonmonotone with a peak
  where sensor dwell (plateau 4 deg / speed) matches the absorption
  timescale; speed* shifts with wlr.

## H8 verdict (A1): CONFIRMED — and the sweep's box was wrong

score_late (12 CRN seeds): peak **wlr=0.1, tlr=0.01 → 0.633, 12/12 seeds
≥0.35** vs 2024-default wlr=1.0 → 0.325 (4/12). wlr=0 → statue (f≈1.0,
0.250). wlr=3.0 → numerical blowup (inf/nan; plasticity has an UPPER
stability bound between 1 and 3 — treat wlr≥3 cells as "exploded" class).
Channel competition behavioral: (wlr=0.1, tlr=0.1) → 0.267 with f=0.977
(fast targets starve the weight channel → near-statue). NOTE: the historic
241-sweep never varied weight_lr — its winners lived on the wlr=1.0 slice.

## H9 verdict (A2): REFUTED — churn is input-powered, not self-exciting

f_late = 0.000 in total darkness at every (wlr, tlr) tested, wlr=1.0
included. With K1: at defaults the network's activity is input-POWERED but
stimulus-statistics-INDEPENDENT — plasticity converts input energy into
sustained churn whose level it sets itself.

## H10 verdict (A3): the matched-timescale ridge (law candidate)

leak x wlr score ridge is diagonal: best (leak, wlr) = (0.05, 0.03) 0.681 |
(0.25, 0.1) 0.633 | (0.5, 0.3-1.0) 0.43 | (0.75, 1.0) 0.494. Score needs
plasticity rate ~ dissipation rate. **Unification: sweep winners w1/w2 both
chose leak≈0.56-0.57 under forced wlr=1.0 — they sat on this same ridge.**
Off-diagonal death: high-leak/low-wlr dead (0.250), low-leak/high-wlr poor
(~0.27-0.29).

## H11 verdict (A4): medium is regional, weight channel universal

w1' (seeds 0-11): full 0.850±0.184; lesion 0.250±0.433 (bimodal collapse —
medium real here, unlike defaults where lesion won); freeze-mid 0.632;
+resetT 0.619 (still not a threshold artifact); shuffle-mid 0.762 (ongoing
process ≥ structure EVEN in the medium regime); freeze-T-only 0.654 (target
adaptation helps at w1', unlike defaults); **freeze-W-only 0.250 — dead in
every regime tested. Weight plasticity is THE universal necessary channel.**

## H12 verdict (A5): bandpass confirmed, with structure

f(speed) at wlr=0.1: high at 0.05-0.13 deg/step (bimodal, silence-vs-chase),
minimum ~0.8, secondary bump at 1.3-2.0, collapse by 8.0 (0.016). Fast slip
is absorbed as if stationary (averages out inside the absorption window) —
**spiking requires non-stationarity ON the plasticity timescale: a bandpass.**
Secondary bump near sensor-spacing/speed resonance — open thread.

## H13 (preregistered before K4): the seed lottery is structural

The wiring-only reflex-kernel slope (sensor→node→effector, no simulation)
predicts per-seed tracking: sign(slope) → above/below-chance; |slope|
correlates with |score-0.25|; prediction strongest in reflex-leaning regimes
(defaults), weaker at w1' (medium), intermediate at the new best regime
(wlr=0.1).

## H13 verdict (K4): UNSUPPORTED at defaults — the lottery is not the static kernel

48 seeds/variant: default sign-agreement 0.44 (≤ chance), Spearman(slope,
score) −0.175; wlr=0.1 weak support (+0.32 score, +0.41 dir-agree, sign 0.62);
w1' nothing (and 48/48 seeds ≥0.35 there — w1' is essentially lottery-free:
0.819±0.189 score_late). The initial wiring's direct reflex kernel is largely
overwritten by adaptation; the lottery's microfoundation stays OPEN.
New 48-seed anchors (score_late): defaults 0.386±0.138 (58% work),
wlr=0.1-only 0.467±0.200 (73%), w1' 0.819±0.189 (100%).

## H14 (preregistered before the cluster ridge grid)

Fine (leak x wlr) grid at tlr=0.01, 48 CRN seeds, 25% checkerboard held out:
(a) the ridge is 1-D: argmax_wlr score(leak) follows a power law
wlr* = c * leak^b with b in [0.8, 1.5]; (b) fitting c,b on 75% of cells
predicts the held-out cells' argmax within one grid step; (c) the ridge's
peak height decays toward high leak; (d) N-line at matched (leak=0.25,
ridge wlr): score and ridge position are N-stable from N=100 to 800
(the lr-ratio, not N, is the organizing variable).

## H15-H17 (preregistered before Act II batch 2)

- **H15 (nulls)**: a saturated P-controller on the retinal centroid scores
  near-ceiling (>0.9) — large headroom above every homeostatic config; the
  1-step flow-greedy controller lands within a few points of the
  P-controller (flow maximization ≈ centering); random-turn ≈ 0.25.
- **H16 (tlr=0)**: at the ridge (leak .25/wlr .1 and leak .05/wlr .03),
  target_lr = 0 exactly matches or beats tlr=0.01 — target adaptation is
  vestigial-to-harmful for tracking in the absorption regime (in-task
  version of K3's freeze-T-only result).
- **H17 (transfer function)**: the spike-readout reconstruction gain G(P)
  of a sinusoidal slip (amp 20°) is bandpass in period P: low at short P
  (averaging), low at long P (absorbed), peaked at P* that scales inversely
  with wlr (higher plasticity rate → faster absorption → peak moves to
  shorter periods).

## H18 (preregistered): the flow-ratchet — behavioral self-repair after effector inversion

Mechanism claim: tracking self-organizes because turns that kill input flow
self-terminate (darkness → silence → no turning, per A2) while turns that
keep flow continue — a ratchet needing no correct innate kernel (consistent
with K4's negative). Decisive test: SWAP the effectors at t=3600 (left spikes
now turn right). Predictions: (a) on the absorption ridge (leak .25, wlr .1)
and at w1', score collapses at the swap then RECOVERS within ~the absorption
timescale (hundreds of steps), ending well above chance in segments 9-10;
(b) recovery is weaker/absent at 2024 defaults (churn regime, wlr=1.0);
(c) in the policy sufficient statistics, out-of-view heading-error bins show
near-zero |dH| (the silence-stops-turning ratchet pawl), already checkable
in recorded A1/K3 runs.

## H15 verdict (B1): CONFIRMED — ceiling is 0.999, flow-greedy ≈ centering

P-controller on the retinal centroid: 0.999 (flow 4.23). 1-step flow-greedy:
0.898 (flow 3.94). Random: 0.251. The embodiment supports near-perfect
tracking; w1' reaches 82% of ceiling, paper defaults 39%. Flow maximization
and centering are behaviorally near-equivalent in this embodiment — the
geometric basis of the input-flow thesis, now quantified.

## H16 verdict (B2): CONFIRMED — target adaptation is vestigial in-task

tlr=0 vs 0.01 (24 seeds): ridge25 0.571/0.503, ridge05 0.586/0.606, default
0.375/0.376 — no benefit anywhere. With K3's freeze-T-only>full and A4's w1'
counterexample: targets matter only as slow gain-normalization where
target_init is mis-set for the (leak, drive) point (testable: target_init
sweep at w1' with tlr=0).

## H17 verdict (B3): the ridge is a SIGNAL-TO-NOISE optimum

Spike-readout reconstruction gain of a 20-deg sinusoidal slip:
wlr=0.1 → 0.22-0.23 flat over P=30-120, absorbed above P≈240-480 (0.04 at
1920). wlr=0.03 → ≈0 (saturation destroys selectivity; f≈1 ceiling).
wlr=1.0 → ≈0.02 at all P (endogenous churn buries the stimulus).
**wlr=0.1 carries 3-10x more stimulus information in its spikes than any
other regime probed** — the matched-timescale ridge is where plasticity
whitens saturation away without self-noise. (Highpass corner below P=30 not
reached; absorption cutoff confirmed; P*-vs-wlr scaling untestable on this
flat-top — refine with shorter periods if needed.)

## H18 verdict (B4): swap-immunity — the loop is self-organizing, not a wired reflex

Effector inversion at t=3600: recovery ratio (swap/full, segments 9-10) =
1.03 ridge25 / 0.93 w1' / 0.96 default — the dip is barely visible at
720-step resolution ANYWHERE (H18b's regime-dependence refuted; recovery is
universal and fast). Ratchet pawl confirmed in recorded policy stats:
RMS dH out-of-view 0.174 vs in-view 1.200 at wlr=0.1 (dark = still), 0.81 at
wlr=1.0 (churn keeps moving).

## H16b verdict (B5): REFUTED — targets at w1' are not static gain-normalization

w1' tlr=0 with target_init 2/3/4 = total death (0.25x; duty law: raising T
drops mu/T below leak → silence); only ti=1 lives (0.654) and true target
ADAPTATION adds the remaining ~0.2 to reach 0.85. Whatever targets do at w1',
it is dynamic.

## H19 + B6 verdict: the agent is a VELOCITY-ENTRAINED FOLLOWER, not a servo

Follow ratio (agent net rotation / stimulus net rotation, per segment):
w1' 0.837 overall (0.88 of segments > 0.5, up to 0.98 late) — genuine
following through reversals. ridge25 0.565 and LEARNING over the session
(0.28 → 0.89 by segment). Yet the signed in-view response is ~zero (w1'
+0.0006) or ANTI-corrective (ridge25 −0.120): **no position-error feedback
exists**. Mechanism: slip → spikes (bandpass) → turning, with the turn BIAS
as the slow learned variable dragged by the stimulus; darkness stalls the
agent (ridge25 dark-stall 0.93) and the periodic world returns the stimulus
(periodicity is load-bearing, as the toy-world result predicted). Swap
immunity follows: inversion just re-learns the bias. "Gibsonian resonance"
is mechanically accurate: entrainment, not servo control.

## B6b: re-entrainment after reversal (measured, prereg partially wrong)

Windowed dH toward the new direction: w1' re-locks in ~30 steps to asymptote
0.94; wlr=0.1 tau~180/asym 0.68; wlr=0.3 and 1.0 tau~270/asym 0.82 with an
initial NEGATIVE lobe (-0.14/-0.40: the old bias persists — behavioral
entrenchment); wlr=0.03 fast but weak (asym 0.36, near-statue). tau does NOT
scale as 1/wlr (H19b wrong): higher churn raises the entrainment ceiling but
slows re-locking. Score again tracks the SNR compromise.

## B7: LAW 3 — the weight channel is a gated integral controller (exact)

d(sum_in W)_n = -wlr * E_n per step on nodes with >=1 presynaptic spike,
else 0 — verified to <=1.2e-14 across defaults/ridge25/w1', 600 closed-loop
steps each. The reduced model is now DERIVED: per node, f slaved by the duty
law; drive servo integrates -wlr*E (spike-gated); target servo integrates
tlr*E (floored). Closure check: predicted absorption cutoff P* ~ 2pi/(wlr*f)
= ~420 steps at (wlr .1, f .15) vs observed 240-960 in B3.

## H20 (preregistered before B8): the turn bias lives in the recurrent weights

Decomposing pool duty differences into input vs recurrent components: at w1'
the RECURRENT component's L-R difference flips with stimulus direction (the
follower's velocity is W-stored); the input component contributes with the
sign of the retinal lag; at defaults neither correlates strongly.

## H20 verdict (B8): SUPPORTED at w1' — the velocity is W-stored; carrier is regional

Sanity: corr(pool Δf, dH) = +1.000 (pipeline exact); Δf tracks stimulus
direction +0.878 at w1'. Carrier decomposition (duty components, L-R):
w1' — recurrent +0.370 vs input −0.042 (couplings to Δf: +0.414 / −0.031):
**pure W-stored bias**. ridge25 — mixed (input +0.534, recurrent +0.388).
default — churn-carried recurrent (+0.919 coupling). Lag-servo alternative
REFUTED at w1'/ridge25: heading error keeps a constant +7–9° offset in BOTH
directions (corr with direction ≈ 0) — there is no direction-flipping lag.
The entrainment loop is now specified end-to-end: stimulus direction →
drive-servo writes pool-asymmetric recurrent drive → duty law reads it out
as the turn command → reversal rewrites it in ~30–200 steps.

## H21 (preregistered): what dynamic targets do at w1'

Arms at w1', 24 seeds: full vs freeze-T-mid (T frozen at its EVOLVED
heterogeneous per-node values at t=3600, W keeps learning) vs freeze-T-only
(T pinned homogeneous at 1.0 from init; prior: 0.654) vs full. Prediction:
if targets' contribution is building a static heterogeneous dynamic-range
profile, freeze-T-mid ≈ full (0.85); if T must keep moving (per-window gain
control), freeze-T-mid ≈ freeze-T-only.

## H22 (preregistered): the bandpass upper edge

Extending B3 at wlr=0.1 with P ∈ {8, 15}: reconstruction gain falls at short
periods (integration/averaging highpass corner), locating the passband's
other side.

## H21 verdict: SUPPORTED — targets are a CALIBRATION channel

w1', 24 seeds: full 0.791±0.205 | freeze-T-mid (T frozen at its evolved
heterogeneous values, W learning on) 0.789±0.195 | freeze-T-only (homogeneous
T=1) 0.700±0.248. Freezing the evolved profile costs NOTHING. The target
channel's entire contribution is building a static per-node dynamic-range
profile; after calibration it can stop. Weights = computation (must keep
running); targets = one-shot-ish calibration.

## H22 verdict: no averaging corner — the fast edge is sparse-and-informative

recon gain at wlr=0.1 RISES to 0.505 at P=8 (peak slip ~16 deg/step) while
activity collapses (f 0.003, spikes on 30% of steps): fast slip sparsifies
spiking into hyper-selective events. Absorption kills slow signals' activity
AND information; fast signals lose activity but keep information (fewer,
sharper spikes). The bandpass is a property of ACTIVITY, not information.

## H23 (preregistered): N-invariance of the ridge

Closed-loop at leak=0.25, wlr {0.05, 0.1, 0.2}, N {50, 100, 200, 400}, 12
CRN seeds: score and ridge position are N-stable for N in 100-400 (the
lr-ratio organizes, not size); N=50 degrades (the toy-world result put the
medium's floor at N~64).

## H23 verdict: REFUTED as designed — and the design was confounded

At FIXED p_link=0.1, w0=0.75: N=50 works (0.44-0.54; no N~64 floor at these
params), N=200 peaks (0.633 at wlr .1), N=400 DEGRADES (0.34-0.43). But
in-degree = N*p grows with N, so this "N-line" confounds size with recurrent
gain (as does cluster R3 — flagged). Corrected experiment: dual scaling.
Ridge pre-fit from coarse A3: wlr* ≈ 1.5 * leak^1.5 (grid-quantized; H14's
cluster fine grid tests b in [1.2, 1.8]).

## H23b (preregistered): dual-scaled N-line

Holding in-degree fixed (p = 20/N) or holding total recurrent weight fixed
(w0 = 15/(N*0.1)) at wlr=0.1: score becomes N-stable for N in 100-400; the
two scalings separate at N=50 (in-degree fluctuations vs weight granularity).

## H23b/c verdict: "size effects" were mostly wiring confounds

Dual scaling alone did not rescue N-invariance (H23b), because p_link also
wires input and output layers: p-scaling halved input drive at N=400. With
recurrent density scaled to in-degree 20 and input_p_link pinned at 0.1
(H23c): N=100 0.456, N=200 0.633, N=400 0.507, N=800 0.536 (92% seeds) —
roughly N-stable, N=200 nominally best at ~1.7 SE. Organizing variables:
in-degree, input drive, lr ratio — not N. (Cluster R3 keeps fixed p and will
reproduce the confounded curve; treat it as the confound demo.)

## The lottery partially dissolves (exploratory, no prereg)

Across seeds: corr(segment-1 score, late score) is only +0.45 (wlr .1) /
+0.39 (wlr 1) / ~0 elsewhere; and LATE seg-to-seg correlation within a run
is ~0 (+0.03 at wlr .1) — performance WANDERS segment to segment. A large
share of "seed lottery" variance is sampling noise of a wandering process,
not fixed wiring fate; w1's 100%-of-seeds reliability = its wander floor
clears 0.35. The remaining fixed effect (early-late +0.45) is real but
modest. Reframes open question 1: ask what sets the wander floor, not which
seeds are blessed.

## H24 (preregistered): the ridge law predicts w1's optimum out-of-family

The coarse ridge law wlr* ≈ 1.5·leak^1.5 gives wlr* ≈ 0.65 at w1's
leak=0.574. Prediction: score(w1', wlr=0.65) ≥ score(w1', wlr=1.0), with
wlr=0.3 below the peak. Also attribution: grafting w1' ingredients onto
ridge25 (gain 28.35, rho 1.525, input_weight 0.828) — bet: gain is the main
floor-lifter (authority speeds re-acquisition).

## H24 verdict: REFUTED both ways — the ridge law is family-local; w1' is epistatic

(a) At w1' (leak .574), wlr=1.0 → 0.842 ≥ wlr=0.65 → 0.789 (16 seeds): the
coarse ridge law does NOT extrapolate across w1's other parameter changes
(rho 1.525, gain 28, N=100/p=.21). wlr=0.3 collapses (0.383) so the ridge's
low side holds. Scope the law to the default family pending the cluster fine
grid. (b) Every w1' ingredient grafted onto ridge25 HURTS (gain28 0.556,
rho1.5 0.289, win0.83 0.534, gain+rho 0.373 vs base 0.633): the best-known
config is a jointly-tuned whole — ingredient attribution by grafting fails
(epistasis). The "what lifts the wander floor" question stays open and is
now known to be non-additive.

## H25 (preregistered): the transplantable core is {high leak, matched wlr, high gain}

From the existing reversion table (w1 loses most when gain, leak, or n_nodes
revert) + H24b (gain grafted at LOW leak hurts): prediction — (leak .574,
wlr 1.0, gain 28.35) on an otherwise-default N=200 network scores ≥ 0.65
(near w1-level), i.e. the epistatic core is gain x leak x lr-matching and
small N is not required.

## H25 verdict: PARTIALLY REFUTED — w1' resists reduction

essence (leak .574 + gain 28.35, N=200 defaults-else): 0.534±0.319 (62%),
short of the ≥0.65 prediction and far from w1's 0.84. With H24b: necessity
analysis names {gain, leak, N} as load-bearing, but NO subset tried is
sufficient — the best configs are non-decomposable in this design space.
gain-only 0.308, leak-only 0.456, essence+wlr0.65 no better. The wander-floor
question stays open and is now known to live in ≥3-way interactions
(plausibly: small N ↔ per-node input share ↔ authority ↔ damping).

## Metric accounting: the canonical dir-agree was an entrainment metric all along

analysis.tracking_metrics.direction_agreement compares the agent's SMOOTHED
TURNING DIRECTION to the STIMULUS direction (chance 0.5 for a turner, 0 for
a non-mover) — it measures velocity-following, not error correction. The
paper's 0.83 anchor is thus a follow statistic, consistent with the
entrainment mechanism. The lab harness's error-sign variant reads ~0.47-0.49
on constant-offset followers by construction (offset keeps err positive in
both travel directions) — do not interpret it as "no direction sense".

## H14 verdict (cluster, 4800 runs, 48 seeds/cell): SUPPORTED with a broad crest

Fine (leak x wlr) ridge: **wlr* = 1.04 · leak^1.41** fit on the 75% non-held
cells (b=1.41 inside the preregistered [1.2, 1.8]); 7/10 held-out rows'
peaks within factor 2 of the law — the misses are rows whose crest is FLAT
(argmax ill-defined at low leak, where near-peak score spans a decade of
wlr). Structure: dead-statue triangle upper-left with an escape threshold in
wlr that RISES with leak; crest height ~0.46-0.55 along the whole diagonal;
crest narrows at high leak. R2 confirms the headline at 48 seeds: wlr 0.1 →
0.467-0.525 vs default 0.386 (regression from the 12-seed 0.633 as the
wander analysis predicted). R3 reproduces the fixed-p N-collapse (confound
demo). Bonus: remote (wlr 1.0, tlr 0.01) cell = 0.386 on 48 seeds — EXACTLY
the local K4 value; cross-machine determinism verified. Figure:
scripts/out/lab/fig_ridge_fine.png. Total cluster compute: 4m19s on
mit_quicktest (the whole batch was ~3 core-hours; the fat-job provisioning
was 40x oversized — lesson recorded).

## H26 (preregistered): ridge-law transfer to Pong — the theory says it should NOT transfer naively

Pong's published init is net-SUBCRITICAL (75% N(0,.2) + 25% N(-1,.1) → mean
w ≈ -0.25, g_init < 0), the opposite regime from tracking's g_init=7.5. The
weight servo must BUILD drive up, not absorb it down, and most nodes ride
the target floor (T=1=floor, E<0 pins them; tlr is inert while pinned).
Predictions for the published Pong config (hit rate, 40 seeds, 100k steps):
(a) tlr 0.1 → 0.01 changes hit rate by < 0.03 (tlr-insensitive, UNLIKE
tracking's channel competition); (b) hit rate is non-decreasing in wlr over
{0.3, 0.65, 1.0, 2.0} (published wlr=1.0 near-optimal — subcritical growth
wants fast weights; the tracking ridge does NOT transfer); (c) wlr=0.1
degrades materially (>0.05).

## H27a (preregistered): wall-avoidance replication anchors (paper is qualitative)

Port follows the released Julia (input_weight=4 not the paper's 2; recurrent
init Normal(4,.1); wlr effectively 1.0; clamp off; ±45° random kick on
contact). Predictions from the paper's claims, quantified: (a) baseline —
hits concentrated in the first ~600 steps; majority of seeds have ZERO hits
in the last 1000 of 3600; (b) stable circling late (steady nonzero mean
turn rate, bounded radius); (c) learning-off — near-total saturation, agent
bounces continually (late hit rate ≫ baseline's); (d) sensor inversion at
t=1000 — hits resume briefly, re-stabilization within ~500 steps; (e) noise
U(±0.2) — avoidance persists (few late hits) though perfect stability is
lost.

## H27b (preregistered): the flow-sign inversion — the input-flow thesis's designed counterexample

On tracking/Pong, input flow correlates POSITIVELY with performance (rho
+0.77/+0.95) because those embodiments make disengagement starve the
sensors. Wall avoidance inverts the geometry: sensors read wall PROXIMITY,
so high flow = hugging walls and success = keeping flow LOW and stable.
Predictions across a random config screen (same battery as
screen_metrics): (a) rho(input_flow, avoidance performance) < 0 (sign
flip); (b) an input-stability metric (negated flow variance or
1/(1+std(flow))) correlates POSITIVELY; (c) the paper's own baseline agent
ends at LOW flow relative to a wall-hugger. Performance = 1 - late hit
rate. This tests "internal metrics generalize over embodiments, the body
decides the sign" (docs + memory item 6) with a preregistered sign.

## H27a verdict: wall-avoidance replication — all five anchors PASS

(24 seeds) Baseline hits 0.168→0.004/step by 600-bins, 96% of seeds
zero-late-hit; late behavior constant-direction circling (|signed dH| 0.145
= |dH| 0.146 rad/step; radius 5.0±1.4; f stays ~0.32 — the churn regime
spins the circle). Learning-off: f=1.000, |dH|=0, straight-bounce forever
at 0.45 hits/step. Inversion at t=1000: hits resume (55.9/500-window) then
restabilize (8.6 by the last window; 62% seeds clean late; paper's "<500
steps" was one representative run — ours is slower on average, noted).
Noise ±0.2: avoidance persists degraded (0.10-0.11 hits/step; zero seeds
fully stable) — matches "cannot completely stabilize".

## H27b verdict: FLOW SIGN FLIP CONFIRMED — the embodiment decides the sign

91-config screen x 6 seeds: among ALIVE configs (51/91),
rho(input_flow, avoidance) = **−0.859** (tracking +0.77, Pong +0.95);
flow STABILITY +0.884 and flow_sd −0.885 (as preregistered); activity f
−0.922; wall-proximity r_mean −0.756. Caveat handled openly: 40/91 dead
configs trivially score perf=1.0 (no motion = no hits) — wall avoidance is
the family's only task SOLVABLE BY DEATH, so alive-only is the meaningful
population (all-config rho −0.20 reported too; the alive restriction is
post-hoc but forced by the task's structure). The input-flow thesis's full
form is now demonstrated across all three case studies: internal metrics
generalize over embodiments; the BODY decides the sign of the
flow-performance coupling.

## H28 (preregistered): stabilization is event-driven rewiring until surprise ceases

Unifying reading of all three case studies: each task's failure event
(stimulus escape / ball reset / wall hit+kick) is an INPUT DISCONTINUITY;
such events inject error, error drives weight-churn bursts (Law 3), churn
changes behavior, and the loop settles exactly when events stop — behavior
as externalized absorption. Wall-avoidance predictions: (a) event-triggered
average of mean|E| and of per-step total |dW| shows a sharp burst at hits
(vs matched non-hit baseline); (b) cumulative |dW| rises step-like with
cumulative hits early and flattens when hits cease; (c) in the learning-off
arm hits produce the |E| burst but no rewiring and no stabilization (the
pawl without the ratchet).

## H28 verdict: REFUTED as event-driven search — corrected to transient convergence + stability selection

With time-local baselines (1444 events): |dW| at hits = 1.04x, |E| = 1.02x
(flow 1.26x) — individual collisions are nearly invisible to the network;
the naive 5-7x "burst" was an epoch confound (hits cluster in the early
high-churn phase). The +0.66 window correlation is co-occurrence, not
causation. Also: the learning-off statue shows ZERO |E| response to
collisions (saturation = deafness to events). Corrected account:
stabilization = convergence of the global churn transient; the settled
orbit persists where input variation along it is absorbable (small circles
away from walls/corners) — STABILITY SELECTION, not surprise-driven search.

## H28c (preregistered): displacement probe for stability selection

Teleport a settled (t=3600) agent, preserving all network state, to (a)
near a wall (2.0, 7.5) vs (b) a flat mid-arena point (7.5, 7.5).
Predictions: wall-adjacent placement destabilizes the orbit (elevated |E|
fluctuation and turning change) and the agent re-forms a wall-clear orbit
within ~600 steps with few hits (<5 median); flat placement just continues
circling (near-zero behavioral change).

## H28c verdict: SUPPORTED — the orbit is stability-selected

Teleport-to-wall (state preserved): |E| jumps to 0.155 (vs 0.075 at a flat
point, 2.1x), the agent drifts AWAY from the wall (2.74 -> 3.18 over the
window) as |E| relaxes to 0.068; median 6 post-hits (predicted <5 — near
miss, heavy tail: mean 46.6 from a few struggling seeds). Teleport-to-flat:
median 0 hits, behavior unchanged. Wall-adjacent orbits are dynamically
UNSTABLE (input variation along them exceeds the absorbable band); flat-
region orbits persist. Wall avoidance = stability selection among orbits,
with collisions as near-invisible perturbations — not punishment learning.

## H29 (preregistered): the boundary of homeostatic competence is absorbability, not difficulty

The edge-hold task: success = keeping the stimulus at |heading error| in
[50, 90] degrees (the steep Gaussian flank of the retina; the fovea plateau
|err|<=4 is FLAT and hence perfectly absorbable, the flank is not).
Homeostatic dynamics are score-blind, so occupancy under standard dynamics
IS the agent's best "performance" on any re-scored task. Predictions over
the 4800-run cluster R1 grid (policy occupancy histograms already
recorded): (a) NO (leak, wlr) cell reaches mean flank-band occupancy above
~0.30, while many cells exceed 0.50 fovea-band occupancy (uniform baselines:
0.222 flank, 0.25 fovea) — the family can hold the flat region but not the
steep one; (b) flank occupancy does not correlate positively with the ridge
(the best tracking configs are no better at edge-holding). If confirmed:
"reward becomes necessary" exactly where success requires operating outside
the absorbable band — and evolution's role (the input-flow work) is to
build bodies that fold tasks INTO that band.

## H29 verdict: SUPPORTED at the design level, with a repertoire wrinkle

Over the 4800-run R1 grid: max CELL flank-band ([50,90] deg) occupancy is
0.303 (1/100 cells above 0.30; uniform 0.222) vs fovea-band max 0.530 (6
cells > 0.50); flank occupancy is uncorrelated with the tracking ridge
(+0.11). No design knob aims the family at the steep band. BUT one run
reaches 0.782 — (leak .12, wlr .05, seed 33) parks at +65 deg with 56%
single-bin occupancy and within45 = 0.000: an offset-entrained orbit with a
large locked lag. Refined conclusion: **edge-holding exists in the
dynamical repertoire but homeostasis cannot select for it** — the offset is
a comfort-blind per-seed constant. "Reward becomes necessary" precisely as
a SELECTOR among dynamically available orbits (and evolution's role, per
the flow work, is to build bodies whose available orbits already fold the
task into the absorbable band).

## H30 (preregistered): selection finds the edge-holders homeostasis cannot aim at

GA over the design space (fitness = flank-band occupancy over the last
half; within45 recorded, never selected; seeds resampled per generation,
the repo's evolve-* protocol): prediction — mean population flank occupancy
exceeds 0.40 within 12 generations (vs 0.303 max random-grid cell), the
champion exceeds 0.60, and champions' within45 stays near 0 (edge-holding
is orthogonal to tracking). Confirms "reward = selector over the
homeostatic repertoire" constructively.

## H26 verdict: all three predictions REFUTED — channel competition is SIGNED by drive regime

400 cluster runs (published sensors, 100k steps, 40 seeds/cell). Published
cell (wlr 1.0, tlr 0.1) = 0.600±0.111 — EXACTLY the prior local validation
(second bit-identical cross-machine reproduction). But: (a) tlr-insensitivity
refuted at wlr=1.0 — (1.0, 0.01) collapses to 0.346; insensitivity holds
only at wlr<=0.3 (±0.03) as the floor argument said; (b) "published
near-optimal" refuted — **wlr 0.3/tlr 0.01 → 0.658** and 0.1/0.1 → 0.650
both beat it; the moderate-wlr peak TRANSFERS; (c) "wlr 0.1 degrades"
refuted (0.628-0.650). wlr=2.0 → 0.24 (the upper stability bound
transfers). Corrected law: channel competition is SIGNED by the drive
regime — supercritical (tracking): targets steal the error weights need to
ERODE gain (harmful); subcritical (Pong): targets DAMP the weight channel's
overshooting growth (necessary at wlr=1.0, unnecessary at moderate wlr).
Practical: one lr change improves all three case studies' published
configs.

## H30 verdict: STRONGLY SUPPORTED — perfect edge-holders in 3 generations

GA on flank occupancy: gen-1 champion 0.855, gen-3 champion **1.000** (100%
of steps in the 50-90 deg band, within45 = 0.000), population mean 0.689 by
gen 11 (prereg asked mean > 0.40, champion > 0.60 — exceeded). Champion:
N=217, w0_mean 0.1 (subcritical), gain 38.8, leak 0.47, wlr 0.74, tlr
0.0014. Reward-as-selector demonstrated constructively; the repertoire was
rich all along.

## H31 (preregistered): the body, not the synapse, is the slow variable

The tau(wlr) puzzle (b6b: 60/180/270/270 steps RISING with wlr) is proposed
to be behavioral, not synaptic. Predictions: (a) OPEN-LOOP pool-asymmetry
flip time after a scripted slip reversal scales roughly ~1/wlr and is
faster than the closed-loop tau63 at wlr >= 0.1; (b) closed-loop tau63
correlates with post-reversal EXCURSION DEPTH (integrated old-direction
motion) across wlr and across seeds; (c) a 2-ODE reduced model (bias
written toward slip at rate ~wlr; heading kinematics + the stall-in-dark
ratchet) reproduces the tau ordering only when the behavioral geometry is
included.

## H31 verdict: SUPPORTED — the body is the slow variable, and the reduced model closes

(a) Open-loop pool-asymmetry flip after a scripted slip reversal: 15-60
steps at EVERY wlr (vs closed-loop tau63 90-225) — synaptic rewrite is
never the bottleneck. The ~1/wlr scaling part was wrong (non-monotone;
stored asymmetry deepens with wlr: |pre| 0.006 -> 0.076). (b) Excursion
depth orders tau perfectly across configs (0/0/8.1/12.5 deg vs
30/90/135/225 steps); pooled per-seed Spearman(tau, excursion) = +0.64.
(c) The 2-ODE reduced model (theta' = v*d − b*vis(theta); b' =
vis*(alpha*(v*d−b) − beta*b); alpha, beta DERIVED from the open-loop flip
time and the measured asymptote; the pawl vis(|theta|<=92) is the geometry)
predicts tau63 with ZERO free parameters: 49/24/276/276 vs measured
60/180/270/270 — 3/4 within ~15%, reproducing the counterintuitive rise of
tau with wlr. The wlr=0.1 miss traces to a noise-dominated open-loop tau_w
estimate (pre-asymmetry 0.009; threshold-crossing detector fired early) —
better estimator flagged, not patched. Resolution of the b6b puzzle:
deeper stored bias -> longer post-reversal excursion -> the ratchet's
stall-and-reacquire pays the time; entrenchment is stored momentum in
BEHAVIOR, not synaptic sluggishness.

## H32 (pursuit, exploratory rounds 1-5): the family's first hard competence boundary

New task (src/homeostasis/pursuit.py): tracking's bearing retina (optional
1/(1+d/3) intensity falloff) on the wall-avoidance Braitenberg body, moving
stimulus (orbit/waypoint/still) in the 15x15 box. A P-controller scores
dist 0.81 / near3 1.00 — the task is servo-trivial. The homeostatic family
FAILS at every configuration tried (~40 arms x 8 seeds):

1. Baseline grids: agents die (f→0) — Law 1 in space: the intensity field
   has an ABSORBING dead basin (wander far → starve → still → dead), which
   rotation-only tracking geometrically lacked.
2. Bearing-only (cannot starve by distance) STILL dies → the true basin is
   **wall-pinned outward-facing poses**: parked in a corner facing out, the
   whole stimulus orbit stays behind the ±92° view forever. Walls +
   translation manufacture absorbing darkness; tracking's periodicity
   rescue needs a body that cannot leave or look away for good.
3. 360° retina deletes the basin (f 0.17-0.44 sustained ✓) — but pursuit
   still fails: orientation at chance, distance ≈ random-turn.
4. Motor-grain hypothesis (wheel_base 1→16, max turn 57→3.6°/step):
   REFUTED — gentler turning is worse. Best whisper: forward retina, wb 4,
   wlr 0.1 (orientation 0.41-0.42 vs 0.25 chance; dist 6.06 vs still-agent
   4.50).
5. Velocity-floor hypothesis (churn-forced cruise cannot match a slower
   target): REFUTED cleanly — faster stimuli are WORSE (orient 0.41 → 0.20
   at speed 0.3-0.5) and cruise straddled the target speed anyway.

Reading: tracking worked because its effector-to-slip map is 1-DOF,
sign-stable, and stillness-capable. The 2D body breaks all three at once
(parallax adds a distance-dependent second coupling that diverges on
approach; translation is churn-forced so the ratchet's waiting state is
unreachable; dark poses exist). The mentors' ladder rung 3 is genuinely
OPEN — not an incremental extension of the paper. Next candidates recorded:
evolution over the pursuit design space (H30 pattern: can selection find
pursuers where hand-design cannot?), tonic-drive restlessness, and
stimulus-locked orbiting (approach-free following) as the achievable form.

## H30 addendum (from the wall-viewer verification): the evolved edge-holder is seed-brittle

Across fresh seeds 0-7 at 7200 steps the champion's band fraction is
{0.91, 0.75, 0.41, 0.33, 0.32, 0.23, 0.10, 0.00} — a works-or-fails wiring
lottery, the same pattern as the historical flow-evolution champions.
Selection finds seed-specific solutions when fitness is averaged over few
resampled seeds.

## H33 (preregistered): can selection find the pursuit hand-design could not?

GA over the pursuit design space (360-retina fixed for aliveness; genes:
n_nodes, p_link, input_weight, w0_mean, leak, tlr, rho, wlr, wheel_base,
intensity_scale; fitness = near3 − dist/15, 3 resampled seeds/gen, orbit
0.15). Predictions: (a) IF the repertoire contains pursuit, champion near3
≥ 0.5 within 14 generations; a null result at N≤320 marks a harder
boundary than H30's; (b) any champion will be seed-brittle (the family's
lottery pattern).

## H33 verdict: pursuit exists as (genome x wiring) jackpots; nothing yet makes it reliable

Evolution reached best-of-generation near3 0.71 / dist 3.35 (gen 7) — far
beyond any hand-designed arm (0.16) — with the predicted generation-to-
generation noise. Champion genome is the SUBCRITICAL recipe (w0 0.11,
wlr 0.02 at the range floor, tlr 0.06 fast-damping, inputs 9.2, agile
wheel_base 1.06 — motor-grain intuition refuted a second time). Fresh-seed
verification: near3 = [1.00, 0.49, 0.25, ...] median 0.00 — ONE perfect
pursuer among 16 wirings. Conclusion: the repertoire contains perfect 2D
pursuit, but (a) homeostasis cannot aim at it (H32), and (b) genome-only
selection cannot fix the wiring lottery at N<=320 and 14 generations.
Reliable rung-3 competence requires something qualitatively new: wiring-
level/developmental selection, larger N, or architectural priors. This is
the sharpest available statement of the mentors' "when does reward/
evolution become necessary" question: reward selects orbits within a
lifetime (H29/H30); evolution selects genomes; NEITHER, so far, selects
past the wiring lottery.

## H33c (autopsy): the perfect pursuer is a phase-locked co-rotating orbit

Seed 1004: the agent circles concentrically JUST INSIDE the stimulus's
orbit — revolution rate +1.9 deg/step = the stimulus's own angular rate
(phase lock), dist 2.40±0.27, bearing ~34 deg constant, f=0.08 (deep
absorbed regime), 9 hits all early. It does not chase; it found the
co-rotating frame in which its retina is STATIC — the wall task's
stability-selected circle transported into the target's moving frame.
Velocity entrainment (tracking), orbit stability-selection (wall), and
pursuit are one mechanism: **behavior settles on absorbable sensory
manifolds**. The competence boundary refines to: such manifolds exist for
pursuit but are rare in wiring space (the lottery); servo controllers need
no luck because they use sign information explicitly.
Figure: scripts/out/lab/fig_perfect_pursuer.png.

## H34 (preregistered): the lottery is fixable by selecting wirings

Joint evolution of (genome, seed) — the wiring seed as a gene (re-drawn
with p 0.25 on mutation), fitness on the individual's own wiring.
Predictions: champion near3 >= 0.8 by generation 8; final population mean
>= 0.4; and the champion GENOME re-tested across fresh seeds remains a
lottery (the fix is keeping the wiring, not improving the genome).

## H34b verdict: a permanent orbit resonance, not general pursuit

Champion pair holds near3 1.00 / dist 0.84 for 14400 steps (3 hits) — a
stable attractor. Genome on fresh wirings: median 0.11 with TWO perfect
(2/16; jackpot rate raised, lottery intact — prereg (c) ✓). WAYPOINT
(unpredictable) motion: collapse to 0.12 / dist 6.68 / 351 hits. The
evolved solution is an orbit-commensurate limit cycle — entrainment onto
the target's co-moving frame — and unpredictable motion offers no such
frame. Empirical settlement of the mentors' meeting-note conjecture: the
"predictable following" half of ladder rung 3 IS an entirely homeostatic
process (their baseball intuition, confirmed); the unpredictable half is
where the family's competence ends.

## H35 (preregistered): minimal reward-gating at the established boundary

Reward-modulated weight servo on WAYPOINT pursuit: after each step, rescale
the just-applied weight delta by (1 + beta*r), r = clip(approach rate /
0.15, -1, 1) (targets untouched; beta=0 is the control). Champion genome,
fresh wirings 2000-2015, beta {0, 2, 5}. Prediction (risky): if the missing
ingredient is merely credit-for-approach, near3 rises well above the 0.12
baseline within a lifetime; failure would mean unpredictable pursuit needs
machinery beyond modulated homeostasis (prediction/planning). Either
outcome sharpens the boundary.

## H35 verdict: REFUTED — reward-as-gain on homeostasis is not reinforcement

Scaling the weight servo's magnitude by an approach reward does nothing on
waypoint pursuit (beta 2: near3 0.14 vs control 0.13; beta 5: 0.10 —
slightly harmful). Mechanism: the homeostatic update's DIRECTION is fixed
by drive error; reward-scaling changes when it learns, never what. The
sharpened boundary statement: crossing it requires reward-DIRECTED credit
(e.g., a three-factor term), not reward-scaled comfort. (H36 candidate.)

## H36 (preregistered): does reward-DIRECTED credit cross the boundary?

Minimal three-factor addition ON TOP of the intact homeostatic rules:
after each step, dw[pre,post] += eta_r * r * pre_spike(t-1) * post_spike(t)
on existing links (same eligibility window as the homeostatic gate), r =
approach rate as in H35. This deliberately ADDS machinery (a marked
design-space extension). Arms: eta_r {0, 0.01, 0.05, 0.2} x 12 fresh
wirings on waypoint pursuit + a time-shuffled-r control at the best eta_r
(same reward statistics, no contingency). Predictions: (a) if directed
credit suffices at N=64, some eta_r lifts median near3 to >= 0.3 (baseline
0.13) and shuffled-r does not; (b) failure at all eta_r means the boundary
needs more than scalar-modulated Hebbian credit (structured
prediction/planning) — either way the boundary sharpens.

## H36 verdict: REFUTED — one-step three-factor credit does not cross it either

eta {0.01, 0.05, 0.2}: best 0.14 vs control 0.13; shuffled-reward control
identical (0.13). Neither reward-scaled (H35) nor reward-directed one-step
Hebbian (H36) modulation produces unpredictable-target pursuit at N=64
within a lifetime. Scope caveats on the record: one genome, global scalar
reward, no eligibility traces, 7200 steps — traces/multi-step credit are
the canonical next rung. As a minimal-additions statement it stands: the
boundary needs structured machinery, not scalar neuromodulation.

## H37 (preregistered): the followability law

Stimulus "wander" motion: constant speed 0.15, heading diffusing with
per-step sigma (direction correlation time T_c ~ 1/sigma^2), reflecting at
the walls. Champion pair (H34, seed 66777). Predictions: (a) near3 is a
decreasing function of 1/T_c with a knee where T_c crosses the agent's own
re-lock time (order 10^2 steps): near3 >= 0.5 for T_c >= ~1000 and
collapse toward the 0.12 waypoint level by T_c ~ 100; (b) the law's form —
followable iff the target's co-moving frame outlives re-entrainment — is
the quantitative boundary of ladder rung 3.

## H37 verdict: unmeasurable as designed — and the probe found something better

Two failures on the record: (1) the rep-variation hack (extra tail steps)
did not vary trajectories (identical runs, ±0.00 SDs) — env randomness
draws from the same rng stream regardless; (2) the sigma=0 "persistence
reference" is a straight-line wall-bouncing shuttle, not the orbit — and
the perfect orbit-pursuer scores **0.02** on it (vs 1.00 on its orbit).
PERFECTLY PREDICTABLE but differently-SHAPED motion destroys the solution:
the champion is a resonator tuned to one closed absorbable manifold, not a
persistence-limited follower. The followability-vs-persistence law cannot
be measured because the family produces no general follower to measure.
Final rung-3 statement: **the homeostatic family produces target-specific
resonators** — following exists only as entrainment onto a specific
stationary-izable manifold (tracking's circle, wall's orbit, pursuit's
co-rotating ring); no tested mechanism (hand design, genome selection,
joint wiring selection, reward-scaled or reward-directed plasticity)
yields motion-general pursuit. Theory note: a shuttle also has a periodic
co-moving frame, but with heading-flip discontinuities at the turning
points that absorption cannot smooth — shape-specificity is
absorbability-of-the-frame, consistent with everything above.

## H38 (preregistered): the manifold theory's discriminator

If competence = entrainment onto a smooth closed absorbable manifold, joint
(genome, wiring) evolution should find a resonator for an ELLIPSE orbit
(smooth, curvature varying 4:1) but NOT for the straight-line SHUTTLE
(heading-flip discontinuities at the turning points). Predictions: 10-gen
joint GA reaches best near3 >= 0.8 on ellipse and stays < 0.4 on shuttle.
Both succeeding would weaken the smoothness requirement; both failing would
mean the circle itself is special (constant curvature).

## H38 verdict: NO following evolved for either shape — both champions are TOLL-BOOTHS

Autopsy: shuttle champion speed 0.000/spread 0.00 (perfectly parked at a
favorable point, near3 0.54 as the target passes); ellipse champion speed
0.039/spread 0.98 (parked with wobble, 0.69). The prereg dichotomy was
mis-posed: for non-circular periodic motion, selection finds STILLNESS at a
good location, not co-motion. Refined law: **entrainable manifolds are
those requiring CONSTANT CONTROL** — the circular orbit is special because
its co-moving frame is reachable by an agent turning at a constant rate
(constant curvature); an ellipse's frame demands modulated turning, so the
control signal itself is non-stationary and unabsorbable. Sensing
stationarity AND control stationarity must hold simultaneously.
Figure: fig_h38_champions.png.

## H38c (preregistered): the constant-control interpolation

A nearly-circular ellipse (a=4.5, b=4.0; curvature ratio ~1.27) should be
within the constant-control tolerance: 10-generation joint GA finds a
GENUINE FOLLOWER (agent speed > 0.05, positional spread > 1, near3 >= 0.8),
unlike the 4:1 ellipse's toll-booth.

## H38c verdict: CONFIRMED — the constant-control law completes

Near-circular ellipse (ratio 1.27): joint GA evolves a GENUINE FOLLOWER in
8 generations — near3 1.00, dist 0.58, agent speed 0.159 ≈ stimulus 0.15
(velocity-matched), spread 4.5 (traverses the path). Full interpolation:
circle → follower (0.80); ratio 1.27 → follower (0.58/1.00); ratio 4 →
toll-booth (0.69); shuttle → parked toll-booth (0.54); aperiodic → nothing
(0.12). **The competence law: homeostatic agents entrain to motions whose
co-moving frame is holdable with ~constant control; past that tolerance
selection substitutes stillness at favorable points; without periodicity,
nothing.** The tolerance boundary lies in curvature ratio (1.27, 4.0) —
open interval, bisectable later. This is the quantitative form of the
mentors' ladder rung 3 and of "prediction vs Gibsonian loops": within the
law's reach, no prediction is needed (the baseball intuition); beyond it,
the family cannot go.

## H38d verdict: the tolerance is ~2x control modulation, with a soft edge

Curvature ratio 1.6: FOLLOWER (near3 0.96, dist 2.39, speed 0.096, spread
2.39 — degraded but genuine). Ratio 2.5: toll-booth, parked at speed 0.000
(near3 0.43). The constant-control boundary sits in (1.6, 2.5), and the
follower's quality degrades smoothly toward it (dist 0.58 → 2.39 over
1.27 → 1.6): an absorbability TOLERANCE, not a cliff. Final hierarchy:
1.0 follower 0.80 | 1.27 follower 0.58 | 1.6 follower 2.39 | 2.5 toll-booth
| 4.0 toll-booth | discontinuous toll-booth | aperiodic nothing.

## H39 (preregistered): the mixed-sign embodiment

Pursuit with 2 wall-proximity sensors APPENDED to the 91-sensor bearing
retina (flow-positive + flow-negative channels in one body), stimulus orbit
radius 6.0 (path within ~1.5 of the walls — approach and wall-comfort in
conflict). Joint GA (10 gens) with vs without wall sensors. Predictions:
(a) wall sensors reduce wall hits (early search made wall-averse);
(b) the conflict produces a measurable COMPROMISE in champions with wall
sensors: agent-stimulus distance increases where the stimulus is
wall-close (positive corr(dist, stimulus wall proximity) across orbit
phase), absent without them; (c) peak near3 is not worse with wall sensors
(the channels are compatible mid-arena).

## H39 verdict: (a)(b) REFUTED, (c) confirmed — the manifold law subsumes the sign story

Both arms evolve perfect wall-adjacent followers (near3 1.00, dist
1.01/1.13, hits 46=46); no compromise signature (corr -0.05/-0.08). Channel
conflict never materializes because the entrained solution makes ALL
channels stationary simultaneously — the co-rotating frame renders
wall-proximity input periodic too. Signs shape screen-level correlations
(H27b); solutions are joint-stationarity orbits.

## H40 (preregistered): is the wiring lottery decided early?

Pursuit champion genome x 32 fresh wirings: record first-200-step features
(mean flow, flow trend, net rotation vs stimulus direction, f) and final
near3. Prediction: early features separate eventual lockers from failures
with >= 80% accuracy — the lottery is an opening-basin property, enabling
audition/developmental selection.

## H41 (preregistered): discomfort-triggered annealing fixes the lottery WITHOUT reward

Homeostasis-native reliability rule built from K3's shuffle-recovery: every
600 steps, if the agent's own input is non-stationary (flow_sd/flow above a
threshold — self-detectable), SHUFFLE the recurrent weights (learning stays
on) and try again; keep the configuration once input goes stationary.
Prediction: lock rate on the pursuit champion genome rises from ~2/16
(baseline jackpot rate) to >= 8/16 within 14400 steps — reliability from
the network's own comfort signal, no external reward.

## H40 verdict: UNDERPOWERED (1/32 lockers); H41 as-designed under-triggered

H40: the champion genome's jackpot rate at seeds 3000+ is 1/32 — one
positive case cannot establish a predictor (feature deltas suggestive but
n=1; a ~300-wiring version would be needed). H41: the "non-stationarity"
trigger fired ~0.9 times/run and lifted lock rate only 2/16 vs 0/16 —
because failed agents are STATIONARY-AT-MEDIOCRE (absorbed wandering, low
steady flow), not unstable. Corrected trigger preregistered: shuffle when
600-window mean flow < 2.2 (insufficient engagement — flow-seeking
annealing, the input-flow thesis as a reset rule). Same prediction: >= 8/16
locked.

## H41b verdict: REFUTED — and the lottery's home is found

Flow-seeking annealing fires every check (21 shuffles/run) and locks 0/16.
Diagnosis: the shuffle operator permutes weight VALUES on the FIXED
adjacency — but the lottery lives in the ADJACENCY DRAW, which no
within-life weight operation can resample (K3's shuffle recovered function
on good structure; it cannot repair bad structure). Closing statement of
the lottery thread: **the family's reliability ceiling is set by frozen
wiring structure; learning, shuffling, and annealing all act in weight
space and cannot cross it. Structural plasticity (grow/prune) or
wiring-level selection is the missing mechanism** — precisely the mentors'
connectivity/DNA agenda, now with an empirical mandate.

## H42 (preregistered): structural homeostasis — can grow/prune cross the ceiling?

Marked model EXTENSION (like input_plastic; implemented in the lab loop,
core untouched): every 120 steps each node with window-mean error < -0.05
(persistently starved) GROWS one incoming link from a random non-neighbor
(weight = weight_init_mean draw); each node with window-mean error > +0.05
(persistently overdriven) PRUNES its weakest incoming link. In-degree
bounded [2, 3*p*N]. Test: pursuit champion genome x 16 fresh wirings x
{structural on, off} x 14400 steps. Prediction (risky): lock rate rises
from 0-1/16 to >= 6/16 — the agent rewires its way into the lock basin.
Failure narrows the fix to wiring-level SELECTION (structure search may
need population-level exploration, not local hill-climbing).

## H42 verdict: PARTIAL — structural plasticity is the first within-life mechanism to cross the ceiling

Grow/prune lifts lock rate 0/16 → 3/16 (mean near3 0.09 → 0.25; 1132
grown / 758 pruned, mean degree 6.4 → 13). Below the preregistered >= 6/16
(recorded), but qualitatively decisive: the mechanism HIERARCHY for the
structure ceiling is now measured — weight-space operations 0/16; local
structural hill-climbing 3/16; population selection over wirings (H34)
reliable on its pair. First-guess parameters (theta ±0.05, window 120,
random growth source) untuned; the direction, not the number, is the
finding. This hands the mentors' connectivity/DNA agenda its empirical
mandate and its first working local rule.

## H40b verdict: REFUTED at proper power — the lottery is not early-legible

300 wirings, 20 lockers (6.7%): balanced accuracy 0.49-0.52 for every
early-200-step feature (flow, flow trend, revolution rate, dist trend) —
chance. With K4 (static kernel fails): the lock outcome is decided late in
the run by structure finer than these summaries. Prediction-based audition
strategies are out; selection/structural search remain.

## H42b/H43 (preregistered): tuning structural homeostasis; the tracking wander floor

H42b variants (16 wirings each, pursuit): (i) gentler+slower (win 240,
theta ±0.03); (ii) sensor-biased growth (new afferents drawn from nodes
with above-median input in-degree — "grow toward the sensors"); (iii) both.
Prediction: at least one variant beats 3/16 (>=5/16). H43: base grow/prune
rule on TRACKING at ridge25 (24 seeds, 14400 steps): prediction —
structural plasticity raises frac(score_late >= 0.35) by >= 0.10 over
baseline and reduces late-segment wander (higher lag-1 correlation of
segment scores).

## H42b/H43 verdicts: tunings fail; structural plasticity is a stability-plasticity trade at the structure level

H42b: no variant beats base 3/16 (gentle 1, sensor-bias 1, both 2). H43:
on TRACKING (a solved task) the same rule is CATASTROPHIC — 0.284 vs 0.491,
frac>=0.35 0.88 -> 0.00: perpetual rewiring destroys the entrained drive
balance (the rule's threshold fires on ordinary churn-regime errors).
Structural plasticity helps only where frozen structure is hopeless and
harms where structure suffices — the stability-plasticity dilemma at the
structural level.

## H44 (preregistered): the developmental window resolves the structural dilemma

Rewiring permitted only during the first 3600 steps, frozen after.
Predictions: (a) tracking recovers to baseline (frac >= 0.80, vs 0.00 with
lifelong rewiring); (b) pursuit locks >= the lifelong rule's 3/16 (early
exploration is when it matters).

## H44 verdict: SUPPORTED — the developmental window resolves the structural dilemma

Tracking recovers fully (0.481 / frac 0.92 vs baseline 0.491/0.88;
lifelong rewiring was 0.284/0.00); pursuit keeps the structural benefit
(2/16 locked, mean 0.26 vs lifelong 3/16, 0.25 — comparable; the >=3/16
count is a near-miss, recorded). Conclusion of the structural trilogy
(H42-H44): grow/prune helps exactly where frozen structure is hopeless,
harms where structure suffices, and a developmental window — structural
exploration that CLOSES — captures the benefit at no cost. The family,
minimally extended, rediscovers why development exists.

## H45 (preregistered): N-scaling of the ridge and duty law

Tracking N-line {200, 500, 1000, 2000} x wlr {0.05, 0.1, 0.2} at leak .25,
in-degree pinned (p = 20/N) and input wiring pinned (input_p_link = 0.1),
24 CRN seeds (the H23c clean protocol, cluster). Predictions: (a) the ridge
position in wlr is N-invariant; (b) score at the ridge is N-stable within
±0.05 from N=500 to 2000; (c) the duty law holds per-node at N=2000 with
corr >= 0.99 (open-loop check, local).

## H45c (local part): duty law at N=2000 — holds, mildly degraded

Per-(node, window) fit over 46,000 points: corr +0.9881 (a shade under the
preregistered 0.99), median |resid| 0.021 (vs 0.0025 at N=200). The law is
N-robust; the extra residual is consistent with more heterogeneous
within-window drift at 10x nodes.

## H46 (preregistered): the two-body problem — mutual tracking

Two tracking agents on the circle, each sensing the OTHER's heading
direction as its stimulus (rotation-only bodies, standard retina; no
external stimulus). Predictions from the entrainment/absorption theory:
(a) at ridge params (wlr 0.1) the pair ALIGNS (|h_A − h_B| → small) and
then goes quiet (mutual absorption: alignment is a stationary manifold —
a two-body statue), f dropping well below the solo-tracking level;
(b) at churn params (wlr 1.0) alignment is looser and activity persists
(mutual chase); (c) no sustained anti-alignment or drift regime at either.

## H46 verdict: the two-body ground state is stillness; churn animates but cannot lead

(Design lesson first: an out-of-view start freezes the pair forever —
mutual darkness is absorbing; rerun with in-view start.) Homogeneous
absorption pair (wlr 0.1): INSTANT mutual statue — f 0.000, zero motion,
gap frozen at its initial value; no aligning phase (a static partner is
already a stationary stimulus, so stillness is self-consistent from t=0;
prereg (a) half-wrong, recorded). Homogeneous churn pair (1.0): perpetual
weak mutual chase (f 0.13, 18% aligned, no lock). Mixed pair (churner +
absorber): the churner ANIMATES the absorber (f_B 0.000 -> 0.038, 20%
aligned) but no leader-follower lock — because churn motion is APERIODIC,
exactly the class the constant-control law says the family cannot follow.
Coherent multi-agent conclusion: interaction alone generates no sustained
dynamics; collective entrainment needs a periodically-moving member (e.g.,
a wall-avoidance circler as the "pacemaker") — a concrete, testable design
principle for the mentors' multi-agent program.

## H47 (preregistered): the pacemaker — can the family follow ITS OWN circler?

Record a stable wall-avoidance circler's late trajectory (seed 0, steps
1800-3600) and replay it cyclically as the pursuit stimulus; joint
(genome, wiring) GA on following it. Prediction: because the circler is
periodic with ~constant curvature, a genuine follower evolves (near3 >=
0.8 within 10 generations) — closing the ecology loop: agent-generated
motion is followable exactly when it is the periodic kind, so homeostatic
collectives CAN sustain entrainment chains with a circler as pacemaker.

## H47 verdict: REFUTED at native speed, CONFIRMED with the band clause — the law gains its second condition

Design confounds fixed on the record (wrap-jump replay; then a clean
single-period loop, period 56, closure 0.19). At the circler's native
6.4 deg/step: best 0.50 = center-parking with the target skimming the
3-unit line — no follower. Replayed at 1/3 speed (2.1 deg/step, the H34
band): **perfect follower by generation 5 (near3 1.00, dist 1.55)**. The
competence law is two-clause: (i) the co-moving frame requires ~constant
control (shape), AND (ii) the frame's angular rate must sit inside the
entrainment band (speed). Ecology corollary: entrainment chains with a
circler pacemaker are possible iff the pacemaker is slow — a quantitative
design principle for the multi-agent program (wall-circlers at default
params turn ~8 deg/step: too fast; slow-circling wall configs are the
predicted viable pacemakers).

## H45 verdict: CONFIRMED — the phenomenology is N-invariant (200 → 2000)

Cluster N-line (288 runs, in-degree pinned at 20, input_p_link pinned 0.1,
24 CRN seeds/cell): ridge position in wlr does not move with N; scores are
trendless across the table (ridge column 0.503 / 0.390 / 0.441 / 0.560 —
spread slightly wider than the preregistered ±0.05, with N=500 dipping;
no monotone trend, and N=2000 is nominally the BEST cell, 0.560/0.83).
Internal coordinates are rock-steady across 10x size: f 0.144-0.164,
|E| 0.534-0.592, flow 2.15-2.52. With the duty law at 0.988 (N=2000):
**the answer to "scaling up network sizes" is that size per se is inert —
the organizing variables are degrees and rates, not counts.** (Batch
lesson: shuffled chunks must respect the lane's wall-time; size-4 chunks on
quicktest finished 288 runs incl. N=2000 in ~9 min.)

## H48 (preregistered): the live ecology — a slow circler as a real pacemaker

(a) Wall task at wlr {0.05, 0.1, 0.2}: prediction — absorption-regime wall
agents still avoid (late hits ~0 for most seeds) and circle SLOWER at
similar radius (angular rate scales with f; radius = v/omega is
f-independent), yielding a natural pacemaker at <= 2.5 deg/step. (b) Live
one-way chain (circler blind to follower; follower senses circler's
position): an evolved follower entrains to the LIVE slow circler (near3 >=
0.8 over the last half), completing the first sustained two-agent
homeostatic ecology.

## H48 verdict: THE LIVE ECOLOGY EXISTS (with its requirements measured)

(a) Naturally slow circlers do NOT exist in the paper's arena: absorption-
regime wall agents are dead statues (avoidance-by-death), churn circlers
turn at >= 5.6 deg/step — above the follower band. The family cannot pace
itself AT THE PAPER'S SCALE. But morphology+world rescaling opens it: wall
wheel_base 2.5-4 in a 30-unit arena yields clean circlers at 1.9-2.6
deg/step (radius ~4.5-7.8). (b) Cold-start GA on the live chain fails
(toll-booth, 0.17) and champion transfer is partial (h34 follower: 0.57);
(c) WARM-STARTED evolution (population seeded from the h34 follower) locks
by generation ~9: **near4 1.00, dist 3.77, stable through 10,800 steps —
the blind wall-circling pacemaker entrains a live follower into a
concentric phase-locked orbit** (fig_live_chain.png). First sustained
two-agent homeostatic system. Requirements, all measured: a pacemaker in
the entrainment band (needs morphological wheel-base + arena rescaling),
one-way sensory coupling, and selection warm-started from prior follower
competence. The mentors' multi-agent question has its constructive answer
and its recipe.

## H49 (preregistered): transitive entrainment — the three-agent chain

Agent C (warm-started from B's follower genome) evolved against the LIVE
follower B's position while B follows the pacemaker A. B's orbit is a clean
circle at ~1.9 deg/step (band-compatible). Prediction: C locks onto B
(near4 >= 0.8), giving A -> B -> C — entrainment propagates through a
chain of comfort-seeking agents. Failure mode to watch: B's orbit may be
noisier than A's (jitter from B's own churn) — if C fails, measure B's
orbit regularity vs A's.

## H49 verdict: ENTRAINMENT PROPAGATES — the three-agent chain locks

C (warm-started from B's genome) locks onto the LIVE B while B follows A:
C-B dist 2.88, near4 1.00, stable through 10,800 steps — three concentric
phase-locked circles (fig_chain3.png). A blind pacemaker's periodicity
cascades down a sensing chain of comfort-seeking agents. The rings shrink
~15% per link (radius 7.8 -> 6.5 -> 5.6), predicting a FINITE CHAIN DEPTH:
successive orbits contract until curvature/speed leaves the entrainment
band — an open, quantitative prediction for the multi-agent program
(measure max depth vs pacemaker radius).

## H50 (preregistered): the telephone-game law — chain depth and jitter accumulation

Extend the chain link by link (D follows live C, E follows live D...), each
link warm-started from its predecessor's genome. The phase-locked angular
rate is constant down the chain, so the band-rate criterion does not break
it; the candidate failure mode is JITTER ACCUMULATION (each follower's
orbit is noisier than its target's). Predictions: (a) D locks (near4 >=
0.8); (b) per-link orbit jitter (sd of distance-to-target) increases
monotonically with depth; (c) the chain fails at the depth where
accumulated jitter exceeds the follower band — reported as the measured
depth limit (or "no break by depth 5" if it holds).

## H50 first pass: CONFOUNDED — established links are start-position sensitive

Moving link C's start from its native y=5.0 to 7.5 unlocked it entirely
(C-B dist 8.74 vs native 2.88): the locks are BASIN-dependent in initial
position as well as wiring — the IC-sensitivity theme again, now measured
at the collective level. Rerun with per-link native starts pinned.

## H50 verdict (clean): chain depth 4, and the jitter law is a threshold, not a ramp

With per-link native starts pinned: B (dist 3.77, sd 0.048) -> C (2.88, sd
0.055) -> D LOCKS (3.16, sd 0.38) -> E fails (0.39). (a) D locks ✓ — after
one false start from an inside-the-rings position: link evolvability is
START-BASIN sensitive, like everything else in this family. (b) Jitter is
NEGLIGIBLE for two links then jumps 6x at the third — threshold
amplification, not linear accumulation; (c) the fourth evolution fails on
that noisier target. Measured chain depth at this budget/scale: a pacemaker
plus THREE followers. The telephone-game law: entrainment chains are nearly
lossless until a link's orbit noise crosses the band, then the next link
cannot form.

## H51 (preregistered): noise robustness of the ridge

Tracking with per-step uniform sensor noise U(±σ), σ {0, 0.1, 0.2} × wlr
{0.03, 0.1, 0.3, 1.0} (leak .25, tlr .01), 16 CRN seeds, harness-level
noise (clamped at 0; core untouched). Predictions (the SNR theory's):
(a) noise costs score most at wlr=1.0 (external noise compounds the
servo's self-churn); (b) the ridge peak stays at or shifts BELOW wlr=0.1
(self-churn becomes redundant when the environment supplies fluctuation);
(c) at wlr=0.03 noise partially RESCUES (breaks the statue: external
fluctuation substitutes for the absent weight churn).

## H51 verdict: stochastic resonance across the plane; strong noise starves absorption

sigma=0.1 RESCUES the under-plastic side (wlr 0.03: 0.367 -> 0.533,
reliability 0.44 -> 0.81 — external fluctuation substitutes for missing
weight churn; prereg (c) strongly confirmed) and even lifts wlr=1.0
(0.340 -> 0.419/0.88): the ridge FLATTENS, not shifts ((b) partial).
sigma=0.2: the absorption regime's activity collapses (f 0.17 -> 0.04 —
noise accelerates erosion; score still 0.442) and wlr=1.0 degrades ((a)
holds only at strong noise). Noise is a design RESOURCE for under-plastic
networks and a tax on over-plastic ones.

## H51b (preregistered): the noise-rescue mechanism is desaturation

Open-loop transfer probe at wlr=0.03 (the saturated, information-blind
regime; baseline recon gain ~0.05): adding sigma=0.1 sensor noise
desaturates (f drops from ~1 toward ~0.2) and recon gain rises above 0.15.

## H51b verdict: mixed, and the miss is the finding

sigma=0.2 desaturates open-loop exactly as predicted (gain 0.002 -> 0.196,
f 0.99 -> 0.38: rectified noise adds mean drive, the integral controller
bites). But sigma=0.1 — the level that rescues closed-loop wlr=0.03 —
does NOTHING open-loop (gain 0.001, f 0.98). The rescue is not a readout
effect. (h51b_desat.json)

## H51c: the rescue is a dark-trap escape (loop-level, not reservoir-level)

At wlr=0.03/sigma=0 the failing agent spends 35% of steps in darkness
(input_duty 0.65) — zero input flow, hence zero learning signal: an
absorbing sensory dead state (the ratchet clause "dark -> still" is a trap
for under-plastic networks). sigma=0.1 abolishes darkness (duty 1.00, flow
1.98 -> 3.68) while the motor barely changes (eff_diff 0.049 -> 0.056):
the flow ratchet stays engaged, score 0.367 -> 0.533. sigma=0.2 keeps
duty=1.00 but washes out contrast (dir-agree 0.297 -> 0.227, eff_sat up):
inverted-U in sigma, optimum = darkness abolished AND contrast preserved.
A sensory noise floor is a design resource specifically for under-plastic
networks. (h51c_movement.json)

## H53 (preregistered): homeostasis is self-repair — mid-run node death

Setup: ridge cell (wlr .1, tlr .01, leak .25), n=14400, at step 7200 kill
k% of nodes (full adjacency removal in+out, input cut, x=0; caches rebuilt
so plasticity cannot regrow; dead stay in output-pool denominators — silent
neurons dilute the readout, gain must be re-earned). Arms: kill-mid
(learning on) vs kill-mid-frozen (learning off at kill), 16 CRN seeds,
k in {10,30,50}. Predictions: (a) k=10: learning recovers to >=80% of its
own pre-kill late score within 2 segments; frozen degrades more and stays
flat; (b) k=30: learning >=60% of pre-kill, frozen well below; (c) k=50:
partial recovery, ordering learning>frozen everywhere; (d) mechanism per
the comfort laws: survivor drive drops -> E<0 -> T falls and incoming W
potentiates -> f returns to pre-kill (duty law); frozen f stays depressed.

## H53 verdict: physiology always repairs; behavior only needs it for big wounds

Score predictions (a)-(c) largely REFUTED — behavioral redundancy buffers
structural damage: frozen 10% kill is free (rec 0.506 ~ baseline 0.510),
frozen 30% keeps ~76% of baseline, and at k=0.1 the repair TRANSIENT is
net harmful (paired learning-frozen -0.098, frozen wins 10/16) — the
inflammation cost exceeds the wound. Repair pays only at big wounds:
k=0.5 median 0.432 (learning) vs 0.295 (frozen). Learning abolishes the
catastrophic tail at k=0.1 (dead seeds 0/16 vs 3/16 frozen).

Mechanism (d) STRONGLY confirmed: f recovers under learning (k=0.3:
0.174 -> 0.115 -> 0.136 vs frozen 0.046 -> 0.057 flat; k=0.5 doubles back
0.042 -> 0.096 vs 0.029). The repair channel is Law 3: kill halves
Sigma-w_in -> persistent E<0 -> integral controller re-inflates surviving
weights (w_mean 0.106 -> 0.210 at k=0.5, near-exact 2x for halved
in-degree; T gives only 0.07, blocked by floor 1.0). The model reproduces
deafferentation-induced compensatory synaptic scaling (Turrigiano) from
eq. 5 with nothing added. (h53_selfrepair.json)

## H54 (preregistered): sparsity acts through drive variance, not drive mean

p_link in {.02,.05,.1,.2,.4,.8}, input wiring pinned (input_p_link=.1) and
output pools pinned (rebuilt at p=.1, side rng) so ONLY recurrent sparsity
varies; wlr {0.1 ridge, 0.03 statue} x 16 CRN seeds; plus two unpinned
output cells (p=.02,.8 at wlr .1) to measure the readout-pool confound.
Predictions from the laws: (i) controller renormalization — w_mean*p*N
(total recurrent input) spread <25% across the 40x p range, g_final
~p-invariant; (ii) f ~invariant (duty law: rates set by mu/T); (iii) RISKY:
dense (.4-.8) DEGRADES score at the ridge — fluctuation starvation, drive
variance ~1/sqrt(pN) too small to break stillness; (iv) sparse (.02-.05)
RESCUES wlr=0.03 the way sigma=0.1 sensor noise did (internal fluctuation
substitutes for external).

## H54 verdict: conservation confirmed; dense-harm refuted; sparse pre-adaptation

(i) CONFIRMED at ridge: w*p*N = 2.01-2.17 over 40x p (8% spread << 25%),
g_final 0.93-1.02 — total recurrent input is conserved; criticality is
wiring-invariant. At wlr=0.03 renormalization goes UNFINISHED for dense
wiring (w*p*N 7.12 at p=.8, f 0.70 spiking storm) — lawful violation.
(ii) f roughly invariant at ridge (0.111-0.176). (iii) REFUTED: dense is
NOT worse at the ridge (flat 0.42-0.48) — no fluctuation starvation.
(iv) CONFIRMED and then some: p=.02 at wlr=.03 scores 0.566 with 15/16
reliability — best cell of the campaign. Mechanism: sparsity sets the
INITIAL Sigma-w near comfort (g_init ~1.5 vs 7.5), so nearly-frozen
plasticity suffices — sparse wiring is PRE-ADAPTED and avoids churn.
Readout confound measured: unpinned dense output pools (~160-node) average
away L/R asymmetry (0.251 vs 0.443 pinned); sparse readout pools are the
motor symmetry-breakers. (h54_sparsity.json)

## H54b (preregistered): the ridge is matched to initial distance-from-comfort

If wlr* exists to renormalize Sigma-w_in to comfort without churn, optimal
wlr should INCREASE with p. Predictions: at p=.02, wlr=1.0 is terrible
(<0.30 — 4-link columns take huge per-link kicks) and wlr<=0.03 optimal;
at p=.8, score becomes monotone-increasing in wlr over {.03,.1,.3,1.0}.

## H54b verdict: sparse lowers wlr* (confirmed); dense narrows, not raises

p=.02 monotone decreasing in wlr (0.566/0.94 at .03 -> 0.299/0.31 at 1.0):
optimum matched to the small initial distance-from-comfort, as predicted.
p=.8 REFUTES the monotone-increase claim: peak stays at wlr=.1 (0.443) and
the band NARROWS — wlr .3 and 1.0 both collapse to 0/16. Two distinct
dense failures: wlr=1.0 is runaway potentiation (w*p*N -> +235; the
explosion threshold falls with density); wlr=.3 renormalizes fine
(w*p*N 2.41, f 0.23) yet competence dies — a carrier-level loss (same
total spread over 160 links), mechanism open. Summary law: sparsity
WIDENS the viable plasticity window and shifts it down; density narrows
it. Sparse is forgiving, dense is brittle. (h54b_matched.json)

## H55 (preregistered): ballistic interception — the baseball test, run forward

New pursuit stimulus_motion "ballistic": spawn on a random edge aimed
inward (+-60deg), straight flight at 0.15/step through the 15-box, respawn
on exit (~40 crossings per 3600-step run). Catch = dist < 1.5 any step of
a crossing. The constant-control law PREDICTS interception is learnable —
a constant-bearing approach has a static retina in the closing frame,
unlike the refuted aperiodic wander. Predictions: (a) fresh GA (same
protocol as H33/H34) reaches per-crossing catch rate >= 0.5 by gen 10;
(b) champion autopsy shows constant-bearing geometry: during closing
stretches, bearing SD < ~15 deg around a NONZERO lead angle (CBD), not
tail-chase (bearing pinned at 0 with heading matched to target's); (c) the
H34 orbital champion transfers only weakly (its solution is an orbit lock,
not a general chase); (d) a blinded variant (input_weight ~ 0) anchors
chance well below 0.2.

## H55 verdict: interception REFUTED at native speed — below the lock horizon

Blind-body chance anchor 0.240; H34 orbital champion transfers to 0.296;
10-gen GA flat (best ~0.31-0.39 train, champion fresh-seed 0.274+-0.040 ~
chance). Crossings last ~70 steps (51/3600) while re-entrainment takes
90-225 (H31): each encounter ends before a lock can form. Proposed third
clause of the competence law: engagement must OUTLAST the re-entrainment
horizon (periodic motion qualifies because it never ends). (h55.log,
h55_intercept.json)

## H55b (preregistered): stretch the encounter past the horizon

Same champions, ballistic speed 0.15 -> {0.08, 0.04} (crossings ~140 and
~275 steps, spanning the 90-225 horizon), n=7200, 8 seeds, blind anchors
at each speed. Prediction: champion-minus-blind catch gap grows from
~0.03 at 0.15 to >=0.15 at 0.04; if the gap stays ~0 while both rise,
the horizon story is wrong and the failure is acquisition geometry.

## H55b verdict: CONFIRMED — the lock horizon is the third clause

H34 champion's skill gap over blind: +0.027 at ~70-step crossings,
+0.209 at ~140, +0.310 at ~275 (catch 0.535 vs blind 0.225, which FALLS
with slowing — a crossover, not a tide). The gap opens exactly as
crossings outlast the 90-225-step re-entrainment horizon measured
independently in H31. Competence law, third clause: THE ENGAGEMENT MUST
OUTLAST THE RE-LOCK HORIZON. Bonus: the H55 GA champion is numerically
identical to its own blinded control at all speeds — its genome has
input_weight at the range floor (1.15) drowned by recurrent 1.23 and
wheel_base at max: below the horizon, vision has zero marginal fitness
and SELECTION EVOLVES A BLIND SWEEPER. Catching a baseball is homeostatic
only because the ball is watched continuously from launch — one long
engagement. (h55b_horizon.json)

## H56 (preregistered): eligibility traces vs the absorbability barrier

H36 protocol verbatim (waypoint pursuit, H34 champion, r = clipped
distance-closing rate, 12 CRN seeds) with the one-step outer product
replaced by a decaying eligibility trace E <- lambda*E + outer(pre,post)
(*adj), dw = eta*r*E; lambda {0.9, 0.97} x eta {0.003, 0.01, 0.03}, plus
lambda=0/eta=0.01 as the H36 anchor. NEW ARM "frozen-half": homeostatic
learning disabled from step 3600, reward channel continues — the
discriminator. Predictions: if H36 failed from credit DELAY, traces cross
(some cell median near3 >= 0.4, shuffled control <= 0.15) in BOTH modes.
If the barrier is ABSORBABILITY (Law 3 erodes reward-installed structure
as drive error), traces fail under full homeostasis but frozen-half
IMPROVES markedly over its own full twin. Stated prior: absorbability —
traces alone do not cross; frozen-half > full.

## H56 verdict: BOTH accounts refuted — the barrier is task-side, not rule-side

Every cell 0.09-0.14 = eta=0 baseline (0.13) = shuffled control (0.12).
Eligibility traces do not cross (credit delay was not the missing
ingredient), and frozen-half does not help either (so Law-3 absorbability
was not the blocker — my stated prior, wrong). Reward ladder complete:
reward-scaled x, one-step three-factor x, eligibility x, eligibility
without homeostatic erosion x. Synthesis: the waypoint task's random legs
last ~30-50 steps — BELOW the 90-225 re-lock horizon (third clause). The
reward mechanisms were being asked to cross a boundary the competence law
forbids on the task side; no weight-tweak rule can install a policy for
engagements shorter than the lock. (h56_eligibility.json)

## H57 (preregistered): "aperiodic -> nothing" is really "sub-horizon -> nothing"

If the third clause is fundamental and periodicity merely its easiest
satisfier, then SLOWING the waypoint stimulus (legs lengthen past the
horizon) should make fully-aperiodic pursuit WORK. H34 champion + blinded
control on waypoint at speeds {0.15, 0.08, 0.04} (legs ~35 -> ~65 -> ~130
steps), 8 seeds, n=7200. Predictions: (a) champion-minus-blind near3 gap
opens with slowing, reaching >= +0.25 at 0.04; (b) if instead the gap
stays ~0 at all speeds, aperiodicity per se is the barrier and the
competence law keeps its periodicity clause.

## H57 verdict: refuted — slowing does not unlock SUSTAINED aperiodic pursuit

Skill gap (champ - blind, near3) is -0.28 at ALL speeds: blind cruising
scores 0.43-0.46 (a moving body covers the 15-box; the target never
leaves) while the sighted champion sits at 0.14-0.17. Slow 130-step legs
(> the 1D horizon) do not produce following. ALSO EXPOSED: the doc table's
"waypoint -> nothing (0.12)" was the H34 champion's TRANSFER number (no GA
was ever evolved on waypoint - h38 ran ellipse+shuttle only), and 0.43 is
trivially attainable blind — the row must be restated as "entrainment
fails; blind cruising beats vision". (h57_slowway.json)

## H57b (preregistered): interception and following are different competences

Same runs, apples-to-apples metric: per-LEG catch (dist<1.5 during a leg;
legs bounded by stimulus heading kinks) for champ vs blind on waypoint
{0.08, 0.04}. Prediction: per-leg catch gap champ-blind >= +0.20 at 0.04
(matching ballistic interception) while near3 gap stays negative -
i.e. clause (iii) governs transient interception, clauses (i)+(ii) govern
sustained following, and waypoint kinks only break the latter. If the
per-leg catch gap is also ~0 or negative, the ballistic success was
geometry-specific (edge-to-edge crossings), not a general competence.

## H57b verdict: refuted — and the blind anchor was PARKED, not cruising

Per-leg catch gap -0.135 (speed .08) / +0.004 (.04), not the predicted
+0.20. The blind H34 body has agent speed 0.000 — a statue that the
milling waypoint target visits often (near3 0.43, leg-catch 0.37). The
sighted champion MOVES (0.065-0.084) yet catches no more than the statue
and holds far less. Length is NOT the separator: ballistic ~140-step
crossings gap +0.21 vs waypoint ~160-step legs +0.00 (matched length,
opposite outcome). Hypothesis sharpened: ballistic resets occur in
DARKNESS (clean; the ratchet pawl holds), waypoint kinks occur IN VIEW
(flow flips mid-lock and poisons the loop) — vision under in-view-kinked
flow is worse than blindness. (h57b_legcatch.json)

## H57c (preregistered): in-view kinks poison the lock (causal test)

Ballistic at speed 0.04 with per-step kink hazard 1/150 (heading jumps
uniform +-90deg mid-flight, env rng). Within the same runs, classify
crossings: kink-free vs >=1 in-view kink (detected from stimulus heading).
Predictions: (a) champion catch(kink-free) - catch(kinked) >= +0.20;
(b) the blind body shows no such split (|diff| < 0.05); (c) kink-free
crossings match the H55b no-hazard catch (~0.5).

## H57c verdict: DESIGN INVALID — the blind negative control caught it

Kinked crossings catch MORE for both agents (champ 0.596 vs 0.472, blind
0.313 vs 0.124). Prediction (b) failed exactly as a confound flag: hazard
classification selects for LONG crossings (P(kink) grows with duration)
and kinks redirect flights inward — more time in the box, more catch.
Within-run outcome classification under a hazard is selection-biased by
construction; lesson recorded. (h57c_kink.json)

## H57d (preregistered): randomized in-view kink at flight age 100

ballistic_kink_at=100: every crossing surviving to age 100 gets exactly
one +-90deg kink (env rng); same seeds 41-48, speed 0.04, vs the H55b
no-kink runs (same CRN). Intent-to-treat on crossings reaching age 100.
Predictions: (a) champion catch on kinked crossings falls >= 0.15 below
its H55b no-kink level (0.535 -> <= 0.38) — in-view kinks break locks;
(b) blind changes little (< 0.08 either way); (c) if champion holds
~0.53, kinks are NOT the waypoint poison and the failure is loitering
geometry instead.

## H57d verdict: kinks are INNOCENT — prediction (c) is the outcome

Randomized kink at flight age 100, intent-to-treat on 422 crossings/arm:
champion kink effect +0.007 (placebo -0.014); blind +0.041 (flights
redirected inward last slightly longer — the H57c confound's direction,
confirming its diagnosis). The lock re-forms within a long engagement even
through a +-90 deg in-view redirect. With aperiodicity (H57), length
(H55b/H57b matched), and kinks (H57d) all cleared, the live hypothesis
for the waypoint failure is TRAVERSAL vs LOITERING geometry: ballistic
targets approach from far with a monotone intensity ramp the flow-ratchet
climbs (champ pre-100 catch 0.332, blind 0.000 — blind's catches are all
late target-comes-to-you luck); loitering targets flutter at mid-range
with no ramp. Open, recorded. (h57d_kinkfixed.json)
