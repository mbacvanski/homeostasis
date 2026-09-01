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
