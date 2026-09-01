# How homeostatic reservoirs work: laws, phases, and the entrainment mechanism

*Findings of the design-space campaign of 2026-08-31/09-01 (scripts/lab/;
50 preregistered hypotheses H1–H50 in
[scripts/lab/LEDGER.md](../scripts/lab/LEDGER.md), ~20,000 local runs plus
three cluster batches; four tasks — tracking, Pong, wall avoidance, and the
new pursuit task — plus two-to-four-agent ecologies). Method: every
hypothesis preregistered before its experiment; common-random-number wiring
seeds; all scripts deterministic (two bit-identical cross-machine
reproductions); every number below traceable to a JSON in
`scripts/out/lab/`. Interactive verification: six `/lab` pages in the
visualizer. Chance score is 0.25; "score" is within-45° occupancy over
reversal segments 6–10 unless noted.*

## The three exact laws

The model's update rules imply, and experiment confirms, three laws that carry
all of the mechanics:

**Law 1 — comfort split.** A node with stationary mean drive μ is
silent-comfortable (T → μ/leak, E → 0, no spikes) iff μ ≥ leak·T_floor, and
dead-uncomfortable (T pinned at the floor, E < 0 forever) below. Verified
674/674 single-node cells (k2). The cold-start spike boundary is
μ > ρ·leak·T₀ to first order (86%), with the residue explained by the *race*
between x rising (~1/leak) and the threshold chasing (target_lr).

**Law 2 — duty law.** Where spiking persists, the spike fraction satisfies
f = (μ/T − leak)/ρ with **zero free parameters** (μ = total drive, T = current
target, ρ = threshold_ratio). Median residual 0.0000 over 419 single-node
cells; corr +0.999, median |resid| 0.0025 per (node, 120-step window) *inside
the full recurrent network in its most violent (churn) regime* (k1b).
Activity level is fully slaved to (drive, target) — everything dynamic lives
in how W moves μ and how T moves.

**Law 3 — the weight channel is a gated integral controller.** Per step, the
total incoming recurrent weight of node n changes by exactly −weight_lr·E_n
when ≥1 of its afferents spiked at t−1, else not at all. Verified to ≤1.2e-14
in closed loop across three regimes (b7). So eq. 5, for all its per-synapse
bookkeeping, is a pure drive servo: **∫gated −wlr·E dt on each node's input
current**, alongside the explicit target integrator ∫tlr·E dt (floored).

The reduced model is therefore *derived*, not fitted: N coupled integral
controllers — a fast drive servo (wlr) and a slow set-point servo (tlr) —
sharing one spike-gating fabric, with f slaved by Law 2. First-order closure:
the drive servo's absorption corner sits at period P* ≈ 2π/(wlr·f̄); measured
240–960 steps at (wlr=0.1, f̄≈0.15), predicted ≈420 (b3).

## The phase geography

Single nodes end in one of four states (k2): dead-floor, silent-comfortable,
periodic-spiking (T marginal), and frozen mode-locked cycles (cycle-mean E ≈ 0
with large instantaneous |E| — *generic* across all leaks tested, not a
leak-0.25 curiosity; median |E| 0.43). Bistability exists (47/625 cells): a
hot start lets T inflate into silence while a cold start sustains marginal
spiking — **target inflation is the silencing mechanism**, and the "statue"
cheat is this basin reached by overdrive.

At network level the same coordinates organize *activity* but *not
performance*:

- g_init = N·p·w̄/(ρ·T₀) organizes prop_spiked (rho +0.38) and the extremes —
  dead at low drive, saturated at high g — but has **zero correlation with
  score** (+0.02 over the 241-config historic sweep; k0). The paper baseline
  starts at g_init = 7.5, deep in the supercritical regime.
- Aliveness follows a drive threshold: a single split on μ_in/(leak·ρ)
  separates the sweep's 91 dead configs from the living at 85% (k0). (Using
  the exact retinal gain S₀ = 3.76 in place of the proxy *worsens* the split
  to 73% — closed-loop drive depends on behavior, so no static drive formula
  can be exact. Honest limit, on the record.)
- **Score lives on a different axis the historic sweep never varied:
  weight_lr.** wlr = 0 is the statue (f ≈ 1.0, score 0.250 — comfortable and
  dead-to-the-world, because targets can only *relabel* drive as comfortable;
  they cannot reduce it). wlr ≥ 3 explodes numerically. In between:

| wlr (tlr=0.01) | 0.0 | 0.01 | 0.03 | 0.1 | 0.3 | 1.0 (paper) | 3.0 |
|---|---|---|---|---|---|---|---|
| score, 12 seeds | .250 | .250 | .380 | **.633** | .414 | .325 | .250 |
| seeds ≥ 0.35 | 0/12 | 0/12 | 6/12 | **12/12** | 6/12 | 4/12 | 0/12 |

  A single change — weight_lr 1.0 → 0.1, i.e. the 2021 paper's value instead
  of the 2024 code's full-error convention — roughly doubles tracking and
  removes the wiring lottery (48-seed check: 0.467±0.200 vs 0.386±0.138).

- The (leak × wlr) plane has a **diagonal ridge**: score peaks where the
  plasticity rate matches dissipation. Confirmed at cluster scale (4800 runs,
  48 CRN seeds/cell, 25% held-out checkerboard;
  [fig_ridge_fine](../scripts/out/lab/fig_ridge_fine.png)): **wlr* =
  1.04·leak^1.41** (fit on non-held cells; 7/10 held-out rows within factor
  2, the misses being flat-crest rows where the argmax is ill-defined).
  Crest height ≈ 0.46–0.55 along the whole diagonal, broad at low leak
  (near-peak across a decade of wlr) and narrow at high leak; below it, a
  dead-statue triangle whose escape threshold in wlr rises with leak. This
  retroactively explains the historic sweep winners: both chose leak ≈
  0.56–0.57 under forced wlr = 1.0 — they were the wlr=1.0 slice of this
  same ridge. (48-seed replication of the headline: wlr 0.1 → 0.467–0.525
  vs default 0.386; and the cluster's default cell reproduced the local run
  EXACTLY, 0.386 on the same 48 seeds — cross-machine determinism.)

**Why the ridge: signal-to-noise.** Driving the retina with a sinusoidal slip
and reconstructing stimulus position from spikes (b3): reconstruction gain at
wlr=0.1 is 0.22–0.23 across P=30–480 — **3–10× every other regime**. Below
the ridge (wlr=0.03) saturation destroys selectivity (f≈1 → the readout sees
everything, hence nothing); above it (wlr=1.0) the servo's own churn buries
the stimulus (gain ≈ 0.02). The matched-timescale ridge is where plasticity
whitens saturation away without generating self-noise. Two corollaries,
both measured: activity under a *stationary* retina at wlr=1.0 stays at
f ≈ 0.39 for every stimulus schedule (the canonical prop_spiked ≈ 0.34 is
churn-set, not stimulus-set; k1), yet **total darkness silences every
regime** (a2) — the churn is input-*powered* but input-statistics-*independent*.

## How tracking actually works: entrainment, not servo control

The ceiling: a hand-written P-controller on the same 62 sensors scores
**0.999** (b1); the 1-step flow-greedy controller scores 0.898 — in this
embodiment, maximizing input flow ≈ centering, which is the geometric basis
of the input-flow thesis. The best homeostatic config known (w1′) reaches
0.82 — 82% of ceiling; paper defaults reach 39%.

But the homeostatic agent is not a worse P-controller — it is a different
kind of controller altogether (b6):

- It **follows**: per-segment (agent net rotation)/(stimulus net rotation) =
  0.84 at w1′ (rising to 0.98 late), through every reversal; ridge25 *learns*
  to follow within the session (0.28 → 0.89 by segment).
- It has **no position feedback**: the signed in-view response
  mean(sign(err)·dH) is +0.0006 at w1′ and −0.120 (anti-corrective!) at
  ridge25.
- It is **swap-immune**: inverting the effectors mid-run (left spikes now
  turn right) costs almost nothing — recovery ratios 0.93–1.03 across
  regimes, dip barely visible at 720-step resolution (b4). A wired reflex
  cannot do this; and indeed the wiring's static reflex kernel does *not*
  predict per-seed success (K4: sign agreement 0.44 at defaults).
- The **ratchet pawl**: out of view, the agent goes still (RMS dH 0.17 vs
  1.20 in view at wlr=0.1; dark-stall fraction 0.93 at ridge25), and the
  *periodic* world brings the stimulus back to it. Periodicity is
  load-bearing, as the earlier toy-world impossibility result predicted.
- Re-entrainment after reversal takes ~30 steps at w1′ (to 0.94 velocity
  match), ~180–270 elsewhere, with a *negative momentum lobe* at high wlr —
  the old bias persists briefly (behavioral entrenchment; b6b).

Mechanism, assembled: retinal slip passes the plasticity bandpass (Law 3
absorbs anything slower than P*) and drives spiking; spikes drive turning;
the *turn bias* is the slow learned variable, dragged along by the stimulus.
Tracking is **velocity entrainment with a flow ratchet** — the 2024 paper's
phrase "Gibsonian resonance" is mechanically accurate, and the classical
reading of the agent as an error-correcting servo is wrong.

**Where the bias lives** (b8): decomposing the pools' duty difference (which
*is* the turn command, corr +1.000) into input vs recurrent components: at
w1′ the bias is carried by the **recurrent drive** (+0.37 with direction;
input pathway nil) — genuinely W-stored, written by the servo and rewritten
at each reversal. At ridge25 the carrier is mixed (input +0.53, recurrent
+0.39); at defaults, churn-carried. And the lag-servo alternative is dead:
heading error keeps a constant +7–9° offset in *both* directions (corr with
direction ≈ 0) — there is no direction-flipping lag anywhere tested.

**The bandpass is in activity, not information** (h22): at slip period P=8
(peak speed ~16°/step) activity collapses to f = 0.003, yet reconstruction
gain *rises* to 0.505 — fast slip sparsifies spiking into hyper-selective
events. Absorption kills slow signals' activity and information together;
fast signals lose activity but keep information.

## The third embodiment: wall avoidance and the sign flip

Case study 3 (implemented and replicated 2026-09-01; `src/homeostasis/wall.py`,
`scripts/lab/wall_rep.py`; all five of the paper's qualitative anchors pass —
see LEDGER H27a) completes the input-flow thesis:

| task | sensors read | rho(input flow, success) |
|---|---|---|
| tracking | stimulus presence (egocentric) | **+0.77** |
| Pong | ball bearing (egocentric) | **+0.95** |
| wall avoidance | wall **proximity** | **−0.86** (alive configs) |

The internal quantity is identical; **the body decides the sign of its
coupling to success** (H27b, preregistered). On wall avoidance the positive
correlate is flow *stability* (+0.88). Two structural facts sharpen the
picture: (1) this is the family's only task **solvable by death** — 40/91
random configs never move and never collide, so only living configs
discriminate theories; (2) the settled solution is a small constant-direction
circle in flat sensory territory, spun by the churn regime (f ≈ 0.32 with
near-stationary input).

**How stabilization actually works here** (H28 series — the preregistered
event-ratchet story was *refuted* and corrected): with time-local baselines,
individual collisions are nearly invisible to the network (|dW| at hits =
1.04× baseline; the saturated learning-off statue shows literally zero |E|
response to hits — saturation is deafness). Stabilization is the convergence
of the global early churn transient, and the surviving orbit is
**stability-selected**: orbits persist where input variation along them fits
inside the absorbable band. Causal probe (H28c): teleporting a settled agent
next to a wall doubles |E| and the agent drifts back to flat territory
(median 6 hits); teleporting to a flat point changes nothing (median 0).

**The unification across all three case studies**: behavior settles where —
and only where — the sensory consequences of action fall inside the weight
servo's absorbable band. Tracking's stimulus moves autonomously, so no
stationary solution exists and the loop *chases* absorbability (velocity
entrainment). Pong's rally is the absorbable regime and resets are not. Wall
avoidance has genuinely stationary solutions, so the loop parks in one.
"Behavior as externalized absorption" — with the H28 caveat that the loop
does not need to *sense* failure events for this; failure-free regions are
simply the only places the dynamics can settle.

## The fourth task: pursuit, and the family's competence boundary

`src/homeostasis/pursuit.py` (new, beyond the paper — the mentors' ladder
rung 3/4): tracking's bearing retina on the wall-avoidance body, moving
stimulus in the walled arena. A P-controller solves it near-perfectly (dist
0.81, 100% within 3 units). The homeostatic family **fails at every hand
configuration tried** (~40 arms; LEDGER H32), and the five-round failure
taxonomy is the finding:

1. **Law 1 in space**: the intensity field has an *absorbing dead basin*
   (wander far → starve → still → dead) that rotation-only tracking
   geometrically could not have.
2. Even distance-blind sensing dies: **wall-pinned outward-facing poses**
   keep the stimulus behind the ±92° view forever — walls + translation
   manufacture absorbing darkness; tracking's periodicity rescue needed a
   body that cannot leave or permanently look away.
3. A 360° retina restores aliveness — and pursuit *still* fails
   (orientation at chance): the deficit is not survival.
4. Motor-grain and velocity-floor hypotheses both refuted cleanly.

Diagnosis: tracking's success rested on an effector-to-slip map that is
1-DOF, sign-stable, and stillness-capable; the 2D body breaks all three
(parallax adds a distance-dependent coupling that diverges on approach,
churn forces a cruise so the ratchet's waiting state is unreachable, and
dark poses exist).

**Selection changes the answer — and reveals what the solution IS** (H33,
H34): genome-level GA reaches near3 = 0.71 in 7 generations; *joint*
(genome, wiring-seed) evolution finds a **perfect pursuer in 3 generations**
(near3 1.00, dist 0.80 — matching the P-controller), stable for 14,400 steps.
The autopsy
([fig_perfect_pursuer](../scripts/out/lab/fig_perfect_pursuer.png)) is the
campaign's most beautiful exhibit: the agent runs a **concentric circle
phase-locked just inside the stimulus's orbit** (revolution rate +1.9°/step
= the target's own angular rate; bearing pinned at 34°; f = 0.08). It does
not chase — it found the co-moving frame in which its retina is *static*:
the wall task's stability-selected circle transported into the target's
moving frame. Velocity entrainment (tracking), orbit selection (wall), and
pursuit are one mechanism.

**And the boundary holds against everything tried:**
- The perfect pursuer scores **0.12 on unpredictable waypoint motion** and
  **0.02 on a perfectly predictable straight-line shuttle** — it is a
  resonator tuned to one closed manifold (a shuttle's co-moving frame has
  heading-flip discontinuities absorption cannot smooth), not a follower.
- Reward-*scaled* homeostasis does nothing (H35: 0.14 vs 0.13 control —
  scaling the drive servo changes *when* it learns, never *what*).
- One-step reward-*directed* three-factor credit does nothing either (H36:
  best 0.14; time-shuffled-reward control identical). Caveats on record
  (N=64, scalar reward, no eligibility traces).
- The wiring lottery persists through genome selection (fresh-seed medians
  0.00–0.11, with rare perfect jackpots).

**The constant-control law** (H38–H38c) completes the picture. Evolving
against other target shapes: a 4:1 ellipse and a straight-line shuttle both
yield **toll-booths** — champions that *park* at a favorable point (the
shuttle's at speed exactly 0.000) and let the passing target pay them near3
credit — while a *near-circular* ellipse (curvature ratio 1.27) evolves a
genuine velocity-matched follower (near3 1.00, dist 0.58, agent speed 0.159
≈ the target's 0.15). The full hierarchy:

| target motion | curvature ratio | evolved solution |
|---|---|---|
| circle | 1.0 | follower (dist 0.80) |
| near-circle | 1.27 | follower (dist 0.58, near3 1.00) |
| ellipse | 1.6 | follower, degraded (dist 2.39, 0.96) |
| ellipse | 2.5 | toll-booth, parked (0.43) |
| ellipse | 4.0 | toll-booth (0.69) |
| shuttle | discontinuous | toll-booth, parked (0.54) |
| waypoint | aperiodic | nothing (0.12) |

**Entrainment reaches exactly those motions whose co-moving frame can be
held with ~constant control** (constant curvature ⇒ constant turn rate ⇒
the control signal is itself stationary and absorbable); past that
tolerance (boundary in ratio (1.6, 2.5) — about 2x control modulation, with
a soft edge: follower quality degrades smoothly toward it), selection
substitutes stillness
at favorable points; without periodicity, nothing. This is the quantitative
form of the mentors' rung 3, and of the "prediction vs Gibsonian loops"
question: within the law's reach the baseball intuition is exactly right —
no prediction is needed; beyond it, the family cannot go, and neither
reward-scaled (H35) nor one-step reward-directed (H36) plasticity crosses,
leaving *minimal structural additions* (eligibility traces? prediction?) as
the live question.

**The lottery's home, and the mechanism hierarchy** (H40–H42): the wiring
lottery lives in the frozen adjacency — weight-space operations cannot touch
it (flow-seeking annealing fires 21 times/run and locks 0/16, because
shuffles permute values on fixed structure). A minimal *structural
homeostasis* extension (persistently starved nodes grow a random afferent,
persistently overdriven nodes prune their weakest; core untouched) becomes
the first within-life mechanism to cross the ceiling: 3/16 locks vs 0/16.
The measured hierarchy — weight ops 0/16, local structural plasticity 3/16,
wiring-level selection reliable-on-its-pair — hands the mentors'
connectivity/DNA agenda both an empirical mandate and a working first local
rule. And the rule's cost was measured too (H43/H44): left on for life it
*destroys* solved tasks (tracking working-seed fraction 0.88 → 0.00 —
perpetual rewiring breaks the entrained balance), while a **developmental
window** — rewiring allowed for the first 3,600 steps, frozen after —
keeps the exploration benefit at zero cost to solved behavior. Minimally
extended, the family rediscovers why development exists.

## Scaling: the phenomenology is N-invariant

The cluster N-line (288 runs; in-degree pinned at 20, input wiring pinned;
N ∈ {200, 500, 1000, 2000} × wlr {0.05, 0.1, 0.2}, 24 CRN seeds/cell): the
ridge does not move, scores show no monotone trend (N=2000 is nominally the
best cell, 0.560 with 83% of seeds working), and the internal coordinates
are constant across a 10× size range (f 0.14–0.16, |E| 0.53–0.59, flow
2.2–2.5). The duty law holds at N=2000 (corr 0.988). **Size per se is
inert; the organizing variables of this model family are degrees and
rates, not counts** — the direct answer to "scaling up network sizes."

## The two-body problem, and the band clause

First multi-agent data (H46–H47, `scripts/lab/h46_mutual.py`,
`h47_pacemaker.py`):

- **Two mutually-tracking homeostats collapse to collective stillness** in
  the absorption regime — instantly, wherever they start in view of each
  other, because a static partner is already a stationary stimulus (and an
  out-of-view start freezes them forever: mutual darkness is absorbing).
  Churn pairs chase weakly without locking; a mixed pair shows the churner
  *animating* the absorber (f 0.000 → 0.038) but not leading it — churn
  motion is aperiodic, exactly the class the competence law forbids.
- **The pacemaker test** closes the loop on the family's own artifacts: a
  recorded wall-circler orbit is unfollowable at its native 6.4°/step (best
  evolved agent: center-parking, 0.50) yet **perfectly followed at 2.1°/step**
  (near3 1.00 by generation 5). The competence law is therefore two-clause:
  *(i) the target's co-moving frame must require ~constant control (shape,
  tolerance ≈ 2× curvature modulation), and (ii) the frame's angular rate
  must sit inside the entrainment band (speed).*
- Ecology corollary for the multi-agent program: sustained collective
  entrainment requires a **slow periodic pacemaker** — interaction alone
  yields stillness or unfollowable churn.
- **And the ecology was then built** (H48,
  [fig_live_chain](../scripts/out/lab/fig_live_chain.png)): the paper-scale
  family cannot pace itself (absorption wall-agents are dead statues; churn
  circlers turn ≥5.6°/step, above the band), but a wall circler with
  wheel_base 2.5 in a 30-unit arena circles cleanly at 1.9°/step — and a
  warm-start-evolved follower locks onto it live: **near4 1.00, dist 3.77,
  stable through 10,800 steps. A blind pacemaker, circling because of walls
  it avoids, entrains a follower that senses it — the first sustained
  two-agent homeostatic system.** Requirements measured: band-compatible
  pacemaker (morphology + arena rescaling), one-way coupling, and selection
  warm-started from prior follower competence (cold-start GAs find
  toll-booths).
- **Entrainment propagates — to depth four** (H49–H50,
  [fig_chain4](../scripts/out/lab/fig_chain4.png)): successive agents lock
  onto the previous *live* follower — a blind pacemaker plus three
  phase-locked followers, four concentric rings, all stable through 10,800
  steps. The chain's law is a **threshold, not a ramp**: orbit jitter stays
  negligible for two links (sd 0.048 → 0.055), jumps 6× at the third
  (0.38) as the rings contract toward a geometric floor (radii 7.8 → 6.5 →
  5.6 → 2.3), and a fifth agent cannot evolve on the noisier, tighter
  target. Link evolvability is start-basin sensitive (one false start
  caught and fixed on the record) — IC-dependence at the collective level.

## What the two homeostatic channels actually do

- **The weight servo is the universal necessary channel**: freeze-W-only is
  dead (0.250) in every regime tested (k3, a4). It is the only channel that
  can change effective drive; and behavior needs it live — freezing all
  learning mid-run degrades score with huge variance (entrenchment), while
  freeze-from-init is plain dead.
- **The target servo is vestigial-to-harmful for tracking at defaults**:
  tlr=0 matches tlr=0.01 at the ridge and defaults (b2); freeze-T-only
  *improves* defaults (0.404 vs 0.325; k3). The channels **compete for the
  same error**: fast targets absorb E before the weights can act (at wlr=0.1,
  tlr=0.1 the gain never erodes; f stays 0.98 and score collapses to 0.267).
  At w1′ target adaptation genuinely adds ~0.1–0.2 — and NOT as static
  *homogeneous* gain normalization (any raised constant T is death by Law 2;
  b5). The resolution (h21): freezing T at its **evolved heterogeneous**
  per-node values costs nothing (0.789 vs full 0.791; homogeneous T=1 costs
  ~0.09). **Targets are a calibration channel** — they build a static
  per-node dynamic-range profile and can then stop; weights are the
  computation channel and must keep running.
- **The learned structure is not the computation — the ongoing process is**:
  shuffling all learned weights mid-run (learning on) recovers to or above
  the unshuffled run everywhere (defaults 0.394 vs 0.325; w1′ 0.762 vs
  0.850). Self-entrenchment is real and mildly maladaptive at defaults.
- **Reflex vs medium is a *place*, not a property**: at paper defaults the
  recurrent medium is a net liability (lesion scores 0.478 > full 0.325);
  at w1′ lesion collapses (0.250±0.433, bimodal). The prior reflex/medium
  dichotomy is the design space talking, not the model.

## Refuted along the way (all preregistered)

1. "Homeostasis silences stationary input" — only the weight channel
   regulates; targets relabel. The statue (f=1, E≈0) is the target channel's
   own fixed point.
2. "Spiking tracks stimulus non-stationarity" at 2024 defaults — activity is
   churn-set (f ≈ 0.3–0.4 for every schedule).
3. "The network self-excites" — darkness silences everything; churn is
   input-powered.
4. "The seed lottery is the wired reflex kernel" — no (0.44 sign agreement);
   the lottery's microfoundation is still open.
5. "g_init organizes performance" — it organizes activity only.
6. "Re-entrainment time ∝ 1/wlr" — no; higher wlr raises the entrainment
   ceiling but slows re-locking (noise), another face of the SNR trade.
7. "Targets at w1′ are static gain normalization" — any raised constant T is
   death; the contribution is genuinely dynamic.

## Open questions, ranked

(Former #1 — dynamic targets — answered by h21: calibration channel. Former
#3 — the turn-bias variable — answered by b8: W-stored at w1′, carrier
regional. Former upper-edge question answered by h22: no averaging corner;
the fast edge is sparse-and-informative.)

1. **What sets the wander floor.** The "seed lottery" partially dissolves:
   late segment scores within a run are mutually uncorrelated (+0.03) — much
   of the seed variance is sampling noise of a *wandering* process, and w1′'s
   100%-seed reliability is a wander floor above threshold. The fixed seed
   effect that remains (early↔late +0.45 at the ridge) is modest. The right
   question is what design choices raise the floor of the wander — and it is
   now known to be **non-additive** (h24/h25): necessity analysis names
   {gain, leak, N} as w1′'s load-bearing trio, yet no subset transplants
   (grafts onto ridge25 all hurt; the trio at N=200 reaches only 0.534 vs
   w1′'s 0.84), and the ridge law itself failed to extrapolate to w1′'s
   corner (wlr=1.0 ≥ 0.65 there). The best configs are jointly-tuned wholes;
   the floor lives in ≥3-way interactions.
2. **Formalize information-per-spike**: the h22 sparse-informative fast edge
   suggests an efficiency curve (bits/spike vs slip frequency) worth one
   clean experiment; also the secondary resonance near sensor-spacing/speed.
3. **Why does the carrier regionalize** — pure W-storage at w1′ vs mixed at
   ridge25? (Candidate: leak sets how long input context persists in x.)
4. **Why the exact retinal drive formula underperforms the proxy** for the
   dead boundary — a closed-loop selection effect worth one figure.
5. **What breaks entrainment chains** — H50 measured the threshold (jitter
   6× at link 3, rings contracting to a floor) but not its mechanism; and
   the max depth vs pacemaker radius curve is one clean sweep away.
6. **Eligibility traces / multi-step credit** — the canonical untested rung
   between scalar neuromodulation (refuted, H35–H36) and full prediction
   machinery, at the motion-generality boundary.
7. ~~Cluster-scale confirmations~~ — landed (H14, H45). The Slurm lane runs
   a 6k-run batch in ~4 minutes on mit_quicktest; batch chunks must respect
   the lane's wall-time.

## Instruments

`scripts/lab/`: `common.py` (closed-loop arms incl. lesion / seven freeze
variants / shuffle / effector-swap, open-loop scripted retina with sine and
per-node law recording, observables), the k0–k4 kill tests, act2 batches,
b6–b8, h21–h50 experiment scripts, cluster runner + batch generators, and
`monday_update.md` (a pasteable summary). Six verified viewers: `/lab`
(single-node explorer, duty law live), `/lab/phase` (the real planes,
per-seed inspection), `/lab/traj` (entrainment trajectories, bit-exact
against archived runs, effector swap), `/lab/wall` (arena + perturbation +
the evolved edge-holder), `/lab/pursuit` (the perfect pursuer and its
failure mode), `/lab/ecology` (the live two-agent chain, self-validating).
Verification culture: preregistration ledger, CRN seeds, exact-law checks
to machine precision, honest negatives — and eight of my own design bugs
caught and recorded rather than papered over.
