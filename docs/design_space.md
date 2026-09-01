# How homeostatic reservoirs work: laws, phases, and the entrainment mechanism

*Findings of the design-space campaign of 2026-08-31/09-01 (scripts/lab/;
94 preregistered hypotheses H1–H94 in
[scripts/lab/LEDGER.md](../scripts/lab/LEDGER.md), ~26,000 local runs plus
six cluster batches; four tasks — tracking, Pong, wall avoidance, and the
new pursuit task — plus two-to-four-agent ecologies). Method: every
hypothesis preregistered before its experiment; common-random-number wiring
seeds; all scripts deterministic (three bit-identical cross-machine
reproductions); every number below traceable to a JSON in
`scripts/out/lab/`. Interactive verification: seven `/lab` pages in the
visualizer (incl. `/lab/repair`, the noise slider, and ballistic pursuit,
each bit-exact against its campaign JSON). Chance score is 0.25; "score" is within-45° occupancy over
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

**The laws compose into a simulation-free predictor — and expose a cliff**
(H59, `h59_dial.py`). Freeze both channels and the network's cold-start
fate is computable from the wiring file alone: iterate the per-node map
f_n ← clip((drive_n/T_n − leak)/ρ, 0, 1) with the Law-1 reachability gate
(a node only ignites if drive ≥ leak·ρ·T, since x plateaus at
drive/leak) from f = 0. Result: **r = 1.0000, median error 0.0%** over
72 frozen runs spanning dead and saturated cells — including a
bistable-basin case the ungated duty law gets wrong. But inverting the
model to *design* an interior rate fails constructively: predicted f is
discontinuous (≈0 → ≈0.6 at a knob cliff; simulated networks snap to
1.000 on the live side). **Homogeneous frozen networks have no stable
interior rates at all.** Every competent behavior in this campaign lives
at f ≈ 0.1–0.3 — a state that exists only as the homeostat's dynamically
maintained equilibrium; and since freezing *evolved heterogeneous*
targets costs nothing (act-2 autopsies), adaptation both holds the
network on the cliff face and manufactures the per-node target profile
that turns interior rates into a legal static design. This is
self-organized marginality made precise.

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

- The (leak × wlr) plane has a **diagonal ridge**: score peaks along a
  rising diagonal in (leak, wlr). Confirmed at cluster scale (4800 runs,
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
the stimulus (gain ≈ 0.02). The ridge is where plasticity
whitens saturation away without generating self-noise — an SNR optimum
as *measured* characterization. (Its once-favored "matched-timescale"
*mechanism* — the absorption corner P* = 2π/(wlr·f̄) meeting a
behavioral period — was later refuted at plane scale, H94: crest wlr
rises *with* duty rate, the opposite sign, so the b3-era single-point
match was coincidence. The ridge exponent remains empirical and
underived — the sharpest theory question the campaign leaves open.) Two corollaries,
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
| waypoint | aperiodic, loitering | entrainment fails — vision *harms* (see below) |

**Entrainment reaches exactly those motions whose co-moving frame can be
held with ~constant control** (constant curvature ⇒ constant turn rate ⇒
the control signal is itself stationary and absorbable); past that
tolerance (boundary in ratio (1.6, 2.5) — about 2x control modulation, with
a soft edge: follower quality degrades smoothly toward it), selection
substitutes stillness
at favorable points; without periodicity, nothing. This is the quantitative
form of the mentors' rung 3, and of the "prediction vs Gibsonian loops"
question: within the law's reach the baseball intuition is exactly right —
no prediction is needed; beyond it, the family cannot go, and no scalar-reward plasticity crosses:
reward-scaled (H35), one-step three-factor (H36), eligibility traces at
two decay constants (H56), and — decisively — traces with the
homeostatic channel *switched off* mid-run so nothing could erode what
reward installs (H56's frozen-half arm) all sit exactly at the η=0
baseline and its shuffled-reward control (0.09–0.14). Neither credit
delay nor Law-3 absorbability was the barrier; the barrier is
task-side — waypoint legs are sub-horizon engagements of loitering
geometry (see the H57 series below), so there is no gradient any
weight-tweak rule could climb. This retro-explains H29's "reward =
selector, not shaper": selection evaluates whole trajectories and can
find repertoire members; within-life scalar broadcast has nothing to
work with here.

**The third clause: engagements must outlast the lock** (H55/H55b,
`h55_intercept.py`, ballistic stimulus mode). The baseball test run
forward: targets fly straight through the arena (spawn on an edge, exit
the other side; catch = closing within 1.5). A constant-bearing approach
is constant-control, so the law seems to promise interception — yet a
10-generation GA stays flat at the blind-body chance level (champion
0.274 ± 0.040 vs blind 0.240). The timescales say why: crossings last
~70 steps while re-entrainment takes 90–225 (the H31 body-clock
measurement) — each encounter ends before a lock can form. Stretch the
encounter and skill appears exactly on schedule: the orbital H34
champion's gap over blind grows +0.03 → +0.21 → **+0.31** as crossings
lengthen to ~140 and ~275 steps (catch 0.535 while blind *falls* to
0.225 — a crossover, not a tide). So the competence law has three
clauses: (i) the co-moving frame must be holdable with ~constant control,
(ii) the frame rate must sit in the entrainment band, and (iii) **the
engagement must outlast the re-lock horizon** — periodic motion
qualifies by never ending. The evolutionary footnote writes itself: the
GA champion bred at native speed turns out to be *numerically identical
to its own blinded control* — its genome drove input_weight to the range
floor and the wheel base to maximum. Below the horizon, vision has zero
marginal fitness, and selection evolves a blind sweeper. Catching a fly
ball is homeostatic only because the fielder watches the ball
continuously from launch: one long engagement, not forty short ones.

The H57 series then stress-tested clause (iii) and corrected the table's
old aperiodic row (whose "nothing (0.12)" was in fact the orbital
champion's *transfer* score — no GA was ever evolved on waypoint). Three
successive preregistered predictions failed, each failure narrowing the
truth: slowing waypoint legs past the horizon does *not* produce
following (H57); it's not the sustained-vs-transient metric (H57b:
per-leg catch gap ≈ 0, and the "blind cruiser" turned out to be a
*parked statue* that the loitering target visits — near3 0.43 vs the
sighted champion's 0.17, so **vision actively harms on loitering
motion**); and it's not the kinks (H57d, randomized mid-flight ±90°
kinks at n=422 crossings: champion effect +0.007, placebo clean — the
lock re-forms through an in-view redirect; the hazard-classified first
attempt H57c was caught as duration-confounded by its own preregistered
blind control). What separates success from failure at *matched
engagement length* (~140–160 steps: ballistic gap +0.21, waypoint
+0.00) is **traversal versus loitering geometry**: a through-flying
target approaches from far with a monotone intensity ramp the
flow-ratchet climbs (the champion catches 33% of crossings inside 100
steps; the blind control, 0.000 — its catches are all
target-comes-to-you luck), while a loitering target flutters at
mid-range and offers no ramp. And the harm mechanism is now measured
(H61): entrained turning is not an internal motor program but a
closed-loop product of *stable bearing geometry* — the true champion
turns 6.14°/step on orbit with its bearing pinned, and 0.00°/step
(median, two-sided) under loitering flow, where fluttering bearings
cancel and only mean forward throttle survives. It cruises
near-straight, ends wall-pinned (34–48% wall time), far from the
interior the target loiters in, while a parked blind statue wins
passively. **Flow is a throttle; stable geometry is the steering** —
vision under rampless flow unparks the agent without aiming it. A
footnote worth keeping: slow-ballistic interception was measured on
fresh wirings and is wiring-robust, while the orbit lock is a wiring
jackpot — the lottery differentiates by task.

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

## Robustness: noise, damage, and the sparse advantage

Three late-night studies (H51–H54) probed what the laws imply when the
world or the network is degraded. All three returned the same shape of
answer: **the comfort machinery always restores physiology; whether
behavior benefits depends on where you start relative to comfort.**

**Sensor noise is a resource for under-plastic networks** (H51,
`h51_noise.py`). Moderate uniform sensor noise (σ=0.1 on activations)
*rescues* the statue regime — wlr=0.03 goes 0.367 → 0.533 with
reliability 0.44 → 0.81 — and even lifts wlr=1.0, flattening the ridge;
strong noise (σ=0.2) hurts everywhere and collapses the absorption
regime's activity (f 0.17 → 0.04). The mechanism is *not* desaturation of
the reservoir: in open loop, σ=0.1 changes nothing (recon gain stays
0.001) while σ=0.2 does desaturate (gain 0.196). The rescue is
loop-level (H51c): at wlr=0.03 the failing agent spends 35% of its steps
in darkness — zero input flow, hence zero learning signal, an absorbing
sensory dead state. σ=0.1 abolishes darkness (input duty 0.65 → 1.00)
while barely touching the motor (eff-diff 0.049 → 0.056), keeping the
flow ratchet engaged. σ=0.2 keeps duty at 1.00 but washes out stimulus
*contrast* (dir-agree 0.297 → 0.227). The optimum sits exactly where
darkness is abolished and contrast survives.

**Node death: physiology always repairs; behavior only needs it for big
wounds** (H53, `h53_selfrepair.py`). Killing 10–50% of nodes mid-run
(full adjacency removal, caches rebuilt so plasticity cannot regrow;
silent nodes stay in readout denominators) at the ridge: spike rate
recovers under learning (k=0.3: 0.174 → 0.115 → 0.136 vs frozen
0.046 → 0.057 flat; k=0.5 doubles back 0.042 → 0.096) via **Law 3
acting as deafferentation-induced synaptic scaling** — the kill halves
Σw_in, E goes persistently negative, and the integral controller
re-inflates surviving weights (w̄ 0.106 → 0.210 at k=0.5, a near-exact 2×
for the halved in-degree; T contributes only 0.07, blocked by its floor).
This is Turrigiano-style compensatory scaling from eq. 5 with nothing
added. Behaviorally, though, redundancy buffers the damage: a *frozen*
network shrugs off 10% death (0.506 vs baseline 0.510) and keeps ~76% of
score at 30% — and at small wounds the repair transient is **net
harmful** (paired learning−frozen −0.098; the inflammation costs more
than the injury), while learning wins at k=0.5 (median 0.432 vs 0.295)
and abolishes the catastrophic tail at k=0.1 (0/16 dead vs 3/16).

**Sparsity relocates the starting point** (H54, `h54_sparsity.py`, with
input and output wiring pinned so only recurrent density varies). At the
ridge, the conservation law is exact: w̄·p·N = 2.01–2.17 across a 40×
density range (g_final 0.93–1.02) — total recurrent input is conserved
and criticality is wiring-invariant; score is flat (the risky
"dense-degrades" prediction was refuted). But at wlr=0.03 the table
turns: **p=0.02 scores 0.566 with 15/16 reliability at 16 seeds — the
best cell of the campaign** (48-seed cluster calibration: 0.509/0.79,
still +0.05 mean and +0.18 reliability over the dense pair, with a third
bit-identical cross-machine determinism check) — because 4-link columns
start at g≈1.5 instead of 7.5:
sparse wiring is *pre-adapted*, needing almost no renormalization, so
nearly-frozen plasticity suffices and churn never enters. The ridge is
really about time-to-renormalize versus churn: sparse shifts the optimal
wlr down (monotone: 0.566 at 0.03 → 0.299 at 1.0) while density
*narrows* the viable window instead of shifting it up (H54b: at p=0.8,
wlr 0.3 and 1.0 both collapse to 0/16 — the former with physiology fully
renormalized, a carrier-level loss; the latter by runaway potentiation,
w̄pN → +235, the explosion threshold falling with density). Sparse is
forgiving; dense is brittle. Two corollaries: internal fluctuations from
sparse wiring substitute for external noise (the H51 rescue, achieved
architecturally), and dense *readout pools* are their own failure —
unpinned p=0.8 output pools (~160 nodes) average away the left/right
asymmetry that steers the body (0.251 vs 0.443 pinned). Sparse readout
pools are the motor symmetry-breakers.

The generality check across embodiments (H58, H62) returned a signed
answer. On Pong — which starts at Σw ≈ 0 (mean-zero weights, 25%
inhibitory) so the controller *grows* input to comfort, the mirror image
of tracking's erosion — sparse+slow is again the best cell (hit 0.679 at
p=.02/wlr=.03 vs published 0.600), running nearly silent (f ≈ 0.02: ten
active neurons of five hundred). But on wall avoidance the transfer
FAILS, and the failure is the flow-sign law resurfacing: wall's
"perfect" slow-wlr cells are dead statues (the task's degenerate death
solution; an aliveness check caught the confounded metric), and the best
sparse cell (wlr 0.3) reaches only 8/16 alive-and-clean versus the paper
cell's 15/16. The follow-up (H79/H79b) then split this in two: the
slow-wlr wall *deaths* were entirely eq. 4's doing — freeze targets at
birth and 16/16 revive at any density (f 0.000 → 0.12–0.17) — while
the *clean-circling* rate still roughly doubles with fast weight
erosion (paper cell 15/16 vs ~6–7/16 slow, frozen-T). So the storm is
not the guardian of aliveness (that was T-poisoning's victim count),
but it remains a genuine search accelerator for attractor quality on
the flow-negative embodiment. Whether the storm helps or hurts still
depends on the body's sign — but the death toll everywhere belongs to
the target channel.

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
- **There is no figure-ground** (H81/H81b, `h81_choice.py`): give the
  proven follower TWO visible pacemakers — even one it cannot follow
  anyway — and it locks *nothing* (A1 lock 1.000 alone → 0.034 with the
  second bump present; a distractor at just 10% salience already
  abolishes the lock, 0.064). Superposed flows at different rates leave
  no co-moving frame with constant control: the mixture violates clause
  (i), and the architecture has no source separation to escape it. Two
  corollaries: the depth-4 chain's per-link exclusive sensing (each
  follower sees only its own target) was *load-bearing*, not a
  simplification — an all-visible ecology would collapse; and with
  H67's ~1% jitter tolerance this completes the knife-edge theme —
  perfect locks, self-repair, and N-invariance inside the single-source
  clean-flow manifold, collapse at every measured edge. **Selective
  attention is the missing machinery for real multi-agent ecologies**,
  arguably ahead of "prediction" in the structural-additions queue —
  and its minimal sufficient form is now measured (H82–H84): evolution
  over the genome space cannot find it (GA plateau 0.21), a *memoryless*
  winner-take-all retinal filter fails too (0.074 — the selected source
  flickers, teleporting the effective stimulus), but winner-take-all
  **with persistence** (keep the source unless the rival is 2× brighter
  for 100 straight steps) restores the lock completely (1.000, zero
  switches). One latched state variable is the whole difference between
  a collapsed ecology and a working one — and the shared-world test
  passes end-to-end (H85/H85b): with every agent seeing every other,
  sticky attention lets follower C lock follower B (a genuine
  all-visible chain), exposes one last emergent pathology — *the
  follower seduces the leader* (C orbits close, becomes B's brightest
  source, and captures B's attention away from the pacemaker; naive
  salience inverts hierarchies) — and a conservative switching
  threshold (rival must be 5× brighter for 300 steps) cures it:
  **B holds A at 1.000 and C holds B at 1.000 simultaneously, zero
  switches — the first stable all-visible homeostatic ecology.** The
  full saved depth-4 chain then passes the same test (H86): all three
  links at 1.000 under shared visibility, distances identical to the
  exclusive-sensing era — with one more knife-edge corollary banked in
  passing: giving mid-chain followers one-step-stale target positions
  (parallel instead of sequential update) destroys a link completely
  (0.011 vs 1.000); the ecology needs within-step sensory freshness.
- **Entrainment propagates — to depth four, and the ceiling's mechanism
  is now closed** (H49–H50, H60–H68b,
  [fig_chain4](../scripts/out/lab/fig_chain4.png),
  [chain_truth](../scripts/out/lab/chain_truth.png)): successive agents
  lock onto the previous *live* follower — a blind pacemaker plus three
  phase-locked followers on four concentric rings about a shared
  *off-center* point (≈(19.7, 19.7); measuring radii about the arena
  center produced a night of eccentric-frame artifacts, caught, retracted,
  and re-retracted on the ledger). All four agents co-rotate at the same
  ω ≈ 1.9°/step, so **ring contraction (7.8 → 6.5 → 5.6 → 2.6) converts
  directly into linear-speed loss** (0.255 → 0.212 → 0.184 → 0.086 =
  ω·r at every link). The fifth agent fails because its target now crawls,
  stop-and-go, below the followable speed band — the known unfollowable
  near-stationary class. The band's edges were measured the same night:
  0.184 and 0.237 followable, 0.086 and 0.313 not — and the full
  psychometric (H75) has a plateau at 0.10–0.26 (near4 0.94–1.00)
  falling to 0.59 at 0.06 and 0.48 at 0.30, so **followable target speed
  ≈ 0.08–0.28 arena-units/step for this body class**, with the low edge
  jointly set by speed and speed-*regularity* (a clean 0.06 circle
  nearly locks; D's stop-and-go 0.086 does not); the chain's
  depth limit is a *speed floor*, not noise (the earlier
  jitter-amplification reading is retracted: phase-aligned residuals
  *shrink* down the chain, 0.31 → 0.11 — each follower is a regularizer,
  broadcasting a cleaner signal than it receives). Controlled replays
  bracket the rest: a follower tolerates the pacemaker's entire
  deterministic waveform but almost no *stochastic* jitter (OU sd 0.1,
  ~1% of ring radius, destroys the lock), and full-trajectory replay of
  link D fails exactly as the live chain does — the signal is
  intrinsically too slow, not too noisy. Link evolvability remains
  start-basin sensitive (one false start caught and fixed on the record)
  — IC-dependence at the collective level.

## The long horizon: locks decay, plasticity re-locks, targets erode

Three-fold extension of the channel story over 21,600-step runs
(H66–H72, 48-seed cluster + local arms). Performance sags everywhere on
long horizons (ridge −0.109 early→late, t=−3.6, n=48). The wander's bad
segments are *dark excursions* (within-run r(segment score, segment
input-duty) = +0.52 ridge / +0.81 sparse — the H51 trap, visited
transiently by healthy agents), but the sag itself is not accumulating
darkness (duty holds at 0.73–0.75 throughout). Two causal arms then
split the channels cleanly:

- **Freezing everything makes it worse** (H69: freeze-mid sag −0.136 vs
  full −0.064): the erosion is metastable *lock decay*, and the weight
  channel is the re-locking mechanism (as the re-entrainment experiments
  said) — lifelong weight plasticity earns its keep.
- **Freezing only the targets abolishes the sag and wins outright** —
  now at 48-seed cluster power (H71/H76): freeze-T-at-birth late 0.466
  vs full 0.351 (+0.115, t=+3.24), and even the *first 3600 steps* of T
  adaptation do lasting damage (freeze-at-3600 loses to freeze-at-birth
  by 0.090, t=+2.65, and still sags — the early-adapted profile itself
  keeps interacting badly with ongoing W dynamics). On ridge tracking,
  eq. 4 — the family's namesake set-point mechanism — is a pure
  liability. The harmful component is boxed by three nulls: not the
  mean (+0.018, uncorrelated), not anatomical input selectivity (H72),
  not lifetime-activity selectivity (H77). Scope: the historic
  "frozen homogeneous T is death" claim was config-specific and does
  not hold at the ridge; Pong's sparse-slow optimum finds tlr neutral
  (H74).

Prescription: **keep W plastic, freeze T after calibration** — on
supercritical tracking. (Pong's subcritical regime still wants fast
targets as a damper at high wlr; at the slow-wlr cells that now win
there, the question is open.)

## Relation to the large-scale ecology program (arXiv 2510.18221)

The meeting notes point at Bejjani et al.'s 60,000-agent evolutionary
ecologies — reward-free populations of evolved networks where foraging,
predation, and vision emerge under survival pressure, with some behaviors
appearing only past a scale threshold. Tonight's results describe the
*individual-agent substrate* such a program needs, and make three
predictions for homeostat-based versions of it: (i) reward-free selection
is exactly the mechanism our boundary experiments certify (selection
finds repertoire members that no within-life scalar rule can shape —
H29/H35/H36/H56); (ii) stable inter-agent behavior will not emerge from
raw mutual visibility — it requires persistent selective attention (the
H81–H86 progression: collapse → flicker → capture → stability), so their
scale thresholds may partly be *attention-architecture* thresholds; and
(iii) network size per se contributes nothing (H45) — the leverage is in
degrees, rates, and ecological structure (band-compatible pacemakers,
supra-horizon engagements), which is where scale plausibly buys its
emergent behaviors: bigger worlds contain more niches that satisfy the
competence clauses.

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
  computation channel and must keep running. (The long-horizon section
  below sharpens this to its limit: at the ridge, target adaptation is
  net harmful *from step one* at 48-seed power, its damage rides
  entirely in the adapted profile, and freezing T at birth is the best
  policy measured — the calibration role survives only where evolved
  heterogeneous profiles genuinely pay, as at w1′.)
- **The learned structure is not the computation — the ongoing process is**:
  shuffling all learned weights mid-run (learning on) recovers to or above
  the unshuffled run everywhere (defaults 0.394 vs 0.325; w1′ 0.762 vs
  0.850). Self-entrenchment is real and mildly maladaptive at defaults.
- **Reflex vs medium is a *place*, not a property**: at paper defaults the
  recurrent medium is a net liability (lesion scores 0.478 > full 0.325);
  at w1′ lesion collapses (0.250±0.433, bimodal). The prior reflex/medium
  dichotomy is the design space talking, not the model.

## A designer's cheat-sheet

Everything above compresses into working rules. Given a new task for this
family:

1. **Screen the task first** (no training needed) with the three-clause
   competence law: the target's co-moving frame must be holdable with
   ~constant control (≤ ~2× control modulation), its frame rate must sit
   in the entrainment band (linear speed ≈ 0.09–0.28 arena-units/step,
   angular ≤ ~2.1°/step for these bodies), and engagements must outlast
   the re-lock horizon (90–225 steps). If the task fails a clause, no
   learning-rate choice and no scalar-reward scheme will cross it —
   only wiring-level selection finds repertoire members, and only if the
   clauses are satisfiable at all.
2. **Set the weight rate by initial distance-to-comfort.** Compute
   Σw_init = p·N·w̄₀ against the comfort total (~2 for tracking-class
   drive; the controller conserves this quantity). Starting far above
   (supercritical) needs fast erosion (wlr ≈ 1) — or better, sparsify so
   you *start* near comfort and slow plasticity suffices (p=0.02 was the
   most reliable tracking cell). Starting at or below (mean-zero Pong)
   grows to comfort at any wlr given enough steps; avoid wlr ≥ 1
   (churn/runaway — and the explosion threshold falls with density).
   At p = 0.1 the ridge rule is wlr* ≈ 1.04·leak^1.41.
3. **Respect the flow sign.** Flow-positive embodiments (approach raises
   input): skip the storm — sparse pre-adapted starts win on score and
   reliability. Flow-negative (avoidance): the storm IS the search that
   finds the live attractor — keep the supercritical start, and never
   trust a zero-collision metric without an aliveness check (death
   solves avoidance).
4. **Keep readout pools sparse** (~10–20 in-neighbors): dense pools
   average away the left/right asymmetry that steers the body.
5. **A sensor-noise floor (σ ≈ 0.1) is a free rescue** for under-plastic
   networks — it abolishes the absorbing dark state while sparing
   contrast. Sparse wiring's internal fluctuations buy the same thing
   architecturally.
6. **Keep the weight channel plastic; freeze the target channel — from
   birth, on supercritical tracking.** Long-horizon competence decays by
   metastable lock loss, and weight plasticity is the re-locking
   mechanism (full-frozen networks sag faster) — while *any* target
   adaptation is net harmful at the ridge (48 seeds: freeze-T-at-birth
   +0.115 over full, t=+3.24; even the first 3600 steps cost +0.090
   lastingly). Elsewhere the knob is task-dependent: neutral at Pong's
   sparse-slow optimum, a needed damper at Pong's fast-weight cells.
   Structural rewiring: developmental window only.
7. **Targets are the calibration surface.** Only heterogeneous, adapted
   per-node target profiles hold interior spike rates statically;
   homogeneous frozen networks are dead-or-saturated (the cliff).
8. **The stacked recommendation** (H83, 48-seed calibrated): p=0.02,
   wlr=0.03, targets frozen, input/output pins, optional σ=0.1 noise
   floor — 0.536–0.570 short-run, and on long horizons *sag-free* at
   0.562 (the full-homeostatic ridge decays to 0.351). Noise buys mean
   and costs the reliability tail; choose by application.
9. **Size is inert.** Pick N for readout smoothness; pin in-degree
   (~20) and input wiring; check ignition with Law 1 (drive ≥
   leak·ρ·T). Predict the frozen cold-start fate from the wiring file
   with the Laws-1+2 vector map before spending any simulation.

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

1. **What sets the wander floor** — now half-answered. The wander's SD is
   *universal* (~0.26–0.28 in every long-run cell tested; H66): design
   moves the mean, and "reliability" is threshold-crossing statistics
   around it. Its bad segments are *dark excursions* (r(segment score,
   segment duty) = +0.52–0.81; H70) — the H51 trap visited transiently.
   What remains open is the microdynamics generating that invariant SD
   (and the earlier non-additivity results stand: the best configs are
   jointly-tuned wholes; the floor lives in ≥3-way interactions).
2. ~~Formalize information-per-spike~~ — DONE (H78): with the Gaussian
   -channel identity on decoding gain, bits/spike rises ~11× with
   density (1.1 → 3.8 → 11.9 mbits/spike across p = .02/.1/.4 at
   P=120), and dense wins the absolute rate too (0.077 vs 0.027
   bits/step at a third of the spikes). Wiring density buys code
   sparsity buys information efficiency. Residue: gain sags ~25% at
   P=240 in every density, below the nominal absorption corner — the
   corner formula mispredicts for quiet nets. The sensor-spacing
   "resonance" was retested at fine grain and retired — flat within
   noise (H92).
3. ~~Why does the carrier regionalize~~ — ANSWERED (H93): leak is the
   storage-locus dial. Along the ridge, W-purity of the turn bias rises
   0.38 → 0.74 as leak goes 0.15 → 0.55 (Spearman +0.90): slow-leak
   agents lean on x-carried input context, fast-leak agents must store
   the bias in the weights; w1′'s leak (~0.59) sits exactly at the
   high-purity cell. (Residue: a dip to 0.61 at leak 0.7.)
4. **Why the exact retinal drive formula underperforms the proxy** for the
   dead boundary — a closed-loop selection effect worth one figure.
5. ~~What breaks entrainment chains~~ — CLOSED (H60–H68b): a speed
   floor — co-rotating rings contract until the target's linear speed
   falls below the followable band (~0.09–0.28 units/step). The live
   follow-up: can a link be *designed* to hold radius (a speed
   regenerator), where the naive heavy-wheel repeater failed (H63)?
6. ~~Eligibility traces / multi-step credit~~ — CLOSED (H56): traces
   fail at two decay constants, and so does the decisive
   erosion-free arm (homeostasis frozen mid-run while reward continues).
   The barrier is task-side (the competence clauses), not rule-side;
   beyond this only structured machinery (prediction) remains untested.
7. **The target channel's slow erosion** (H71): freeze-T-only abolishes
   the long-horizon sag and wins outright; the per-node selectivity of
   the harmful drift is dynamic, not anatomical (H72 null) — one
   activity-conditioned measurement away.
8. **Why the sag's cure and the statics cliff point opposite ways** — T
   adaptation is needed early (calibration; cliff escape) and harmful
   late (erosion): the natural synthesis is a target-channel
   developmental window, untested.
9. ~~Cluster-scale confirmations~~ — landed (H14, H45, plus the sparse
   and durability batches). The Slurm lane runs a 6k-run batch in ~4
   minutes on mit_quicktest; batch chunks must respect the lane's
   wall-time.

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
