# Overnight research update (2026-09-01) — pasteable for Slack

**TL;DR for the 4pm meeting.** The design-space campaign ran overnight to
94 preregistered hypotheses (~26k runs, six cluster batches, all
pushed). The model family now has: three exact laws that compose into a
simulation-free predictor (r = 1.0000); a mechanistic account of every
task in the repo (tracking = velocity entrainment; wall = stability
selection; Pong best at ~10 active neurons; pursuit bounded by a
three-clause competence law with measured constants); answers to the
meeting's "simple questions" (sparsity: a conservation law + a
pre-adaptation effect; target adaptation: calibration early, erosion
late — freeze it after calibration; why stable: self-organized
marginality, now precise — frozen homogeneous networks have NO interior
rates); a four-agent ecology whose depth ceiling has a closed geometric
mechanism (speed floor); and a designer's cheat-sheet in
docs/design_space.md. Interactive verification: seven /lab pages, each
bit-exact against its campaign JSON. Read order: this file → the
"Three Laws and a Ridge" artifact → docs/design_space.md →
scripts/lab/LEDGER.md for any specific number.


Ran a preregistered design-space campaign on the homeostatic reservoirs
(~42 hypotheses H1–H42 in scripts/lab/LEDGER.md, ~15k runs local + 6.4k on
engaging; every number re-derivable; all pushed). Headlines:

**Three exact laws** now govern the mechanics: comfort split (alive iff
drive ≥ leak·T_floor), the duty law f=(μ/T−leak)/ρ (zero free parameters,
holds per-node inside the full network), and eq. 5 is exactly a gated
integral controller on each node's total input (−wlr·E, verified to 1e-14).

**weight_lr is the axis our old sweep never varied — and it matters most.**
The (leak × wlr) plane has a diagonal ridge, wlr* = 1.04·leak^1.41
(4,800-run cluster fit with held-out validation). One change (wlr 1.0→0.1,
the 2021 paper's value) roughly doubles tracking. The ridge is a
signal-to-noise optimum: below it saturation blinds the readout, above it
plasticity churn drowns the stimulus. Both old sweep winners' leak≈0.57 was
this ridge seen through the forced wlr=1.0 slice.

**Tracking is velocity entrainment, not servo control.** Followers with
ZERO signed error response (follow ratio 0.84–0.98); effector inversion
mid-run barely dents them; the turn bias is physically stored in the
recurrent weights; out of view they go still and the periodic world returns
the stimulus. Re-entrainment time is set by the BODY (excursion + ratchet),
not the synapse — a 2-ODE model with both parameters measured (none fitted)
predicts 3/4 regimes within ~15%.

**Case study 3 (wall avoidance) is now implemented and replicated** (all
five anchors; discrepancy table in README — e.g. input weights are 4 in the
code vs 2 in the paper). And it confirms the flow-sign prediction:
rho(input flow, success) = −0.86 there vs +0.77/+0.95 on tracking/Pong —
the body decides the sign. Wall stabilization is stability selection
(collisions are nearly invisible to the network), not punishment learning.

**Pong at 48 seeds/cell:** published config reproduces exactly (0.600);
wlr 0.3 / tlr 0.01 beats it (0.658). Channel competition is SIGNED:
supercritical (tracking) → target adaptation harms; subcritical (Pong) →
it's a necessary damper at wlr=1.0.

**New task: pursuit (ladder rung 3) — and the family's competence law.**
Hand design fails five ways; joint (genome, wiring) evolution finds a
PERFECT pursuer in 3 generations — which turns out to orbit phase-locked
inside the target's orbit (its retina is static in the co-moving frame).
Sweeping target shapes: circle → follower; ellipse ratio 1.27 → follower;
ratio 1.6 → degraded follower; ratio 2.5+ and shuttle → the GA evolves
TOLL-BOOTHS (agents that park where the target passes); aperiodic → nothing.
**The constant-control law: entrainment reaches exactly the motions whose
co-moving frame is holdable with ~constant control (tolerance ≈ 2×
curvature modulation).** The "baseball" intuition from our meeting is
exactly right within the law's reach — no prediction needed — and provably
insufficient beyond it: reward-scaled and one-step reward-directed
plasticity both fail to cross (preregistered, controlled).

**The wiring lottery's home:** it lives in the frozen adjacency.
Weight-space fixes lock 0/16; a minimal grow/prune structural-homeostasis
rule is the first within-life mechanism to cross (3/16); wiring-level
selection is reliable on its pair. Mechanism hierarchy measured — direct
empirical mandate for the connectivity/DNA direction.

Everything is interactive: /lab pages in the visualizer (single-node
explorer, phase maps, entrainment trajectories, wall arena, the perfect
pursuer and its failure mode). Synthesis: docs/design_space.md. Shareable
writeup: the "Three Laws and a Ridge" artifact.

**Late-night additions:** (1) Structural trilogy — the wiring lottery lives
in the frozen adjacency (no weight-space operation fixes it, 0/16); a
minimal grow/prune structural-homeostasis rule is the first within-life fix
(3/16); lifelong rewiring destroys solved tasks; a developmental window
(structural exploration that closes) keeps the benefit at zero cost — the
family, minimally extended, rediscovers why development exists. (2) First
multi-agent data — two mutually-tracking homeostats collapse to collective
stillness (a static partner is already stationary input); churn pairs chase
without locking; the pacemaker test shows a recorded wall-circler orbit is
unfollowable at its native 6.4°/step but PERFECTLY followed at 2.1°/step:
the competence law is two-clause (constant-control shape AND frame rate in
the entrainment band), and collective entrainment needs a slow periodic
pacemaker. (3) Duty law verified at N=2000 (corr 0.988); full N-scaling
ridge batch on the cluster.

**The capstone: a live multi-agent homeostatic ecology.** The family cannot
pace itself at the paper's scale (absorption wall-agents are dead statues;
churn circlers turn too fast to follow) — but with a wider wheel base in a
doubled arena, a wall-avoider circles cleanly at 1.9 deg/step, inside the
entrainment band, and an evolved follower locks onto it LIVE (dist 3.77,
stable 10,800 steps). Then a third agent locks onto the follower: **three
concentric phase-locked circles — a blind pacemaker's periodicity cascading
down a sensing chain of comfort-seeking agents** (figs in the artifact and
scripts/out/lab/). The rings contract ~15% per link, predicting a finite
chain depth. This is a constructive answer to "what if agents interact":
interaction alone gives stillness or unfollowable churn (measured); add one
band-compatible pacemaker and entrainment propagates.

**Night two (H51–H69, ~3,000 more runs, three more cluster batches).**
The campaign kept going and the ledger now runs to 69 preregistered
hypotheses — including five refuted-backwards results that each replaced
a plausible story with a measured one:

- **Robustness chapter.** Moderate sensor noise *rescues* under-plastic
  networks (0.367→0.533, reliability 0.44→0.81) — not by desaturation but
  by abolishing the absorbing dark state (the agent otherwise spends 35%
  of its time in darkness where no learning signal exists). Killing
  10–50% of nodes mid-run: spike rate recovers via eq. 5 acting as
  Turrigiano-style synaptic scaling (surviving weights re-inflate ~2× for
  a halved in-degree, nothing added to the model); behavior barely needs
  it (redundancy buffers 10% death entirely) and at small wounds the
  repair transient costs more than the injury.
- **Sparsity (your "simple question") has a law.** Total recurrent input
  after adaptation is conserved across a 40× density range (w̄·p·N ≈
  2.1); the interesting effect is the *starting point*: p=0.02 begins
  near comfort, so nearly-frozen plasticity suffices — best tracking
  reliability of the campaign (0.566, 15/16; 48-seed calibration
  0.509/0.79) and best Pong cell (0.679 vs published 0.600, running at
  ~10 active neurons in 500). But it FAILS on wall avoidance — there the
  supercritical storm is the *search* that finds the live attractor
  (flow sign strikes again). And density *sharpens* the spike code
  (info-per-spike 9.02 vs 1.42 sparse): the sparse-wiring advantage is
  dynamical, not informational.
- **The baseball question, closed with a boundary.** Ballistic
  interception fails at native speeds — crossings (~70 steps) end before
  the re-lock horizon (90–225 steps) — and selection, given that task,
  evolves a literally blind sweeper (vision has zero marginal fitness
  below the horizon). Slow the targets past the horizon and interception
  works (+0.31 over blind, wiring-robust). Third clause of the
  competence law: **engagements must outlast the lock**. Mid-flight ±90°
  kinks are harmless (randomized test); what kills loitering pursuit is
  ramplessness — entrained turning is closed-loop bearing geometry, so
  fluttering bearings cancel to pure forward throttle: vision *unparks
  the agent without aiming it* and a parked blind statue wins. Flow is a
  throttle; stable geometry is the steering.
- **The reward ladder is complete and the barrier is task-side.**
  Eligibility traces fail; traces with homeostatic erosion switched OFF
  fail identically — neither credit delay nor absorbability was the
  blocker. The tasks reward was aimed at violate the competence clauses;
  there is no gradient any weight-tweak rule could climb.
- **The chain's depth ceiling has a closed mechanism** — after two wrong
  ones, caught by our own controls. The four rings are concentric about
  an off-center point and co-rotate at one angular rate, so ring
  contraction converts to linear-speed loss (0.255→0.086 = ω·r at every
  link); link five fails because its target crawls below the followable
  speed band, whose edges we measured (~0.09–0.28 units/step). Followers
  are *regularizers* (each broadcasts a cleaner signal than it receives);
  stochastic jitter tolerance is ~1% of ring radius.
- **Statics vs dynamics, made precise.** Laws 1+2 compose into a
  simulation-free predictor of frozen-network fate (r = 1.0000 from the
  wiring file alone) — and the same model shows homogeneous frozen
  networks have NO stable interior spike rates: the f ≈ 0.1–0.3 regime
  where all competence lives exists only as the homeostat's dynamically
  held equilibrium. Long-horizon competence decays by metastable lock
  loss, and freezing makes it WORSE — plasticity is the re-locking
  mechanism. Homeostasis doesn't just tune the network; it *is* the
  thing keeping the working state in existence.

Practical payoff: docs/design_space.md now ends with a designer's
cheat-sheet (task screen via the three-clause law, wlr from
distance-to-comfort, flow-sign rules, readout sparsity, noise floors,
never-freeze). Three new interactive pages (/lab/repair, noise slider,
ballistic pursuit) reproduce their campaign JSONs bit-exactly.

**Late addition (48-seed powered): the target channel is a liability on
the flagship task.** Freezing targets at birth beats full homeostasis by
+0.115 late-run score (t=+3.24, n=48, 21,600-step runs); even the first
3,600 steps of target adaptation do lasting damage (+0.090 vs freezing
at birth, t=+2.65). Long-horizon competence decays by metastable lock
loss; weight plasticity is the re-locking mechanism (freezing weights
makes decay worse), while the target channel's profile drift causes the
decay (mechanism boxed by three preregistered nulls: not the mean, not
anatomically-, not activity-selective). Practical rule: keep W plastic,
freeze T — on supercritical tracking; Pong's sparse-slow optimum finds
the target knob simply neutral. The speed band of followable motion also
got its full psychometric (plateau 0.10–0.26 units/step), and
information-per-spike is now in units: density buys code sparsity buys
~11x bits/spike (dense-quiet networks carry more information absolutely
at a third of the spikes).

**And the last boundary of the night: no figure-ground.** Give the
proven follower two visible pacemakers — even one it cannot follow
anyway — and it locks nothing (1.000 alone → 0.034 with the second bump;
a distractor at 10% salience already abolishes the lock). Superposed
flows leave no constant-control frame, and the architecture has no
source separation. The depth-4 chain worked only because each link sees
exclusively its own target — selective attention, not prediction, looks
like the first structural addition a real multi-agent homeostatic
ecology needs.

**Postscript — attention, solved minimally; and a recommended config.**
The figure-ground deficit has a measured minimal cure: evolution can't
find attention in the parameter space (GA plateau 0.21), a memoryless
winner-take-all filter fails (0.074 — the selection flickers), but WTA
*with persistence* (stick with a source unless the rival is 2x brighter
for 100 steps) restores the lock to 1.000. One latched state variable.
Separately, stacking the night's discoveries (p=.02, wlr=.03, targets
OFF, sensor-noise floor 0.1) yields the best tracking cell measured
(0.607 short-run) and — with targets off — the long-horizon sag is gone
entirely (late-run 0.582 vs 0.351 for the full-homeostatic ridge; the
eq.-4 causal story validated end-to-end).

**Final result of the night: the first stable shared-visibility
homeostatic ecology.** The progression, each step measured: all-visible
agents with no attention collapse (0.034); a memoryless winner-take-all
filter still collapses (0.074 — selection flicker); sticky attention
forms chains but exposes a new emergent pathology — the follower orbits
close, becomes its leader's brightest stimulus, and seduces the
leader's attention away from the pacemaker (salience inverts
hierarchies); raising the switching threshold (5x for 300 steps) cures
it: pacemaker→B→C all mutually visible, B holds A at 1.000 and C holds
B at 1.000 with zero attention switches. One latched selection bit plus
a conservative switch rule is the complete minimal machinery.

**On the arXiv paper from the notes (2510.18221, Bejjani et al.):** their
60k-agent reward-free evolutionary ecologies are the macro version of the
direction; our campaign supplies the micro substrate and makes three
predictions for homeostat-based versions — selection (not within-life
reward rules) is the certified mechanism; stable inter-agent behavior
needs persistent selective attention (raw mutual visibility collapses);
and scale buys emergence through niches that satisfy the competence
clauses, not through network size (which is inert).

**Coda — reproduction as action, measured (the budding theorem).** A
locked follower spawning offspring at its own position: position
inheritance alone fails (fresh wiring), adding wiring heredity fails
(cold state), a pure clone fails (state, not mutation, is the barrier)
— only full-state budding works, and then completely (population
cascades to cap, 6/6 locked). The heritable unit is the entire
dynamical state, not the structure: you can't inherit a lock, because
competence is a held equilibrium. What DNA-like heredity needs — and
this family lacks — is scaffolded development that re-derives state.
That's the measured gap for the reproduction/DNA direction.
