# Overnight research update (2026-09-01) — pasteable for Slack

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
