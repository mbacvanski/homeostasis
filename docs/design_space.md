# How homeostatic reservoirs work: laws, phases, and the entrainment mechanism

*Findings of the design-space campaign of 2026-08-31/09-01 (scripts/lab/, ~2600
local runs + cluster replication in flight). Method: every hypothesis was
preregistered in [scripts/lab/LEDGER.md](../scripts/lab/LEDGER.md) before its
experiment; common-random-number wiring seeds across cells; all scripts
deterministic; every number below traceable to a JSON in `scripts/out/lab/`.
Interactive verification: the `/lab` pages of the visualizer (single-node
explorer; phase maps; trajectory viewer). Chance score is 0.25; "score" is
within-45° occupancy over reversal segments 6–10 unless noted.*

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
  plasticity rate matches dissipation (leak 0.05 → wlr* ≈ 0.03, score 0.681;
  0.25 → 0.1, 0.633; 0.75 → 1.0, 0.494). This retroactively explains the
  historic sweep winners: both chose leak ≈ 0.56–0.57 under forced wlr = 1.0
  — they were the wlr=1.0 slice of this same ridge.

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
   question is what design choices raise the floor of the wander.
2. **Formalize information-per-spike**: the h22 sparse-informative fast edge
   suggests an efficiency curve (bits/spike vs slip frequency) worth one
   clean experiment; also the secondary resonance near sensor-spacing/speed.
3. **Why does the carrier regionalize** — pure W-storage at w1′ vs mixed at
   ridge25? (Candidate: leak sets how long input context persists in x.)
4. **Why the exact retinal drive formula underperforms the proxy** for the
   dead boundary — a closed-loop selection effect worth one figure.
5. Cluster-scale confirmations in flight: 48-seed fine ridge (H14: ridge
   follows wlr* = c·leak^b; held-out checkerboard), N-line dual-scaling.

## Instruments

`scripts/lab/`: `common.py` (closed-loop arms incl. lesion/freeze/shuffle/
swap variants, open-loop scripted retina, observables), k0–k4, act2 batches,
b6/b6b/b7, cluster runner + batch generator. Viewers: `/lab` single-node
explorer (all four end states as verified presets; duty law live), `/lab/phase`
and `/lab/traj` (phase-map browser; entrainment trajectory viewer).
Verification culture: preregistration ledger, CRN seeds, exact-law checks to
machine precision, and honest negatives kept on the record.
