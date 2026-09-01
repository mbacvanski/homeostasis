# homeostasis4

Re-implementation of the homeostatic reservoir model from:

> Falandays, J.B., Yoshimi, J., Warren, W.H., & Spivey, M.J. (2024). A potential
> mechanism for Gibsonian resonance: behavioral entrainment emerges from local
> homeostasis in an unsupervised reservoir network. *Cognitive Neurodynamics*
> 18:1811–1834. ([PDF in repo root](./A%20potential%20mechanism%20for%20Gibsonian%20resonance-%20behavioral%20entrainment%20emerges%20from%20local%20homeostasis%20in%20an%20unsupervised%20reservoir%20network.pdf))

Implements **case study 1 (moving-object tracking)** and **case study 2
(Pong)**: an agent controlled by a reservoir of homeostatic leaky
integrate-and-fire nodes spontaneously entrains to a stimulus orbiting
around it, and — with the same network core and no change to the learning
rule — plays Pong well above chance. No reward signal, no trained readout, no
supervision. The authors' original Julia code (OSF:
[osf.io/6hqrt](https://osf.io/6hqrt/)) is archived in
[reference/original_julia/](reference/original_julia/) as ground truth.

## Layout

- [src/homeostasis/reservoir.py](src/homeostasis/reservoir.py) — the
  task-agnostic network core (eqs. 1–5; timing semantics documented in the
  module docstring)
- [src/homeostasis/tracking.py](src/homeostasis/tracking.py) — case-study-1
  environment: sensor geometry (eq. 6), effector mapping (eq. 7), stimulus
  kinematics
- [src/homeostasis/pong.py](src/homeostasis/pong.py) — case-study-2
  environment: ball physics, paddle collision, egocentric/allocentric sensor
  encodings, hit/miss scoring
- [src/homeostasis/simulation.py](src/homeostasis/simulation.py) — the
  action–perception loop wiring and run recording
- [src/homeostasis/analysis.py](src/homeostasis/analysis.py) — tracking
  metrics, with reference values from the authors' published run
- [tests/](tests/) — 66 tests: hand-computed mechanics on tiny manual
  networks, sensor/effector geometry, wiring/reproducibility, and behavioral
  regressions on fixed seeds
- [scripts/run_tracking_experiment.py](scripts/run_tracking_experiment.py) —
  multi-seed experiment + Fig. 4-style plots
- [scripts/fingerprint.py](scripts/fingerprint.py) — deterministic state
  fingerprint for cross-checking the visualizer against batch runs
- [viz/](viz/) — interactive browser visualizer (FastAPI + WebSocket; the
  simulation it runs *is* the tested package, no duplicated model logic)

## Setup

```bash
uv venv --python 3.12
uv pip install -e ".[dev,viz]"
```

## Tests

```bash
.venv/bin/python -m pytest
```

The `slow`-marked behavioral tests run five full 7200-step simulations plus
ablations (~5 s); `-m "not slow"` skips them.

## Run the experiment

```bash
.venv/bin/python scripts/run_tracking_experiment.py --seeds 10
```

Prints per-seed tracking metrics with learning-off ablations and writes a
Fig. 4-style plot to `scripts/out/tracking_run.png`.

## Irregular-motion tracking variant

The published constant-speed task remains the default. An opt-in extension
varies stimulus speed smoothly and samples each speed-retarget and direction-
reversal interval independently:

```bash
.venv/bin/python scripts/run_variable_tracking_experiment.py --seed 0
```

This writes `scripts/out/variable_tracking_run.pdf` without replacing the
published-task output. Programmatically, use `VariableTrackingConfig`,
`VariableTrackingSimulation`, or `run_variable_tracking`. The defaults vary
speed from 0.5 to 1.5 degrees per step, retarget speed every 180–540 steps,
and reverse direction every 480–960 steps. Speed approaches each new target
smoothly rather than jumping. The reservoir seed also determines the motion
schedule; reservoir initialization consumes its established draws first, so
the original tracking trajectory and fingerprint remain unchanged.

## Visualizer

```bash
.venv/bin/python -m uvicorn viz.server:app --port 8471
```

Then open http://localhost:8471. Features, all aimed at verifying the model
by eye and by hand:

- **Arena** with agent, eyes, per-sensor activation ticks, and the orbiting
  stimulus; **drag the green dot** to take manual control of the stimulus
  (the reversal clock suspends; Reset restores the standard experiment).
- **Single-stepping** (Step ×1/×10/×100, `space`/`s` keys) with every number
  visible: sensor bars, effector activities, ΔH, per-node inspector
  (activation, target, threshold, error, degrees — click a raster row).
- **Fig. 4-style live charts**: heading vs. stimulus, wrapped heading error
  with the ±45° band, spike raster (1 px per step), proportion spiking.
- **Live toggles**: learning on/off (watch the network saturate and the agent
  decouple, then recover), constant/irregular stimulus motion, stimulus
  direction flip, speed up to ~2000 steps/s, all model parameters (applied on
  Reset). Irregular mode displays current and target stimulus speeds, the next
  schedule events, and a live speed trace.
- **Named loadouts**: the paper configuration plus search results 236 and 234
  can be loaded from the controls. The displayed score, direction-agreement,
  and spike-proportion values are the reference results supplied with those
  configurations; loading a preset rebuilds the simulation at step 0.
- **Determinism fingerprint**: the page shows Σx/ΣT/ΣW at step t; run
  `scripts/fingerprint.py --seed S --steps T` and the sums match exactly
  (for an untouched default-parameter run), proving the browser session is
  bit-for-bit the batch trajectory.

## Pong (case study 2)

```bash
.venv/bin/python scripts/run_pong_experiment.py --runs 40
```

Runs four conditions and compares each to the published numbers (chance is
0.20 — a 100 px paddle in a 500 px field): `published` (released sensors),
`fixed-sensors` (see below), `no-learning`, and `allocentric`. The
visualizer's Pong page is at **`/pong`** on the same server, with the field,
the egocentric sensor fan, live hit rate against the paper's 0.582, the
`|Δθ|` trace that the paper's proposed mechanism turns on, and the same
determinism fingerprint (cross-check with `scripts/pong_fingerprint.py`).

### One fixed bug: the blind spot straight ahead

The 46-sensor egocentric array spans ±90° in 4° steps, so it *straddles* zero
(…, −6, −2, +2, +6, …) — there is no sensor pointing straight ahead. The
released code tests `< 2` strictly, so a ball at an exact multiple of 4°,
most importantly **0° (dead ahead)**, sits 2° from both neighbours and
activates *nothing*. Measured over full runs this fires on ~0.5% of steps and
**every occurrence is at exactly 0°**, because the paddle clamps at y = 50/450
and the ball's height moves in multiples of 5 — so the network goes blind
precisely when a parked paddle is level with the ball.

We treat this as an oversight rather than a design choice, because the same
authors wrote `<= 5` in their *allocentric* variant of the same function.
`PongConfig` therefore defaults to `sensor_inclusive=True` (`<= 2`), which
closes every gap without changing the sensor count, spacing, or ±90° span:
both neighbours fire at those angles, and every other angle still activates
exactly one sensor. `PongConfig.published()` restores the strict test for
bit-exact replication; the two score within noise of each other (numbers in
`scripts/out/pong/results.json`).

## Fidelity: paper text vs. released code

Where the paper's methods text and the released Julia code disagree, **this
implementation follows the code**, since the code produced the published
results. Verified discrepancies (tracking model):

| Topic | Paper text | Released code (= this repo) |
|---|---|---|
| Recurrent weight init | Normal(0, 1) | Normal(0.75, 0.1) — all-positive (`rand(Normal(input_amp, .1))`); the code's inhibitory-node branch is dead code (fraction overridden to 0) |
| Sensor activation | Gaussian `exp(-θ²/10)` (eq. 6) | Gaussian, **plus activation = 1 for any sensor within 4°** of the stimulus — so the two bracketing sensors per eye always read 1.0 (the full-height bars in Fig. 3) |
| Spike delivery timing | eq. 1's `W_{t-1}` subscript implies emission-time weights | spikes integrate through the **current** weight matrix at receipt time (`get_acts` uses `wmat` after the previous step's update) |
| Spike condition | "activity exceeds the threshold" | `>=` (at-or-above) |
| Weight update size | full error split across spiking in-neighbors (eq. 5) | same — the code's `lrate_wmat = .01` parameter is defined but **never used** |
| Negative activations | not discussed | allowed at runtime (`acts_neg` clamp switch disabled in the run script) |

With these settled, the re-implementation reproduces the published behavior.
On the authors' own published run (OSF `Data/ObjectTracking`), our metrics
give within-45° = 0.38, median |error| = 62°, direction-agreement = 0.83,
mean proportion spiking = 0.34; our seeds 0–9 bracket those values
(within-45° 0.24–0.53, direction-agreement 0.65–0.79, spiking 0.29–0.38),
and disabling learning abolishes movement entirely, as in the paper's
ablations.

### Pong: further discrepancies

The Pong scripts disagree with the paper *and* with the tracking code, so the
network core carries per-case-study settings (`PONG_RESERVOIR_CONFIG`):

| Topic | Paper text | Released Pong code (= this repo) |
|---|---|---|
| Target learning rate | 0.01 (eq. 4) | **0.1** — ten times the paper's value and the tracking code's (`lrate_targ = .1` in both Pong scripts) |
| Recurrent weight init | Normal(0, 1) | **per-synapse mixture**: 25% drawn from Normal(−1, 0.1), 75% from Normal(0, 0.2) — a third scheme, distinct from tracking's Normal(0.75, 0.1) |
| Sensor tuning | 4°-wide arcs, value 1 inside | binary, `< 2` strictly — leaving the 0° blind spot described above (we default to `<= 2`; see that section) |
| Ball radius | 15 px | never used in the physics: collision intersects the ball's *path segment* with the paddle's line segment, so the ball is a point |
| Miss detection | ball "passed the paddle" | recorded only when the ball reaches x ≤ 0 — 100 px and 20 steps *past* the paddle |
| Wall bounces | ball changes y at top/bottom, x at the right wall | the wall and miss tests are an `if/elseif` chain, so at most one fires per step (a paddle bounce is separate and can combine with them) |

Validation, 40 runs × 100,000 steps each (chance 0.20), against the paper's
500-run figures: see `scripts/out/pong/results.json` and
`scripts/out/pong/pong_validation.png`.

## Beyond the paper: the design-space campaign

`scripts/lab/` is an open investigation into why these networks work: three
exact laws (comfort split, duty law, gated-integral weight servo), the phase
geography (dead / statue / absorption ridge / churn / exploded), the
matched-timescale ridge as a signal-to-noise optimum, and the finding that
tracking is velocity entrainment with a flow ratchet rather than servo
control. Every hypothesis was preregistered in
[scripts/lab/LEDGER.md](scripts/lab/LEDGER.md); the synthesis with verified
numbers is [docs/design_space.md](docs/design_space.md); interactive
verification lives at `/lab` in the visualizer.

## Beyond the paper: the language track

`scripts/stage1*`–`scripts/stage5a_*` are an exploratory track applying the
same reservoir to the authors' earlier *language* model (Falandays, Nguyen &
Spivey 2021, PDF in repo root): passive listening, prediction-as-pattern-
completion, and then attempts to close the loop through a mouth (speaking,
contingent dialogue). Findings — including the full 2021 replication, the
prediction-by-absorption mechanism, the staleness result, and the
context-residue confound — are distilled with verified numbers in
[docs/language_track.md](docs/language_track.md).
