"""Homeostatic reservoir network from Falandays, Yoshimi, Warren & Spivey (2024),
"A potential mechanism for Gibsonian resonance: behavioral entrainment emerges
from local homeostasis in an unsupervised reservoir network", Cognitive
Neurodynamics 18:1811-1834.

This module implements the task-agnostic network core: an input layer feeding a
recurrent reservoir of homeostatic leaky integrate-and-fire nodes, plus a
non-updating output (effector) layer read out as spike proportions.

Where the paper's text and the released Julia code (OSF: https://osf.io/6hqrt/,
archived in ``reference/original_julia/``) disagree, this implementation
follows the code, because the code is what produced the published results.
The known discrepancies are listed in the project README; the two that matter
here:

- Recurrent weights are initialized ``Normal(weight_init_mean, weight_init_sd)``
  with defaults (0.75, 0.1) — all-positive — as in the code (``rand(Normal(
  input_amp, .1))``), not the Normal(0, 1) stated in the paper.
- Spikes are delivered through the *current* weight matrix at receipt time
  (``spikes_prev . wmat_current`` in ``get_acts``), i.e. after the previous
  step's homeostatic update, not the emission-time weights that a literal
  reading of the paper's eq. 1 subscripts would suggest.

Timing semantics (one call to :meth:`HomeostaticReservoir.step` = timestep t)
-----------------------------------------------------------------------------
1. **Integrate** (eq. 1)::

       x_t = x_{t-1} * (1 - leak) + i_t @ W_in + s(x_{t-1}) @ W

   where ``W`` is the current recurrent weight matrix (receipt-time, see
   above). If ``clamp_negative_activations`` is set, x is then clamped at 0
   (the original code's ``acts_neg`` switch; disabled in the published runs).

2. **Spike** (eq. 2): any node with ``x_t >= threshold`` (the code uses >=;
   threshold = ``threshold_ratio * T_t``) spikes, at most once per step, and
   immediately subtracts its threshold: ``x'_t = x_t - s(x_t) * threshold_t``.

3. **Homeostatic update** (eqs. 3-5), driven by the error ``E_t = x'_t - T_t``:

   - Targets (eq. 4): ``T_{t+1} = max(T_t + target_lr * E_t, target_floor)``.
     Spike thresholds stay coupled at ``threshold_ratio * T``.
   - Incoming recurrent weights (eq. 5): each node n adjusts only the weights
     from in-neighbors that spiked on the *previous* step (the spikes just
     integrated in phase 1), moving opposite to the error and splitting the
     full error equally across those weights::

         W[s, n] -= E_n / (number of in-neighbors of n that spiked at t-1)

     If no in-neighbor spiked at t-1, node n's weights are unchanged. (The
     full error is applied: the original code defines a weight learning rate
     but never uses it.)

4. **Effector readout**: each output node's activity is the proportion of its
   incoming reservoir connections whose source node spiked *this* step
   (range [0, 1]; weights to the output layer play no role in the readout).

Input nodes are stateless: the caller supplies the input activation vector
each step, it is scaled by the fixed input weight matrix, and nothing is
retained. Input weights and output connectivity never update; the reservoir's
adjacency is fixed while its weights and targets update every step.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

__all__ = ["ReservoirConfig", "HomeostaticReservoir", "StepState"]


@dataclass(frozen=True)
class ReservoirConfig:
    """Parameters of the network core (defaults = released code, case study 1)."""

    n_nodes: int = 200          # N, reservoir size
    n_inputs: int = 62          # sensor nodes
    n_outputs: int = 2          # effector nodes
    p_link: float = 0.1         # connection probability for all three link types
    input_weight: float = 0.75  # fixed weight of every input->reservoir link
    leak: float = 0.25          # proportion of activation leaked per step
    weight_init_mean: float = 0.75  # recurrent weights ~ Normal(mean, sd);
    weight_init_sd: float = 0.1     # tracking code uses Normal(input_amp, 0.1)
    # Per-synapse inhibitory draw, used by the Pong code: with probability
    # inhibitory_fraction a link is drawn from Normal(inhibitory_weight_mean,
    # inhibitory_weight_sd) instead. Zero (the default) reproduces the
    # tracking scheme and consumes no extra random draws.
    inhibitory_fraction: float = 0.0
    inhibitory_weight_mean: float = -1.0
    inhibitory_weight_sd: float = 0.1
    target_init: float = 1.0    # initial target activation T
    target_floor: float = 1.0   # hard lower bound on T (eq. 4)
    target_lr: float = 0.01     # proportion of error applied to targets
    threshold_ratio: float = 2.0  # spike threshold = ratio * target
    allow_self_connections: bool = False  # code: `if row == col continue`
    clamp_negative_activations: bool = False  # code's acts_neg switch (off in runs)

    def __post_init__(self) -> None:
        if not 0.0 <= self.p_link <= 1.0:
            raise ValueError("p_link must be in [0, 1]")
        if not 0.0 <= self.leak <= 1.0:
            raise ValueError("leak must be in [0, 1]")
        if self.n_nodes < 1 or self.n_inputs < 1 or self.n_outputs < 1:
            raise ValueError("layer sizes must be positive")


@dataclass(frozen=True)
class StepState:
    """Snapshot of one timestep, for tests, analysis, and visualization.

    All arrays are copies; mutating them does not affect the network.
    """

    t: int                    # index of the step that produced this state
    inputs: np.ndarray        # (n_inputs,) input activations fed this step
    x: np.ndarray             # (N,) post-spike-subtraction activations x'_t
    spiked: np.ndarray        # (N,) bool, spiked this step
    targets: np.ndarray       # (N,) targets T_{t+1} (after this step's update)
    error: np.ndarray         # (N,) E_t = x'_t - T_t (pre-update targets)
    outputs: np.ndarray       # (n_outputs,) effector activations in [0, 1]

    @property
    def prop_spiked(self) -> float:
        return float(np.mean(self.spiked))


class HomeostaticReservoir:
    """The three-layer homeostatic reservoir network.

    Reproducibility: all connectivity and initial weights are drawn from
    ``numpy.random.default_rng(seed)`` in a fixed order (input adjacency,
    reservoir adjacency, reservoir weights, output adjacency). The dynamics
    themselves are deterministic, so a given seed and input sequence fully
    determine every trajectory.
    """

    def __init__(self, config: ReservoirConfig = ReservoirConfig(), seed: int | None = None):
        self.config = config
        self.seed = seed
        # Keep the generator so opt-in environments can draw reproducible
        # task schedules only after the reservoir's initialization draws have
        # completed. The published tracking task never consumes it again.
        self.rng = np.random.default_rng(seed)
        rng = self.rng
        c = config

        # --- Fixed structure ------------------------------------------------
        # Input -> reservoir: Bernoulli(p_link) per (input, node) pair, all
        # existing links share one fixed weight.
        self.input_adjacency = rng.random((c.n_inputs, c.n_nodes)) < c.p_link
        self.input_weights = self.input_adjacency * c.input_weight

        # Reservoir recurrent: directed Bernoulli(p_link) per ordered pair.
        # adjacency[s, n] means s -> n. The adjacency never changes.
        self.adjacency = rng.random((c.n_nodes, c.n_nodes)) < c.p_link
        if not c.allow_self_connections:
            np.fill_diagonal(self.adjacency, False)

        # Reservoir -> output: Bernoulli(p_link); readout is the proportion of
        # in-neighbors spiking, so these links carry no weight value.
        self.output_adjacency = rng.random((c.n_nodes, c.n_outputs)) < c.p_link
        self._rebuild_structure_caches()

        # --- Mutable state --------------------------------------------------
        # weights[s, n]: current weight of link s -> n (zero where no link).
        weights = rng.normal(c.weight_init_mean, c.weight_init_sd, (c.n_nodes, c.n_nodes))
        if c.inhibitory_fraction > 0.0:
            is_inhibitory = rng.random((c.n_nodes, c.n_nodes)) < c.inhibitory_fraction
            weights = np.where(
                is_inhibitory,
                rng.normal(
                    c.inhibitory_weight_mean, c.inhibitory_weight_sd, (c.n_nodes, c.n_nodes)
                ),
                weights,
            )
        self.weights = np.where(self.adjacency, weights, 0.0)
        self.x = np.zeros(c.n_nodes)                    # x'_{t-1}
        self.targets = np.full(c.n_nodes, c.target_init)  # T_t
        self.spiked = np.zeros(c.n_nodes, dtype=bool)     # s(x_{t-1})
        self.t = 0
        # When False, targets and weights are frozen (the paper's
        # "learning turned off" ablation); dynamics still run.
        self.learning_enabled = True

    def _rebuild_structure_caches(self) -> None:
        """Precompute float views of the fixed boolean structure.

        Called from __init__; call again after overwriting adjacency matrices
        by hand (as the tests' manual-network helper does). Caching avoids
        re-allocating NxN float casts on every step.
        """
        self._adjacency_f = self.adjacency.astype(float)
        self._output_adjacency_f = self.output_adjacency.astype(float)
        self._output_in_degree = self.output_adjacency.sum(axis=0)
        self._spiked_f = getattr(self, "spiked", np.zeros(self.adjacency.shape[0], dtype=bool)).astype(float)

    # -- properties ---------------------------------------------------------

    @property
    def thresholds(self) -> np.ndarray:
        """Spike thresholds T' = threshold_ratio * T (always coupled)."""
        return self.config.threshold_ratio * self.targets

    # -- dynamics -----------------------------------------------------------

    def step(self, inputs: np.ndarray) -> StepState:
        """Advance the network one timestep given input activations.

        Parameters
        ----------
        inputs : (n_inputs,) float array of sensor activations for this step.
        """
        c = self.config
        inputs = np.asarray(inputs, dtype=float)
        if inputs.shape != (c.n_inputs,):
            raise ValueError(f"expected inputs of shape ({c.n_inputs},), got {inputs.shape}")

        # 1. Integrate: leak, then add external input and the previous step's
        # spikes through the *current* weights (receipt-time; see module doc).
        # _spiked_f is the cached float view of last step's spike vector (a
        # boolean-boolean matmul would be a logical OR, not a sum).
        x = (
            self.x * (1.0 - c.leak)
            + inputs @ self.input_weights
            + self._spiked_f @ self.weights
        )
        if c.clamp_negative_activations:
            x = np.maximum(x, 0.0)

        # 2. Spike: at or above threshold, once per step, subtract threshold.
        thresholds = self.thresholds
        spiked = x >= thresholds
        x_adj = x - spiked * thresholds

        # 3. Homeostatic update from error E = x' - T.
        error = x_adj - self.targets
        if self.learning_enabled:
            # Spikes integrated in phase 1 (previous step's), as row indices.
            prev_rows = np.flatnonzero(self.spiked)
            if prev_rows.size:
                counts = self._spiked_f @ self._adjacency_f
                with np.errstate(divide="ignore", invalid="ignore"):
                    per_weight = np.where(counts > 0, error / counts, 0.0)
                # Only links whose source spiked at t-1 move; each absorbs an
                # equal share of the full error, opposite in sign (eq. 5).
                self.weights[prev_rows] -= self._adjacency_f[prev_rows] * per_weight
            self.targets = np.maximum(self.targets + c.target_lr * error, c.target_floor)

        # 4. Effector readout: proportion of in-neighbors spiking this step.
        spiked_f = spiked.astype(float)
        outputs = (spiked_f @ self._output_adjacency_f) / np.maximum(self._output_in_degree, 1)

        # Commit state.
        self.x = x_adj
        self.spiked = spiked
        self._spiked_f = spiked_f
        self.t += 1

        return StepState(
            t=self.t - 1,
            inputs=inputs.copy(),
            x=x_adj.copy(),
            spiked=spiked.copy(),
            targets=self.targets.copy(),
            error=error.copy(),
            outputs=outputs.copy(),
        )
