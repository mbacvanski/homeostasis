"""Wiring of the homeostatic reservoir to its environments.

Tracking (case study 1) couples the pieces in this order each step:

1. Read the sensor array from the current agent/stimulus geometry.
2. Step the network on those inputs (integrate, spike, homeostatic update).
3. Apply the effector outputs to turn the agent (eq. 7).
4. Advance the stimulus (with its periodic direction reversal).

So the network's step t sees the world as it was *after* step t-1's movement,
and its output moves the agent before the stimulus advances. The paper does
not spell out the sub-step ordering of agent vs. stimulus motion; at 1
degree/step it is immaterial, but it is fixed here for reproducibility.

Pong (case study 2) uses the ordering the released code makes observable,
because collisions depend on it: sense, step the network, move the *ball*
(resolving any bounce against the paddle's current position), then move the
paddle.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .pong import PongConfig, PongEnv
from .reservoir import HomeostaticReservoir, ReservoirConfig, StepState
from .tracking import TrackingConfig, TrackingEnv

__all__ = [
    "TrackingSimulation",
    "History",
    "run_tracking",
    "PongSimulation",
    "PongHistory",
    "run_pong",
    "PONG_RESERVOIR_CONFIG",
]


@dataclass
class History:
    """Per-step records of a tracking run (arrays of length n_steps)."""

    heading: np.ndarray          # agent heading after this step's turn
    stimulus_angle: np.ndarray   # stimulus angle before advancing (what was sensed)
    stimulus_direction: np.ndarray  # +1 / -1 while this step was sensed
    stimulus_speed: np.ndarray   # unsigned speed applied when the stimulus advances
    error: np.ndarray            # signed heading error at sensing time
    d_heading: np.ndarray        # turn applied this step
    outputs: np.ndarray          # (n_steps, 2) effector activations
    prop_spiked: np.ndarray      # fraction of reservoir spiking
    mean_target: np.ndarray      # mean of T across nodes
    mean_abs_error: np.ndarray   # mean |E| across nodes
    spikes: np.ndarray           # (n_steps, N) bool raster

    def __len__(self) -> int:
        return len(self.heading)


class TrackingSimulation:
    """A reservoir-controlled agent in the tracking environment."""

    def __init__(
        self,
        reservoir_config: ReservoirConfig = ReservoirConfig(),
        tracking_config: TrackingConfig = TrackingConfig(),
        seed: int | None = None,
    ):
        if reservoir_config.n_inputs != tracking_config.n_sensors:
            raise ValueError(
                f"reservoir expects {reservoir_config.n_inputs} inputs but the "
                f"environment has {tracking_config.n_sensors} sensors"
            )
        if reservoir_config.n_outputs != 2:
            raise ValueError("the tracking task needs exactly 2 effectors")
        self.network = HomeostaticReservoir(reservoir_config, seed=seed)
        self.env = TrackingEnv(tracking_config)
        self.t = 0

    def step(self, advance_stimulus: bool = True) -> tuple[StepState, float]:
        """Advance one timestep; returns (network state, applied heading change).

        ``advance_stimulus=False`` holds the stimulus in place (and suspends
        its reversal clock) while the network and agent still run — used by
        the visualizer's manual-stimulus mode.
        """
        inputs = self.env.sense()
        state = self.network.step(inputs)
        e_left, e_right = state.outputs
        d_heading = self.env.apply_action(e_left, e_right)
        if advance_stimulus:
            self.env.advance_stimulus()
        self.t += 1
        return state, d_heading

    def run(self, n_steps: int, record_spikes: bool = True) -> History:
        """Run n_steps and record the trajectory."""
        n_nodes = self.network.config.n_nodes
        heading = np.empty(n_steps)
        stimulus_angle = np.empty(n_steps)
        stimulus_direction = np.empty(n_steps, dtype=int)
        stimulus_speed = np.empty(n_steps)
        error = np.empty(n_steps)
        d_heading = np.empty(n_steps)
        outputs = np.empty((n_steps, 2))
        prop_spiked = np.empty(n_steps)
        mean_target = np.empty(n_steps)
        mean_abs_error = np.empty(n_steps)
        spikes = (
            np.zeros((n_steps, n_nodes), dtype=bool)
            if record_spikes
            else np.zeros((0, n_nodes), dtype=bool)
        )

        for i in range(n_steps):
            stimulus_angle[i] = self.env.stimulus_angle
            stimulus_direction[i] = self.env.stimulus_direction
            stimulus_speed[i] = self.env.current_stimulus_speed
            error[i] = self.env.heading_error()
            state, dh = self.step()
            heading[i] = self.env.heading
            d_heading[i] = dh
            outputs[i] = state.outputs
            prop_spiked[i] = state.prop_spiked
            mean_target[i] = float(np.mean(state.targets))
            mean_abs_error[i] = float(np.mean(np.abs(state.error)))
            if record_spikes:
                spikes[i] = state.spiked

        return History(
            heading=heading,
            stimulus_angle=stimulus_angle,
            stimulus_direction=stimulus_direction,
            stimulus_speed=stimulus_speed,
            error=error,
            d_heading=d_heading,
            outputs=outputs,
            prop_spiked=prop_spiked,
            mean_target=mean_target,
            mean_abs_error=mean_abs_error,
            spikes=spikes,
        )


def run_tracking(
    n_steps: int = 7200,
    seed: int | None = None,
    learning_enabled: bool = True,
    reservoir_config: ReservoirConfig = ReservoirConfig(),
    tracking_config: TrackingConfig = TrackingConfig(),
    record_spikes: bool = True,
) -> History:
    """Convenience one-shot run of the tracking experiment."""
    sim = TrackingSimulation(reservoir_config, tracking_config, seed=seed)
    sim.network.learning_enabled = learning_enabled
    return sim.run(n_steps, record_spikes=record_spikes)


# Network parameters for case study 2, from the released Pong scripts. Note
# target_lr = 0.1, ten times both the paper's stated 0.01 and the tracking
# code's value, and the per-synapse inhibitory weight draw.
PONG_RESERVOIR_CONFIG = ReservoirConfig(
    n_nodes=500,
    n_inputs=46,
    p_link=0.1,
    input_weight=2.75,
    leak=0.25,
    target_lr=0.1,
    weight_init_mean=0.0,
    weight_init_sd=0.2,
    inhibitory_fraction=0.25,
    inhibitory_weight_mean=-1.0,
    inhibitory_weight_sd=0.1,
)


@dataclass
class PongHistory:
    """Per-step records of a Pong run, plus the hit/miss sequence.

    ``hits`` is the ordered outcome of each scoring opportunity (1 = the
    paddle intercepted the ball, 0 = it got past and reached the left edge);
    its mean is the headline hit rate. The per-step arrays are empty when a
    run was made with ``record=False``.
    """

    hits: np.ndarray             # (n_opportunities,) 1 = hit, 0 = miss
    ball_x: np.ndarray
    ball_y: np.ndarray
    paddle_y: np.ndarray
    d_paddle: np.ndarray         # paddle movement applied this step
    outputs: np.ndarray          # (n_steps, 2) up/down effector activations
    prop_spiked: np.ndarray
    mean_target: np.ndarray
    event: np.ndarray            # per step: +1 hit, -1 miss, 0 otherwise
    spikes: np.ndarray           # (n_steps, N) bool raster (optional)

    @property
    def hit_rate(self) -> float:
        return float(np.mean(self.hits)) if len(self.hits) else float("nan")

    @property
    def n_opportunities(self) -> int:
        return len(self.hits)


class PongSimulation:
    """A reservoir-controlled paddle in the Pong environment."""

    def __init__(
        self,
        reservoir_config: ReservoirConfig = PONG_RESERVOIR_CONFIG,
        pong_config: PongConfig = PongConfig(),
        seed: int | None = None,
    ):
        if reservoir_config.n_inputs != pong_config.n_sensors:
            raise ValueError(
                f"reservoir expects {reservoir_config.n_inputs} inputs but the "
                f"environment has {pong_config.n_sensors} sensors"
            )
        if reservoir_config.n_outputs != 2:
            raise ValueError("the Pong task needs exactly 2 effectors")
        self.network = HomeostaticReservoir(reservoir_config, seed=seed)
        self.env = PongEnv(pong_config, seed=seed)
        self.t = 0

    def step(self) -> tuple[StepState, str | None, float]:
        """Advance one timestep.

        Returns (network state, "hit" / "miss" / None, paddle movement).
        """
        inputs = self.env.sense()
        state = self.network.step(inputs)
        up, down = state.outputs
        event = self.env.step_ball()          # ball moves first (see module docstring)
        d_paddle = self.env.apply_action(up, down)
        self.t += 1
        return state, event, d_paddle

    def run(
        self, n_steps: int, record: bool = True, record_spikes: bool = False
    ) -> PongHistory:
        """Run n_steps. ``record=False`` keeps only the hit/miss sequence."""
        n_nodes = self.network.config.n_nodes
        empty = np.zeros(0)
        if record:
            ball_x = np.empty(n_steps)
            ball_y = np.empty(n_steps)
            paddle_y = np.empty(n_steps)
            d_paddle = np.empty(n_steps)
            outputs = np.empty((n_steps, 2))
            prop_spiked = np.empty(n_steps)
            mean_target = np.empty(n_steps)
            event = np.zeros(n_steps, dtype=int)
        spikes = (
            np.zeros((n_steps, n_nodes), dtype=bool)
            if record_spikes
            else np.zeros((0, n_nodes), dtype=bool)
        )

        for i in range(n_steps):
            state, ev, dp = self.step()
            if record:
                ball_x[i] = self.env.ball_x
                ball_y[i] = self.env.ball_y
                paddle_y[i] = self.env.paddle_y
                d_paddle[i] = dp
                outputs[i] = state.outputs
                prop_spiked[i] = state.prop_spiked
                mean_target[i] = float(np.mean(state.targets))
                event[i] = 1 if ev == "hit" else (-1 if ev == "miss" else 0)
            if record_spikes:
                spikes[i] = state.spiked

        return PongHistory(
            hits=np.array(self.env.hits, dtype=int),
            ball_x=ball_x if record else empty,
            ball_y=ball_y if record else empty,
            paddle_y=paddle_y if record else empty,
            d_paddle=d_paddle if record else empty,
            outputs=outputs if record else np.zeros((0, 2)),
            prop_spiked=prop_spiked if record else empty,
            mean_target=mean_target if record else empty,
            event=event if record else np.zeros(0, dtype=int),
            spikes=spikes,
        )


def run_pong(
    n_steps: int = 100_000,
    seed: int | None = None,
    learning_enabled: bool = True,
    reservoir_config: ReservoirConfig = PONG_RESERVOIR_CONFIG,
    pong_config: PongConfig = PongConfig(),
    record: bool = False,
    record_spikes: bool = False,
) -> PongHistory:
    """Convenience one-shot run of the Pong experiment."""
    sim = PongSimulation(reservoir_config, pong_config, seed=seed)
    sim.network.learning_enabled = learning_enabled
    return sim.run(n_steps, record=record, record_spikes=record_spikes)
