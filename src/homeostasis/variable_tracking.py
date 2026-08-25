"""Irregular-motion extension of the moving-object tracking task.

The published task in :mod:`homeostasis.tracking` remains unchanged. This
module adds an opt-in stimulus whose unsigned speed eases toward randomly
retargeted values and whose direction reverses after randomly sampled
intervals. All schedule draws use the reservoir's seeded generator after its
topology and weight initialization, preserving the default task's trajectory.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .reservoir import ReservoirConfig
from .simulation import History, TrackingSimulation
from .tracking import TrackingConfig, TrackingEnv

__all__ = [
    "VariableTrackingConfig",
    "VariableTrackingEnv",
    "VariableTrackingSimulation",
    "run_variable_tracking",
]


@dataclass(frozen=True)
class VariableTrackingConfig(TrackingConfig):
    """Configuration for smooth speed variation and irregular reversals.

    ``stimulus_speed`` inherited from :class:`TrackingConfig` is the initial
    speed. Subsequent targets are sampled uniformly from the configured speed
    range. Each target lasts for a randomly sampled number of steps, while the
    current speed approaches it by ``speed_smoothing`` of the remaining gap per
    step. Direction intervals are sampled independently and inclusively.
    """

    stimulus_speed_min: float = 0.5
    stimulus_speed_max: float = 1.5
    speed_smoothing: float = 0.02
    speed_change_min_steps: int = 180
    speed_change_max_steps: int = 540
    reverse_min_steps: int = 480
    reverse_max_steps: int = 960

    def __post_init__(self) -> None:
        if self.stimulus_speed_min < 0.0:
            raise ValueError("stimulus_speed_min must be non-negative")
        if self.stimulus_speed_max < self.stimulus_speed_min:
            raise ValueError("stimulus_speed_max must be >= stimulus_speed_min")
        if not self.stimulus_speed_min <= self.stimulus_speed <= self.stimulus_speed_max:
            raise ValueError("initial stimulus_speed must lie within the variable speed range")
        if not 0.0 < self.speed_smoothing <= 1.0:
            raise ValueError("speed_smoothing must be in (0, 1]")
        for lower_name, upper_name in (
            ("speed_change_min_steps", "speed_change_max_steps"),
            ("reverse_min_steps", "reverse_max_steps"),
        ):
            lower = getattr(self, lower_name)
            upper = getattr(self, upper_name)
            if lower < 1 or upper < lower:
                raise ValueError(f"{upper_name} must be >= {lower_name} >= 1")


class VariableTrackingEnv(TrackingEnv):
    """Tracking environment with a seeded, piecewise-smooth motion schedule."""

    def __init__(
        self,
        config: VariableTrackingConfig = VariableTrackingConfig(),
        *,
        rng: np.random.Generator,
    ):
        super().__init__(config)
        self.config = config
        self._rng = rng
        self._current_speed = float(config.stimulus_speed)
        self._target_speed = self._draw_speed()
        self._speed_steps_remaining = self._draw_interval(
            config.speed_change_min_steps, config.speed_change_max_steps
        )
        self._direction_steps_remaining = self._draw_interval(
            config.reverse_min_steps, config.reverse_max_steps
        )

    def _draw_speed(self) -> float:
        c = self.config
        if c.stimulus_speed_min == c.stimulus_speed_max:
            return float(c.stimulus_speed_min)
        return float(self._rng.uniform(c.stimulus_speed_min, c.stimulus_speed_max))

    def _draw_interval(self, minimum: int, maximum: int) -> int:
        return int(self._rng.integers(minimum, maximum + 1))

    @property
    def current_stimulus_speed(self) -> float:
        return self._current_speed

    @property
    def target_stimulus_speed(self) -> float:
        return self._target_speed

    @property
    def steps_until_speed_change(self) -> int:
        return self._speed_steps_remaining

    @property
    def steps_until_direction_change(self) -> int:
        return self._direction_steps_remaining

    def advance_stimulus(self) -> None:
        """Move, update the smooth speed schedule, and reverse when scheduled."""
        c = self.config
        self.stimulus_angle = (
            self.stimulus_angle + self.stimulus_direction * self._current_speed
        ) % 360.0
        self.stimulus_steps += 1

        self._speed_steps_remaining -= 1
        if self._speed_steps_remaining == 0:
            self._target_speed = self._draw_speed()
            self._speed_steps_remaining = self._draw_interval(
                c.speed_change_min_steps, c.speed_change_max_steps
            )
        self._current_speed += c.speed_smoothing * (
            self._target_speed - self._current_speed
        )
        self._current_speed = float(
            np.clip(self._current_speed, c.stimulus_speed_min, c.stimulus_speed_max)
        )

        self._direction_steps_remaining -= 1
        if self._direction_steps_remaining == 0:
            self.stimulus_direction *= -1
            self._direction_steps_remaining = self._draw_interval(
                c.reverse_min_steps, c.reverse_max_steps
            )

    def flip_stimulus_direction(self) -> None:
        """Flip immediately and start a fresh irregular reversal interval."""
        self.stimulus_direction *= -1
        self._direction_steps_remaining = self._draw_interval(
            self.config.reverse_min_steps, self.config.reverse_max_steps
        )


class VariableTrackingSimulation(TrackingSimulation):
    """A tracking simulation using :class:`VariableTrackingEnv`."""

    def __init__(
        self,
        reservoir_config: ReservoirConfig = ReservoirConfig(),
        tracking_config: VariableTrackingConfig = VariableTrackingConfig(),
        seed: int | None = None,
    ):
        # The parent constructs the reservoir first. Replacing only the
        # environment lets its schedule continue from that reservoir's seeded
        # generator without changing any reservoir initialization draws.
        super().__init__(reservoir_config, tracking_config, seed=seed)
        self.env = VariableTrackingEnv(tracking_config, rng=self.network.rng)


def run_variable_tracking(
    n_steps: int = 7200,
    seed: int | None = None,
    learning_enabled: bool = True,
    reservoir_config: ReservoirConfig = ReservoirConfig(),
    tracking_config: VariableTrackingConfig = VariableTrackingConfig(),
    record_spikes: bool = True,
) -> History:
    """Run the opt-in irregular-motion tracking experiment."""
    sim = VariableTrackingSimulation(reservoir_config, tracking_config, seed=seed)
    sim.network.learning_enabled = learning_enabled
    return sim.run(n_steps, record_spikes=record_spikes)
