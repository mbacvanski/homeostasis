"""Case study 1 from Falandays et al. (2024): moving-object tracking.

The agent sits at a fixed position and can only rotate. A single stimulus
moves along a circle of radius 1 around the agent at 1 degree/step, reversing
direction every 720 steps (two full rotations). The agent has two "eyes" at
+30 and -30 degrees relative to its heading; each eye is an arc of 31 sensors
spaced 4 degrees apart spanning +/-60 degrees around the eye's center (so each
eye has a 120-degree field of view, the two overlap by 60 degrees in the
middle, and together they span 180 degrees). Sensor activation is a Gaussian
of the angular distance between the sensor's direction and the stimulus
(eq. 6): ``i = exp(-theta^2 / 10)`` with theta in degrees — except that the
released code additionally sets activation to exactly 1 for any sensor within
``plateau_width`` (4) degrees of the stimulus (``sens_acts[sens_dists .<= 4]
.= 1``). With 4-degree sensor spacing this means the two grid sensors
bracketing the stimulus in each eye read 1.0 at all times (visible as the
full-height bars in the paper's Fig. 3), which roughly doubles the sensory
drive relative to the paper's eq. 6 alone. We follow the code.

Angle conventions: all angles are in degrees, counter-clockwise positive
(standard math convention; 0 = east, 90 = north), stored wrapped to [0, 360).
A positive heading change is a left turn, matching the paper's effector gain
equation (eq. 7): ``dH = gain * (e_left - e_right)``, where e_left = 1 and
e_right = 0 turns the agent 10 degrees left.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

__all__ = ["TrackingConfig", "TrackingEnv", "wrap_angle", "angular_difference"]


def wrap_angle(angle):
    """Wrap angle(s) to [0, 360)."""
    return np.asarray(angle) % 360.0 if isinstance(angle, np.ndarray) else angle % 360.0


def angular_difference(a, b):
    """Signed minimal angular difference a - b, wrapped to [-180, 180)."""
    return (np.asarray(a) - np.asarray(b) + 180.0) % 360.0 - 180.0


@dataclass(frozen=True)
class TrackingConfig:
    """Parameters of the tracking task (defaults = paper, case study 1)."""

    eye_offsets: tuple[float, ...] = (30.0, -30.0)  # left eye, right eye
    sensors_per_eye: int = 31
    sensor_spacing: float = 4.0        # degrees between adjacent sensors
    tuning_width: float = 10.0         # denominator in exp(-theta^2 / width)
    plateau_width: float = 4.0         # |theta| <= this reads exactly 1.0
    gain: float = 10.0                 # effector gain (eq. 7)
    stimulus_speed: float = 1.0        # degrees per step
    reverse_every: int = 720           # steps between stimulus direction flips
    stimulus_radius: float = 1.0       # only matters for display
    initial_heading: float = 90.0      # agent starts facing north
    initial_stimulus_angle: float = 0.0  # stimulus starts east
    initial_stimulus_direction: int = 1  # +1 = counter-clockwise

    @property
    def n_sensors(self) -> int:
        return self.sensors_per_eye * len(self.eye_offsets)

    @property
    def sensor_offsets(self) -> np.ndarray:
        """Direction of every sensor relative to the agent's heading, in
        degrees, ordered eye by eye (left eye first with default offsets).

        Each eye's sensors span +/- (sensors_per_eye - 1)/2 * spacing around
        the eye's center: with the defaults, -60..+60 in steps of 4.
        """
        half_span = (self.sensors_per_eye - 1) / 2.0 * self.sensor_spacing
        within_eye = np.linspace(-half_span, half_span, self.sensors_per_eye)
        return np.concatenate([eye + within_eye for eye in self.eye_offsets])


class TrackingEnv:
    """The rotating agent and orbiting stimulus.

    The environment is deterministic; all randomness in the model lives in the
    network. Call :meth:`sense` to read the sensor array for the current
    configuration, :meth:`apply_action` with the two effector outputs to turn
    the agent, and :meth:`advance_stimulus` to move the stimulus one step
    (handling the periodic direction reversal).
    """

    def __init__(self, config: TrackingConfig = TrackingConfig()):
        self.config = config
        self.heading = config.initial_heading
        self.stimulus_angle = config.initial_stimulus_angle
        self.stimulus_direction = config.initial_stimulus_direction
        self.stimulus_steps = 0  # steps the stimulus has taken (drives reversals)
        self._sensor_offsets = config.sensor_offsets  # cached, (n_sensors,)

    def sensor_angles(self) -> np.ndarray:
        """Absolute direction of each sensor, wrapped to [0, 360)."""
        return (self.heading + self._sensor_offsets) % 360.0

    def sense(self) -> np.ndarray:
        """Sensor activations for the current agent/stimulus configuration:
        Gaussian of angular distance (eq. 6), with the released code's plateau
        of exactly 1.0 within plateau_width degrees."""
        theta = np.abs(angular_difference(self.stimulus_angle, self.sensor_angles()))
        acts = np.exp(-(theta**2) / self.config.tuning_width)
        acts[theta <= self.config.plateau_width] = 1.0
        return acts

    def apply_action(self, e_left: float, e_right: float) -> float:
        """Turn the agent by dH = gain * (e_left - e_right) degrees (eq. 7).

        Positive dH is a left (counter-clockwise) turn. Returns dH.
        """
        d_heading = self.config.gain * (e_left - e_right)
        self.heading = (self.heading + d_heading) % 360.0
        return d_heading

    def advance_stimulus(self) -> None:
        """Move the stimulus one step; flip direction every reverse_every steps."""
        c = self.config
        self.stimulus_angle = (
            self.stimulus_angle + self.stimulus_direction * c.stimulus_speed
        ) % 360.0
        self.stimulus_steps += 1
        if self.stimulus_steps % c.reverse_every == 0:
            self.stimulus_direction *= -1

    @property
    def current_stimulus_speed(self) -> float:
        """Unsigned speed that will be applied on the next automatic step."""
        return float(self.config.stimulus_speed)

    @property
    def target_stimulus_speed(self) -> float:
        """Speed target; constant motion is always already at its target."""
        return self.current_stimulus_speed

    @property
    def steps_until_speed_change(self) -> int | None:
        """Countdown to a speed retarget, or ``None`` for constant motion."""
        return None

    @property
    def steps_until_direction_change(self) -> int:
        """Number of automatic advances remaining before the next reversal."""
        return self.config.reverse_every - self.stimulus_steps % self.config.reverse_every

    def flip_stimulus_direction(self) -> None:
        """Flip immediately; used by interactive controls."""
        self.stimulus_direction *= -1

    def heading_error(self) -> float:
        """Signed angular distance from heading to stimulus, in [-180, 180)."""
        return float(angular_difference(self.stimulus_angle, self.heading))
