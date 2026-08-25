"""Case study 2 from Falandays et al. (2024): playing Pong.

A paddle at a fixed x-position moves up and down in a 1000 x 500 pixel field,
trying to intercept a ball that travels at 5 px/step in each axis. The ball
bounces off the top, bottom, and right walls and off the paddle; if it passes
the paddle and reaches the left edge it counts as a miss and is re-served from
the right at a random height. The paddle is 100 px tall in a 500 px field, so
chance performance is 20%.

Sensor encodings
----------------
Egocentric (the published condition): 46 sensors spaced 4 degrees apart from
-90 to +90, encoding the angle of the ball *relative to the paddle*. A sensor
reads 1 when the ball's angle is within 2 degrees of the sensor's angle, else
0 (binary, not the Gaussian tuning used in case study 1). Because the sensors
only span the right half-plane, the paddle is blind to a ball behind it.

The 46-sensor grid straddles zero (..., -6, -2, +2, +6, ...): with an even
number of sensors spanning +/-90 there is no sensor pointing straight ahead.
The released egocentric code tests ``< 2`` strictly, so a ball at an exact
multiple of 4 degrees -- most importantly 0, straight ahead -- lands 2 degrees
from both neighbours and activates *nothing*. That blind spot fires on roughly
0.5% of steps, and every occurrence is at exactly 0 degrees, because the
paddle clamps at y = 50/450 and the ball's height moves in multiples of 5: the
network goes blind exactly when a parked paddle is level with the ball.

``sensor_inclusive`` (the default here) tests ``<= 2`` instead, so both
neighbours fire at those angles and coverage is gap-free; every other angle
still activates exactly one sensor. This matches the convention the same
authors used in their *allocentric* variant (``<= 5``), which is why we treat
the strict test as an oversight rather than a design choice. Use
:meth:`PongConfig.published` for bit-exact replication of the paper's
condition; the two score within noise of each other (see the README).

Allocentric (the control condition): 50 sensors spaced 10 px apart from y=5 to
y=495, encoding the ball's absolute height, active when the ball is within 5
px (inclusive). Under this encoding the paddle's own motion does not change
the sensory input, and the paper reports performance collapsing to near
chance.

Fidelity notes (released code vs. paper text; see README table)
--------------------------------------------------------------
- The ball is a point for collision purposes: the stated 15 px radius never
  enters the physics, which intersects the ball's path segment with the
  paddle's line segment.
- A miss is recorded when the ball reaches x <= 0 -- 100 px *past* the paddle,
  20 steps later -- not at the moment it passes the paddle.
- The wall/miss tests are an if/elseif chain, so at most one of them fires per
  step; a paddle bounce is tested separately and can combine with them.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

__all__ = ["PongConfig", "PongEnv"]


@dataclass(frozen=True)
class PongConfig:
    """Parameters of the Pong task (defaults = released code, case study 2)."""

    width: float = 1000.0
    height: float = 500.0
    paddle_x: float = 100.0
    paddle_half_height: float = 50.0   # paddle_radius: paddle is 100 px tall
    paddle_start_y: float = 250.0
    ball_start_x: float = 500.0
    ball_start_y: float = 250.0
    ball_speed_x: float = 5.0
    ball_speed_y: float = 5.0
    gain: float = 100.0                # movement_amp
    hit_push: float = 5.0              # extra x-offset applied on a paddle bounce
    # Field limits used by the bounce/miss tests.
    x_bounce: float = 995.0            # right wall
    x_miss: float = 0.0                # ball at or past this counts as a miss
    y_max: float = 495.0
    y_min: float = 5.0
    reserve_x: float = 995.0           # x the ball is re-served from
    reserve_y_low: float = 1.0         # re-serve height ~ Uniform(low, high)
    reserve_y_high: float = 499.0
    # Sensor array: values from sensor_min to sensor_max in sensor_step
    # increments; a sensor fires when |sensor value - measurement| is within
    # sensor_tolerance (strictly, unless sensor_inclusive).
    sensor_mode: str = "egocentric"    # "egocentric" | "allocentric"
    sensor_min: float = -90.0
    sensor_max: float = 90.0
    sensor_step: float = 4.0
    sensor_tolerance: float = 2.0
    sensor_inclusive: bool = True      # see the module docstring; False = published

    def __post_init__(self) -> None:
        if self.sensor_mode not in ("egocentric", "allocentric"):
            raise ValueError(f"unknown sensor_mode {self.sensor_mode!r}")

    @classmethod
    def published(cls, **overrides) -> "PongConfig":
        """The egocentric condition exactly as released, strict ``< 2`` test.

        Reproduces the paper's blind spot at 0 degrees; use this for
        replication, and the default constructor for new work.
        """
        params = dict(sensor_inclusive=False)
        params.update(overrides)
        return cls(**params)

    @classmethod
    def allocentric(cls, **overrides) -> "PongConfig":
        """The paper's control condition: 50 sensors on the ball's y-position."""
        params = dict(
            sensor_mode="allocentric",
            sensor_min=5.0,
            sensor_max=495.0,
            sensor_step=10.0,
            sensor_tolerance=5.0,
            sensor_inclusive=True,
        )
        params.update(overrides)
        return cls(**params)

    @property
    def sensor_values(self) -> np.ndarray:
        """Sensor tuning values: degrees (egocentric) or pixels (allocentric)."""
        n = int(round((self.sensor_max - self.sensor_min) / self.sensor_step)) + 1
        return self.sensor_min + self.sensor_step * np.arange(n)

    @property
    def n_sensors(self) -> int:
        return len(self.sensor_values)

    @property
    def chance_hit_rate(self) -> float:
        """Fraction of the field the paddle covers (0.2 for the defaults)."""
        return 2.0 * self.paddle_half_height / self.height


class PongEnv:
    """The Pong field, ball, and paddle.

    Unlike the tracking environment this one is stochastic (the ball's initial
    y-direction, and its height and direction after each miss), so it carries
    its own generator. It is seeded from a stream derived from ``seed`` and
    independent of the network's, keeping runs reproducible without coupling
    ball trajectories to network structure.
    """

    def __init__(self, config: PongConfig = PongConfig(), seed: int | None = None):
        self.config = config
        self.rng = np.random.default_rng(None if seed is None else [seed, 0xB0A7])
        self.paddle_y = config.paddle_start_y
        self.ball_x = config.ball_start_x
        self.ball_y = config.ball_start_y
        self.dx = -config.ball_speed_x            # served leftward, toward the paddle
        self.dy = float(self.rng.choice([-config.ball_speed_y, config.ball_speed_y]))
        self.hits: list[int] = []                 # 1 per paddle bounce, 0 per miss
        self._sensor_values = config.sensor_values

    # -- sensing ------------------------------------------------------------

    def ball_angle(self) -> float:
        """Angle of the ball relative to the paddle, in degrees (-180, 180]."""
        return float(
            np.degrees(np.arctan2(self.ball_y - self.paddle_y, self.ball_x - self.config.paddle_x))
        )

    def sense(self) -> np.ndarray:
        """Binary sensor activations for the current configuration."""
        c = self.config
        measurement = self.ball_angle() if c.sensor_mode == "egocentric" else self.ball_y
        distance = np.abs(self._sensor_values - measurement)
        active = distance <= c.sensor_tolerance if c.sensor_inclusive else distance < c.sensor_tolerance
        return active.astype(float)

    # -- dynamics -----------------------------------------------------------

    def _paddle_intersection(self, new_x: float, new_y: float) -> float | None:
        """Fraction along the ball's path at which it crosses the paddle, or None.

        The ball's path this step is the segment (ball -> new position); the
        paddle is the vertical segment of half-height ``paddle_half_height``
        at ``paddle_x``.
        """
        c = self.config
        if self.dx == 0.0:
            return None
        t = (c.paddle_x - self.ball_x) / self.dx
        if not 0.0 <= t <= 1.0:
            return None
        y_at_paddle = self.ball_y + t * self.dy
        if abs(y_at_paddle - self.paddle_y) > c.paddle_half_height:
            return None
        return t

    def step_ball(self) -> str | None:
        """Advance the ball one step. Returns "hit", "miss", or None.

        Collisions are resolved against the paddle's *current* position: the
        released code moves the ball before the paddle each step.
        """
        c = self.config
        new_x = self.ball_x + self.dx
        new_y = self.ball_y + self.dy
        event: str | None = None

        t = self._paddle_intersection(new_x, new_y)
        if t is not None:
            # Push the ball back out by the distance it travelled past the
            # paddle, plus a fixed nudge, and reverse its x-direction.
            step_length = float(np.hypot(self.dx, self.dy))
            new_x += (1.0 - t) * step_length + c.hit_push
            self.dx *= -1.0
            self.hits.append(1)
            event = "hit"

        if new_x >= c.x_bounce:
            new_x = c.x_bounce
            self.dx *= -1.0
        elif new_x <= c.x_miss:
            new_x = c.reserve_x
            new_y = float(self.rng.uniform(c.reserve_y_low, c.reserve_y_high))
            self.dx = -c.ball_speed_x
            self.dy = float(self.rng.choice([-c.ball_speed_y, c.ball_speed_y]))
            self.hits.append(0)
            event = "miss"
        elif new_y >= c.y_max:
            new_y = c.y_max
            self.dy *= -1.0
        elif new_y <= c.y_min:
            new_y = c.y_min
            self.dy *= -1.0

        self.ball_x = new_x
        self.ball_y = new_y
        return event

    def apply_action(self, up: float, down: float) -> float:
        """Move the paddle by gain * (up - down) px, clamped to the field.

        Returns the change actually applied to the paddle's position.
        """
        c = self.config
        delta = c.gain * (up - down)
        new_y = min(max(self.paddle_y + delta, c.paddle_half_height),
                    c.height - c.paddle_half_height)
        applied = new_y - self.paddle_y
        self.paddle_y = new_y
        return applied

    # -- metrics ------------------------------------------------------------

    @property
    def n_opportunities(self) -> int:
        return len(self.hits)

    @property
    def hit_rate(self) -> float:
        return float(np.mean(self.hits)) if self.hits else float("nan")
