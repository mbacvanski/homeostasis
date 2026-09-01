"""Pursuit: following a moving stimulus in a walled 2D arena (beyond the paper).

The mentors' environment-ladder rungs 3-4 ("following an object in 2D",
"following without hitting walls"), built from the family's own parts:

- Body: the wall-avoidance Braitenberg agent (radius 0.5, differential
  drive, translation (e1+e2)/2, rotation (e2-e1)/(2r), wall clamp with a
  random +/-45 degree kick). Walls are INVISIBLE to the sensors here —
  they act purely as motion disruptions.
- Retina: tracking's two-eye bearing array (62 sensors, Gaussian tuning
  with the released code's plateau), aimed at the stimulus's bearing, and
  scaled by an intensity falloff with distance:

      act = bearing_tuning(theta) * 1 / (1 + dist / intensity_scale)

  so approach raises input flow and escape starves it (a flow-positive
  embodiment by design; the walls contribute no sensory sign).
- Stimulus motion: "orbit" (circle around the arena center), "waypoint"
  (straight lines to random targets — unpredictable), or "still".

This is an exploratory task, not a paper replication: there is no released
code to follow. All randomness (stimulus waypoints, wall kicks) draws from
the RNG handed in (the simulation passes the network's generator, so seeds
fully determine trajectories).
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .tracking import angular_difference
from .wall import ray_wall_distance

__all__ = ["PursuitConfig", "PursuitEnv"]


@dataclass(frozen=True)
class PursuitConfig:
    """Parameters of the pursuit task."""

    box_size: float = 15.0
    agent_radius: float = 0.5
    bounce_turn_deg: float = 45.0
    # retina (tracking geometry)
    eye_offsets: tuple[float, ...] = (30.0, -30.0)
    sensors_per_eye: int = 31
    sensor_spacing: float = 4.0
    tuning_width: float = 10.0
    plateau_width: float = 4.0
    # intensity falloff: 1 / (1 + dist / intensity_scale)
    intensity_scale: float = 3.0
    # Optional wall-proximity channel: two extra sensors (the wall task's
    # +/-45 deg rays, activation 1 - dist/diagonal) APPENDED to the retina.
    # This makes the embodiment mixed-sign: stimulus bearing is the
    # flow-positive channel, wall proximity the flow-negative one.
    wall_sensors: bool = False
    # motor geometry: omega = (e2 - e1) / wheel_base; None = 2 * agent_radius
    # (the released wall-avoidance kinematics). Larger wheel_base = gentler
    # turning per step - the morphological knob that matches motor authority
    # to the retina's angular grain.
    wheel_base: float | None = None
    # stimulus
    stimulus_motion: str = "orbit"        # orbit | waypoint | still | wander | ellipse
    ellipse_a: float = 5.0                # ellipse semi-axes (ellipse mode)
    ellipse_b: float = 2.5
    wander_sigma: float = 0.05            # per-step heading diffusion (rad), wander mode
    stimulus_speed: float = 0.15          # arena units per step
    orbit_radius: float = 4.5
    waypoint_margin: float = 2.0          # keep targets this far from walls
    initial_agent_x: float = 7.5
    initial_agent_y: float = 7.5
    initial_heading_deg: float = 90.0

    @property
    def n_sensors(self) -> int:
        return self.sensors_per_eye * len(self.eye_offsets) + (2 if self.wall_sensors else 0)

    @property
    def sensor_offsets(self) -> np.ndarray:
        half = (self.sensors_per_eye - 1) / 2.0 * self.sensor_spacing
        within = np.linspace(-half, half, self.sensors_per_eye)
        return np.concatenate([eye + within for eye in self.eye_offsets])


class PursuitEnv:
    """Walled arena with a moving stimulus and a bearing-plus-intensity retina."""

    def __init__(self, config: PursuitConfig = PursuitConfig(), rng: np.random.Generator | None = None):
        self.config = config
        self.rng = rng if rng is not None else np.random.default_rng()
        c = config
        self.x = c.initial_agent_x
        self.y = c.initial_agent_y
        self.heading = np.deg2rad(c.initial_heading_deg)
        self._offsets = c.sensor_offsets
        self.steps = 0
        self.hits = 0
        center = c.box_size / 2.0
        if c.stimulus_motion == "orbit":
            self._phase = 0.0
            self.sx = center + c.orbit_radius
            self.sy = center
        elif c.stimulus_motion == "wander":
            self.sx = center + 2.0
            self.sy = center
            self._sphi = 0.0
        elif c.stimulus_motion == "ellipse":
            self._phase = 0.0
            self.sx = center + c.ellipse_a
            self.sy = center
        else:
            self.sx = center
            self.sy = center + 3.0
            self._target = self._new_waypoint()

    def _new_waypoint(self):
        c = self.config
        lo, hi = c.waypoint_margin, c.box_size - c.waypoint_margin
        return (float(self.rng.uniform(lo, hi)), float(self.rng.uniform(lo, hi)))

    # -- observation --------------------------------------------------------

    def stimulus_bearing_deg(self) -> float:
        """Bearing of the stimulus relative to the agent's heading, degrees."""
        ang = np.rad2deg(np.arctan2(self.sy - self.y, self.sx - self.x))
        return float(angular_difference(ang, np.rad2deg(self.heading)))

    def distance(self) -> float:
        return float(np.hypot(self.sx - self.x, self.sy - self.y))

    def sense(self) -> np.ndarray:
        c = self.config
        theta = np.abs(angular_difference(self.stimulus_bearing_deg(), self._offsets))
        acts = np.exp(-(theta ** 2) / c.tuning_width)
        acts[theta <= c.plateau_width] = 1.0
        acts = acts / (1.0 + self.distance() / c.intensity_scale)
        if c.wall_sensors:
            diag = np.sqrt(2.0) * c.box_size
            wall = np.empty(2)
            for i, off in enumerate((45.0, -45.0)):
                ang = self.heading + np.deg2rad(off)
                px = self.x + c.agent_radius * np.cos(ang)
                py = self.y + c.agent_radius * np.sin(ang)
                wall[i] = 1.0 - ray_wall_distance(px, py, ang, c.box_size) / diag
            acts = np.concatenate([acts, wall])
        return acts

    # -- dynamics -----------------------------------------------------------

    def apply_action(self, e_first: float, e_second: float) -> tuple[float, bool]:
        """Wall-avoidance kinematics verbatim (translation from old heading)."""
        c = self.config
        vel = (e_first + e_second) / 2.0
        dx = vel * np.cos(self.heading)
        dy = vel * np.sin(self.heading)
        wb = c.wheel_base if c.wheel_base is not None else 2.0 * c.agent_radius
        omega = (e_second - e_first) / wb
        self.heading += omega
        nx, ny = self.x + dx, self.y + dy
        hit = False
        lo, hi = c.agent_radius, c.box_size - c.agent_radius
        if nx > hi:
            nx, hit = hi, True
        if nx < lo:
            nx, hit = lo, True
        if ny > hi:
            ny, hit = hi, True
        if ny < lo:
            ny, hit = lo, True
        if hit:
            kick = float(self.rng.choice((-c.bounce_turn_deg, c.bounce_turn_deg)))
            self.heading += np.deg2rad(kick)
            self.heading = float(np.remainder(self.heading + np.pi, 2 * np.pi) - np.pi)
            self.hits += 1
        self.x, self.y = float(nx), float(ny)
        return float(omega), hit

    def advance_stimulus(self) -> None:
        c = self.config
        if c.stimulus_motion == "still":
            pass
        elif c.stimulus_motion == "orbit":
            center = c.box_size / 2.0
            self._phase += c.stimulus_speed / c.orbit_radius
            self.sx = center + c.orbit_radius * np.cos(self._phase)
            self.sy = center + c.orbit_radius * np.sin(self._phase)
        elif c.stimulus_motion == "ellipse":
            # advance phase for ~constant arc speed
            a, b = c.ellipse_a, c.ellipse_b
            center = c.box_size / 2.0
            r = np.hypot(a * np.sin(self._phase), b * np.cos(self._phase))
            self._phase += c.stimulus_speed / max(r, 1e-6)
            self.sx = center + a * np.cos(self._phase)
            self.sy = center + b * np.sin(self._phase)
        elif c.stimulus_motion == "wander":
            self._sphi += float(self.rng.normal(0.0, c.wander_sigma))
            nx = self.sx + c.stimulus_speed * np.cos(self._sphi)
            ny = self.sy + c.stimulus_speed * np.sin(self._sphi)
            lo, hi = c.waypoint_margin, c.box_size - c.waypoint_margin
            if nx < lo or nx > hi:
                self._sphi = np.pi - self._sphi
                nx = float(np.clip(nx, lo, hi))
            if ny < lo or ny > hi:
                self._sphi = -self._sphi
                ny = float(np.clip(ny, lo, hi))
            self.sx, self.sy = float(nx), float(ny)
        elif c.stimulus_motion == "waypoint":
            tx, ty = self._target
            d = np.hypot(tx - self.sx, ty - self.sy)
            if d < c.stimulus_speed:
                self.sx, self.sy = tx, ty
                self._target = self._new_waypoint()
            else:
                self.sx += c.stimulus_speed * (tx - self.sx) / d
                self.sy += c.stimulus_speed * (ty - self.sy) / d
        else:
            raise ValueError(f"unknown stimulus_motion {c.stimulus_motion!r}")
        self.steps += 1
