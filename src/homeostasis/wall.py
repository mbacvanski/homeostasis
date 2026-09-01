"""Case study 3 from Falandays et al. (2024): wall avoidance.

A circular Braitenberg-style agent (radius 0.5) lives in a 15x15 box. Two
sensors at +/-45 degrees from the heading each cast a ray from the agent's
rim and read proximity to the nearest wall: activation = 1 - dist/diagonal
(1 at touch, ~0 at the far corner). Two effectors are wheel speeds: the
agent translates by (e_left + e_right)/2 per step along its heading and
rotates by (e_right_node - e_left_node)/(2 * radius) radians. On contact
with a wall the position is clamped and the heading receives a random
+/-45 degree kick (the environment's only randomness), and a "hit" is
recorded — the case study's outcome measure.

As with the other case studies, where the paper's text and the released
Julia code (reference/original_julia/WallAvoidance/) disagree, this module
follows the code. Discrepancies are tabled in the README; the ones that
matter:

- Input weights: paper says "all input->reservoir weights now set to 2";
  the code sets input_amp = 4 (used for BOTH the input weights and the
  recurrent weight-init mean Normal(4, 0.1)).
- The code defines lrate_wmat = .01 and movement_amp = 10 but uses neither:
  the weight update applies the full error (weight_lr = 1.0) and the motion
  uses raw effector activations.
- The paper's kinematics example says (L=0, R=1) "did not change position";
  the code always translates by (L+R)/2 (here 0.5) while rotating.
- Julia's effector columns: omega = (out[2] - out[1]) / (2r) in 1-based
  indexing, i.e. the SECOND effector node turns the agent counter-clockwise.
- The published main script runs with the sensory perturbation enabled
  (perturb_at = 1000: sensors swapped and doubled from step 1001 on) and
  sensor noise disabled.

The reservoir core is the shared HomeostaticReservoir; this module only adds
the environment. Randomness: wall-kick (and optional sensor-noise) draws
come from the RNG handed to WallEnv — the simulation passes the network's
own generator, whose reservoir-initialization draws have already completed,
so trajectories are fully determined by the seed (the variable-tracking
pattern).
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

__all__ = ["WallConfig", "WallEnv", "ray_wall_distance"]


def ray_wall_distance(px: float, py: float, angle: float, box_size: float) -> float:
    """Distance from (px, py) along `angle` to the nearest face of the box."""
    c, s = np.cos(angle), np.sin(angle)
    ts = []
    if c > 1e-12:
        ts.append((box_size - px) / c)
    elif c < -1e-12:
        ts.append((0.0 - px) / c)
    if s > 1e-12:
        ts.append((box_size - py) / s)
    elif s < -1e-12:
        ts.append((0.0 - py) / s)
    ts = [t for t in ts if t >= 0.0]
    return float(min(ts)) if ts else float(np.sqrt(2.0) * box_size)


@dataclass(frozen=True)
class WallConfig:
    """Parameters of the wall-avoidance task (defaults = released code)."""

    box_size: float = 15.0
    agent_radius: float = 0.5
    sensor_angles: tuple[float, ...] = (45.0, -45.0)  # degrees from heading
    initial_x: float = 7.5
    initial_y: float = 7.5
    initial_heading_deg: float = 90.0   # radians pi/2 in the code
    bounce_turn_deg: float = 45.0       # magnitude of the random kick on contact
    sensor_noise: float = 0.0           # uniform(-noise, +noise) per sensor per step
    # Sensory perturbation (the published run): from steps > perturb_at the two
    # sensor values are SWAPPED and multiplied by perturb_gain. None disables.
    perturb_at: int | None = None
    perturb_gain: float = 2.0
    # Morphological knob (beyond the paper): omega = (e2 - e1) / wheel_base;
    # None = 2 * agent_radius (the released kinematics).
    wheel_base: float | None = None

    @property
    def n_sensors(self) -> int:
        return len(self.sensor_angles)

    @property
    def max_dist(self) -> float:
        return float(np.sqrt(2.0) * self.box_size)


class WallEnv:
    """The box, the embodied agent, and its proximity sensors."""

    def __init__(self, config: WallConfig = WallConfig(), rng: np.random.Generator | None = None):
        self.config = config
        self.rng = rng if rng is not None else np.random.default_rng()
        self.x = config.initial_x
        self.y = config.initial_y
        self.heading = np.deg2rad(config.initial_heading_deg)
        self.steps = 0
        self.hits = 0

    # -- sensing ------------------------------------------------------------

    def _ray_wall_distance(self, px: float, py: float, angle: float) -> float:
        return ray_wall_distance(px, py, angle, self.config.box_size)

    def sense(self) -> np.ndarray:
        """Sensor activations, with the published perturbation and optional
        noise applied. Sensors sit ON the rim at their angles and read
        1 - dist/diagonal toward the nearest wall along their own direction."""
        c = self.config
        acts = np.empty(c.n_sensors)
        for i, off in enumerate(c.sensor_angles):
            ang = self.heading + np.deg2rad(off)
            px = self.x + c.agent_radius * np.cos(ang)
            py = self.y + c.agent_radius * np.sin(ang)
            d = self._ray_wall_distance(px, py, ang)
            acts[i] = 1.0 - d / c.max_dist
        if c.sensor_noise > 0.0:
            acts = acts + self.rng.uniform(-c.sensor_noise, c.sensor_noise, c.n_sensors)
        if c.perturb_at is not None and self.steps >= c.perturb_at:
            acts = acts[::-1] * c.perturb_gain
        return acts

    # -- acting -------------------------------------------------------------

    def apply_action(self, e_first: float, e_second: float) -> tuple[float, bool]:
        """Move per the released kinematics; returns (d_heading_rad, hit).

        e_first / e_second are the two effector readouts in their wiring
        order; the SECOND node minus the FIRST sets the rotation (the Julia
        code's (out[2]-out[1])/(2r)). Translation is the mean speed along
        the (pre-rotation-updated) heading — the code updates heading first,
        but computes the translation direction from the heading BEFORE the
        update (pos_centre uses agent.heading read prior to the omega
        increment ordering in Julia: translation direction is computed
        first, then omega is added). We follow the code's order exactly:
        translation direction from the OLD heading, then rotate.
        """
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
        self.steps += 1
        return float(omega), hit
