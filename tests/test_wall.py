"""Mechanics tests for the wall-avoidance environment (case study 3).

Hand-computed sensor geometry, the released kinematics (translation from the
pre-rotation heading; omega = (e2 - e1)/(2r)), wall clamp + random kick, the
published sensory perturbation, and seed determinism of the full simulation.
"""

import numpy as np
import pytest

from homeostasis.simulation import WALL_RESERVOIR_CONFIG, WallSimulation, run_wall
from homeostasis.wall import WallConfig, WallEnv


def make_env(x, y, heading_deg, **cfg):
    env = WallEnv(WallConfig(**cfg), rng=np.random.default_rng(0))
    env.x, env.y = x, y
    env.heading = np.deg2rad(heading_deg)
    return env


def test_sensor_geometry_symmetric():
    # Agent at (3, 7.5) facing east: both sensors see the top/bottom walls at
    # the same hand-computed distance 10.10660, act = 1 - d/21.21320 = 0.52357.
    env = make_env(3.0, 7.5, 0.0)
    acts = env.sense()
    assert acts.shape == (2,)
    assert acts[0] == pytest.approx(0.5235702, abs=1e-6)
    assert acts[1] == pytest.approx(0.5235702, abs=1e-6)


def test_sensor_geometry_asymmetric_near_floor():
    # Agent at (2, 3) facing east: left ray exits via the top wall at
    # t = 16.47035 (act 0.22359), right ray hits the floor at t = 3.74246
    # (act 0.82358).
    env = make_env(2.0, 3.0, 0.0)
    acts = env.sense()
    assert acts[0] == pytest.approx(0.2235702, abs=1e-6)
    assert acts[1] == pytest.approx(0.8235702, abs=1e-6)


def test_kinematics_straight_and_turn():
    env = make_env(7.5, 7.5, 90.0)
    dh, hit = env.apply_action(1.0, 1.0)
    assert (env.x, env.y) == (pytest.approx(7.5), pytest.approx(8.5))
    assert dh == 0.0 and not hit

    env = make_env(7.5, 7.5, 90.0)
    dh, hit = env.apply_action(0.0, 1.0)
    # translation uses the OLD heading; rotation of +1 rad applies after
    assert (env.x, env.y) == (pytest.approx(7.5), pytest.approx(8.0))
    assert dh == pytest.approx(1.0)
    assert env.heading == pytest.approx(np.deg2rad(90.0) + 1.0)
    assert not hit


def test_wall_clamp_and_random_kick():
    env = make_env(14.4, 7.5, 0.0)
    heading_before = env.heading
    dh, hit = env.apply_action(1.0, 1.0)
    assert hit and env.hits == 1
    assert env.x == pytest.approx(14.5)  # box_size - radius
    kick = env.heading - heading_before
    assert abs(abs(kick) - np.deg2rad(45.0)) < 1e-12

    # deterministic given the rng seed
    env2 = make_env(14.4, 7.5, 0.0)
    env2.apply_action(1.0, 1.0)
    assert env2.heading == pytest.approx(env.heading)


def test_perturbation_swaps_and_doubles():
    base = make_env(2.0, 3.0, 0.0)
    normal = base.sense()
    env = make_env(2.0, 3.0, 0.0, perturb_at=2, perturb_gain=2.0)
    a0 = env.sense()
    assert np.allclose(a0, normal)
    env.apply_action(0.0, 0.0)  # no motion, steps -> 1
    assert np.allclose(env.sense(), normal)
    env.apply_action(0.0, 0.0)  # steps -> 2 == perturb_at
    perturbed = env.sense()
    assert np.allclose(perturbed, normal[::-1] * 2.0)


def test_simulation_determinism_and_config_guard():
    h1 = run_wall(n_steps=300, seed=5)
    h2 = run_wall(n_steps=300, seed=5)
    assert np.array_equal(h1.x, h2.x)
    assert np.array_equal(h1.hit, h2.hit)
    assert np.array_equal(h1.heading, h2.heading)
    h3 = run_wall(n_steps=300, seed=6)
    assert not np.array_equal(h1.x, h3.x)

    from dataclasses import replace
    with pytest.raises(ValueError):
        WallSimulation(replace(WALL_RESERVOIR_CONFIG, n_inputs=3))


def test_learning_off_saturates():
    # The paper: with updating off, activity goes to a maximum with all nodes
    # spiking, and the agent moves straight and bounces off walls.
    h = run_wall(n_steps=400, seed=0, learning_enabled=False)
    assert h.prop_spiked[-100:].mean() > 0.95
    assert h.hit.sum() > 0
