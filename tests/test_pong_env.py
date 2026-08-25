"""Tests of the Pong environment: sensor encodings, ball physics, collision
resolution, scoring, and the paddle mapping, against hand-computed values
taken from the released Julia code."""

import numpy as np
import pytest

from homeostasis import PongConfig, PongEnv


def env_at(ball_xy, paddle_y=250.0, dxy=(-5.0, 5.0), config=None, seed=0):
    """An environment placed in an exact state, bypassing the random serve."""
    env = PongEnv(config or PongConfig(), seed=seed)
    env.ball_x, env.ball_y = ball_xy
    env.paddle_y = paddle_y
    env.dx, env.dy = dxy
    return env


# ---------------------------------------------------------------------------
# Configuration and sensor layout
# ---------------------------------------------------------------------------


class TestSensorLayout:
    def test_egocentric_array(self):
        c = PongConfig()
        assert c.n_sensors == 46
        assert c.sensor_values[0] == -90.0
        assert c.sensor_values[-1] == 90.0
        assert np.allclose(np.diff(c.sensor_values), 4.0)

    def test_allocentric_array(self):
        c = PongConfig.allocentric()
        assert c.n_sensors == 50
        assert c.sensor_values[0] == 5.0
        assert c.sensor_values[-1] == 495.0
        assert np.allclose(np.diff(c.sensor_values), 10.0)

    def test_chance_rate_is_twenty_percent(self):
        assert PongConfig().chance_hit_rate == pytest.approx(0.2)

    def test_bad_mode_rejected(self):
        with pytest.raises(ValueError):
            PongConfig(sensor_mode="retinal")


class TestEgocentricSensing:
    def test_sensor_grid_straddles_zero(self):
        # collect(-90:4:90) yields ..., -6, -2, 2, 6, ...: there is no sensor
        # pointing straight ahead.
        values = PongConfig().sensor_values
        assert 0.0 not in values
        assert -2.0 in values and 2.0 in values

    def test_published_config_is_blind_straight_ahead(self):
        # The released strict `< 2` test: a ball level with the paddle sits 2
        # degrees from both neighbours, so the network receives no input at
        # all. Preserved under PongConfig.published() for replication.
        env = env_at((500.0, 250.0), paddle_y=250.0, config=PongConfig.published())
        assert env.ball_angle() == pytest.approx(0.0)
        assert env.sense().sum() == 0.0

    def test_default_config_closes_the_blind_spot(self):
        # Our default uses `<= 2`, so both bracketing sensors fire straight
        # ahead instead of neither.
        env = env_at((500.0, 250.0), paddle_y=250.0)
        assert env.ball_angle() == pytest.approx(0.0)
        acts = env.sense()
        assert set(PongConfig().sensor_values[acts > 0]) == {-2.0, 2.0}

    def test_fix_only_affects_boundary_angles(self):
        # Away from the exact midpoints both variants agree: one sensor fires.
        for ball in [(500.0, 450.0), (400.0, 120.0), (900.0, 300.0)]:
            strict = env_at(ball, paddle_y=250.0, config=PongConfig.published()).sense()
            fixed = env_at(ball, paddle_y=250.0).sense()
            assert strict.sum() == 1.0
            assert np.array_equal(strict, fixed)

    def test_single_sensor_fires_off_axis(self):
        # atan2(200, 400) = 26.565 deg, within 2 of the sensor at 26.
        env = env_at((500.0, 450.0), paddle_y=250.0)
        acts = env.sense()
        assert env.ball_angle() == pytest.approx(np.degrees(np.arctan2(200.0, 400.0)))
        assert acts.sum() == 1.0
        assert PongConfig().sensor_values[int(np.argmax(acts))] == 26.0

    def test_activation_is_binary(self):
        env = env_at((500.0, 300.0), paddle_y=250.0)
        acts = env.sense()
        assert set(np.unique(acts)) <= {0.0, 1.0}

    def test_angle_sign_convention(self):
        # Ball above the paddle -> positive angle, below -> negative.
        assert env_at((500.0, 650.0), paddle_y=250.0).ball_angle() == pytest.approx(45.0)
        assert env_at((500.0, -150.0), paddle_y=250.0).ball_angle() == pytest.approx(-45.0)

    def test_tolerance_is_strict(self):
        # The egocentric encoding tests `< tolerance`, not `<=`. Checked on a
        # pixel-valued array so the boundary arithmetic is exact (an angle
        # built from tan/atan2 lands a few ulps off the boundary).
        strict = PongConfig(
            sensor_mode="allocentric", sensor_min=5.0, sensor_max=495.0,
            sensor_step=10.0, sensor_tolerance=5.0, sensor_inclusive=False,
        )
        env = env_at((500.0, 260.0), config=strict)
        assert env.sense().sum() == 0.0  # |255-260| = |265-260| = 5, not < 5

    def test_blind_behind_the_paddle(self):
        # Sensors span only the right half-plane.
        env = env_at((50.0, 250.0), paddle_y=250.0)
        assert abs(env.ball_angle()) == pytest.approx(180.0)
        assert env.sense().sum() == 0.0

    def test_paddle_motion_changes_egocentric_input(self):
        env = env_at((500.0, 300.0), paddle_y=250.0)
        before = env.sense()
        env.paddle_y = 300.0
        assert not np.array_equal(before, env.sense())


class TestAllocentricSensing:
    def test_fires_on_ball_height(self):
        env = env_at((500.0, 255.0), config=PongConfig.allocentric())
        acts = env.sense()
        values = PongConfig.allocentric().sensor_values
        assert acts.sum() == 1.0
        assert values[int(np.argmax(acts))] == 255.0

    def test_tolerance_is_inclusive(self):
        # |250 - 255| = 5 and |260 - 255| = 5 both count (the code tests `<= 5`).
        env = env_at((500.0, 260.0), config=PongConfig.allocentric())
        env.ball_y = 260.0
        acts = env.sense()
        values = PongConfig.allocentric().sensor_values
        assert set(values[acts > 0]) == {255.0, 265.0}

    def test_paddle_motion_does_not_change_input(self):
        cfg = PongConfig.allocentric()
        env = env_at((500.0, 300.0), paddle_y=250.0, config=cfg)
        before = env.sense()
        env.paddle_y = 450.0
        assert np.array_equal(before, env.sense())


# ---------------------------------------------------------------------------
# Ball physics
# ---------------------------------------------------------------------------


class TestBallMotion:
    def test_free_flight(self):
        env = env_at((500.0, 250.0), dxy=(-5.0, 5.0))
        assert env.step_ball() is None
        assert (env.ball_x, env.ball_y) == pytest.approx((495.0, 255.0))

    def test_top_wall_bounce(self):
        env = env_at((500.0, 493.0), dxy=(-5.0, 5.0))
        assert env.step_ball() is None
        assert env.ball_y == pytest.approx(495.0)
        assert env.dy == pytest.approx(-5.0)

    def test_bottom_wall_bounce(self):
        env = env_at((500.0, 7.0), dxy=(-5.0, -5.0))
        env.step_ball()
        assert env.ball_y == pytest.approx(5.0)
        assert env.dy == pytest.approx(5.0)

    def test_right_wall_bounce(self):
        env = env_at((993.0, 250.0), dxy=(5.0, 5.0))
        env.step_ball()
        assert env.ball_x == pytest.approx(995.0)
        assert env.dx == pytest.approx(-5.0)

    def test_wall_tests_are_mutually_exclusive(self):
        # x and y bounces are an if/elseif chain: only the x bounce fires.
        env = env_at((993.0, 493.0), dxy=(5.0, 5.0))
        env.step_ball()
        assert env.ball_x == pytest.approx(995.0)
        assert env.dx == pytest.approx(-5.0)
        assert env.ball_y == pytest.approx(498.0)  # not clamped this step
        assert env.dy == pytest.approx(5.0)


class TestCollision:
    def test_hit_when_crossing_paddle_within_reach(self):
        env = env_at((102.0, 250.0), paddle_y=250.0, dxy=(-5.0, 5.0))
        assert env.step_ball() == "hit"
        assert env.dx == pytest.approx(5.0)
        assert env.hits == [1]

    def test_hit_pushes_ball_forward_of_the_paddle(self):
        # t = (100 - 102) / -5 = 0.4; step length = hypot(5, 5);
        # new_x = 97 + 0.6 * 7.0711 + 5 = 106.2426.
        env = env_at((102.0, 250.0), paddle_y=250.0, dxy=(-5.0, 5.0))
        env.step_ball()
        assert env.ball_x == pytest.approx(97.0 + 0.6 * np.hypot(5.0, 5.0) + 5.0)
        assert env.ball_y == pytest.approx(255.0)  # y is untouched by a bounce

    def test_miss_when_ball_passes_above_the_paddle(self):
        # Paddle spans 200..300; the ball crosses x=100 at y=460.
        env = env_at((102.0, 455.0), paddle_y=250.0, dxy=(-5.0, 5.0))
        assert env.step_ball() is None
        assert env.dx == pytest.approx(-5.0)
        assert env.hits == []

    def test_edge_of_paddle_counts_as_hit(self):
        # Crossing exactly at paddle_y + half height (300.0).
        env = env_at((102.0, 299.2), paddle_y=250.0, dxy=(-5.0, 2.0))
        assert env.step_ball() == "hit"

    def test_just_past_edge_is_not_a_hit(self):
        env = env_at((102.0, 299.4), paddle_y=250.0, dxy=(-5.0, 2.0))
        assert env.step_ball() is None

    def test_no_collision_when_path_does_not_reach_paddle(self):
        env = env_at((130.0, 250.0), paddle_y=250.0, dxy=(-5.0, 0.0))
        assert env.step_ball() is None
        assert env.ball_x == pytest.approx(125.0)

    def test_collision_uses_current_paddle_position(self):
        # Same ball state, two paddle positions: one reaches, one does not.
        assert env_at((102.0, 250.0), paddle_y=250.0).step_ball() == "hit"
        assert env_at((102.0, 250.0), paddle_y=450.0).step_ball() is None


class TestScoring:
    def test_miss_recorded_at_left_edge_and_reserves(self):
        env = env_at((3.0, 250.0), dxy=(-5.0, 5.0), seed=3)
        assert env.step_ball() == "miss"
        assert env.hits == [0]
        assert env.ball_x == pytest.approx(995.0)
        assert 1.0 <= env.ball_y <= 499.0
        assert env.dx == pytest.approx(-5.0)
        assert abs(env.dy) == pytest.approx(5.0)

    def test_no_score_between_paddle_and_left_edge(self):
        # Past the paddle but not yet at the edge: no opportunity recorded.
        env = env_at((60.0, 250.0), dxy=(-5.0, 0.0))
        assert env.step_ball() is None
        assert env.hits == []

    def test_hit_rate_and_opportunities(self):
        env = env_at((500.0, 250.0))
        env.hits = [1, 0, 1, 1]
        assert env.hit_rate == pytest.approx(0.75)
        assert env.n_opportunities == 4


class TestPaddle:
    def test_up_and_down(self):
        env = env_at((500.0, 250.0))
        assert env.apply_action(1.0, 0.0) == pytest.approx(100.0)
        assert env.paddle_y == pytest.approx(350.0)
        assert env.apply_action(0.0, 1.0) == pytest.approx(-100.0)
        assert env.paddle_y == pytest.approx(250.0)

    def test_balanced_effectors_do_not_move(self):
        env = env_at((500.0, 250.0))
        assert env.apply_action(0.4, 0.4) == pytest.approx(0.0)

    def test_clamped_to_field(self):
        env = env_at((500.0, 250.0), paddle_y=420.0)
        env.apply_action(1.0, 0.0)
        assert env.paddle_y == pytest.approx(450.0)  # 500 - half height
        env.paddle_y = 80.0
        env.apply_action(0.0, 1.0)
        assert env.paddle_y == pytest.approx(50.0)

    def test_applied_delta_reflects_clamping(self):
        env = env_at((500.0, 250.0), paddle_y=400.0)
        assert env.apply_action(1.0, 0.0) == pytest.approx(50.0)


class TestReproducibility:
    def test_same_seed_same_serves(self):
        a, b = PongEnv(seed=11), PongEnv(seed=11)
        assert a.dy == b.dy
        for _ in range(2000):
            a.step_ball()
            b.step_ball()
        assert (a.ball_x, a.ball_y) == (b.ball_x, b.ball_y)
        assert a.hits == b.hits

    def test_env_stream_independent_of_network(self):
        # The environment must not consume the network's generator.
        from homeostasis import PONG_RESERVOIR_CONFIG, HomeostaticReservoir, PongSimulation

        sim = PongSimulation(seed=5)
        solo = HomeostaticReservoir(PONG_RESERVOIR_CONFIG, seed=5)
        assert np.array_equal(sim.network.weights, solo.weights)
