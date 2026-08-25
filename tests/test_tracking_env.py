"""Tests of the tracking environment: sensor geometry (eq. 6), effector
mapping (eq. 7), and stimulus kinematics, all against hand-computed values."""

import numpy as np
import pytest

from homeostasis import TrackingConfig, TrackingEnv, angular_difference


class TestAngles:
    def test_angular_difference_basics(self):
        assert angular_difference(10, 350) == pytest.approx(20)
        assert angular_difference(350, 10) == pytest.approx(-20)
        assert angular_difference(90, 90) == pytest.approx(0)
        assert angular_difference(270, 90) == pytest.approx(-180)  # wraps to -180

    def test_angular_difference_vectorized(self):
        d = angular_difference(np.array([0.0, 359.0]), np.array([359.0, 0.0]))
        assert d == pytest.approx([1.0, -1.0])


class TestSensorGeometry:
    def test_offsets_layout(self):
        cfg = TrackingConfig()
        offsets = cfg.sensor_offsets
        assert cfg.n_sensors == 62
        # Left eye: centered at +30, spanning -30..+90 in steps of 4.
        assert offsets[:31] == pytest.approx(np.arange(-30.0, 91.0, 4.0))
        # Right eye: centered at -30, spanning -90..+30 in steps of 4.
        assert offsets[31:] == pytest.approx(np.arange(-90.0, 31.0, 4.0))

    def test_total_field_of_view(self):
        offsets = TrackingConfig().sensor_offsets
        assert offsets.min() == -90.0 and offsets.max() == 90.0

    def test_sensor_angles_wrap(self):
        env = TrackingEnv()
        env.heading = 350.0
        angles = env.sensor_angles()
        assert np.all((angles >= 0.0) & (angles < 360.0))
        # Left-eye sensor with offset +50 points at 350 + 50 = 40.
        idx = np.argwhere(np.isclose(TrackingConfig().sensor_offsets, 50.0)).ravel()[0]
        assert angles[idx] == pytest.approx(40.0)


class TestSensing:
    def test_plateau_within_four_degrees(self):
        # Released code: any sensor within 4 degrees of the stimulus reads
        # exactly 1.0 (`sens_acts[sens_dists .<= 4] .= 1`). A stimulus on the
        # sensor grid therefore lights the aligned sensor AND its two
        # neighbors (at distance exactly 4) in each eye.
        env = TrackingEnv()
        env.heading = 90.0
        env.stimulus_angle = 92.0  # offset +2: on both eyes' grids
        acts = env.sense()
        plateau = [7, 8, 9, 31 + 22, 31 + 23, 31 + 24]  # sensors at 88/92/96
        assert acts[plateau] == pytest.approx(np.ones(6))
        others = np.ones(62, dtype=bool)
        others[plateau] = False
        assert np.all(acts[others] < 1.0)

    def test_plateau_between_grid_points(self):
        # Stimulus midway between sensors (distance 2 to both): both read 1.0,
        # the next ones out (distance 6) fall on the Gaussian.
        env = TrackingEnv()
        env.heading = 90.0
        env.stimulus_angle = 90.0  # offsets -2 and +2 bracket it
        acts = env.sense()
        assert acts[7] == pytest.approx(1.0)   # sensor at 88, distance 2
        assert acts[8] == pytest.approx(1.0)   # sensor at 92, distance 2
        assert acts[6] == pytest.approx(np.exp(-36.0 / 10.0))  # 84, distance 6
        assert acts[9] == pytest.approx(np.exp(-36.0 / 10.0))  # 96, distance 6

    def test_gaussian_tuning_beyond_plateau(self):
        # eq. 6 applies outside the plateau: theta = 8 -> exp(-6.4).
        env = TrackingEnv()
        env.heading = 90.0
        env.stimulus_angle = 92.0
        acts = env.sense()
        assert acts[6] == pytest.approx(np.exp(-64.0 / 10.0))   # sensor at 84
        assert acts[10] == pytest.approx(np.exp(-64.0 / 10.0))  # sensor at 100

    def test_out_of_view_is_silent(self):
        env = TrackingEnv()
        env.heading = 90.0
        env.stimulus_angle = 270.0  # directly behind
        assert np.all(env.sense() < 1e-100)

    def test_sense_wraps_across_zero(self):
        env = TrackingEnv()
        env.heading = 350.0
        env.stimulus_angle = 40.0  # sensor offset +50 points exactly there
        acts = env.sense()
        idx = np.argwhere(np.isclose(TrackingConfig().sensor_offsets, 50.0)).ravel()[0]
        assert acts[idx] == pytest.approx(1.0)


class TestEffectorMapping:
    def test_left_turn(self):
        env = TrackingEnv()
        dh = env.apply_action(1.0, 0.0)
        assert dh == pytest.approx(10.0)
        assert env.heading == pytest.approx(100.0)

    def test_right_turn(self):
        env = TrackingEnv()
        dh = env.apply_action(0.0, 1.0)
        assert dh == pytest.approx(-10.0)
        assert env.heading == pytest.approx(80.0)

    def test_balanced_effectors_no_turn(self):
        env = TrackingEnv()
        env.apply_action(0.6, 0.6)
        assert env.heading == pytest.approx(90.0)

    def test_heading_wraps(self):
        env = TrackingEnv()
        env.heading = 355.0
        env.apply_action(1.0, 0.0)
        assert env.heading == pytest.approx(5.0)


class TestStimulusKinematics:
    def test_speed_and_wrap(self):
        env = TrackingEnv()
        for _ in range(5):
            env.advance_stimulus()
        assert env.stimulus_angle == pytest.approx(5.0)
        env.stimulus_angle = 359.0
        env.advance_stimulus()
        assert env.stimulus_angle == pytest.approx(0.0)

    def test_direction_reversal_every_720(self):
        env = TrackingEnv()
        directions = []
        for _ in range(1441):
            directions.append(env.stimulus_direction)
            env.advance_stimulus()
        assert all(d == 1 for d in directions[:720])
        assert all(d == -1 for d in directions[720:1440])
        assert env.stimulus_direction == 1  # flipped back at step 1440


class TestHeadingError:
    def test_sign_convention(self):
        # Stimulus counter-clockwise (left) of heading => positive error,
        # matching positive dH = left turn.
        env = TrackingEnv()
        env.heading = 90.0
        env.stimulus_angle = 100.0
        assert env.heading_error() == pytest.approx(10.0)
        env.stimulus_angle = 80.0
        assert env.heading_error() == pytest.approx(-10.0)
