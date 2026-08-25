"""Tests for the opt-in irregular-motion tracking extension."""

import numpy as np
import pytest

from homeostasis import (
    VariableTrackingConfig,
    VariableTrackingSimulation,
    run_variable_tracking,
)


class TestVariableTrackingConfig:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"stimulus_speed_min": -0.1},
            {"stimulus_speed_min": 2.0, "stimulus_speed_max": 1.0},
            {"stimulus_speed": 3.0, "stimulus_speed_max": 2.0},
            {"speed_smoothing": 0.0},
            {"speed_smoothing": 1.1},
            {"speed_change_min_steps": 0},
            {"speed_change_min_steps": 20, "speed_change_max_steps": 10},
            {"reverse_min_steps": 0},
            {"reverse_min_steps": 20, "reverse_max_steps": 10},
        ],
    )
    def test_invalid_schedule_rejected(self, kwargs):
        with pytest.raises(ValueError):
            VariableTrackingConfig(**kwargs)


class TestVariableMotion:
    @staticmethod
    def compact_config() -> VariableTrackingConfig:
        return VariableTrackingConfig(
            stimulus_speed=1.0,
            stimulus_speed_min=0.5,
            stimulus_speed_max=1.5,
            speed_smoothing=0.15,
            speed_change_min_steps=5,
            speed_change_max_steps=15,
            reverse_min_steps=20,
            reverse_max_steps=60,
        )

    def test_speed_varies_smoothly_within_bounds(self):
        config = self.compact_config()
        history = run_variable_tracking(
            n_steps=1000, seed=3, tracking_config=config, record_spikes=False
        )
        assert np.all(history.stimulus_speed >= config.stimulus_speed_min)
        assert np.all(history.stimulus_speed <= config.stimulus_speed_max)
        assert np.ptp(history.stimulus_speed) > 0.25
        # Smoothing prevents target changes from becoming discontinuous jumps.
        assert np.max(np.abs(np.diff(history.stimulus_speed))) < (
            config.stimulus_speed_max - config.stimulus_speed_min
        )

    def test_reversal_intervals_are_bounded_and_irregular(self):
        config = self.compact_config()
        history = run_variable_tracking(
            n_steps=2000, seed=4, tracking_config=config, record_spikes=False
        )
        flips = np.flatnonzero(np.diff(history.stimulus_direction) != 0) + 1
        intervals = np.diff(np.concatenate([[0], flips]))
        assert len(intervals) > 10
        assert np.all(intervals >= config.reverse_min_steps)
        assert np.all(intervals <= config.reverse_max_steps)
        assert len(np.unique(intervals)) > 1

    def test_same_seed_replays_motion_and_network(self):
        config = self.compact_config()
        a = run_variable_tracking(n_steps=500, seed=9, tracking_config=config)
        b = run_variable_tracking(n_steps=500, seed=9, tracking_config=config)
        assert np.array_equal(a.stimulus_speed, b.stimulus_speed)
        assert np.array_equal(a.stimulus_direction, b.stimulus_direction)
        assert np.array_equal(a.stimulus_angle, b.stimulus_angle)
        assert np.array_equal(a.spikes, b.spikes)
        assert np.array_equal(a.heading, b.heading)

    def test_different_seed_changes_motion_schedule(self):
        config = self.compact_config()
        a = run_variable_tracking(
            n_steps=300, seed=9, tracking_config=config, record_spikes=False
        )
        b = run_variable_tracking(
            n_steps=300, seed=10, tracking_config=config, record_spikes=False
        )
        assert not np.array_equal(a.stimulus_speed, b.stimulus_speed)
        assert not np.array_equal(a.stimulus_direction, b.stimulus_direction)

    def test_manual_hold_suspends_the_motion_schedule(self):
        sim = VariableTrackingSimulation(tracking_config=self.compact_config(), seed=2)
        env = sim.env
        before = (
            env.stimulus_angle,
            env.current_stimulus_speed,
            env.target_stimulus_speed,
            env.steps_until_speed_change,
            env.steps_until_direction_change,
        )
        for _ in range(10):
            sim.step(advance_stimulus=False)
        after = (
            env.stimulus_angle,
            env.current_stimulus_speed,
            env.target_stimulus_speed,
            env.steps_until_speed_change,
            env.steps_until_direction_change,
        )
        assert after == before

    def test_history_records_speed_for_every_step(self):
        history = run_variable_tracking(
            n_steps=25, seed=1, tracking_config=self.compact_config()
        )
        assert history.stimulus_speed.shape == (25,)


class TestPublishedTaskCompatibility:
    def test_constant_tracking_still_records_constant_speed(self):
        from homeostasis import run_tracking

        history = run_tracking(n_steps=100, seed=0, record_spikes=False)
        assert np.array_equal(history.stimulus_speed, np.ones(100))
