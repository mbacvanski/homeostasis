"""Tests of the simulation wiring: step ordering, recording, reproducibility."""

import numpy as np
import pytest

from homeostasis import (
    ReservoirConfig,
    TrackingConfig,
    TrackingSimulation,
    run_tracking,
)


class TestWiring:
    def test_config_mismatch_raises(self):
        with pytest.raises(ValueError):
            TrackingSimulation(ReservoirConfig(n_inputs=10), TrackingConfig())
        with pytest.raises(ValueError):
            TrackingSimulation(ReservoirConfig(n_outputs=3), TrackingConfig())

    def test_step_order_sense_then_move_then_advance(self):
        sim = TrackingSimulation(seed=0)
        h = sim.run(3)
        # Step 0 sensed the initial configuration...
        assert h.stimulus_angle[0] == pytest.approx(0.0)
        assert h.stimulus_direction[0] == 1
        # ...then turned the agent from its initial heading...
        assert h.heading[0] == pytest.approx((90.0 + h.d_heading[0]) % 360.0)
        # ...then the stimulus advanced 1 degree counter-clockwise.
        assert h.stimulus_angle[1] == pytest.approx(1.0)

    def test_d_heading_matches_outputs(self):
        h = run_tracking(n_steps=50, seed=1)
        expected = 10.0 * (h.outputs[:, 0] - h.outputs[:, 1])
        assert h.d_heading == pytest.approx(expected)

    def test_recorded_error_matches_angles(self):
        h = run_tracking(n_steps=50, seed=1)
        # error[i] is measured at sensing time: against the heading left by
        # the previous step (90 at the start).
        prev_heading = np.concatenate([[90.0], h.heading[:-1]])
        expected = (h.stimulus_angle - prev_heading + 180.0) % 360.0 - 180.0
        assert h.error == pytest.approx(expected)


class TestRecording:
    def test_shapes(self):
        h = run_tracking(n_steps=40, seed=2)
        assert len(h) == 40
        assert h.spikes.shape == (40, 200)
        assert h.outputs.shape == (40, 2)

    def test_spikes_optional(self):
        h = run_tracking(n_steps=10, seed=2, record_spikes=False)
        assert h.spikes.shape == (0, 200)


class TestReproducibility:
    def test_same_seed_identical(self):
        a = run_tracking(n_steps=200, seed=7)
        b = run_tracking(n_steps=200, seed=7)
        assert np.array_equal(a.heading, b.heading)
        assert np.array_equal(a.spikes, b.spikes)
        assert np.array_equal(a.outputs, b.outputs)

    def test_different_seeds_differ(self):
        a = run_tracking(n_steps=200, seed=7)
        b = run_tracking(n_steps=200, seed=8)
        assert not np.array_equal(a.heading, b.heading)
