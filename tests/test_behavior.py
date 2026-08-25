"""Behavioral regression tests: emergent tracking, as in the paper's Fig. 4.

These run the full 7200-step experiment on fixed seeds, so outcomes are
deterministic. Thresholds are set with margin below the observed values
(seeds 0-4: direction_agreement 0.65-0.79, within45 0.30-0.40, prop_spiked
0.29-0.38) and anchored against the authors' published run, which scores
within45 = 0.38, direction_agreement = 0.83, prop_spiked = 0.34 on the same
metrics (see homeostasis.analysis).
"""

import numpy as np
import pytest

from homeostasis import run_tracking, tracking_metrics

SEEDS = [0, 1, 2, 3, 4]


@pytest.fixture(scope="module")
def runs():
    return {seed: run_tracking(n_steps=7200, seed=seed, record_spikes=False) for seed in SEEDS}


@pytest.fixture(scope="module")
def runs_no_learning():
    return {
        seed: run_tracking(n_steps=7200, seed=seed, learning_enabled=False, record_spikes=False)
        for seed in SEEDS
    }


@pytest.mark.slow
class TestEmergentTracking:
    def test_agent_entrains_to_stimulus_direction(self, runs):
        # The signature behavior: the agent turns with the stimulus well above
        # chance (0.5) on every seed.
        agreement = [tracking_metrics(h)["direction_agreement"] for h in runs.values()]
        assert min(agreement) > 0.55
        assert float(np.median(agreement)) > 0.65

    def test_heading_stays_near_stimulus_above_chance(self, runs):
        # Chance (frozen or uniformly random heading) is 0.25.
        within = [tracking_metrics(h)["within45"] for h in runs.values()]
        assert float(np.median(within)) > 0.28

    def test_reservoir_activity_in_published_band(self, runs):
        # Paper Fig. 4B: structured activity, roughly 0.2-0.45 of the
        # reservoir spiking; published run averages 0.34.
        for h in runs.values():
            prop = tracking_metrics(h)["prop_spiked_mean"]
            assert 0.15 < prop < 0.5

    def test_learning_off_agent_never_moves(self, runs_no_learning):
        # Without homeostatic updating the network saturates: both effectors
        # read identically and the agent stops turning entirely (cf. the
        # paper's learning-off ablations).
        for h in runs_no_learning.values():
            assert np.all(h.d_heading[720:] == 0.0)

    def test_learning_beats_no_learning(self, runs, runs_no_learning):
        for seed in SEEDS:
            on = tracking_metrics(runs[seed])
            off = tracking_metrics(runs_no_learning[seed])
            assert on["direction_agreement"] > off["direction_agreement"]
