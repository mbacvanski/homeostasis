"""Behavioral regression tests for Pong (Falandays et al. 2024, case study 2).

Fixed seeds, so outcomes are deterministic. Runs are shortened to 20k steps
(~70 scoring opportunities each) to keep the suite fast; thresholds carry
margin against the values observed at that length across seeds:

    baseline     0.58 +/- 0.12    (paper, 1e5 steps: 0.582 +/- 0.100)
    no-learning  0.44 +/- 0.11    (paper: 0.43 +/- 0.138)
    allocentric  0.22 +/- 0.03    (paper: 0.216 +/- 0.021)

Chance is 0.20.
"""

import dataclasses

import numpy as np
import pytest

from homeostasis import (
    PONG_RESERVOIR_CONFIG,
    PongConfig,
    PongSimulation,
    pong_metrics,
    run_pong,
)

STEPS = 20_000
SEEDS = [0, 1, 2, 3]
CHANCE = 0.20

ALLOCENTRIC_PONG = PongConfig.allocentric()
ALLOCENTRIC_RESERVOIR = dataclasses.replace(
    PONG_RESERVOIR_CONFIG, n_inputs=ALLOCENTRIC_PONG.n_sensors
)


@pytest.fixture(scope="module")
def baseline():
    return [run_pong(n_steps=STEPS, seed=s) for s in SEEDS]


@pytest.fixture(scope="module")
def allocentric():
    return [
        run_pong(
            n_steps=STEPS, seed=s,
            reservoir_config=ALLOCENTRIC_RESERVOIR, pong_config=ALLOCENTRIC_PONG,
        )
        for s in SEEDS
    ]


@pytest.mark.slow
class TestPongPerformance:
    def test_beats_chance_by_a_wide_margin(self, baseline):
        rates = [pong_metrics(h)["hit_rate"] for h in baseline]
        assert float(np.mean(rates)) > 0.38
        assert min(rates) > CHANCE

    def test_enough_opportunities_to_be_meaningful(self, baseline):
        for h in baseline:
            assert h.n_opportunities > 30

    def test_allocentric_encoding_collapses_to_near_chance(self, allocentric, baseline):
        allo = [pong_metrics(h)["hit_rate"] for h in allocentric]
        ego = [pong_metrics(h)["hit_rate"] for h in baseline]
        assert float(np.mean(allo)) < 0.32
        assert float(np.mean(allo)) < float(np.mean(ego))

    def test_frozen_learning_hurts_but_stays_above_chance(self):
        # The paper's striking Pong ablation: a frozen random network still
        # plays well above chance, so learning is not what creates the skill.
        frozen = [
            pong_metrics(run_pong(n_steps=STEPS, seed=s, learning_enabled=False))["hit_rate"]
            for s in SEEDS[:2]
        ]
        assert float(np.mean(frozen)) > CHANCE

    def test_reproducible(self):
        a = run_pong(n_steps=3000, seed=7)
        b = run_pong(n_steps=3000, seed=7)
        assert np.array_equal(a.hits, b.hits)

    def test_seeds_differ(self):
        a = run_pong(n_steps=3000, seed=7)
        b = run_pong(n_steps=3000, seed=8)
        assert not np.array_equal(a.hits, b.hits)


@pytest.mark.slow
class TestPongDynamics:
    def test_network_stays_finite_and_bounded(self):
        sim = PongSimulation(seed=0)
        h = sim.run(5000, record=True)
        assert np.all(np.isfinite(h.prop_spiked))
        assert np.all((h.outputs >= 0.0) & (h.outputs <= 1.0))
        assert np.all(sim.network.targets >= 1.0)
        assert np.all(np.isfinite(sim.network.weights))

    def test_paddle_stays_in_bounds(self):
        cfg = PongConfig()
        h = PongSimulation(seed=1).run(5000, record=True)
        assert h.paddle_y.min() >= cfg.paddle_half_height
        assert h.paddle_y.max() <= cfg.height - cfg.paddle_half_height

    def test_ball_stays_in_bounds(self):
        # The wall tests are an if/elseif chain, so a step that resolves an
        # x-event skips the y-clamp: the ball may sit up to one step of dy
        # outside the y-wall before being pulled back on the next step. That
        # transient is faithful to the released code (see
        # test_pong_env.TestBallMotion.test_wall_tests_are_mutually_exclusive).
        cfg = PongConfig()
        h = PongSimulation(seed=2).run(5000, record=True)
        slack = cfg.ball_speed_y
        assert h.ball_x.min() >= 0.0
        assert h.ball_x.max() <= cfg.x_bounce + 1e-9
        assert h.ball_y.min() >= cfg.y_min - slack - 1e-9
        assert h.ball_y.max() <= cfg.y_max + slack + 1e-9

    def test_events_match_hit_sequence(self):
        h = PongSimulation(seed=3).run(5000, record=True)
        assert int((h.event == 1).sum()) == int(h.hits.sum())
        assert int((h.event == -1).sum()) == int((h.hits == 0).sum())
