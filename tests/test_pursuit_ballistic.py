"""Mechanics of the ballistic stimulus mode (H55/H57 instruments)."""
import numpy as np

from homeostasis.pursuit import PursuitConfig, PursuitEnv


def _env(seed=3, **kw):
    return PursuitEnv(PursuitConfig(stimulus_motion="ballistic", **kw),
                      rng=np.random.default_rng(seed))


def test_flight_steps_and_bounds():
    env = _env()
    xs, ys = [], []
    for _ in range(1200):
        env.advance_stimulus()
        xs.append(env.sx)
        ys.append(env.sy)
    xs, ys = np.array(xs), np.array(ys)
    step = np.hypot(np.diff(xs), np.diff(ys))
    in_flight = step[step < 1.0]
    assert np.allclose(in_flight, 0.15, atol=1e-9)
    lo, hi = 2.0, 13.0  # waypoint_margin box
    assert xs.min() >= lo - 1e-9 and xs.max() <= hi + 1e-9
    assert ys.min() >= lo - 1e-9 and ys.max() <= hi + 1e-9


def test_respawn_counts_crossings():
    env = _env()
    for _ in range(1200):
        env.advance_stimulus()
    # every jump is one respawn; +1 for the spawn at construction
    assert env.crossings >= 2


def test_deterministic_under_seed():
    a, b = _env(seed=11), _env(seed=11)
    for _ in range(800):
        a.advance_stimulus()
        b.advance_stimulus()
    assert a.sx == b.sx and a.sy == b.sy and a.crossings == b.crossings


def test_kink_at_fixed_age_changes_heading_once():
    base = _env(seed=5)
    kinked = PursuitEnv(
        PursuitConfig(stimulus_motion="ballistic", ballistic_kink_at=40),
        rng=np.random.default_rng(5))
    diverged = False
    for i in range(60):
        base.advance_stimulus()
        kinked.advance_stimulus()
        if i < 39:
            assert (base.sx, base.sy) == (kinked.sx, kinked.sy)
        if (base.sx, base.sy) != (kinked.sx, kinked.sy):
            diverged = True
    assert diverged
