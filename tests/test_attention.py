"""Mechanics of the sticky attention rule (LEDGER H82-H86)."""
import numpy as np
import pytest

from homeostasis.attention import StickyAttention


def _bumps(a, b):
    return [np.full(4, a / 4.0), np.full(4, b / 4.0)]


def test_holds_current_source_below_ratio():
    att = StickyAttention(ratio=5.0, patience=3)
    for _ in range(50):
        out = att.select(_bumps(1.0, 4.9))  # rival stronger but < 5x
        assert att.selected == 0
        assert out.sum() == pytest.approx(1.0)
    assert att.switch_times == []


def test_switches_after_patience_consecutive_steps():
    att = StickyAttention(ratio=2.0, patience=3)
    att.select(_bumps(1.0, 2.5))
    att.select(_bumps(1.0, 2.5))
    assert att.selected == 0
    att.select(_bumps(1.0, 2.5))
    assert att.selected == 1
    assert att.switch_times == [2]


def test_streak_resets_on_interruption():
    att = StickyAttention(ratio=2.0, patience=3)
    att.select(_bumps(1.0, 2.5))
    att.select(_bumps(1.0, 2.5))
    att.select(_bumps(1.0, 1.0))  # interruption
    att.select(_bumps(1.0, 2.5))
    att.select(_bumps(1.0, 2.5))
    assert att.selected == 0  # streak restarted, no switch yet


def test_returns_attended_vector_verbatim():
    att = StickyAttention()
    bumps = [np.arange(3.0), np.arange(3.0) * 10]
    out = att.select(bumps)
    assert out is bumps[0]


def test_validation():
    with pytest.raises(ValueError):
        StickyAttention(ratio=0.5)
    with pytest.raises(ValueError):
        StickyAttention(patience=0)
