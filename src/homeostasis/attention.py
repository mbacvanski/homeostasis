"""Sticky source attention for multi-agent homeostatic ecologies.

The design-space campaign (LEDGER H81-H86) established that homeostatic
entrainment has no figure-ground: summing the retinal bumps of two moving
sources abolishes the lock on either (superposed flows leave no co-moving
frame with ~constant control), and a *memoryless* winner-take-all filter
fails as well, because the selected source flickers with relative
intensity and the effective stimulus teleports.

The minimal sufficient machinery is source selection *with persistence*:
attend one source, and switch only when a rival has been at least
``ratio`` times stronger for ``patience`` consecutive steps. With a
conservative threshold (ratio 5, patience 300) this yields stable
shared-visibility ecologies — including the full depth-4 entrainment
chain with every agent visible to every other — and prevents the
hierarchy inversion observed at lax thresholds, where a follower orbiting
close becomes its own leader's brightest stimulus and captures its
attention ("the follower seduces the leader").

This module owns only the attention rule. Sensing stays in the
environment (each candidate source is rendered to a retinal activation
vector by the caller); the reservoir never changes.
"""

from __future__ import annotations

import numpy as np

__all__ = ["StickyAttention"]


class StickyAttention:
    """Latched winner-take-all over candidate sources.

    Parameters
    ----------
    ratio:
        A rival must be at least ``ratio`` times stronger (by summed
        activation) than the currently attended source to make progress
        toward a switch. Campaign values: 2.0 admits leader capture;
        5.0 is capture-resistant.
    patience:
        The rival must clear the ratio for this many *consecutive* steps
        before the switch happens.
    initial:
        Index of the initially attended source.
    """

    def __init__(self, ratio: float = 5.0, patience: int = 300, initial: int = 0):
        if ratio < 1.0:
            raise ValueError("ratio must be >= 1")
        if patience < 1:
            raise ValueError("patience must be >= 1")
        self.ratio = float(ratio)
        self.patience = int(patience)
        self.selected = int(initial)
        self._streak = 0
        self.switch_times: list[int] = []
        self._t = 0

    def select(self, bumps: list[np.ndarray]) -> np.ndarray:
        """Pick this step's attended activation vector.

        ``bumps`` is one retinal activation vector per candidate source,
        in a fixed order across the run. Returns the attended vector
        (the caller feeds it to the reservoir).
        """
        if self.selected >= len(bumps):
            raise ValueError("selected source index out of range")
        sums = [float(b.sum()) for b in bumps]
        rival = int(np.argmax(sums))
        if rival != self.selected and sums[rival] >= self.ratio * max(
            sums[self.selected], 1e-9
        ):
            self._streak += 1
        else:
            self._streak = 0
        if self._streak >= self.patience:
            self.selected = rival
            self._streak = 0
            self.switch_times.append(self._t)
        self._t += 1
        return bumps[self.selected]
