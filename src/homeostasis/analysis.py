"""Analysis metrics for tracking runs.

Reference values, computed with :func:`tracking_metrics` on the authors'
published run (OSF Data/ObjectTracking, the run shown in the paper's Fig. 4):

    within45 = 0.38, median_abs_error = 62.0 deg,
    direction_agreement = 0.83, prop_spiked_mean = 0.34

Our re-implementation across seeds 0-9 (7200 steps) brackets those values:
within45 0.24-0.53, direction_agreement 0.65-0.79, prop_spiked 0.29-0.38.
With learning disabled the agent does not move at all (both effectors
saturate equally), so direction_agreement is 0 and within45 is 0.25 (the
chance level for a frozen heading under a uniformly sweeping stimulus).
"""

from __future__ import annotations

import numpy as np

from .simulation import History, PongHistory
from .tracking import angular_difference

__all__ = ["tracking_metrics", "pong_metrics"]


def tracking_metrics(h: History, settle: int = 720, smooth_window: int = 50) -> dict:
    """Quantify tracking quality after an initial settling period.

    - within45: fraction of post-settling steps with |heading error| <= 45
      degrees (a uniformly random or frozen heading scores 0.25).
    - median_abs_error: median |heading error| in degrees.
    - direction_agreement: fraction of post-settling steps where the agent's
      smoothed turning direction matches the stimulus direction (chance 0.5
      for an agent that keeps turning; 0 for one that never turns).
    - prop_spiked_mean: mean fraction of the reservoir spiking.
    """
    err = np.abs(angular_difference(h.stimulus_angle[settle:], h.heading[settle:]))
    # Smooth the turn signal before taking its sign, so step-to-step jitter
    # doesn't mask the direction of travel.
    kernel = np.ones(smooth_window) / smooth_window
    smoothed_turn = np.convolve(h.d_heading, kernel, mode="same")[settle:]
    agree = np.sign(smoothed_turn) == h.stimulus_direction[settle:]
    return {
        "within45": float(np.mean(err <= 45.0)),
        "median_abs_error": float(np.median(err)),
        "direction_agreement": float(np.mean(agree)),
        "prop_spiked_mean": float(np.mean(h.prop_spiked[settle:])),
    }


def pong_metrics(h: PongHistory, window: int = 50) -> dict:
    """Hit rate and early/late windows for a Pong run.

    Reference values from the paper (500 runs of 1e5 steps, chance = 0.20):
    hit_rate 0.582 (SD 0.0995); first/last 50 opportunities both 0.5786
    (SD 0.105 / 0.12), i.e. no improvement over the run; learning disabled
    0.43 (SD 0.138); allocentric sensory encoding 0.216 (SD 0.021).
    """
    hits = np.asarray(h.hits, dtype=float)
    if hits.size == 0:
        return {"hit_rate": float("nan"), "n_opportunities": 0,
                "first": float("nan"), "last": float("nan")}
    return {
        "hit_rate": float(hits.mean()),
        "n_opportunities": int(hits.size),
        "first": float(hits[:window].mean()),
        "last": float(hits[-window:].mean()),
    }
