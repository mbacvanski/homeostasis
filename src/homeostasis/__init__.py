"""Re-implementation of the homeostatic reservoir model from Falandays,
Yoshimi, Warren & Spivey (2024), Cognitive Neurodynamics 18:1811-1834."""

from .analysis import pong_metrics, tracking_metrics
from .pong import PongConfig, PongEnv
from .reservoir import HomeostaticReservoir, ReservoirConfig, StepState
from .simulation import (
    PONG_RESERVOIR_CONFIG,
    History,
    PongHistory,
    PongSimulation,
    TrackingSimulation,
    run_pong,
    run_tracking,
)
from .tracking import TrackingConfig, TrackingEnv, angular_difference, wrap_angle
from .variable_tracking import (
    VariableTrackingConfig,
    VariableTrackingEnv,
    VariableTrackingSimulation,
    run_variable_tracking,
)

__all__ = [
    "tracking_metrics",
    "pong_metrics",
    "PongConfig",
    "PongEnv",
    "PongHistory",
    "PongSimulation",
    "run_pong",
    "PONG_RESERVOIR_CONFIG",
    "HomeostaticReservoir",
    "ReservoirConfig",
    "StepState",
    "History",
    "TrackingSimulation",
    "run_tracking",
    "TrackingConfig",
    "TrackingEnv",
    "angular_difference",
    "wrap_angle",
    "VariableTrackingConfig",
    "VariableTrackingEnv",
    "VariableTrackingSimulation",
    "run_variable_tracking",
]
