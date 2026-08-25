"""Server-side checks for visualizer motion-mode controls."""

import numpy as np

from viz.server import CONFIG_LOADOUTS, VizSession


def test_visualizer_defaults_to_published_constant_motion():
    session = VizSession()
    frame = session.frame([])
    assert frame["motion_mode"] == "constant"
    assert frame["now"]["speed"] == 1.0
    assert frame["now"]["target_speed"] == 1.0
    assert frame["now"]["steps_until_speed_change"] is None


def test_visualizer_switches_to_variable_motion_and_streams_speed():
    session = VizSession()
    session.handle(
        {
            "cmd": "motion_mode",
            "mode": "variable",
            "params": {
                "speed_change_min_steps": 3,
                "speed_change_max_steps": 7,
                "reverse_min_steps": 8,
                "reverse_max_steps": 15,
            },
        }
    )
    series = session.advance(100)
    frame = session.frame(series)
    speeds = np.array([entry["speed"] for entry in series])
    assert frame["motion_mode"] == "variable"
    assert frame["custom_params"]
    assert np.ptp(speeds) > 0.0
    assert all("target_speed" in entry for entry in series)
    assert frame["now"]["steps_until_speed_change"] is not None
    assert 1 <= frame["now"]["steps_until_direction_change"] <= 1200


def test_visualizer_normalizes_reversed_parameter_ranges():
    session = VizSession()
    session.reset(
        motion_mode="variable",
        params={
            "stimulus_speed_min": 2.0,
            "stimulus_speed_max": 0.5,
            "speed_change_min_steps": 20,
            "speed_change_max_steps": 5,
            "reverse_min_steps": 50,
            "reverse_max_steps": 10,
        },
    )
    config = session.sim.env.config
    assert (config.stimulus_speed_min, config.stimulus_speed_max) == (0.5, 2.0)
    assert (config.speed_change_min_steps, config.speed_change_max_steps) == (5, 20)
    assert (config.reverse_min_steps, config.reverse_max_steps) == (10, 50)


def test_visualizer_exposes_all_named_loadouts():
    session = VizSession()
    frame = session.frame([])
    ids = [loadout["id"] for loadout in frame["loadouts"]]
    # The three named loadouts come first; evolved champions (scripts/out/
    # evolution*/champions.json) are appended dynamically when present.
    assert ids[:3] == ["paper", "236", "234"]
    assert frame["active_loadout"] == "paper"
    assert not frame["custom_params"]


def test_visualizer_applies_named_loadouts_exactly():
    for loadout in CONFIG_LOADOUTS:
        session = VizSession()
        session.handle({"cmd": "loadout", "id": loadout["id"]})
        frame = session.frame([])
        for name, value in loadout["params"].items():
            assert np.isclose(frame["config"][name], value), (loadout["id"], name)
        assert frame["active_loadout"] == loadout["id"]
        assert frame["custom_params"] == (loadout["id"] != "paper")


def test_loadout_preserves_irregular_motion_and_resets_run():
    session = VizSession()
    session.handle({"cmd": "motion_mode", "mode": "variable"})
    session.advance(5)
    session.handle({"cmd": "loadout", "id": "236"})
    assert session.sim.t == 0
    assert session.motion_mode == "variable"
    assert session.sim.network.config.n_nodes == 100
    assert session.sim.env.config.gain == 28.0
