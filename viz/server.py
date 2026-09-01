"""Interactive visualizer server for the homeostatic reservoir tracking model.

Serves a single-page frontend (viz/static) and a WebSocket at /ws. Each
websocket connection owns an independent TrackingSimulation. The frontend
sends JSON commands; the server streams frames at ~30 fps containing the
per-step series (heading, stimulus, spikes, ...) for every simulated step
since the last frame, plus a snapshot of current network state.

The simulation code is exactly the tested `homeostasis` package — the server
adds no model logic, so what the visualizer shows is what the batch
experiments run. The `fingerprint` field (sums of x, targets, weights at
step t) can be cross-checked against `scripts/fingerprint.py --seed S --steps T`
to confirm the visualized trajectory is bit-for-bit the batch trajectory.

Run:  uvicorn viz.server:app --port 8471   (or via .claude/launch.json)
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
from pathlib import Path

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from homeostasis import (
    ReservoirConfig,
    TrackingConfig,
    TrackingSimulation,
    VariableTrackingConfig,
    VariableTrackingSimulation,
)

STATIC_DIR = Path(__file__).parent / "static"

FRAME_RATE = 30.0  # frames per second sent to the client

# Ranges for user-adjustable parameters: name -> (min, max, type)
PARAM_SPECS = {
    "n_nodes": (10, 1000, int),
    "p_link": (0.01, 1.0, float),
    "input_weight": (0.0, 5.0, float),
    "weight_init_mean": (-2.0, 2.0, float),
    "weight_init_sd": (0.0, 2.0, float),
    "leak": (0.0, 1.0, float),
    "target_lr": (0.0, 1.0, float),
    "weight_lr": (0.0, 2.0, float),
    "threshold_ratio": (1.0, 10.0, float),
    "gain": (0.0, 90.0, float),
    "stimulus_speed": (0.0, 10.0, float),
    "reverse_every": (10, 100000, int),
    "stimulus_speed_min": (0.0, 10.0, float),
    "stimulus_speed_max": (0.0, 10.0, float),
    "speed_smoothing": (0.001, 1.0, float),
    "speed_change_min_steps": (1, 100000, int),
    "speed_change_max_steps": (1, 100000, int),
    "reverse_min_steps": (1, 100000, int),
    "reverse_max_steps": (1, 100000, int),
}
RESERVOIR_PARAMS = {
    "n_nodes", "p_link", "input_weight", "weight_init_mean", "weight_init_sd",
    "leak", "target_lr", "weight_lr", "threshold_ratio",
}
BASE_TRACKING_PARAMS = {"gain", "stimulus_speed", "reverse_every"}
VARIABLE_TRACKING_PARAMS = BASE_TRACKING_PARAMS | {
    "stimulus_speed_min", "stimulus_speed_max", "speed_smoothing",
    "speed_change_min_steps", "speed_change_max_steps",
    "reverse_min_steps", "reverse_max_steps",
}
VARIABLE_DEFAULTS = VariableTrackingConfig()

# Named tracking loadouts supplied from the parameter search.  Keeping these
# server-side gives the visualizer one authoritative mapping from the compact
# labels used in reports (w_in, w0, thr, ...) to the actual model fields.
CONFIG_LOADOUTS = (
    {
        "id": "paper",
        "label": "paper",
        "metrics": {"score": 0.402, "dir_agree": 0.75, "prop_spiked": 0.33},
        "params": {
            "n_nodes": 200,
            "p_link": 0.10,
            "input_weight": 0.75,
            "weight_init_mean": 0.75,
            "weight_init_sd": 0.10,
            "leak": 0.25,
            "target_lr": 0.010,
            "weight_lr": 1.0,
            "threshold_ratio": 2.0,
            "gain": 10.0,
        },
    },
    {
        "id": "236",
        "label": "#1 (id 236)",
        "metrics": {"score": 0.709, "dir_agree": 0.85, "prop_spiked": 0.95},
        "params": {
            "n_nodes": 100,
            "p_link": 0.21,
            "input_weight": 0.83,
            "weight_init_mean": 1.26,
            "weight_init_sd": 0.11,
            "leak": 0.57,
            "target_lr": 0.020,
            "weight_lr": 1.0,
            "threshold_ratio": 1.52,
            "gain": 28.0,
        },
    },
    {
        "id": "234",
        "label": "#2 (id 234)",
        "metrics": {"score": 0.656, "dir_agree": 0.90, "prop_spiked": 0.01},
        "params": {
            "n_nodes": 300,
            "p_link": 0.07,
            "input_weight": 1.78,
            "weight_init_mean": 0.36,
            "weight_init_sd": 0.22,
            "leak": 0.56,
            "target_lr": 0.003,
            "weight_lr": 1.0,
            "threshold_ratio": 3.14,
            "gain": 38.0,
        },
    },
    # Lab-campaign configs (scripts/lab/act2_*): metrics are means over the
    # recorded runs in scripts/out/lab/act2_batch1.json (12 seeds, 7200 steps;
    # score = whole-run within-45).
    {
        "id": "lab-ridge25",
        "label": "lab: ridge25 (leak .25, wlr .1)",
        "metrics": {"score": 0.513, "dir_agree": 0.35, "prop_spiked": 0.17},
        "params": {
            "n_nodes": 200,
            "p_link": 0.10,
            "input_weight": 0.75,
            "weight_init_mean": 0.75,
            "weight_init_sd": 0.10,
            "leak": 0.25,
            "target_lr": 0.010,
            "weight_lr": 0.1,
            "threshold_ratio": 2.0,
            "gain": 10.0,
        },
    },
    {
        "id": "lab-w1prime",
        "label": "lab: w1-prime",
        "metrics": {"score": 0.766, "dir_agree": 0.38, "prop_spiked": 0.95},
        "params": {
            "n_nodes": 100,
            "p_link": 0.21,
            "input_weight": 0.83,
            "weight_init_mean": 0.75,
            "weight_init_sd": 0.11,
            "leak": 0.57,
            "target_lr": 0.020,
            "weight_lr": 1.0,
            "threshold_ratio": 1.52,
            "gain": 28.0,
        },
    },
)
# Champions from the viability-evolution experiment (scripts/evolve_viability.py)
# are appended from disk when present, so evolved winners appear in the same
# loadout picker. The file's entries share the loadout shape exactly.
_CHAMPIONS_FILE = Path(__file__).resolve().parent.parent / "scripts/out/evolution/champions.json"
if _CHAMPIONS_FILE.exists():
    try:
        _known = {loadout["id"] for loadout in CONFIG_LOADOUTS}
        CONFIG_LOADOUTS = CONFIG_LOADOUTS + tuple(
            entry for entry in json.loads(_CHAMPIONS_FILE.read_text())
            if entry.get("id") not in _known
        )
    except (json.JSONDecodeError, OSError) as exc:
        print(f"could not load evolution champions: {exc!r}")

LOADOUT_BY_ID = {loadout["id"]: loadout for loadout in CONFIG_LOADOUTS}
PUBLISHED_CONFIG = {
    **LOADOUT_BY_ID["paper"]["params"],
    "stimulus_speed": TrackingConfig().stimulus_speed,
    "reverse_every": TrackingConfig().reverse_every,
}


class VizSession:
    """One connection's simulation plus playback state."""

    def __init__(self):
        self.seed = 0
        self.params: dict = {}
        self.motion_mode = "constant"
        self.playing = False
        self.steps_per_second = 60.0
        self._step_debt = 0.0
        self.manual_stimulus = False
        self.selected_node = 0
        self.dirty = True  # a frame should be sent even if no step ran
        self.last_state = None
        self.last_d_heading = 0.0
        self._build()

    # -- lifecycle ----------------------------------------------------------

    def _build(self) -> None:
        r_kwargs = {k: v for k, v in self.params.items() if k in RESERVOIR_PARAMS}
        if self.motion_mode == "variable":
            t_kwargs = {
                name: self.params.get(name, getattr(VARIABLE_DEFAULTS, name))
                for name in VARIABLE_TRACKING_PARAMS
            }
            # Keep paired UI ranges valid even if the user enters them in the
            # opposite order. Direct model construction remains strict.
            for lower, upper in (
                ("stimulus_speed_min", "stimulus_speed_max"),
                ("speed_change_min_steps", "speed_change_max_steps"),
                ("reverse_min_steps", "reverse_max_steps"),
            ):
                t_kwargs[lower], t_kwargs[upper] = sorted(
                    (t_kwargs[lower], t_kwargs[upper])
                )
            t_kwargs["stimulus_speed"] = min(
                max(t_kwargs["stimulus_speed"], t_kwargs["stimulus_speed_min"]),
                t_kwargs["stimulus_speed_max"],
            )
            self.sim = VariableTrackingSimulation(
                ReservoirConfig(**r_kwargs), VariableTrackingConfig(**t_kwargs), seed=self.seed
            )
        else:
            t_kwargs = {k: v for k, v in self.params.items() if k in BASE_TRACKING_PARAMS}
            self.sim = TrackingSimulation(
                ReservoirConfig(**r_kwargs), TrackingConfig(**t_kwargs), seed=self.seed
            )
        self.last_state = None
        self.last_d_heading = 0.0
        self.dirty = True

    def reset(
        self,
        seed: int | None = None,
        params: dict | None = None,
        motion_mode: str | None = None,
    ) -> None:
        if seed is not None:
            self.seed = int(seed)
        if params is not None:
            clean = {}
            for name, value in params.items():
                if name not in PARAM_SPECS:
                    continue
                lo, hi, cast = PARAM_SPECS[name]
                try:
                    clean[name] = min(max(cast(value), lo), hi)
                except (TypeError, ValueError):
                    continue
            self.params = clean
        if motion_mode is not None:
            self.motion_mode = "variable" if motion_mode == "variable" else "constant"
        self._build()

    def apply_loadout(
        self,
        loadout_id: str,
        seed: int | None = None,
        motion_mode: str | None = None,
    ) -> bool:
        """Reset to one named loadout, returning whether the id was valid."""
        loadout = LOADOUT_BY_ID.get(str(loadout_id))
        if loadout is None:
            return False
        self.reset(seed=seed, params=loadout["params"], motion_mode=motion_mode)
        return True

    # -- stepping -----------------------------------------------------------

    def advance(self, n_steps: int) -> list[dict]:
        """Run n steps, returning the per-step series for the frame."""
        series = []
        for _ in range(n_steps):
            env = self.sim.env
            entry = {
                "t": self.sim.t,
                "stim": env.stimulus_angle,
                "dir": env.stimulus_direction,
                "speed": env.current_stimulus_speed,
                "target_speed": env.target_stimulus_speed,
            }
            state, dh = self.sim.step(advance_stimulus=not self.manual_stimulus)
            self.last_state, self.last_d_heading = state, dh
            entry.update(
                heading=env.heading,
                dh=round(dh, 4),
                prop=state.prop_spiked,
                err=env.heading_error(),
                spikes=np.flatnonzero(state.spiked).tolist(),
            )
            series.append(entry)
        if series:
            self.dirty = True
        return series

    def due_steps(self, dt: float) -> int:
        """How many steps to run this tick to honor steps_per_second."""
        self._step_debt += self.steps_per_second * dt
        n = int(self._step_debt)
        self._step_debt -= n
        return min(n, 2000)  # safety cap per frame

    # -- frame building -----------------------------------------------------

    def frame(self, series: list[dict]) -> dict:
        sim = self.sim
        net = sim.network
        env = sim.env
        state = self.last_state

        w = net.weights[net.adjacency]
        hist_counts, hist_edges = np.histogram(w, bins=30)
        node = min(self.selected_node, net.config.n_nodes - 1)
        # Sensor bars show the *current* geometry (what the next step will
        # consume) so dragging the stimulus while paused gives live feedback.
        inputs = env.sense()

        config = {
            **{f: getattr(net.config, f) for f in (
                "n_nodes", "p_link", "input_weight", "weight_init_mean",
                "weight_init_sd", "leak", "target_lr", "weight_lr",
                "threshold_ratio",
            )},
            **{f: getattr(env.config, f) for f in (
                "gain", "stimulus_speed", "reverse_every",
            )},
            **{
                f: getattr(env.config, f, getattr(VARIABLE_DEFAULTS, f))
                for f in (
                    "stimulus_speed_min", "stimulus_speed_max", "speed_smoothing",
                    "speed_change_min_steps", "speed_change_max_steps",
                    "reverse_min_steps", "reverse_max_steps",
                )
            },
        }
        active_loadout = next(
            (
                loadout["id"]
                for loadout in CONFIG_LOADOUTS
                if all(np.isclose(config[name], value) for name, value in loadout["params"].items())
            ),
            None,
        )
        custom_params = self.motion_mode == "variable" or any(
            not np.isclose(config[name], value)
            for name, value in PUBLISHED_CONFIG.items()
        )

        return {
            "type": "frame",
            "t": sim.t,
            "seed": self.seed,
            "playing": self.playing,
            "manual_stimulus": self.manual_stimulus,
            "motion_mode": self.motion_mode,
            "learning": net.learning_enabled,
            "steps_per_second": self.steps_per_second,
            "series": series,
            "now": {
                "heading": env.heading,
                "stim": env.stimulus_angle,
                "dir": env.stimulus_direction,
                "speed": env.current_stimulus_speed,
                "target_speed": env.target_stimulus_speed,
                "steps_until_speed_change": env.steps_until_speed_change,
                "steps_until_direction_change": env.steps_until_direction_change,
                "err": env.heading_error(),
                "dh": round(self.last_d_heading, 4),
                "sensors": np.round(inputs, 4).tolist(),
                "sensor_offsets": env.config.sensor_offsets.tolist(),
                "outputs": np.round(state.outputs, 4).tolist() if state is not None else [0, 0],
                "spikes": np.flatnonzero(state.spiked).tolist() if state is not None else [],
                "prop": state.prop_spiked if state is not None else 0.0,
                "mean_x": float(np.mean(net.x)),
                "mean_target": float(np.mean(net.targets)),
                "mean_abs_error": float(np.mean(np.abs(state.error))) if state is not None else 0.0,
            },
            "hist": {
                "counts": hist_counts.tolist(),
                "edges": np.round(hist_edges, 4).tolist(),
            },
            "node": {
                "index": node,
                "x": float(net.x[node]),
                "target": float(net.targets[node]),
                "threshold": float(net.thresholds[node]),
                "error": float(state.error[node]) if state is not None else 0.0,
                "spiked": bool(net.spiked[node]),
                "in_degree": int(net.adjacency[:, node].sum()),
                "out_degree": int(net.adjacency[node, :].sum()),
                "n_input_links": int(net.input_adjacency[:, node].sum()),
                "in_weight_sum": float(net.weights[:, node].sum()),
                "spiking_in_neighbors": int((net.adjacency[:, node] & net.spiked).sum()),
            },
            "fingerprint": {
                "sum_x": round(float(np.sum(net.x)), 6),
                "sum_targets": round(float(np.sum(net.targets)), 6),
                "sum_weights": round(float(np.sum(net.weights)), 6),
            },
            # scripts/fingerprint.py replays with default parameters, so the
            # cross-check hint only applies to an unmodified configuration
            "custom_params": custom_params,
            "config": config,
            "loadouts": CONFIG_LOADOUTS,
            "active_loadout": active_loadout,
        }

    # -- commands -----------------------------------------------------------

    def handle(self, msg: dict) -> list[dict]:
        """Apply a client command; returns per-step series if steps were run."""
        cmd = msg.get("cmd")
        if cmd == "play":
            self.playing = True
            self._step_debt = 0.0
        elif cmd == "pause":
            self.playing = False
            self.dirty = True
        elif cmd == "step":
            n = min(max(int(msg.get("n", 1)), 1), 1000)
            return self.advance(n)
        elif cmd == "reset":
            self.playing = False
            self.manual_stimulus = False
            self.reset(
                seed=msg.get("seed"),
                params=msg.get("params"),
                motion_mode=msg.get("motion_mode"),
            )
        elif cmd == "loadout":
            self.playing = False
            self.manual_stimulus = False
            self.apply_loadout(
                msg.get("id", ""),
                seed=msg.get("seed"),
                motion_mode=msg.get("motion_mode"),
            )
        elif cmd == "motion_mode":
            self.playing = False
            self.manual_stimulus = False
            self.reset(params=msg.get("params"), motion_mode=msg.get("mode"))
        elif cmd == "speed":
            self.steps_per_second = min(max(float(msg.get("sps", 60)), 1.0), 2000.0)
        elif cmd == "learning":
            self.sim.network.learning_enabled = bool(msg.get("enabled", True))
            self.dirty = True
        elif cmd == "stim_mode":
            self.manual_stimulus = bool(msg.get("manual", False))
            self.dirty = True
        elif cmd == "stim_set":
            self.sim.env.stimulus_angle = float(msg.get("angle", 0.0)) % 360.0
            self.dirty = True
        elif cmd == "stim_flip":
            self.sim.env.flip_stimulus_direction()
            self.dirty = True
        elif cmd == "select_node":
            self.selected_node = max(0, int(msg.get("index", 0)))
            self.dirty = True
        return []


app = FastAPI(title="homeostasis visualizer")


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    session = VizSession()
    pending_series: list[dict] = []

    async def reader():
        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue
            try:
                pending_series.extend(session.handle(msg))
            except Exception as exc:  # keep the session alive on bad input
                print(f"command error: {exc!r} for {raw[:200]}")

    reader_task = asyncio.create_task(reader())
    try:
        frame_dt = 1.0 / FRAME_RATE
        while True:
            if session.playing:
                pending_series.extend(session.advance(session.due_steps(frame_dt)))
            if pending_series or session.dirty:
                series, pending_series = pending_series, []
                session.dirty = False
                await ws.send_text(json.dumps(session.frame(series)))
            await asyncio.sleep(frame_dt)
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        reader_task.cancel()


# Case study 2 lives in its own sub-application, served at /pong.
from .pong_server import pong_app  # noqa: E402  (mounted after this app is built)

app.mount("/pong", pong_app)

# Lab viewers (single-node explorer, …) live in their own sub-application, at /lab.
from .lab_server import lab_app  # noqa: E402

app.mount("/lab", lab_app)
