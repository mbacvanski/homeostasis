"""Interactive visualizer for the Pong case study (Falandays et al. 2024, #2).

A self-contained FastAPI sub-application, mounted at /pong by viz.server. It
mirrors the tracking visualizer's design: the frontend is a thin renderer and
every number it shows comes from the tested `homeostasis` package, so what you
watch in the browser is exactly what the batch experiments run. The
`fingerprint` field (sums of x, targets, weights, plus the hit sequence
length) can be cross-checked with `scripts/pong_fingerprint.py`.

Run via the main app:  uvicorn viz.server:app --port 8471  ->  /pong
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

from homeostasis import PONG_RESERVOIR_CONFIG, PongConfig, PongSimulation

STATIC_DIR = Path(__file__).parent / "static"
FRAME_RATE = 30.0

# name -> (min, max, type)
PARAM_SPECS = {
    "n_nodes": (10, 2000, int),
    "p_link": (0.01, 1.0, float),
    "input_weight": (0.0, 20.0, float),
    "weight_init_mean": (-2.0, 2.0, float),
    "weight_init_sd": (0.0, 2.0, float),
    "inhibitory_fraction": (0.0, 1.0, float),
    "inhibitory_weight_mean": (-5.0, 5.0, float),
    "leak": (0.0, 1.0, float),
    "target_lr": (0.0, 1.0, float),
    "threshold_ratio": (1.0, 10.0, float),
    "gain": (0.0, 500.0, float),
    "paddle_half_height": (5.0, 250.0, float),
    "ball_speed_x": (0.5, 50.0, float),
    "ball_speed_y": (0.0, 50.0, float),
}
RESERVOIR_PARAMS = {
    "n_nodes", "p_link", "input_weight", "weight_init_mean", "weight_init_sd",
    "inhibitory_fraction", "inhibitory_weight_mean", "leak", "target_lr",
    "threshold_ratio",
}
PONG_PARAMS = {"gain", "paddle_half_height", "ball_speed_x", "ball_speed_y"}

# Preset loadouts: the published configuration plus any champions produced by
# scripts/evolve_pong.py (read at import; restart the server to refresh).
PONG_LOADOUTS = [{
    "id": "published",
    "label": "published (hit 0.58)",
    "metrics": {"hit_rate": 0.582},
    "params": {},   # empty = the defaults, which are the published values
}]
_CHAMPS = Path(__file__).resolve().parent.parent / "scripts/out/evolution_pong/champions.json"
if _CHAMPS.exists():
    try:
        PONG_LOADOUTS += json.loads(_CHAMPS.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        print(f"could not load pong champions: {exc!r}")
LOADOUT_BY_ID = {entry["id"]: entry for entry in PONG_LOADOUTS}


class PongSession:
    """One connection's Pong simulation plus playback state."""

    def __init__(self):
        self.seed = 0
        self.params: dict = {}
        self.encoding = "egocentric"
        self.playing = False
        self.steps_per_second = 60.0
        self._step_debt = 0.0
        self.selected_node = 0
        self.dirty = True
        self.last_state = None
        self.last_d_paddle = 0.0
        self.last_angle = 0.0
        self._build()

    # -- lifecycle ----------------------------------------------------------

    def _build(self) -> None:
        pong_kwargs = {k: v for k, v in self.params.items() if k in PONG_PARAMS}
        pong_config = (
            PongConfig.allocentric(**pong_kwargs)
            if self.encoding == "allocentric"
            else PongConfig(**pong_kwargs)
        )
        reservoir_config = dataclasses.replace(
            PONG_RESERVOIR_CONFIG,
            n_inputs=pong_config.n_sensors,
            **{k: v for k, v in self.params.items() if k in RESERVOIR_PARAMS},
        )
        self.sim = PongSimulation(reservoir_config, pong_config, seed=self.seed)
        self.last_state = None
        self.last_d_paddle = 0.0
        self.last_angle = self.sim.env.ball_angle()
        self.dirty = True

    def reset(self, seed=None, params: dict | None = None, encoding: str | None = None) -> None:
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
        if encoding is not None:
            self.encoding = "allocentric" if encoding == "allocentric" else "egocentric"
        self._build()

    # -- stepping -----------------------------------------------------------

    def advance(self, n_steps: int) -> list[dict]:
        series = []
        env = self.sim.env
        for _ in range(n_steps):
            angle_before = env.ball_angle()
            state, event, d_paddle = self.sim.step()
            angle = env.ball_angle()
            self.last_state, self.last_d_paddle, self.last_angle = state, d_paddle, angle
            series.append({
                "t": self.sim.t - 1,
                "bx": round(env.ball_x, 2),
                "by": round(env.ball_y, 2),
                "py": round(env.paddle_y, 2),
                "ev": 1 if event == "hit" else (-1 if event == "miss" else 0),
                "prop": state.prop_spiked,
                "ang": round(angle, 3),
                # angular speed of the ball in the paddle's frame: the paper's
                # proposed mechanism is that this spikes as the ball arrives
                "dang": round(abs((angle - angle_before + 180.0) % 360.0 - 180.0), 3),
                "spikes": np.flatnonzero(state.spiked).tolist(),
            })
        if series:
            self.dirty = True
        return series

    def due_steps(self, dt: float) -> int:
        self._step_debt += self.steps_per_second * dt
        n = int(self._step_debt)
        self._step_debt -= n
        return min(n, 4000)

    # -- frame building -----------------------------------------------------

    def frame(self, series: list[dict]) -> dict:
        sim, net, env = self.sim, self.sim.network, self.sim.env
        state = self.last_state
        hits = np.asarray(env.hits, dtype=float)

        w = net.weights[net.adjacency]
        hist_counts, hist_edges = np.histogram(w, bins=40)
        node = min(self.selected_node, net.config.n_nodes - 1)

        return {
            "type": "frame",
            "t": sim.t,
            "seed": self.seed,
            "playing": self.playing,
            "learning": net.learning_enabled,
            "encoding": self.encoding,
            "steps_per_second": self.steps_per_second,
            "series": series,
            "now": {
                "ball": [round(env.ball_x, 2), round(env.ball_y, 2)],
                "vel": [env.dx, env.dy],
                "paddle_y": round(env.paddle_y, 2),
                "d_paddle": round(self.last_d_paddle, 3),
                "angle": round(env.ball_angle(), 3),
                "sensors": env.sense().tolist(),
                "sensor_values": env.config.sensor_values.tolist(),
                "outputs": np.round(state.outputs, 4).tolist() if state is not None else [0, 0],
                "prop": state.prop_spiked if state is not None else 0.0,
                "spikes": np.flatnonzero(state.spiked).tolist() if state is not None else [],
                "mean_x": float(np.mean(net.x)),
                "mean_target": float(np.mean(net.targets)),
                "mean_abs_error": float(np.mean(np.abs(state.error))) if state is not None else 0.0,
            },
            "score": {
                "hits": int(hits.sum()),
                "opportunities": int(hits.size),
                "hit_rate": float(hits.mean()) if hits.size else None,
                "recent": hits[-40:].tolist(),
                "chance": env.config.chance_hit_rate,
                # running hit rate after each opportunity, thinned for display
                "curve": _thin(np.cumsum(hits) / np.arange(1, hits.size + 1), 240),
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
                "n_opportunities": int(hits.size),
            },
            "custom_params": bool(self.params),
            "loadouts": PONG_LOADOUTS,
            "config": {
                **{f: getattr(net.config, f) for f in (
                    "n_nodes", "p_link", "input_weight", "weight_init_mean",
                    "weight_init_sd", "inhibitory_fraction", "inhibitory_weight_mean",
                    "leak", "target_lr", "threshold_ratio",
                )},
                **{f: getattr(env.config, f) for f in (
                    "gain", "paddle_half_height", "ball_speed_x", "ball_speed_y",
                    "width", "height", "paddle_x",
                )},
                "n_sensors": env.config.n_sensors,
            },
        }

    # -- commands -----------------------------------------------------------

    def handle(self, msg: dict) -> list[dict]:
        cmd = msg.get("cmd")
        if cmd == "play":
            self.playing = True
            self._step_debt = 0.0
        elif cmd == "pause":
            self.playing = False
            self.dirty = True
        elif cmd == "step":
            return self.advance(min(max(int(msg.get("n", 1)), 1), 5000))
        elif cmd == "reset":
            self.playing = False
            self.reset(msg.get("seed"), msg.get("params"), msg.get("encoding"))
        elif cmd == "speed":
            self.steps_per_second = min(max(float(msg.get("sps", 60)), 1.0), 4000.0)
        elif cmd == "learning":
            self.sim.network.learning_enabled = bool(msg.get("enabled", True))
            self.dirty = True
        elif cmd == "select_node":
            self.selected_node = max(0, int(msg.get("index", 0)))
            self.dirty = True
        elif cmd == "loadout":
            entry = LOADOUT_BY_ID.get(str(msg.get("id")))
            if entry is not None:
                self.playing = False
                self.reset(msg.get("seed"), dict(entry["params"]), "egocentric")
        return []


def _thin(values: np.ndarray, limit: int) -> list[float]:
    """Downsample a curve to at most `limit` points for transport."""
    if values.size <= limit:
        return np.round(values, 4).tolist()
    idx = np.linspace(0, values.size - 1, limit).astype(int)
    return np.round(values[idx], 4).tolist()


pong_app = FastAPI(title="homeostasis pong visualizer")


@pong_app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "pong.html")


pong_app.mount("/static", StaticFiles(directory=STATIC_DIR), name="pong-static")


@pong_app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    session = PongSession()
    pending: list[dict] = []

    async def reader():
        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue
            try:
                pending.extend(session.handle(msg))
            except Exception as exc:
                print(f"pong command error: {exc!r} for {raw[:200]}")

    reader_task = asyncio.create_task(reader())
    try:
        frame_dt = 1.0 / FRAME_RATE
        while True:
            if session.playing:
                pending.extend(session.advance(session.due_steps(frame_dt)))
            if pending or session.dirty:
                series, pending = pending, []
                session.dirty = False
                await ws.send_text(json.dumps(session.frame(series)))
            await asyncio.sleep(frame_dt)
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        reader_task.cancel()
