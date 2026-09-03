"""Export a simulated network for the blog post's browser port.

The post (personal-website-2026, "Local Learning Beyond Global Objectives")
reruns the talk's tracking and Pong simulations in JavaScript. numpy's PCG64
generator is not ported; instead this script writes out everything the seed
produced — the wiring, the initial weights, and the environment's own random
schedule — plus a reference trace the JS port is validated against.

Three tasks:

  tracking   wiring + irregular-motion schedule + reference trace
  pong       wiring + serve list + reference trace
  neuron     a 240-step replay of one photogenic neuron for the small plate

Examples (run from the repo root with the project venv):

  .venv/bin/python scripts/export_wiring.py --task tracking --loadout 234 --seed 2 \\
      --steps 21600 --horizon 100000 \\
      --out-wiring ../personal-website-2026/src/posts/local-learning-beyond-global-objectives/data/tracking-234-s2.json \\
      --out-reference ../personal-website-2026/scripts/homeostat/reference/tracking-234-s2.reference.json

  .venv/bin/python scripts/export_wiring.py --task pong --loadout pongEvo1 --seed 13 \\
      --steps 228500 --serves 1024 \\
      --out-wiring ../personal-website-2026/src/posts/local-learning-beyond-global-objectives/data/pong-evo1-s13.json \\
      --out-reference ../personal-website-2026/scripts/homeostat/reference/pong-evo1-s13.reference.json

  .venv/bin/python scripts/export_wiring.py --task neuron --loadout 234 --seed 2 \\
      --out-wiring ../personal-website-2026/src/posts/local-learning-beyond-global-objectives/data/neuron-replay.json

Formats: ``homeostat-wiring/1`` (shipped to the browser), ``homeostat-reference/1``
(validation only, never shipped) and ``homeostat-neuron/1``. Large arrays are
``{"dtype", "n", "b64"}`` objects holding little-endian bytes; the recurrent
matrix is CSR by source row (``row_ptr[s] .. row_ptr[s+1]`` lists the links
``s -> col[k]`` with weight ``w[k]``), the input matrix is CSR by node
(sensor indices feeding each node).
"""

from __future__ import annotations

import argparse
import base64
import dataclasses
import datetime as _dt
import hashlib
import json
import math
import pathlib
import subprocess
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from homeostasis import (  # noqa: E402
    HomeostaticReservoir,
    ReservoirConfig,
    TrackingConfig,
    TrackingEnv,
    TrackingSimulation,
    VariableTrackingConfig,
)
from homeostasis.pong import PongConfig, PongEnv  # noqa: E402
from homeostasis.simulation import PONG_RESERVOIR_CONFIG, PongSimulation  # noqa: E402
from homeostasis.tracking import angular_difference  # noqa: E402
from homeostasis.variable_tracking import VariableTrackingEnv  # noqa: E402

WIRING_FORMAT = "homeostat-wiring/1"
REFERENCE_FORMAT = "homeostat-reference/1"
NEURON_FORMAT = "homeostat-neuron/1"
TRACE_STEPS = 300
CHECKPOINTS = (1000, 3000, 8000)
HASH_BLOCK = 100


# --------------------------------------------------------------------------
# encoding helpers
# --------------------------------------------------------------------------

_DTYPES = {"f64": "<f8", "u32": "<u4", "u16": "<u2", "u8": "<u1", "i8": "<i1"}


def encode(arr, dtype: str) -> dict:
    a = np.ascontiguousarray(np.asarray(arr), dtype=_DTYPES[dtype])
    return {"dtype": dtype, "n": int(a.size), "b64": base64.b64encode(a.tobytes()).decode("ascii")}


def csr_by_row(adjacency: np.ndarray, values: np.ndarray | None = None):
    """CSR of a boolean matrix by row (row-major np.nonzero order)."""
    rows, cols = np.nonzero(adjacency)
    row_ptr = np.zeros(adjacency.shape[0] + 1, dtype=np.int64)
    np.add.at(row_ptr, rows + 1, 1)
    row_ptr = np.cumsum(row_ptr)
    out = {"row_ptr": row_ptr, "col": cols}
    if values is not None:
        out["w"] = values[rows, cols]
    return out


def fnv1a32_blocks(spike_lists, block: int = HASH_BLOCK) -> list[str]:
    """FNV-1a (32-bit) over each block of ``block`` steps of spike ids.

    The byte stream per step is the step index (uint32 LE) followed by the
    ascending spike ids (uint16 LE each) — the JS harness hashes the same bytes.
    """
    out = []
    h = 0x811C9DC5
    for t, ids in enumerate(spike_lists):
        if t % block == 0 and t > 0:
            out.append(f"{h:08x}")
            h = 0x811C9DC5
        for b in int(t).to_bytes(4, "little"):
            h = ((h ^ b) * 0x01000193) & 0xFFFFFFFF
        for i in ids:
            for b in int(i).to_bytes(2, "little"):
                h = ((h ^ b) * 0x01000193) & 0xFFFFFFFF
    if len(spike_lists) % block == 0 and len(spike_lists) > 0:
        out.append(f"{h:08x}")
    return out


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def provenance(task: str) -> dict:
    return {
        "script": "scripts/export_wiring.py",
        "homeostasis_commit": git_commit(),
        "numpy": np.__version__,
        "python": sys.version.split()[0],
        "utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "task": task,
    }


def write_json(path: str, payload: dict) -> int:
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, separators=(",", ":"))
    p.write_text(text)
    return len(text.encode())


# --------------------------------------------------------------------------
# recording generators
# --------------------------------------------------------------------------

class RecordingRng:
    """Forwards the draws an environment makes and logs them in order."""

    def __init__(self, rng: np.random.Generator):
        self._rng = rng
        self.log: list[tuple] = []

    def uniform(self, lo, hi):
        v = float(self._rng.uniform(lo, hi))
        self.log.append(("uniform", (float(lo), float(hi)), v))
        return v

    def integers(self, lo, hi):
        v = int(self._rng.integers(lo, hi))
        self.log.append(("integers", (int(lo), int(hi)), v))
        return v

    def choice(self, options):
        v = self._rng.choice(options)
        self.log.append(("choice", tuple(float(o) for o in options), float(v)))
        return v


# --------------------------------------------------------------------------
# wiring
# --------------------------------------------------------------------------

def wiring_payload(net: HomeostaticReservoir) -> dict:
    c = net.config
    rec = csr_by_row(net.adjacency, net.weights)
    inp = csr_by_row(net.input_adjacency.T)          # by node: sensors feeding each node
    effectors = [np.flatnonzero(net.output_adjacency[:, k]).tolist() for k in range(c.n_outputs)]
    in_degree = net.adjacency.sum(axis=0)
    assert not np.any(np.diag(net.adjacency)), "self-connections are never exported"
    assert np.all(net.input_weights[net.input_adjacency] == c.input_weight), "input weights are uniform"
    return {
        "n_links": int(rec["col"].size),
        "n_input_links": int(inp["col"].size),
        "rec": {
            "row_ptr": encode(rec["row_ptr"], "u32"),
            "col": encode(rec["col"], "u16"),
            "w": encode(rec["w"], "f64"),
        },
        "input": {
            "row_ptr": encode(inp["row_ptr"], "u32"),
            "sensor": encode(inp["col"], "u8"),
        },
        "effectors": effectors,
        "checks": {
            "fsum_w": math.fsum(rec["w"].tolist()),
            "sum_col": int(rec["col"].sum()),
            "in_degree_max": int(in_degree.max()),
            "effector_in_degree": [len(e) for e in effectors],
        },
    }


def reservoir_payload(c: ReservoirConfig) -> dict:
    d = dataclasses.asdict(c)
    return {
        "n_nodes": c.n_nodes,
        "n_inputs": c.n_inputs,
        "n_outputs": c.n_outputs,
        "input_weight": c.input_weight,
        "leak": c.leak,
        "target_init": c.target_init,
        "target_floor": c.target_floor,
        "target_lr": c.target_lr,
        "threshold_ratio": c.threshold_ratio,
        "weight_lr": c.weight_lr,
        "clamp_negative_activations": c.clamp_negative_activations,
        "provenance": {k: v for k, v in d.items() if k in (
            "p_link", "weight_init_mean", "weight_init_sd", "inhibitory_fraction",
            "inhibitory_weight_mean", "inhibitory_weight_sd", "input_p_link",
            "input_weight_sd", "input_plastic", "allow_self_connections")},
    }


def net_sums(net: HomeostaticReservoir) -> dict:
    return {
        "sum_x": math.fsum(net.x.tolist()),
        "sum_t": math.fsum(net.targets.tolist()),
        "sum_w": math.fsum(net.weights[net.adjacency].tolist()),
    }


# --------------------------------------------------------------------------
# tracking
# --------------------------------------------------------------------------

def tracking_configs(loadout: str):
    from viz.server import BASE_TRACKING_PARAMS, LOADOUT_BY_ID, RESERVOIR_PARAMS
    entry = LOADOUT_BY_ID[loadout]
    p = entry["params"]
    r_cfg = ReservoirConfig(**{k: v for k, v in p.items() if k in RESERVOIR_PARAMS})
    base = {k: v for k, v in p.items() if k in BASE_TRACKING_PARAMS}
    return entry, r_cfg, VariableTrackingConfig(**base), TrackingConfig(**base)


def run_tracking_reference(sim: TrackingSimulation, n_steps: int, trace_steps: int, checkpoints):
    """Step a tracking simulation, recording the reference trace."""
    net = sim.network
    n = net.config.n_nodes
    trace = {k: [] for k in ("sum_x", "sum_t", "sum_w", "mean_abs_e", "heading",
                             "stim_sensed", "speed_sensed", "direction_sensed", "outputs", "spikes")}
    spike_lists = []
    stim_sensed = np.empty(n_steps)
    heading_after = np.empty(n_steps)
    prop = np.empty(n_steps)
    cps = {}
    for t in range(n_steps):
        stim_sensed[t] = sim.env.stimulus_angle
        speed = sim.env.current_stimulus_speed
        direction = sim.env.stimulus_direction
        state, _dh = sim.step()
        heading_after[t] = sim.env.heading
        prop[t] = state.prop_spiked
        ids = np.flatnonzero(state.spiked).tolist()
        spike_lists.append(ids)
        if t < trace_steps:
            s = net_sums(net)
            trace["sum_x"].append(s["sum_x"])
            trace["sum_t"].append(s["sum_t"])
            trace["sum_w"].append(s["sum_w"])
            trace["mean_abs_e"].append(math.fsum(np.abs(state.error).tolist()) / n)
            trace["heading"].append(float(sim.env.heading))
            trace["stim_sensed"].append(float(stim_sensed[t]))
            trace["speed_sensed"].append(float(speed))
            trace["direction_sensed"].append(int(direction))
            trace["outputs"].append([float(v) for v in state.outputs])
            trace["spikes"].append(ids)
        if (t + 1) in checkpoints or (t + 1) == n_steps:
            cp = net_sums(net)
            cp.update({
                "t": t + 1,
                "heading": float(sim.env.heading),
                "stim": float(sim.env.stimulus_angle),
                "x": encode(net.x, "f64"),
                "targets": encode(net.targets, "f64"),
            })
            if (t + 1) == n_steps:
                cp["w"] = encode(net.weights[net.adjacency], "f64")
            cps[t + 1] = cp
    err = np.abs(angular_difference(stim_sensed[720:], heading_after[720:]))
    metric = {
        "settle": 720,
        "within45": float(np.mean(err <= 45.0)),
        "median_abs_error": float(np.median(err)),
        "prop_spiked_mean": float(np.mean(prop[720:])),
    }
    return {
        "n_steps": n_steps,
        "trace": trace,
        "checkpoints": [cps[k] for k in sorted(cps)],
        "spike_hash": {"block": HASH_BLOCK, "fnv1a32": fnv1a32_blocks(spike_lists)},
        "metric": metric,
    }, stim_sensed


def export_tracking(args) -> None:
    entry, r_cfg, v_cfg, c_cfg = tracking_configs(args.loadout)
    # Mirror VariableTrackingSimulation.__init__: the parent builds the network
    # (consuming the seed contract's draws), then the irregular environment is
    # attached to the same generator — here through a recording proxy.
    sim = TrackingSimulation(r_cfg, c_cfg, seed=args.seed)
    proxy = RecordingRng(sim.network.rng)
    sim.env = VariableTrackingEnv(v_cfg, rng=proxy)
    net = sim.network
    wiring = wiring_payload(net)

    reference, stim_sensed = run_tracking_reference(sim, args.steps, TRACE_STEPS, CHECKPOINTS)
    # Keep drawing the schedule past the reference run so the browser can keep
    # going for as long as the reader watches.
    while sim.env.stimulus_steps < args.horizon:
        sim.env.advance_stimulus()

    speed_targets = [v for kind, _a, v in proxy.log if kind == "uniform"]
    s_range = (v_cfg.speed_change_min_steps, v_cfg.speed_change_max_steps + 1)
    d_range = (v_cfg.reverse_min_steps, v_cfg.reverse_max_steps + 1)
    assert s_range != d_range, "cannot tell the two interval draws apart by range"
    speed_intervals = [v for kind, a, v in proxy.log if kind == "integers" and a == s_range]
    direction_intervals = [v for kind, a, v in proxy.log if kind == "integers" and a == d_range]
    sched = {
        "speed_targets": speed_targets,
        "speed_intervals": speed_intervals,
        "direction_intervals": direction_intervals,
    }
    # Self-check: replaying the lists reproduces the run's stimulus bit-exactly.
    replay = replay_schedule_exact(v_cfg, sched, args.steps)
    assert np.array_equal(replay, stim_sensed), "schedule replay differs from the run"

    env = {
        "kind": "tracking",
        "gain": v_cfg.gain,
        "initial_heading": v_cfg.initial_heading,
        "initial_stimulus_angle": v_cfg.initial_stimulus_angle,
        "initial_stimulus_direction": v_cfg.initial_stimulus_direction,
        "sensor_offsets": v_cfg.sensor_offsets.tolist(),
        "tuning_width": v_cfg.tuning_width,
        "plateau_width": v_cfg.plateau_width,
        "constant": {"stimulus_speed": c_cfg.stimulus_speed, "reverse_every": c_cfg.reverse_every},
        "irregular": {
            "stimulus_speed": v_cfg.stimulus_speed,
            "speed_min": v_cfg.stimulus_speed_min,
            "speed_max": v_cfg.stimulus_speed_max,
            "speed_smoothing": v_cfg.speed_smoothing,
            "speed_change_steps": [v_cfg.speed_change_min_steps, v_cfg.speed_change_max_steps],
            "reverse_steps": [v_cfg.reverse_min_steps, v_cfg.reverse_max_steps],
            "horizon_steps": args.horizon,
            **sched,
        },
    }
    payload = {
        "format": WIRING_FORMAT,
        "generated": provenance("tracking"),
        "task": "tracking",
        "loadout": {"id": entry["id"], "label": entry["label"], "seed": args.seed},
        "reservoir": reservoir_payload(r_cfg),
        "wiring": wiring,
        "env": env,
    }
    n_bytes = write_json(args.out_wiring, payload)
    print(f"wiring: {args.out_wiring} ({n_bytes/1024:.1f} KB) "
          f"links={wiring['n_links']} input_links={wiring['n_input_links']} "
          f"effectors={wiring['checks']['effector_in_degree']} "
          f"draws: speeds={len(speed_targets)} speed_iv={len(speed_intervals)} dir_iv={len(direction_intervals)}")

    # Constant-motion reference on a fresh copy of the same network.
    const_sim = TrackingSimulation(r_cfg, c_cfg, seed=args.seed)
    assert np.array_equal(const_sim.network.weights, wiring_initial_weights(net, r_cfg, args.seed))
    const_ref, _ = run_tracking_reference(const_sim, min(args.steps, 7200), TRACE_STEPS, ())

    reference = {
        "format": REFERENCE_FORMAT,
        "generated": provenance("tracking"),
        "wiring_sha256": hashlib.sha256(pathlib.Path(args.out_wiring).read_bytes()).hexdigest(),
        "task": "tracking",
        "loadout": entry["id"],
        "seed": args.seed,
        "motion": "irregular",
        **reference,
        "constant_motion": const_ref,
    }
    n_bytes = write_json(args.out_reference, reference)
    print(f"reference: {args.out_reference} ({n_bytes/1024:.1f} KB) metric={reference['metric']} "
          f"constant={const_ref['metric']}")


def wiring_initial_weights(net, r_cfg, seed):
    fresh = HomeostaticReservoir(r_cfg, seed=seed)
    return fresh.weights


def replay_schedule_exact(cfg: VariableTrackingConfig, sched: dict, n_steps: int) -> np.ndarray:
    """Pure-Python replay of VariableTrackingEnv.advance_stimulus.

    This is the algorithm the JS port implements: index 0 of every list is the
    constructor draw; each later target draw is immediately followed by its
    interval draw, so the two speed lists advance together.
    """
    targets = sched["speed_targets"]
    s_iv = sched["speed_intervals"]
    d_iv = sched["direction_intervals"]
    angle = cfg.initial_stimulus_angle
    direction = cfg.initial_stimulus_direction
    current = float(cfg.stimulus_speed)
    target, speed_left, dir_left = targets[0], s_iv[0], d_iv[0]
    k_speed, k_dir = 1, 1
    out = np.empty(n_steps)
    for t in range(n_steps):
        out[t] = angle
        angle = (angle + direction * current) % 360.0
        speed_left -= 1
        if speed_left == 0:
            target = targets[k_speed]
            speed_left = s_iv[k_speed]
            k_speed += 1
        current += cfg.speed_smoothing * (target - current)
        current = float(np.clip(current, cfg.stimulus_speed_min, cfg.stimulus_speed_max))
        dir_left -= 1
        if dir_left == 0:
            direction *= -1
            dir_left = d_iv[k_dir]
            k_dir += 1
    return out


# --------------------------------------------------------------------------
# pong
# --------------------------------------------------------------------------

def pong_configs(loadout: str):
    from viz.pong_server import LOADOUT_BY_ID, PONG_PARAMS, RESERVOIR_PARAMS
    entry = LOADOUT_BY_ID[loadout]
    p = entry["params"]
    pong_cfg = PongConfig(**{k: v for k, v in p.items() if k in PONG_PARAMS})
    r_cfg = dataclasses.replace(
        PONG_RESERVOIR_CONFIG,
        n_inputs=pong_cfg.n_sensors,
        **{k: v for k, v in p.items() if k in RESERVOIR_PARAMS},
    )
    return entry, r_cfg, pong_cfg


def export_pong(args) -> None:
    entry, r_cfg, pong_cfg = pong_configs(args.loadout)
    sim = PongSimulation(r_cfg, pong_cfg, seed=args.seed)
    env = sim.env
    initial_dy = float(env.dy)                # the constructor's one draw
    proxy = RecordingRng(env.rng)
    env.rng = proxy                           # every later draw happens in step_ball
    net = sim.network
    n = r_cfg.n_nodes
    wiring = wiring_payload(net)

    trace = {k: [] for k in ("sum_x", "sum_t", "sum_w", "mean_abs_e", "ball_x", "ball_y",
                             "paddle_y", "bearing_sensed", "event", "outputs", "spikes")}
    spike_lists = []
    cps = {}
    serve_y_seen = []
    for t in range(args.steps):
        bearing = env.ball_angle()
        state, ev, _dp = sim.step()
        ids = np.flatnonzero(state.spiked).tolist()
        spike_lists.append(ids)
        if ev == "miss":
            serve_y_seen.append(env.ball_y)
        if t < TRACE_STEPS:
            s = net_sums(net)
            trace["sum_x"].append(s["sum_x"])
            trace["sum_t"].append(s["sum_t"])
            trace["sum_w"].append(s["sum_w"])
            trace["mean_abs_e"].append(math.fsum(np.abs(state.error).tolist()) / n)
            trace["ball_x"].append(float(env.ball_x))
            trace["ball_y"].append(float(env.ball_y))
            trace["paddle_y"].append(float(env.paddle_y))
            trace["bearing_sensed"].append(float(bearing))
            trace["event"].append(1 if ev == "hit" else (-1 if ev == "miss" else 0))
            trace["outputs"].append([float(v) for v in state.outputs])
            trace["spikes"].append(ids)
        if (t + 1) in CHECKPOINTS or (t + 1) == args.steps:
            cp = net_sums(net)
            cp.update({
                "t": t + 1,
                "ball": [float(env.ball_x), float(env.ball_y)],
                "paddle_y": float(env.paddle_y),
                "n_opportunities": len(env.hits),
                "x": encode(net.x, "f64"),
                "targets": encode(net.targets, "f64"),
            })
            if (t + 1) == args.steps:
                cp["w"] = encode(net.weights[net.adjacency], "f64")
            cps[t + 1] = cp
    hits = [int(h) for h in env.hits]

    # Extend the serve list past the run.
    while sum(1 for k, _a, _v in proxy.log if k == "uniform") < args.serves:
        proxy.uniform(pong_cfg.reserve_y_low, pong_cfg.reserve_y_high)
        proxy.choice([-pong_cfg.ball_speed_y, pong_cfg.ball_speed_y])
    serve_y = [v for k, _a, v in proxy.log if k == "uniform"]
    serve_dy = [v for k, _a, v in proxy.log if k == "choice"]
    assert len(serve_y) == len(serve_dy) >= args.serves
    assert serve_y[:len(serve_y_seen)] == serve_y_seen, "serve list disagrees with the run"

    payload = {
        "format": WIRING_FORMAT,
        "generated": provenance("pong"),
        "task": "pong",
        "loadout": {"id": entry["id"], "label": entry["label"], "seed": args.seed},
        "reservoir": reservoir_payload(r_cfg),
        "wiring": wiring,
        "env": {
            "kind": "pong",
            "width": pong_cfg.width,
            "height": pong_cfg.height,
            "paddle_x": pong_cfg.paddle_x,
            "paddle_half_height": pong_cfg.paddle_half_height,
            "paddle_start_y": pong_cfg.paddle_start_y,
            "ball_start": [pong_cfg.ball_start_x, pong_cfg.ball_start_y],
            "ball_speed": [pong_cfg.ball_speed_x, pong_cfg.ball_speed_y],
            "gain": pong_cfg.gain,
            "hit_push": pong_cfg.hit_push,
            "x_bounce": pong_cfg.x_bounce,
            "x_miss": pong_cfg.x_miss,
            "y_max": pong_cfg.y_max,
            "y_min": pong_cfg.y_min,
            "reserve_x": pong_cfg.reserve_x,
            "reserve_y": [pong_cfg.reserve_y_low, pong_cfg.reserve_y_high],
            "step_length": float(np.hypot(pong_cfg.ball_speed_x, pong_cfg.ball_speed_y)),
            "sensor_values": pong_cfg.sensor_values.tolist(),
            "sensor_tolerance": pong_cfg.sensor_tolerance,
            "sensor_inclusive": pong_cfg.sensor_inclusive,
            "chance_hit_rate": pong_cfg.chance_hit_rate,
            "initial_dy": initial_dy,
            "serves": {"y": encode(serve_y, "f64"), "dy": encode(serve_dy, "i8")},
        },
    }
    n_bytes = write_json(args.out_wiring, payload)
    print(f"wiring: {args.out_wiring} ({n_bytes/1024:.1f} KB) links={wiring['n_links']} "
          f"input_links={wiring['n_input_links']} effectors={wiring['checks']['effector_in_degree']} "
          f"serves={len(serve_y)} (run used {len(serve_y_seen)})")
    reference = {
        "format": REFERENCE_FORMAT,
        "generated": provenance("pong"),
        "wiring_sha256": hashlib.sha256(pathlib.Path(args.out_wiring).read_bytes()).hexdigest(),
        "task": "pong",
        "loadout": entry["id"],
        "seed": args.seed,
        "n_steps": args.steps,
        "trace": trace,
        "checkpoints": [cps[k] for k in sorted(cps)],
        "spike_hash": {"block": HASH_BLOCK, "fnv1a32": fnv1a32_blocks(spike_lists)},
        "hits": hits,
        "metric": {"hit_rate": float(np.mean(hits)), "n_opportunities": len(hits)},
    }
    n_bytes = write_json(args.out_reference, reference)
    print(f"reference: {args.out_reference} ({n_bytes/1024:.1f} KB) metric={reference['metric']}")


# --------------------------------------------------------------------------
# one neuron
# --------------------------------------------------------------------------

def export_neuron(args) -> None:
    """A 240-step replay of one moderately active neuron of the tracking net."""
    entry, r_cfg, v_cfg, c_cfg = tracking_configs(args.loadout)
    n_pass1 = args.t_max + args.seg + 10
    sim = TrackingSimulation(r_cfg, c_cfg, seed=args.seed)
    sim.env = VariableTrackingEnv(v_cfg, rng=sim.network.rng)
    net = sim.network
    n = r_cfg.n_nodes
    x_post = np.zeros((n_pass1, n), dtype=np.float32)
    thr = np.zeros((n_pass1, n), dtype=np.float32)
    spiked = np.zeros((n_pass1, n), dtype=bool)
    for t in range(n_pass1):
        thr[t] = net.thresholds
        state, _ = sim.step()
        x_post[t] = state.x
        spiked[t] = state.spiked
    rate = spiked[args.t_min:].mean(0)
    in_deg = net.adjacency.sum(0)
    sens = net.input_adjacency.sum(0)
    cand = np.flatnonzero((rate > 0.10) & (rate < 0.6) & (in_deg >= 8) & (sens >= 1))
    if cand.size == 0:
        cand = np.argsort(np.abs(rate - 0.3))[:20]
    cand = cand[np.argsort(np.abs(rate[cand] - 0.30))][:40]

    def score_window(j, t0):
        sp = spiked[t0:t0 + args.seg, j]
        e = x_post[t0:t0 + args.seg, j] - thr[t0:t0 + args.seg, j] / r_cfg.threshold_ratio
        branches = len(np.unique(np.where(e >= 0, 1, 0) + 2 * sp.astype(int)))
        s = 3.0 * branches - 6.0 * abs(sp.mean() - 0.30)
        runs = np.diff(np.flatnonzero(np.r_[True, sp, True]))
        s -= 0.15 * max(0, int(runs.max()) - 30)
        if sp[:4].any():
            s -= 1.0                                    # a calm start reads better
        seg_t = thr[t0:t0 + args.seg, j] / r_cfg.threshold_ratio
        s += min(float(seg_t.std()) * 40.0, 2.0)        # the target visibly moves
        return s

    best, best_s = None, -1e9
    for j in cand:
        for t0 in range(args.t_min, args.t_max, 20):
            s = score_window(int(j), t0)
            if s > best_s:
                best, best_s = (int(j), t0), s
    j, t0 = best
    print(f"neuron {j}: rate {rate[j]:.3f}, in-degree {in_deg[j]}, sensors {sens[j]}, window {t0}..{t0+args.seg}")

    # pass 2: rerun and record the chosen neuron in detail
    sim = TrackingSimulation(r_cfg, c_cfg, seed=args.seed)
    sim.env = VariableTrackingEnv(v_cfg, rng=sim.network.rng)
    net = sim.network
    in_nbrs = np.flatnonzero(net.adjacency[:, j])
    rec = {k: [] for k in ("x_pre", "x_post", "target", "threshold", "spiked", "n_in_spiked",
                           "mean_in_weight", "sensory", "recurrent")}
    link_w = []
    link_spiked = []
    for t in range(t0 + args.seg):
        if t >= t0:
            prev_spiked = net.spiked.copy()
            w_col = net.weights[in_nbrs, j].copy()
            inputs = sim.env.sense()
            sensory = float(inputs @ net.input_weights[:, j])
            recurrent = float(net._spiked_f @ net.weights[:, j])
            x_pre = float(net.x[j] * (1.0 - r_cfg.leak) + sensory + recurrent)
            threshold = float(net.thresholds[j])
            target = float(net.targets[j])
        state, _ = sim.step()
        if t >= t0:
            rec["x_pre"].append(x_pre)
            rec["x_post"].append(float(state.x[j]))
            rec["target"].append(target)
            rec["threshold"].append(threshold)
            rec["spiked"].append(bool(state.spiked[j]))
            rec["n_in_spiked"].append(int(prev_spiked[in_nbrs].sum()))
            rec["mean_in_weight"].append(float(w_col.mean()))
            rec["sensory"].append(sensory)
            rec["recurrent"].append(recurrent)
            link_w.append(w_col.tolist())
            link_spiked.append(prev_spiked[in_nbrs].tolist())
    link_w = np.asarray(link_w)
    link_spiked = np.asarray(link_spiked)
    # the five in-links that matter most in this window: most arrivals, then weight
    order = np.lexsort((-np.abs(link_w).mean(0), -link_spiked.sum(0)))[:5]
    links = [{
        "source": int(in_nbrs[i]),
        "w": [round(float(v), 5) for v in link_w[:, i]],
        "spiked": [int(v) for v in link_spiked[:, i]],
    } for i in order]
    r5 = lambda xs: [round(float(v), 5) for v in xs]  # noqa: E731
    payload = {
        "format": NEURON_FORMAT,
        "generated": provenance("neuron"),
        "source": {"loadout": entry["id"], "seed": args.seed, "neuron": j, "t0": t0, "n_steps": args.seg,
                   "in_degree": int(len(in_nbrs)), "n_sensor_links": int(sens[j]),
                   "effectors": [int(k) for k in np.flatnonzero(net.output_adjacency[j])]},
        "params": {"leak": r_cfg.leak, "target_lr": r_cfg.target_lr, "threshold_ratio": r_cfg.threshold_ratio,
                   "weight_lr": r_cfg.weight_lr, "target_floor": r_cfg.target_floor},
        "steps": {
            "x_pre": r5(rec["x_pre"]), "x_post": r5(rec["x_post"]), "target": r5(rec["target"]),
            "threshold": r5(rec["threshold"]), "spiked": [int(s) for s in rec["spiked"]],
            "n_in_spiked": rec["n_in_spiked"], "mean_in_weight": r5(rec["mean_in_weight"]),
            "sensory": r5(rec["sensory"]), "recurrent": r5(rec["recurrent"]),
        },
        "links": links,
    }
    n_bytes = write_json(args.out_wiring, payload)
    spikes = sum(rec["spiked"])
    print(f"neuron replay: {args.out_wiring} ({n_bytes/1024:.1f} KB) spikes={spikes}/{args.seg} "
          f"target {min(rec['target']):.3f}..{max(rec['target']):.3f}")


# --------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=("tracking", "pong", "neuron"), required=True)
    ap.add_argument("--loadout", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--steps", type=int, default=21600)
    ap.add_argument("--horizon", type=int, default=100_000, help="tracking: schedule length to export")
    ap.add_argument("--serves", type=int, default=1024, help="pong: serves to export")
    ap.add_argument("--t-min", type=int, default=5000, help="neuron: earliest window start")
    ap.add_argument("--t-max", type=int, default=7000, help="neuron: latest window start")
    ap.add_argument("--seg", type=int, default=240, help="neuron: replay length")
    ap.add_argument("--out-wiring", required=True)
    ap.add_argument("--out-reference")
    args = ap.parse_args()
    if args.task in ("tracking", "pong") and not args.out_reference:
        ap.error("--out-reference is required for tracking and pong")
    {"tracking": export_tracking, "pong": export_pong, "neuron": export_neuron}[args.task](args)


if __name__ == "__main__":
    main()
