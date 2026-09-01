"""Lab viewers: interactive pages for the design-space campaign (scripts/lab/).

A self-contained FastAPI sub-application, mounted at /lab by viz.server. Same
ground rule as the other visualizers: no model logic lives here — every
simulated number comes from the tested `homeostasis` package, run server-side;
the frontend only renders JSON.

Pages:
  /lab        single-node explorer — the scripts/lab/k2_single_node.py
              protocol: one node (no recurrence possible at p_link=0) under
              constant drive mu, end state classified with k2's rules and the
              duty-law prediction f = (mu/T - leak)/rho next to observed f.
  /lab/phase  phase-map browser — heatmaps over the recorded A1 (wlr x tlr)
              and A3 (leak x wlr) sweeps from scripts/out/lab (display only;
              nothing is simulated).
  /lab/traj   trajectory viewer — one tracking run with the exact
              scripts/lab/common.py run_closed_loop step order, plus an
              optional effector swap (the "swap-mid" arm generalized to an
              arbitrary step).
  /lab/wall   arena viewer for the wall-avoidance case study (package
              run_wall, the scripts/lab/wall_rep.py arms), plus the H30
              evolved edge-holder — a TRACKING run — as a fifth variant.
  /lab/pursuit  the pursuit task (package run_pursuit): a champion genome
              (the H34 orbital pair or the H55 blind sweeper) on orbit,
              waypoint, or ballistic stimulus motion with adjustable speed.
              Protocol is exactly scripts/lab/h34b_verify.py's run_one /
              h55b_horizon.py's evaluate; ballistic runs also report the
              per-crossing catch rate of scripts/lab/h55_intercept.py.
  /lab/ecology  the live two-agent homeostatic chain: a blind wall-circling
              pacemaker plus the H48e warm-started follower champion sensing
              its live position. Co-simulation loop is exactly
              scripts/lab/h48c_live_chain.py's cosim.
  /lab/ecology3  the shared-visibility three-agent ecology (H81 -> H85b):
              pacemaker + two followers, every follower sensing ALL other
              agents, under the four measured attention regimes (summed
              retina / memoryless WTA / sticky 2x100 / sticky 5x300). The
              attention rule is scripts/lab/h85_shared.py's StickyFollower,
              imported — not duplicated — and the co-simulation loop is
              exactly h85_shared.run's.
  /lab/repair  the H53 self-repair exhibit: the ridge config with a mid-run
              node kill, learning-on vs learning-frozen side by side. Runs
              go straight through the lab harness (scripts/lab/common.py
              run_closed_loop, arms kill-mid / kill-mid-frozen), so the
              displayed numbers are the H53 campaign's numbers.

Run via the main app:  uvicorn viz.server:app --port 8471  ->  /lab
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import dataclasses

from homeostasis.pursuit import PursuitConfig, PursuitEnv
from homeostasis.reservoir import HomeostaticReservoir, ReservoirConfig
from homeostasis.simulation import (
    WALL_RESERVOIR_CONFIG,
    WallSimulation,
    run_pursuit,
    run_tracking,
    run_wall,
)
from homeostasis.tracking import TrackingConfig, TrackingEnv
from homeostasis.wall import WallConfig

# The lab harness itself (scripts/lab is not a package, so load it by path).
# The repair page replays H53 through run_closed_loop rather than duplicating
# the kill surgery / freeze semantics here — same ground rule as model logic:
# where a lab page replays a campaign experiment, it imports the harness.
import importlib.util

_COMMON_PATH = Path(__file__).resolve().parent.parent / "scripts/lab/common.py"
_common_spec = importlib.util.spec_from_file_location("lab_common", _COMMON_PATH)
lab_common = importlib.util.module_from_spec(_common_spec)
_common_spec.loader.exec_module(lab_common)

# The H85 harness (StickyFollower — the sticky-attention rule — plus the
# pacemaker config and champion pair it runs). It imports its lab-script
# siblings (h50_depth -> h48c_live_chain -> common / h33_evolve_pursuit) by
# bare name, so the lab dir must be importable while it loads; it is removed
# again right after so the server's import space stays clean.
_H85_PATH = _COMMON_PATH.parent / "h85_shared.py"
_h85_spec = importlib.util.spec_from_file_location("lab_h85_shared", _H85_PATH)
lab_h85 = importlib.util.module_from_spec(_h85_spec)
sys.path.insert(0, str(_H85_PATH.parent))
try:
    _h85_spec.loader.exec_module(lab_h85)
finally:
    sys.path.remove(str(_H85_PATH.parent))

STATIC_DIR = Path(__file__).parent / "static"
OUT_DIR = Path(__file__).resolve().parent.parent / "scripts/out"

MAX_STEPS = 20_000
DEFAULT_STEPS = 3_000
TAIL = 500  # classification window, same as scripts/lab/k2_single_node.py

# name -> (min, max, type); values outside are clamped, like the other viewers
PARAM_SPECS = {
    "mu": (0.001, 100.0, float),
    "leak": (0.0, 1.0, float),
    "rho": (1.0, 10.0, float),
    "target_lr": (0.0, 1.0, float),
    "steps": (10, MAX_STEPS, int),
}


def _clamp(name: str, value) -> float | int:
    lo, hi, cast = PARAM_SPECS[name]
    return min(max(cast(value), lo), hi)


def run_single_node(
    mu: float, leak: float, rho: float, target_lr: float, steps: int, ic: str
) -> dict:
    """One node under constant drive mu; the k2_single_node.py protocol.

    Records per step (all straight from StepState, no model logic here):
    x_post = state.x (post-spike-subtraction x'_t), spiked, T = state.targets
    (the post-update T_{t+1}, as k2 records it), threshold = rho*T, and
    E = state.error (x'_t - T_t against the pre-update target).
    """
    cfg = ReservoirConfig(
        n_nodes=1, n_inputs=1, n_outputs=1, p_link=0.0, input_p_link=1.0,
        input_weight=mu, leak=leak, threshold_ratio=rho, target_lr=target_lr,
    )
    net = HomeostaticReservoir(cfg, seed=0)
    if ic == "hot":
        net.x[:] = 3.0 * mu / max(leak, 1e-9)
    inp = np.ones(1)
    x_post = np.empty(steps)
    spiked = np.empty(steps, dtype=bool)
    T = np.empty(steps)
    E = np.empty(steps)
    for i in range(steps):
        s = net.step(inp)
        x_post[i] = s.x[0]
        spiked[i] = s.spiked[0]
        T[i] = s.targets[0]
        E[i] = s.error[0]

    # classification over the tail, exactly k2_single_node.py's rules
    tail = min(TAIL, steps)
    f_l, E_l, T_l = spiked[-tail:], E[-tail:], T[-tail:]
    f_late = float(f_l.mean())
    mean_E = float(E_l.mean())
    mean_absE = float(np.abs(E_l).mean())
    T_late = float(T_l[-1])
    at_floor = bool(T_l[-1] <= cfg.target_floor + 1e-9)
    if f_late == 0.0:
        state = "dead-floor" if (at_floor and mean_E < -0.02) else "silent-comf"
    else:
        state = "spiking" if abs(mean_E) < 0.02 else "frozen-cycle"
    duty_pred = (mu / T_late - leak) / rho if f_late > 0 else None

    return {
        "params": {
            "mu": mu, "leak": leak, "rho": rho, "target_lr": target_lr,
            "steps": steps, "ic": ic,
        },
        "trace": {
            "x_post": np.round(x_post, 5).tolist(),
            "spiked": spiked.astype(int).tolist(),
            "T": np.round(T, 5).tolist(),
            "threshold": np.round(rho * T, 5).tolist(),
            "E": np.round(E, 5).tolist(),
        },
        "summary": {
            "state": state,
            "f_late": round(f_late, 4),
            "duty_pred": None if duty_pred is None else round(duty_pred, 4),
            "T_late": round(T_late, 4),
            "meanE_late": round(mean_E, 4),
            "absE_late": round(mean_absE, 4),
            "tail": tail,
            "target_floor": cfg.target_floor,
            # the two analytic boundaries the page talks about
            "mu_comfort": round(leak * cfg.target_floor, 4),        # comfort split
            "mu_spike_cold": round(rho * leak * cfg.target_init, 4),  # cold-start crossing
        },
    }


# ---------------------------------------------------------------------------
# Phase-map browser (/lab/phase): recorded sweep data, slimmed for transport.
# ---------------------------------------------------------------------------

# First existing file wins; cluster1_results.json (the re-run) supersedes
# act2_batch1.json when present. Same row schema. Checked per request (a
# cheap stat), so a file that appears later is picked up without a restart.
PHASE_SOURCES = ("lab/cluster1_results.json", "lab/act2_batch1.json")
PHASE_TAGS = ("A1", "A3")
PHASE_FIELDS = (
    "tag", "seed", "wlr", "tlr", "leak",
    "score_late", "prop_spiked", "g_final", "mean_abs_E", "input_flow",
)
# Reservoir fields a tracking loadout can pin (mirrors the picker's params)
_LOADOUT_FIELDS = (
    "n_nodes", "p_link", "input_weight", "weight_init_mean", "weight_init_sd",
    "leak", "target_lr", "weight_lr", "threshold_ratio",
)


def _cell_links(rows: list[dict]) -> list[dict]:
    """Map phase cells to tracking-visualizer loadouts, where one matches.

    A cell matches a loadout when the cell's effective configuration (package
    defaults + the sweep's overrides) equals the loadout's effective
    configuration. Pure config comparison — no model logic.
    """
    from .server import CONFIG_LOADOUTS  # late import: viz.server mounts us

    tdef = TrackingConfig()
    rdef = ReservoirConfig(n_inputs=tdef.n_sensors)
    base = {f: getattr(rdef, f) for f in _LOADOUT_FIELDS}
    base["gain"] = tdef.gain

    comparable = []
    for loadout in CONFIG_LOADOUTS:
        params = loadout.get("params", {})
        if set(params) <= set(base):
            comparable.append((loadout["id"], {**base, **params}))

    links = []
    seen: set[tuple] = set()
    for r in rows:
        if r["tag"] == "A1":
            x, y = r["wlr"], r["tlr"]
            overrides = {"weight_lr": x, "target_lr": y}
        else:  # A3
            x, y = r["leak"], r["wlr"]
            overrides = {"leak": x, "weight_lr": y}
        key = (r["tag"], x, y)
        if key in seen:
            continue
        seen.add(key)
        eff = {**base, **overrides}
        for loadout_id, lo_eff in comparable:
            if all(np.isclose(lo_eff[k], v) for k, v in eff.items()):
                links.append({"tag": r["tag"], "x": x, "y": y, "loadout": loadout_id})
                break
    return links


# ---------------------------------------------------------------------------
# Trajectory viewer (/lab/traj): one tracking run, common.py's step order.
# ---------------------------------------------------------------------------

TRAJ_MAX_STEPS = 14_400
TRAJ_DEFAULT_STEPS = 7_200
TRAJ_SUBSAMPLE = 4
SEG = 720  # one stimulus-reversal segment, as scripts/lab/common.py


def _traj_variants() -> dict[str, tuple[dict, dict]]:
    """(reservoir overrides, tracking overrides) per variant.

    Exactly scripts/lab/act2_batch3.py: ridge25 = default config with
    leak .25 (the default) and weight_lr .1; w1prime = sweep config 236 with
    weight_init_mean reverted to 0.75, keeping 236's gain.
    """
    variants = {
        "default": ({}, {}),
        "ridge25": ({"leak": 0.25, "weight_lr": 0.1}, {}),
        # H51's under-plastic "statue" cell: sensor noise sigma=0.1 rescues it
        "statue03": ({"leak": 0.25, "weight_lr": 0.03}, {}),
    }
    sweep = OUT_DIR / "sweep/results.json"
    if sweep.exists():
        cfg = json.loads(sweep.read_text())["configs"][236]
        res = dict(
            n_nodes=cfg["n_nodes"], p_link=cfg["p_link"],
            input_weight=cfg["input_weight"], weight_init_mean=0.75,
            weight_init_sd=cfg["weight_init_sd"], leak=cfg["leak"],
            target_lr=cfg["target_lr"], threshold_ratio=cfg["threshold_ratio"],
        )
        variants["w1prime"] = (res, {"gain": cfg["gain"]})
    return variants


TRAJ_VARIANTS = _traj_variants()


def run_traj(
    variant: str, seed: int, steps: int, swap_at: int | None,
    sensor_noise: float = 0.0,
) -> dict:
    """One closed-loop tracking run, the exact scripts/lab/common.py
    run_closed_loop step order (sense -> net.step -> apply_action ->
    advance_stimulus). The optional effector swap is the "swap-mid" arm
    generalized: from step swap_at on, the two outputs are exchanged before
    being passed to apply_action. sensor_noise is common.py's "sensor_noise"
    task key — environment/harness logic, not model logic: uniform(+-sigma)
    added to the sensed activations, clamped at 0, from its own rng stream
    seed+900001, so numbers match the H51 harness runs bit for bit. The
    simulation itself is entirely the tested package; this function only
    records observables.
    """
    res_over, trk_over = TRAJ_VARIANTS[variant]
    tcfg = TrackingConfig(**trk_over)
    res_kwargs = dict(res_over)
    res_kwargs.setdefault("n_inputs", tcfg.n_sensors)
    rcfg = ReservoirConfig(**res_kwargs)
    net = HomeostaticReservoir(rcfg, seed=seed)
    env = TrackingEnv(tcfg)
    noise_rng = np.random.default_rng(seed + 900001) if sensor_noise > 0 else None

    heading = np.empty(steps)
    stim = np.empty(steps)
    err = np.empty(steps)
    dh_arr = np.empty(steps)
    prop = np.empty(steps)
    n_seg = max(1, steps // SEG)
    seg_in45 = np.zeros(n_seg)
    seg_count = np.zeros(n_seg)

    for i in range(steps):
        herr = env.heading_error()
        heading[i] = env.heading
        stim[i] = env.stimulus_angle
        err[i] = herr
        inputs = env.sense()
        if noise_rng is not None:  # common.py's exact noise convention
            inputs = np.maximum(inputs + noise_rng.uniform(
                -sensor_noise, sensor_noise, inputs.shape), 0.0)
        state = net.step(inputs)
        e_left, e_right = state.outputs
        if swap_at is not None and i >= swap_at:
            e_left, e_right = e_right, e_left
        dh_arr[i] = env.apply_action(e_left, e_right)
        env.advance_stimulus()
        prop[i] = state.prop_spiked
        seg = min(i // SEG, n_seg - 1)
        seg_count[seg] += 1
        if abs(herr) <= 45.0:
            seg_in45[seg] += 1

    seg_scores = (seg_in45 / np.maximum(seg_count, 1)).tolist()
    late = seg_scores[5:] if n_seg >= 10 else seg_scores
    # unwrap at full resolution (presentation math), then subsample
    h_un = np.unwrap(heading, period=360.0)
    s_un = np.unwrap(stim, period=360.0)
    sl = slice(0, steps, TRAJ_SUBSAMPLE)

    return {
        "params": {
            "variant": variant, "seed": seed, "steps": steps,
            "swap_at": swap_at, "sensor_noise": sensor_noise,
            "subsample": TRAJ_SUBSAMPLE, "seg_len": SEG,
        },
        "config": {
            "n_nodes": rcfg.n_nodes, "leak": rcfg.leak,
            "weight_lr": rcfg.weight_lr, "target_lr": rcfg.target_lr,
            "weight_init_mean": rcfg.weight_init_mean, "gain": tcfg.gain,
        },
        "trace": {
            "t": np.arange(steps)[sl].tolist(),
            "heading": np.round(h_un[sl], 2).tolist(),
            "stimulus": np.round(s_un[sl], 2).tolist(),
            "err": np.round(err[sl], 2).tolist(),
            "dh": np.round(dh_arr[sl], 3).tolist(),
            "prop": np.round(prop[sl], 4).tolist(),
        },
        "summary": {
            "score": round(float(np.mean(seg_scores)), 4),
            "score_late": round(float(np.mean(late)), 4),
            "seg_scores": [round(s, 4) for s in seg_scores],
            "prop_spiked": round(float(prop.mean()), 4),
        },
    }


# ---------------------------------------------------------------------------
# Wall-avoidance viewer (/lab/wall): case study 3, plus the evolved
# edge-holder (a TRACKING run) as its fifth variant.
# ---------------------------------------------------------------------------

WALL_MAX_STEPS = 14_400
WALL_DEFAULT_STEPS = 3_600
WALL_SUBSAMPLE = 2
LATE_WINDOW = 1_000  # summary window, as scripts/lab/wall_rep.py
FLANK_FILE = OUT_DIR / "lab/h30_evolve_flank.json"

# The evolved edge-holder from the H30 flank evolution: the last generation's
# champion dict (ReservoirConfig fields plus the tracking gain).
FLANK_CHAMPION = (
    json.loads(FLANK_FILE.read_text())[-1]["champion"] if FLANK_FILE.exists() else None
)

WALL_VARIANTS = {
    # variant -> (WallConfig, learning_enabled); exactly scripts/lab/wall_rep.py's arms
    "base": (WallConfig(), True),
    "perturb": (WallConfig(perturb_at=1000), True),
    "noise": (WallConfig(sensor_noise=0.2), True),
    "no-learn": (WallConfig(), False),
}


def _pair_any(flags: np.ndarray, k: int) -> np.ndarray:
    """OR-reduce consecutive groups of k, so subsampling drops no hit events."""
    n = (len(flags) // k) * k
    return flags[:n].reshape(-1, k).any(axis=1)


def run_wall_variant(variant: str, seed: int, steps: int, wlr: float, tlr: float) -> dict:
    """One wall-avoidance run (the tested package's run_wall), subsampled for
    transport. wlr/tlr replace weight_lr/target_lr on WALL_RESERVOIR_CONFIG
    (defaults 1.0 / 0.01 = the released values); config plumbing only."""
    wall_config, learning = WALL_VARIANTS[variant]
    rcfg = dataclasses.replace(WALL_RESERVOIR_CONFIG, weight_lr=wlr, target_lr=tlr)
    h = run_wall(
        n_steps=steps, seed=seed, learning_enabled=learning,
        reservoir_config=rcfg, wall_config=wall_config,
    )
    sl = slice(0, steps, WALL_SUBSAMPLE)
    late = slice(-min(LATE_WINDOW, steps), None)
    return {
        "kind": "wall",
        "params": {
            "variant": variant, "seed": seed, "steps": steps,
            "wlr": wlr, "tlr": tlr, "subsample": WALL_SUBSAMPLE,
        },
        "config": {
            "box_size": wall_config.box_size,
            "agent_radius": wall_config.agent_radius,
            "sensor_angles": list(wall_config.sensor_angles),
            "perturb_at": wall_config.perturb_at,
            "sensor_noise": wall_config.sensor_noise,
            "learning": learning,
            "n_nodes": rcfg.n_nodes,
        },
        "trace": {
            "t": np.arange(steps)[sl].tolist(),
            "x": np.round(h.x[sl], 3).tolist(),
            "y": np.round(h.y[sl], 3).tolist(),
            "heading": np.round(h.heading[sl], 4).tolist(),  # radians
            # OR of each pair, so no hit event vanishes from the display
            "hit": _pair_any(h.hit, WALL_SUBSAMPLE).astype(int).tolist(),
            "s_left": np.round(h.inputs[sl, 0], 4).tolist(),   # +45 deg sensor
            "s_right": np.round(h.inputs[sl, 1], 4).tolist(),  # -45 deg sensor
            "prop": np.round(h.prop_spiked[sl], 4).tolist(),
        },
        "summary": {  # all at full resolution, as scripts/lab/wall_rep.py
            "hits_total": int(h.hit.sum()),
            "hits_last_1000": int(h.hit[late].sum()),
            "late_mean_abs_dh": round(float(np.abs(h.d_heading[late]).mean()), 4),
            "hit_rate": round(float(h.hit.mean()), 4),
        },
    }


def run_flank_champion(seed: int, steps: int) -> dict:
    """The evolved edge-holder: a TRACKING run of the H30 champion config.
    Selection rewarded holding |heading error| in [50, 90] degrees — a band
    homeostasis alone never aims for — so the agent orbits the stimulus
    off-center indefinitely."""
    if FLANK_CHAMPION is None:
        return {"error": f"{FLANK_FILE} not found"}
    champ = dict(FLANK_CHAMPION)
    gain = champ.pop("gain")
    tcfg = TrackingConfig(gain=gain)
    rcfg = ReservoirConfig(n_inputs=tcfg.n_sensors, **champ)
    hist = run_tracking(
        n_steps=steps, seed=seed, reservoir_config=rcfg,
        tracking_config=tcfg, record_spikes=False,
    )
    sl = slice(0, steps, WALL_SUBSAMPLE)
    abs_err = np.abs(hist.error)
    late = slice(-min(LATE_WINDOW, steps), None)
    return {
        "kind": "tracking",
        "params": {
            "variant": "flank-champion", "seed": seed, "steps": steps,
            "subsample": WALL_SUBSAMPLE,
        },
        "config": {
            "n_nodes": rcfg.n_nodes, "gain": round(gain, 3),
            "leak": round(rcfg.leak, 4), "weight_lr": round(rcfg.weight_lr, 4),
            "target_lr": round(rcfg.target_lr, 5),
        },
        "trace": {
            "t": np.arange(steps)[sl].tolist(),
            "heading": np.round(hist.heading[sl], 2).tolist(),          # degrees
            "stimulus": np.round(hist.stimulus_angle[sl], 2).tolist(),  # degrees
            "err": np.round(hist.error[sl], 2).tolist(),
            "prop": np.round(hist.prop_spiked[sl], 4).tolist(),
        },
        "summary": {
            "band_frac": round(float(np.mean((abs_err >= 50) & (abs_err <= 90))), 4),
            "band_frac_late": round(float(np.mean(
                (abs_err[late] >= 50) & (abs_err[late] <= 90))), 4),
            "within45": round(float(np.mean(abs_err <= 45)), 4),
            "mean_abs_err": round(float(abs_err.mean()), 2),
        },
    }


# ---------------------------------------------------------------------------
# Pursuit viewer (/lab/pursuit): champion genomes on orbit/waypoint/ballistic.
# ---------------------------------------------------------------------------

PURSUIT_MAX_STEPS = 14_400
PURSUIT_DEFAULT_STEPS = 3_600
PURSUIT_SUBSAMPLE = 2
H34_FILE = OUT_DIR / "lab/h34_joint.json"
H55_FILE = OUT_DIR / "lab/h55_champion.json"

# The H34 jointly-evolved champion PAIR: genome (ReservoirConfig fields plus
# wheel_base / intensity_scale for PursuitConfig) and its wiring seed.
_H34_LAST = json.loads(H34_FILE.read_text())[-1] if H34_FILE.exists() else None
H34_CHAMPION = _H34_LAST["champion"] if _H34_LAST else None
H34_CHAMP_SEED = int(_H34_LAST["champ_seed"]) if _H34_LAST else 0

# The H55 ballistic-GA champion: evolved below the re-lock horizon, it is a
# functionally blind sweeper (input_weight at the range floor, drowned by
# recurrence; LEDGER H55/H55b). No champ seed — it is wiring-insensitive.
H55_CHAMPION = (
    json.loads(H55_FILE.read_text())["champion"] if H55_FILE.exists() else None
)

# genome preset -> (champion dict, wiring seed it was evolved with, source)
PURSUIT_GENOMES = {
    "h34-champion": (H34_CHAMPION, H34_CHAMP_SEED, H34_FILE),
    "h55-blind": (H55_CHAMPION, None, H55_FILE),
}
PURSUIT_MOTIONS = ("orbit", "waypoint", "ballistic")
CATCH_R = 1.5  # a catch = closest approach < 1.5, as scripts/lab/h55_intercept.py
_PURSUIT_RES_KEYS = (  # scripts/lab/h34b_verify.py RES_KEYS
    "n_nodes", "p_link", "input_weight", "weight_init_mean",
    "leak", "target_lr", "threshold_ratio", "weight_lr",
)


def _crossing_stats(sx, sy, dist) -> tuple[float, int]:
    """Per-crossing catch rate on ballistic runs — exactly
    scripts/lab/h55_intercept.py crossing_stats: a crossing ends when the
    stimulus jumps >1 unit in one step (a respawn); crossings shorter than
    20 steps are skipped; a catch is any step with dist < CATCH_R."""
    jump = np.hypot(np.diff(sx), np.diff(sy)) > 1.0
    bounds = [0] + (np.flatnonzero(jump) + 1).tolist() + [len(sx)]
    catches, n = 0, 0
    for a, b in zip(bounds[:-1], bounds[1:]):
        if b - a < 20:
            continue
        n += 1
        if float(dist[a:b].min()) < CATCH_R:
            catches += 1
    return (catches / max(n, 1)), n


def run_pursuit_variant(
    genome_key: str, motion: str, speed: float, seed: int, steps: int
) -> dict:
    """One pursuit run of a stored champion genome — the exact construction of
    scripts/lab/h34b_verify.py run_one and h55b_horizon.py evaluate (single
    91-sensor full-circle eye; wheel_base and intensity_scale from the
    genome; stimulus_speed passed through to the config). Summary uses the
    same late window (second half) and definitions as h34b_verify; ballistic
    runs add h55_intercept.py's per-crossing catch stats."""
    champ, champ_seed, src = PURSUIT_GENOMES[genome_key]
    if champ is None:
        return {"error": f"{src} not found"}
    pc = PursuitConfig(
        eye_offsets=(0.0,), sensors_per_eye=91,
        wheel_base=champ["wheel_base"],
        intensity_scale=champ["intensity_scale"],
        stimulus_motion=motion,
        stimulus_speed=speed,
    )
    res = ReservoirConfig(
        n_inputs=pc.n_sensors, **{k: champ[k] for k in _PURSUIT_RES_KEYS}
    )
    h = run_pursuit(n_steps=steps, seed=seed, reservoir_config=res, pursuit_config=pc)
    late = slice(steps // 2, None)
    sl = slice(0, steps, PURSUIT_SUBSAMPLE)
    summary = {  # full resolution, definitions of h34b_verify.py
        "dist_late": round(float(h.dist[late].mean()), 4),
        "near3_late": round(float((h.dist[late] < 3).mean()), 4),
        "hits_total": int(h.hit.sum()),
    }
    if motion == "ballistic":
        catch, n_cross = _crossing_stats(h.sx, h.sy, h.dist)
        summary["catch_rate"] = round(catch, 4)
        summary["n_crossings"] = n_cross
    return {
        "kind": "pursuit",
        "params": {
            "genome": genome_key, "motion": motion, "speed": speed,
            "seed": seed, "steps": steps, "subsample": PURSUIT_SUBSAMPLE,
        },
        "config": {
            "motion": motion,
            "box_size": pc.box_size,
            "agent_radius": pc.agent_radius,
            "orbit_radius": pc.orbit_radius,
            "n_nodes": res.n_nodes,
            "wheel_base": round(pc.wheel_base, 3),
            "intensity_scale": round(pc.intensity_scale, 3),
            "input_weight": round(res.input_weight, 3),
            "champ_seed": champ_seed,
        },
        "trace": {
            "t": np.arange(steps)[sl].tolist(),
            "x": np.round(h.x[sl], 3).tolist(),
            "y": np.round(h.y[sl], 3).tolist(),
            "sx": np.round(h.sx[sl], 3).tolist(),
            "sy": np.round(h.sy[sl], 3).tolist(),
            "heading": np.round(h.heading[sl], 4).tolist(),  # radians
            "dist": np.round(h.dist[sl], 3).tolist(),
            "hit": _pair_any(h.hit, PURSUIT_SUBSAMPLE).astype(int).tolist(),
            "prop": np.round(h.prop_spiked[sl], 4).tolist(),
        },
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# Ecology viewer (/lab/ecology): the live two-agent homeostatic chain.
# ---------------------------------------------------------------------------

ECOLOGY_MAX_STEPS = 21_600
ECOLOGY_DEFAULT_STEPS = 7_200
ECOLOGY_SUBSAMPLE = 3  # cosim's own record stride
H48E_FILE = OUT_DIR / "lab/h48e_warm.json"

# Pacemaker: a blind wall circler, exactly scripts/lab/h48c_live_chain.py
ECOLOGY_PACE_CFG = WallConfig(box_size=30.0, initial_x=15.0, initial_y=15.0,
                              wheel_base=2.5)
ECOLOGY_PACE_SEED = 3

# Follower: the H48e warm-started champion pair (genome + wiring seed)
_H48E = json.loads(H48E_FILE.read_text()) if H48E_FILE.exists() else None
H48E_CHAMPION = _H48E["champion"] if _H48E else None
H48E_CHAMP_SEED = int(_H48E["champ_seed"]) if _H48E else 0


def run_ecology(seed: int, steps: int) -> dict:
    """The live chain, the exact co-simulation loop of h48c_live_chain.cosim:
    each step the pacemaker (an unmodified WallSimulation) moves first, the
    follower's stimulus is set to the pacemaker's live position, then the
    follower senses/steps/acts. Both agents are pure package objects; this
    function only records observables."""
    if H48E_CHAMPION is None:
        return {"error": f"{H48E_FILE} not found"}
    pace = WallSimulation(wall_config=ECOLOGY_PACE_CFG, seed=ECOLOGY_PACE_SEED)
    pc = PursuitConfig(
        eye_offsets=(0.0,), sensors_per_eye=91, box_size=30.0,
        initial_agent_x=15.0, initial_agent_y=10.0,
        wheel_base=H48E_CHAMPION["wheel_base"],
        intensity_scale=H48E_CHAMPION["intensity_scale"],
    )
    res = ReservoirConfig(
        n_inputs=pc.n_sensors, **{k: H48E_CHAMPION[k] for k in _PURSUIT_RES_KEYS}
    )
    net = HomeostaticReservoir(res, seed=seed)
    env_b = PursuitEnv(pc, rng=net.rng)

    dist = np.empty(steps)
    ax = np.empty(steps)
    ay = np.empty(steps)
    bx = np.empty(steps)
    by = np.empty(steps)
    b_hit = np.zeros(steps, dtype=bool)
    pace_dh = np.empty(steps)
    for i in range(steps):
        _, dh_a, _ = pace.step()
        env_b.sx, env_b.sy = pace.env.x, pace.env.y
        dist[i] = env_b.distance()
        state = net.step(env_b.sense())
        _, hit_b = env_b.apply_action(*map(float, state.outputs))
        env_b.steps += 1
        ax[i], ay[i] = pace.env.x, pace.env.y
        bx[i], by[i] = env_b.x, env_b.y
        b_hit[i] = hit_b
        pace_dh[i] = dh_a

    late = slice(steps // 2, None)
    sl = slice(0, steps, ECOLOGY_SUBSAMPLE)
    return {
        "kind": "ecology",
        "params": {"seed": seed, "steps": steps, "subsample": ECOLOGY_SUBSAMPLE},
        "config": {
            "box_size": ECOLOGY_PACE_CFG.box_size,
            "agent_radius": ECOLOGY_PACE_CFG.agent_radius,
            "pace_seed": ECOLOGY_PACE_SEED,
            "pace_wheel_base": ECOLOGY_PACE_CFG.wheel_base,
            "n_nodes": res.n_nodes,
            "wheel_base": round(pc.wheel_base, 3),
            "intensity_scale": round(pc.intensity_scale, 3),
            "champ_seed": H48E_CHAMP_SEED,
        },
        "trace": {
            "t": np.arange(steps)[sl].tolist(),
            "ax": np.round(ax[sl], 3).tolist(),
            "ay": np.round(ay[sl], 3).tolist(),
            "bx": np.round(bx[sl], 3).tolist(),
            "by": np.round(by[sl], 3).tolist(),
            "dist": np.round(dist[sl], 3).tolist(),
            "b_hit": _pair_any(b_hit, ECOLOGY_SUBSAMPLE).astype(int).tolist(),
        },
        "summary": {  # full resolution; h48c cosim's late-half definitions
            "near4_late": round(float((dist[late] < 4).mean()), 4),
            "dist_late": round(float(dist[late].mean()), 4),
            "b_hits_total": int(b_hit.sum()),
            # pacemaker turn rate, for the label (deg/step, late half)
            "pace_turn_deg": round(float(np.rad2deg(np.abs(pace_dh[late]).mean())), 2),
        },
    }


# ---------------------------------------------------------------------------
# Shared-visibility ecology (/lab/ecology3): H81 -> H85b, sticky attention.
# ---------------------------------------------------------------------------

ECOLOGY3_MAX_STEPS = 21_600
ECOLOGY3_DEFAULT_STEPS = 10_800  # the H85/H85b campaign length
ECOLOGY3_SUBSAMPLE = 3
ECOLOGY3_B_START = (15.0, 10.0)
ECOLOGY3_C_START = (15.0, 8.0)  # the twin@15,8 geometry of H85/H85b
ECOLOGY3_MAX_SWITCHES = 200  # switch *times* shipped per follower (counts exact)

# mode -> (label, sticky, ratio, patience): the four measured attention
# regimes of the H81 -> H85b progression. "wta" is StickyFollower at
# ratio 1 / patience 1 — selection follows argmax every step, i.e. H82's
# memoryless per-step WTA filter expressed in the same imported rule.
ECOLOGY3_MODES = {
    "off": ("off (summed retina)", False, 2.0, 100),
    "wta": ("memoryless WTA", True, 1.0, 1),
    "sticky": ("sticky 2×/100", True, 2.0, 100),
    "resist": ("sticky 5×/300 (capture-resistant)", True, 5.0, 300),
}


def _running_lock(near: np.ndarray) -> np.ndarray:
    """Running late-half lock fraction: after step i, mean(near[t//2:t]), t=i+1.

    Presentation math only — its final value is exactly the h85_shared.run
    late-half lock the summary reports.
    """
    s = np.concatenate(([0.0], np.cumsum(near)))
    t = np.arange(1, len(near) + 1)
    half = t // 2
    return (s[t] - s[half]) / np.maximum(t - half, 1)


def run_ecology3(mode: str, steps: int) -> dict:
    """The shared-visibility three-agent ecology under one attention regime.

    Exactly scripts/lab/h85_shared.py run(): pacemaker A (an unmodified
    WallSimulation, the h48c PACE_CFG, seed 3) moves first each step; then
    follower B (h48e champion genome+wiring, start (15,10)) and follower C
    (same champion pair, start (15,8)) each sense BOTH other agents from the
    previous step's positions and move. The attention rule (winner-take-all
    source selection with persistence) is lab_h85.StickyFollower — imported,
    never duplicated here — and the loop mirrors run() bit for bit, including
    its initial view of C at (15,5). This function only records observables;
    the sticky-mode summaries reproduce the H85/H85b ledger numbers exactly.
    """
    label, sticky, ratio, patience = ECOLOGY3_MODES[mode]
    champ_seed = lab_h85.CHAMP["champ_seed"]
    A = WallSimulation(wall_config=lab_h85.PACE_CFG, seed=lab_h85.A_SEED)
    B = lab_h85.StickyFollower(champ_seed, ECOLOGY3_B_START, 2, sticky,
                               ratio=ratio, patience=patience)
    C = lab_h85.StickyFollower(champ_seed, ECOLOGY3_C_START, 2, sticky,
                               ratio=ratio, patience=patience)

    ax = np.empty(steps)
    ay = np.empty(steps)
    bx = np.empty(steps)
    by = np.empty(steps)
    cx = np.empty(steps)
    cy = np.empty(steps)
    D = np.zeros((steps, 3))  # B-A, C-A, C-B, as h85_shared.run's columns
    sel_b = np.full(steps, -1, dtype=int)  # 0 = A, 1 = C; -1 = summed (off)
    sel_c = np.full(steps, -1, dtype=int)  # 0 = A, 1 = B; -1 = summed (off)
    pos_b, pos_c = ECOLOGY3_B_START, (15.0, 5.0)  # run()'s exact init
    for i in range(steps):
        A.step()
        pa = (A.env.x, A.env.y)
        new_b = B.step([pa, pos_c], i)
        new_c = C.step([pa, pos_b], i)
        pos_b, pos_c = new_b, new_c
        if sticky:
            sel_b[i] = B.sel
            sel_c[i] = C.sel
        ax[i], ay[i] = pa
        bx[i], by[i] = pos_b
        cx[i], cy[i] = pos_c
        D[i] = (np.hypot(pos_b[0] - pa[0], pos_b[1] - pa[1]),
                np.hypot(pos_c[0] - pa[0], pos_c[1] - pa[1]),
                np.hypot(pos_c[0] - pos_b[0], pos_c[1] - pos_b[1]))

    late = D[steps // 2:]
    sl = slice(0, steps, ECOLOGY3_SUBSAMPLE)
    return {
        "kind": "ecology3",
        "params": {"mode": mode, "steps": steps, "subsample": ECOLOGY3_SUBSAMPLE},
        "config": {
            "label": label, "sticky": sticky, "ratio": ratio, "patience": patience,
            "box_size": lab_h85.PACE_CFG.box_size,
            "agent_radius": lab_h85.PACE_CFG.agent_radius,
            "pace_seed": lab_h85.A_SEED,
            "champ_seed": champ_seed,
            "n_nodes": lab_h85.CHAMP["champion"]["n_nodes"],
            "b_start": ECOLOGY3_B_START,
            "c_start": ECOLOGY3_C_START,
        },
        "trace": {
            "t": np.arange(steps)[sl].tolist(),
            "ax": np.round(ax[sl], 3).tolist(),
            "ay": np.round(ay[sl], 3).tolist(),
            "bx": np.round(bx[sl], 3).tolist(),
            "by": np.round(by[sl], 3).tolist(),
            "cx": np.round(cx[sl], 3).tolist(),
            "cy": np.round(cy[sl], 3).tolist(),
            "d_ba": np.round(D[sl, 0], 3).tolist(),
            "d_ca": np.round(D[sl, 1], 3).tolist(),
            "d_cb": np.round(D[sl, 2], 3).tolist(),
            "lock_ba": np.round(_running_lock(D[:, 0] < 4)[sl], 4).tolist(),
            "lock_ca": np.round(_running_lock(D[:, 1] < 4)[sl], 4).tolist(),
            "lock_cb": np.round(_running_lock(D[:, 2] < 4)[sl], 4).tolist(),
            "sel_b": sel_b[sl].tolist(),
            "sel_c": sel_c[sl].tolist(),
        },
        "summary": {  # full resolution — h85_shared.run's exact definitions
            "B_A": round(float((late[:, 0] < 4).mean()), 4),
            "C_A": round(float((late[:, 1] < 4).mean()), 4),
            "C_B": round(float((late[:, 2] < 4).mean()), 4),
            "b_switches": len(B.switches),
            "c_switches": len(C.switches),
            "b_switch_times": B.switches[:ECOLOGY3_MAX_SWITCHES],
            "c_switch_times": C.switches[:ECOLOGY3_MAX_SWITCHES],
        },
    }


# ---------------------------------------------------------------------------
# Self-repair exhibit (/lab/repair): H53's mid-run node kill, both arms.
# ---------------------------------------------------------------------------

REPAIR_MAX_STEPS = 21_600
REPAIR_DEFAULT_STEPS = 14_400  # H53's length; the kill lands at the midpoint
REPAIR_RES = {"weight_lr": 0.1, "target_lr": 0.01}  # the ridge cell, as h53_selfrepair.py
REPAIR_KILLS = (0.1, 0.3, 0.5)


def run_repair(kill: float, seed: int, steps: int) -> dict:
    """Both H53 arms for one seed, straight through the lab harness
    (scripts/lab/common.py run_closed_loop). The kill surgery — adjacency-level
    removal of kill*N nodes at the midpoint, caches rebuilt, dead nodes left in
    the output-pool denominators — and the frozen variant's learning shutoff
    live in the harness, not here; this function only slims the returned
    observables. pre/drop/rec are h53_selfrepair.py's segment windows (the 5
    segments before the kill, the 2 after, the last 5)."""
    arms = {}
    for key, arm in (("learning", "kill-mid"), ("frozen", "kill-mid-frozen")):
        r = lab_common.run_closed_loop(dict(
            res=dict(REPAIR_RES), seed=seed, n_steps=steps, arm=arm,
            kill_frac=kill))
        ss = r["seg_scores"]
        ks = len(ss) // 2  # first segment after the kill
        arms[key] = {
            "seg_scores": ss,
            "snaps": r["snaps"],
            "score": round(r["score"], 4),
            "score_late": round(r["score_late"], 4),
            "prop_spiked": round(r["prop_spiked"], 4),
            "pre": round(float(np.mean(ss[max(ks - 5, 0):ks])), 4),
            "drop": round(float(np.mean(ss[ks:ks + 2])), 4),
            "rec": round(float(np.mean(ss[-5:])), 4),
        }
    rcfg, _ = lab_common.make_configs(dict(REPAIR_RES))
    return {
        "kind": "repair",
        "params": {
            "kill": kill, "seed": seed, "steps": steps,
            "kill_at": steps // 2, "seg_len": lab_common.SEG,
        },
        "config": {
            "n_nodes": rcfg.n_nodes, "leak": rcfg.leak,
            "weight_lr": rcfg.weight_lr, "target_lr": rcfg.target_lr,
            "n_killed": int(round(kill * rcfg.n_nodes)),
        },
        "arms": arms,
    }


# ---------------------------------------------------------------------------
# App and routes
# ---------------------------------------------------------------------------

lab_app = FastAPI(title="homeostasis lab viewers")


@lab_app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "lab.html")


@lab_app.get("/phase")
async def phase_page():
    return FileResponse(STATIC_DIR / "lab_phase.html")


@lab_app.get("/traj")
async def traj_page():
    return FileResponse(STATIC_DIR / "lab_traj.html")


@lab_app.get("/wall")
async def wall_page():
    return FileResponse(STATIC_DIR / "lab_wall.html")


@lab_app.get("/pursuit")
async def pursuit_page():
    return FileResponse(STATIC_DIR / "lab_pursuit.html")


@lab_app.get("/ecology")
async def ecology_page():
    return FileResponse(STATIC_DIR / "lab_ecology.html")


@lab_app.get("/ecology3")
async def ecology3_page():
    return FileResponse(STATIC_DIR / "lab_ecology3.html")


@lab_app.get("/repair")
async def repair_page():
    return FileResponse(STATIC_DIR / "lab_repair.html")


lab_app.mount("/static", StaticFiles(directory=STATIC_DIR), name="lab-static")


@lab_app.get("/api/single_node")
def single_node(
    mu: float = 0.75,
    leak: float = 0.25,
    rho: float = 2.0,
    target_lr: float = 0.01,
    steps: int = DEFAULT_STEPS,
    ic: str = "cold",
):
    # sync endpoint: FastAPI runs it in the threadpool, so a 20k-step run
    # never blocks the websocket loops of the other visualizers
    return run_single_node(
        mu=_clamp("mu", mu),
        leak=_clamp("leak", leak),
        rho=_clamp("rho", rho),
        target_lr=_clamp("target_lr", target_lr),
        steps=_clamp("steps", steps),
        ic="hot" if ic == "hot" else "cold",
    )


@lab_app.get("/api/phase")
def phase():
    """Recorded A1/A3 sweep rows, slimmed (no policy/snaps leave the server)."""
    for rel in PHASE_SOURCES:
        path = OUT_DIR / rel
        if path.exists():
            break
    else:
        return {"source": None, "rows": [], "links": [],
                "error": f"none of {PHASE_SOURCES} found under {OUT_DIR}"}
    rows = json.loads(path.read_text())
    # Some archived runs diverged to inf/nan (g_final, mean_abs_E); strict
    # JSON forbids those, so ship them as null and let the page print "—".
    fin = lambda v: v if not isinstance(v, float) or np.isfinite(v) else None
    slim = [
        {k: fin(r[k]) for k in PHASE_FIELDS if k in r}
        for r in rows
        if r.get("tag") in PHASE_TAGS and r.get("arm", "full") == "full"
    ]
    return {"source": rel, "rows": slim, "links": _cell_links(slim)}


@lab_app.get("/api/traj")
def traj(
    variant: str = "ridge25",
    seed: int = 0,
    steps: int = TRAJ_DEFAULT_STEPS,
    swap_at: int = -1,
    noise: float = 0.0,
):
    if variant not in TRAJ_VARIANTS:
        return {"error": f"unknown variant {variant!r}; have {sorted(TRAJ_VARIANTS)}"}
    steps = min(max(int(steps), SEG), TRAJ_MAX_STEPS)
    seed = min(max(int(seed), 0), 1_000_000)
    swap = None if swap_at < 0 else min(int(swap_at), steps)
    noise = min(max(float(noise), 0.0), 0.25)
    return run_traj(variant, seed, steps, swap, noise)


@lab_app.get("/api/wall")
def wall(
    variant: str = "base",
    seed: int = 0,
    steps: int = WALL_DEFAULT_STEPS,
    wlr: float = 1.0,
    tlr: float = 0.01,
):
    if variant not in WALL_VARIANTS and variant != "flank-champion":
        return {"error": f"unknown variant {variant!r}; "
                         f"have {sorted(WALL_VARIANTS) + ['flank-champion']}"}
    steps = min(max(int(steps), 200), WALL_MAX_STEPS)
    steps -= steps % WALL_SUBSAMPLE
    seed = min(max(int(seed), 0), 1_000_000)
    if variant == "flank-champion":
        # wlr/tlr do not apply here: the champion dict fixes its own rates
        return run_flank_champion(seed, steps)
    wlr = min(max(float(wlr), 0.0), 2.0)
    tlr = min(max(float(tlr), 0.0), 1.0)
    return run_wall_variant(variant, seed, steps, wlr, tlr)


@lab_app.get("/api/pursuit")
def pursuit(
    genome: str = "h34-champion",
    motion: str = "orbit",
    speed: float = 0.15,
    seed: int = H34_CHAMP_SEED,
    steps: int = PURSUIT_DEFAULT_STEPS,
):
    if genome not in PURSUIT_GENOMES:
        return {"error": f"unknown genome {genome!r}; have {sorted(PURSUIT_GENOMES)}"}
    if motion not in PURSUIT_MOTIONS:
        return {"error": f"unknown motion {motion!r}; have {sorted(PURSUIT_MOTIONS)}"}
    speed = min(max(float(speed), 0.01), 0.3)
    steps = min(max(int(steps), 200), PURSUIT_MAX_STEPS)
    steps -= steps % PURSUIT_SUBSAMPLE
    seed = min(max(int(seed), 0), 1_000_000)
    return run_pursuit_variant(genome, motion, speed, seed, steps)


@lab_app.get("/api/ecology")
def ecology(seed: int = H48E_CHAMP_SEED, steps: int = ECOLOGY_DEFAULT_STEPS):
    steps = min(max(int(steps), 300), ECOLOGY_MAX_STEPS)
    steps -= steps % ECOLOGY_SUBSAMPLE
    seed = min(max(int(seed), 0), 1_000_000)
    return run_ecology(seed, steps)


@lab_app.get("/api/ecology3")
def ecology3(mode: str = "resist", steps: int = ECOLOGY3_DEFAULT_STEPS):
    if mode not in ECOLOGY3_MODES:
        return {"error": f"unknown mode {mode!r}; have {sorted(ECOLOGY3_MODES)}"}
    steps = min(max(int(steps), 600), ECOLOGY3_MAX_STEPS)
    steps -= steps % ECOLOGY3_SUBSAMPLE
    return run_ecology3(mode, steps)


@lab_app.get("/api/repair")
def repair(kill: float = 0.3, seed: int = 0, steps: int = REPAIR_DEFAULT_STEPS):
    kill = min(max(float(kill), 0.0), 0.9)
    # even segment count keeps the kill on a segment boundary, as in H53
    steps = min(max(int(steps), 2 * SEG), REPAIR_MAX_STEPS)
    steps -= steps % (2 * SEG)
    seed = min(max(int(seed), 0), 1_000_000)
    return run_repair(kill, seed, steps)
