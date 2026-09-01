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
  /lab/pursuit  the pursuit task (package run_pursuit): the H34 champion
              PAIR on its evolved orbit motion (phase-locked pursuit) and
              on waypoint motion (the collapse). Protocol is exactly
              scripts/lab/h34b_verify.py's run_one.

Run via the main app:  uvicorn viz.server:app --port 8471  ->  /lab
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import dataclasses

from homeostasis.pursuit import PursuitConfig
from homeostasis.reservoir import HomeostaticReservoir, ReservoirConfig
from homeostasis.simulation import (
    WALL_RESERVOIR_CONFIG,
    run_pursuit,
    run_tracking,
    run_wall,
)
from homeostasis.tracking import TrackingConfig, TrackingEnv
from homeostasis.wall import WallConfig

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


def run_traj(variant: str, seed: int, steps: int, swap_at: int | None) -> dict:
    """One closed-loop tracking run, the exact scripts/lab/common.py
    run_closed_loop step order (sense -> net.step -> apply_action ->
    advance_stimulus). The optional effector swap is the "swap-mid" arm
    generalized: from step swap_at on, the two outputs are exchanged before
    being passed to apply_action. The simulation itself is entirely the
    tested package; this function only records observables.
    """
    res_over, trk_over = TRAJ_VARIANTS[variant]
    tcfg = TrackingConfig(**trk_over)
    res_kwargs = dict(res_over)
    res_kwargs.setdefault("n_inputs", tcfg.n_sensors)
    rcfg = ReservoirConfig(**res_kwargs)
    net = HomeostaticReservoir(rcfg, seed=seed)
    env = TrackingEnv(tcfg)

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
            "swap_at": swap_at, "subsample": TRAJ_SUBSAMPLE, "seg_len": SEG,
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
# Pursuit viewer (/lab/pursuit): the H34 champion pair, orbit vs waypoint.
# ---------------------------------------------------------------------------

PURSUIT_MAX_STEPS = 14_400
PURSUIT_DEFAULT_STEPS = 3_600
PURSUIT_SUBSAMPLE = 2
H34_FILE = OUT_DIR / "lab/h34_joint.json"

# The H34 jointly-evolved champion PAIR: genome (ReservoirConfig fields plus
# wheel_base / intensity_scale for PursuitConfig) and its wiring seed.
_H34_LAST = json.loads(H34_FILE.read_text())[-1] if H34_FILE.exists() else None
H34_CHAMPION = _H34_LAST["champion"] if _H34_LAST else None
H34_CHAMP_SEED = int(_H34_LAST["champ_seed"]) if _H34_LAST else 0

# preset -> stimulus motion; the champion was evolved on orbit
PURSUIT_PRESETS = {"perfect-pursuer": "orbit", "pursuit-fails": "waypoint"}
_PURSUIT_RES_KEYS = (  # scripts/lab/h34b_verify.py RES_KEYS
    "n_nodes", "p_link", "input_weight", "weight_init_mean",
    "leak", "target_lr", "threshold_ratio", "weight_lr",
)


def run_pursuit_preset(preset: str, seed: int, steps: int) -> dict:
    """One pursuit run of the H34 champion pair — the exact construction of
    scripts/lab/h34b_verify.py run_one (single 91-sensor full-circle eye;
    wheel_base and intensity_scale from the genome). Summary uses the same
    late window (second half) and definitions as that script."""
    if H34_CHAMPION is None:
        return {"error": f"{H34_FILE} not found"}
    motion = PURSUIT_PRESETS[preset]
    pc = PursuitConfig(
        eye_offsets=(0.0,), sensors_per_eye=91,
        wheel_base=H34_CHAMPION["wheel_base"],
        intensity_scale=H34_CHAMPION["intensity_scale"],
        stimulus_motion=motion,
    )
    res = ReservoirConfig(
        n_inputs=pc.n_sensors, **{k: H34_CHAMPION[k] for k in _PURSUIT_RES_KEYS}
    )
    h = run_pursuit(n_steps=steps, seed=seed, reservoir_config=res, pursuit_config=pc)
    late = slice(steps // 2, None)
    sl = slice(0, steps, PURSUIT_SUBSAMPLE)
    return {
        "kind": "pursuit",
        "params": {
            "preset": preset, "seed": seed, "steps": steps,
            "subsample": PURSUIT_SUBSAMPLE,
        },
        "config": {
            "motion": motion,
            "box_size": pc.box_size,
            "agent_radius": pc.agent_radius,
            "orbit_radius": pc.orbit_radius,
            "n_nodes": res.n_nodes,
            "wheel_base": round(pc.wheel_base, 3),
            "intensity_scale": round(pc.intensity_scale, 3),
            "champ_seed": H34_CHAMP_SEED,
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
        "summary": {  # full resolution, definitions of h34b_verify.py
            "dist_late": round(float(h.dist[late].mean()), 4),
            "near3_late": round(float((h.dist[late] < 3).mean()), 4),
            "hits_total": int(h.hit.sum()),
        },
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
):
    if variant not in TRAJ_VARIANTS:
        return {"error": f"unknown variant {variant!r}; have {sorted(TRAJ_VARIANTS)}"}
    steps = min(max(int(steps), SEG), TRAJ_MAX_STEPS)
    seed = min(max(int(seed), 0), 1_000_000)
    swap = None if swap_at < 0 else min(int(swap_at), steps)
    return run_traj(variant, seed, steps, swap)


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
    preset: str = "perfect-pursuer",
    seed: int = H34_CHAMP_SEED,
    steps: int = PURSUIT_DEFAULT_STEPS,
):
    if preset not in PURSUIT_PRESETS:
        return {"error": f"unknown preset {preset!r}; have {sorted(PURSUIT_PRESETS)}"}
    steps = min(max(int(steps), 200), PURSUIT_MAX_STEPS)
    steps -= steps % PURSUIT_SUBSAMPLE
    seed = min(max(int(seed), 0), 1_000_000)
    return run_pursuit_preset(preset, seed, steps)
