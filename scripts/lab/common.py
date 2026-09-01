"""Shared harness for the design-space campaign (scripts/lab/).

Two evaluators built directly on the tested package (no model logic duplicated):

- :func:`run_closed_loop` — the tracking task with intervention "arms"
  (lesion / freeze variants / weight shuffle), rich per-run observables,
  segment-resolved scores, and mergeable policy-curve sufficient statistics.
- :func:`run_open_loop` — the same reservoir driven by a SCRIPTED retinal
  stream (no motor feedback), for law-fitting free of the closed-loop slip
  confound. Records windowed per-node drive/spike/target statistics so the
  steady-state duty law f = (mu/T - leak)/rho can be tested per node with no
  free parameters.

Freeze semantics (all exact, via state save/restore around net.step):
  full            learning on throughout
  no-learn        learning_enabled=False from step 0
  lesion          recurrent adjacency removed before step 0 (weights zeroed,
                  caches rebuilt) — the analyze_winners protocol; regrowth is
                  impossible because the adjacency itself is gone
  freeze-mid      learning_enabled=False at t = n_steps//2
  freeze-mid-resetT   freeze-mid + targets reset to target_init at the freeze
  freeze-mid-resetW   freeze-mid + weights reset to their t=0 values
  shuffle-mid     at t = n_steps//2 (learning stays ON) permute the nonzero
                  recurrent weights among the existing links (preserves the
                  weight distribution and adjacency; destroys learned structure)
  freeze-W-only   weights pinned at init every step; targets keep learning
  freeze-T-only   targets pinned at init every step; weights keep learning

Workers must import this module (module-level functions; macOS spawn-safe).
"""

from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import dataclasses
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from homeostasis.reservoir import HomeostaticReservoir, ReservoirConfig  # noqa: E402
from homeostasis.tracking import TrackingConfig, TrackingEnv, angular_difference  # noqa: E402

ERR_EDGES = np.arange(-180.0, 181.0, 10.0)  # 36 policy bins, as analyze_winners
SEG = 720  # one stimulus-reversal segment
WIN = 120  # window for open-loop law statistics


def make_configs(res_over: dict | None = None, trk_over: dict | None = None):
    """Build (ReservoirConfig, TrackingConfig) with n_inputs kept consistent."""
    trk = TrackingConfig(**(trk_over or {}))
    res_kwargs = dict(res_over or {})
    res_kwargs.setdefault("n_inputs", trk.n_sensors)
    return ReservoirConfig(**res_kwargs), trk


def _weight_stats(net):
    mask = net.adjacency
    w = net.weights[mask]
    if w.size == 0:
        return 0.0, 0.0
    return float(w.mean()), float((w < 0).mean())


def g_of(net):
    """Current recurrent gain N*p_eff*w_mean / (rho * T_mean), from live state."""
    c = net.config
    in_deg = net.adjacency.sum(axis=0).mean()  # mean in-degree
    w_mean, _ = _weight_stats(net)
    t_mean = float(net.targets.mean())
    return float(in_deg * w_mean / (c.threshold_ratio * t_mean)) if t_mean else 0.0


def run_closed_loop(task: dict) -> dict:
    """One tracking run. task keys:
    res (dict), trk (dict), seed (int), n_steps (default 7200),
    arm (default 'full'), snap_every (default 240) for w/T/g trajectories.
    Returns a JSON-safe dict of observables.
    """
    res_over = task.get("res", {})
    trk_over = task.get("trk", {})
    seed = task["seed"]
    n_steps = int(task.get("n_steps", 7200))
    arm = task.get("arm", "full")
    snap_every = int(task.get("snap_every", 240))
    sensor_noise = float(task.get("sensor_noise", 0.0))
    noise_rng = np.random.default_rng(seed + 900001) if sensor_noise > 0 else None

    rcfg, tcfg = make_configs(res_over, trk_over)
    net = HomeostaticReservoir(rcfg, seed=seed)
    env = TrackingEnv(tcfg)

    pin_out = task.get("pin_output_p")
    if pin_out:
        orng = np.random.default_rng(seed + 880008)
        net.output_adjacency = orng.random(
            (rcfg.n_nodes, rcfg.n_outputs)) < float(pin_out)
        net._rebuild_structure_caches()

    if arm == "no-learn":
        net.learning_enabled = False
    elif arm == "lesion":
        net.adjacency[:] = False
        net.weights[:] = 0.0
        net._rebuild_structure_caches()

    w0 = net.weights.copy()
    t0 = net.targets.copy()
    g_init = g_of(net)
    half = n_steps // 2

    n_seg = max(1, n_steps // SEG)
    seg_in45 = np.zeros(n_seg)
    seg_count = np.zeros(n_seg)
    err_sign_agree = 0.0
    flow_sum = 0.0
    duty = 0.0
    f_sum = 0.0
    absE_sum = 0.0
    eff_diff_sum = 0.0
    eff_sat = 0.0
    bin_count = np.zeros(36)
    bin_sum = np.zeros(36)
    bin_sumsq = np.zeros(36)
    snaps = {"t": [], "w_mean": [], "neg_frac": [], "T_mean": [], "g": [], "f_win": []}
    f_win_acc = 0.0

    kill_frac = float(task.get("kill_frac", 0.0))
    t_mid = None
    for i in range(n_steps):
        if arm.startswith("kill-mid") and i == half and kill_frac > 0:
            krng = np.random.default_rng(seed + 770007)
            dead = krng.choice(rcfg.n_nodes, int(round(kill_frac * rcfg.n_nodes)),
                               replace=False)
            net.adjacency[dead, :] = False
            net.adjacency[:, dead] = False
            net.weights[dead, :] = 0.0
            net.weights[:, dead] = 0.0
            net.input_adjacency[:, dead] = False
            net.input_weights[:, dead] = 0.0
            net._rebuild_structure_caches()
            net.x[dead] = 0.0
            if arm == "kill-mid-frozen":
                net.learning_enabled = False
        if arm.startswith("freeze-mid") and i == half:
            net.learning_enabled = False
            if arm == "freeze-mid-resetT":
                net.targets = np.full(rcfg.n_nodes, rcfg.target_init)
            elif arm == "freeze-mid-resetW":
                net.weights = w0.copy()
        elif arm == "freeze-T-mid" and i == half:
            t_mid = net.targets.copy()
        elif arm == "shuffle-mid" and i == half:
            mask = net.adjacency
            vals = net.weights[mask]
            net.weights[mask] = net.rng.permutation(vals)

        herr = env.heading_error()
        inputs = env.sense()
        if noise_rng is not None:
            inputs = np.maximum(inputs + noise_rng.uniform(
                -sensor_noise, sensor_noise, inputs.shape), 0.0)
        state = net.step(inputs)
        if arm == "freeze-W-only":
            net.weights = w0.copy()
        elif arm == "freeze-T-only":
            net.targets = t0.copy()
        elif arm == "freeze-T-mid" and t_mid is not None:
            net.targets = t_mid.copy()
        e_left, e_right = state.outputs
        if arm == "swap-mid" and i >= half:
            e_left, e_right = e_right, e_left
        dh = env.apply_action(e_left, e_right)
        env.advance_stimulus()

        seg = min(i // SEG, n_seg - 1)
        seg_count[seg] += 1
        if abs(herr) <= 45.0:
            seg_in45[seg] += 1
        # dir-agree: sign of applied turn matches sign of heading error (when both nonzero)
        if dh != 0.0 and herr != 0.0 and np.sign(dh) == np.sign(herr):
            err_sign_agree += 1
        flow = float(inputs.sum())
        flow_sum += flow
        duty += flow > 0.05
        f_sum += state.prop_spiked
        f_win_acc += state.prop_spiked
        absE_sum += float(np.mean(np.abs(state.error)))
        eff_diff_sum += abs(e_left - e_right)
        eff_sat += (e_left in (0.0, 1.0)) and (e_right in (0.0, 1.0))
        b = min(max(int(np.digitize(herr, ERR_EDGES)) - 1, 0), 35)
        bin_count[b] += 1
        bin_sum[b] += dh
        bin_sumsq[b] += dh * dh

        if (i + 1) % snap_every == 0:
            w_mean, neg = _weight_stats(net)
            snaps["t"].append(i + 1)
            snaps["w_mean"].append(round(w_mean, 5))
            snaps["neg_frac"].append(round(neg, 5))
            snaps["T_mean"].append(round(float(net.targets.mean()), 5))
            snaps["g"].append(round(g_of(net), 5))
            snaps["f_win"].append(round(f_win_acc / snap_every, 5))
            f_win_acc = 0.0

    seg_scores = (seg_in45 / np.maximum(seg_count, 1)).tolist()
    late = seg_scores[5:] if n_seg >= 10 else seg_scores
    return dict(
        seed=seed, arm=arm, n_steps=n_steps,
        res=res_over, trk=trk_over,
        score=float(np.mean(seg_scores)),
        score_late=float(np.mean(late)),
        seg_scores=[round(s, 4) for s in seg_scores],
        dir_agree=err_sign_agree / n_steps,
        prop_spiked=f_sum / n_steps,
        mean_abs_E=absE_sum / n_steps,
        input_flow=flow_sum / n_steps,
        input_duty=duty / n_steps,
        eff_diff=eff_diff_sum / n_steps,
        eff_sat=eff_sat / n_steps,
        g_init=round(g_init, 5),
        g_final=snaps["g"][-1] if snaps["g"] else round(g_of(net), 5),
        T_final=round(float(net.targets.mean()), 5),
        w_mean_final=snaps["w_mean"][-1] if snaps["w_mean"] else None,
        neg_frac_final=snaps["neg_frac"][-1] if snaps["neg_frac"] else None,
        snaps=snaps,
        policy=dict(count=bin_count.tolist(), sum=bin_sum.tolist(), sumsq=bin_sumsq.tolist()),
    )


def scripted_theta(schedule: dict, n_steps: int) -> np.ndarray:
    """Heading-error trajectory theta(t) in degrees for open-loop drives.

    schedule kinds:
      {"kind": "stationary", "theta": 0.0}
      {"kind": "slip", "speed": s}                  theta advances s deg/step, wraps to [-180,180)
      {"kind": "jump", "theta0": 0, "theta1": 30, "t_jump": 1000}
      {"kind": "dark"}                              stimulus out of view (theta=180)
    """
    kind = schedule["kind"]
    if kind == "stationary":
        return np.full(n_steps, float(schedule.get("theta", 0.0)))
    if kind == "slip":
        s = float(schedule["speed"])
        th = (np.arange(n_steps) * s + 180.0) % 360.0 - 180.0
        return th
    if kind == "jump":
        th = np.full(n_steps, float(schedule.get("theta0", 0.0)))
        th[int(schedule["t_jump"]):] = float(schedule.get("theta1", 30.0))
        return th
    if kind == "dark":
        return np.full(n_steps, 180.0)
    if kind == "sine":
        amp = float(schedule.get("amp", 20.0))
        period = float(schedule["period"])
        return amp * np.sin(2 * np.pi * np.arange(n_steps) / period)
    raise ValueError(f"unknown schedule kind {kind!r}")


def run_open_loop(task: dict) -> dict:
    """Scripted-drive run (no motor loop). task keys: res, trk, seed,
    schedule (see scripted_theta), n_steps (default 3000), per_node (bool):
    when True, also return windowed per-node sums for law fitting.
    """
    res_over = task.get("res", {})
    trk_over = task.get("trk", {})
    seed = task["seed"]
    n_steps = int(task.get("n_steps", 3000))
    per_node = bool(task.get("per_node", False))
    sensor_noise = float(task.get("sensor_noise", 0.0))
    noise_rng = np.random.default_rng(seed + 900001) if sensor_noise > 0 else None

    rcfg, tcfg = make_configs(res_over, trk_over)
    net = HomeostaticReservoir(rcfg, seed=seed)
    env = TrackingEnv(tcfg)
    theta = scripted_theta(task["schedule"], n_steps)
    offs = env._sensor_offsets

    n_win = n_steps // WIN
    N = rcfg.n_nodes
    f_t = np.empty(n_steps)
    E_t = np.empty(n_steps)
    T_t = np.empty(n_steps)
    xprime_t = np.empty(n_steps)
    recon = bool(task.get("recon", False))
    if recon:
        in_deg = np.maximum(net.input_adjacency.sum(axis=0), 1)
        node_centers = (net.input_adjacency.T @ offs) / in_deg  # (N,)
        theta_hat = np.full(n_steps, np.nan)
    if per_node:
        drive_w = np.zeros((n_win, N))   # total drive per node, window mean
        spike_w = np.zeros((n_win, N))
        T_w = np.zeros((n_win, N))
        xp_w = np.zeros((n_win, N))

    for i in range(n_steps):
        d = np.abs((theta[i] - offs + 180.0) % 360.0 - 180.0)
        acts = np.exp(-(d ** 2) / tcfg.tuning_width)
        acts[d <= tcfg.plateau_width] = 1.0
        if noise_rng is not None:
            acts = np.maximum(acts + noise_rng.uniform(
                -sensor_noise, sensor_noise, acts.shape), 0.0)
        drive = acts @ net.input_weights + net._spiked_f @ net.weights
        state = net.step(acts)
        f_t[i] = state.prop_spiked
        E_t[i] = float(np.mean(np.abs(state.error)))
        T_t[i] = float(np.mean(state.targets))
        xprime_t[i] = float(np.mean(state.x))
        if recon and state.spiked.any():
            theta_hat[i] = float(node_centers[state.spiked].mean())
        if per_node and i < n_win * WIN:
            w = i // WIN
            drive_w[w] += drive / WIN
            spike_w[w] += state.spiked / WIN
            T_w[w] += state.targets / WIN
            xp_w[w] += state.x / WIN

    out = dict(
        seed=seed, res=res_over, schedule=task["schedule"], n_steps=n_steps,
        f_t=[round(float(v), 5) for v in f_t[:: max(1, n_steps // 600)]],
        f_mean_late=float(f_t[n_steps // 2:].mean()),
        f_mean_first200=float(f_t[:200].mean()),
        absE_late=float(E_t[n_steps // 2:].mean()),
        T_late=float(T_t[n_steps // 2:].mean()),
        xprime_late=float(xprime_t[n_steps // 2:].mean()),
        g_final=round(g_of(net), 5),
        silence_step=int(np.argmax(np.convolve(f_t == 0, np.ones(50), "valid") >= 50))
        if (np.convolve(f_t == 0, np.ones(50), "valid") >= 50).any() else -1,
    )
    if recon:
        sched = task["schedule"]
        if sched.get("kind") == "sine":
            period = float(sched["period"])
            amp = float(sched.get("amp", 20.0))
            half = slice(n_steps // 2, n_steps)
            t = np.arange(n_steps)[half]
            th_h = theta_hat[half]
            ok = ~np.isnan(th_h)
            if ok.sum() > 50:
                ph = 2 * np.pi * t[ok] / period
                c = np.cos(ph); s = np.sin(ph)
                y = th_h[ok] - th_h[ok].mean()
                a = 2 * np.mean(y * s); b = 2 * np.mean(y * c)
                out["recon_gain"] = float(np.hypot(a, b) / amp)
                out["recon_phase"] = float(np.arctan2(b, a))
                fy = f_t[half] - f_t[half].mean()
                fa = 2 * np.mean(fy * np.sin(2 * np.pi * np.arange(n_steps)[half] / period))
                fb = 2 * np.mean(fy * np.cos(2 * np.pi * np.arange(n_steps)[half] / period))
                out["rate_gain"] = float(np.hypot(fa, fb))
            else:
                out["recon_gain"] = 0.0
                out["recon_phase"] = float("nan")
                out["rate_gain"] = 0.0
            out["recon_valid_frac"] = float(np.mean(~np.isnan(theta_hat[half])))
    if per_node:
        out["law"] = dict(
            drive=drive_w.round(4).tolist(),
            f=spike_w.round(4).tolist(),
            T=T_w.round(4).tolist(),
            xprime=xp_w.round(4).tolist(),
            leak=rcfg.leak, rho=rcfg.threshold_ratio, win=WIN,
        )
    return out


def retinal_gain(trk_over: dict | None = None) -> dict:
    """Exact total retinal activation S(theta) from the sensor model, plus the
    in-view mean S0 — the missing scale in K0's mu_in proxy."""
    tcfg = TrackingConfig(**(trk_over or {}))
    offs = tcfg.sensor_offsets
    thetas = np.arange(-180.0, 180.0, 0.5)
    S = []
    for th in thetas:
        d = np.abs((th - offs + 180.0) % 360.0 - 180.0)
        a = np.exp(-(d ** 2) / tcfg.tuning_width)
        a[d <= tcfg.plateau_width] = 1.0
        S.append(a.sum())
    S = np.array(S)
    return dict(theta=thetas.tolist(), S=S.tolist(),
                S_max=float(S.max()), S_in45=float(S[np.abs(thetas) <= 45].mean()))
