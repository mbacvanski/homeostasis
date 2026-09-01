"""Render the single-neuron update rule as an animated 16:9 explainer video.

A redesign of the paper's Figure-2 decision-tree flowchart (Falandays et al.
2024). The tree is hard to read because it hides a symmetry: the released
code never walks four branches — after an optional spike-and-reset it computes
ONE error, E = x' - T, and every leaf of the tree is just the sign of E moving
the same two knobs in opposite directions. So instead of a tree, the video
shows the step as a three-stage pipeline, with as little text as possible
(it is meant to be narrated over on a slide):

    1 - integrate   leak 25%, add sensor input, add weighted spikes from
                    last step's spiking in-neighbors
    2 - spike?      a vertical gauge: the activation ball rises through three
                    zones (below target = hungry, above target = sated, above
                    threshold = spike). Crossing the threshold fires a spike
                    and visibly drops the ball by 2T (the reset).
    3 - adapt       one two-sided card replaces the tree's four leaf boxes:
                    landed low  -> target down (floor 1), weights up;
                    landed high -> target up, weights down.
                    Every nudge is proportional to the error (dT = 0.01*E,
                    dw = -E/k on the k in-links that spiked).

Mapping to the paper's flowchart: its first two diamonds are the gauge's zone
boundaries (target, threshold = 2*target); its post-spike re-check diamond is
the fact that E is computed from the LANDING (post-reset) activation; its
four leaf boxes are the two halves of the adapt card.

Below the pipeline sit two half-width panels. Left: the WIRING — the neuron
with its sensor feed, its strongest in-neighbors (plus a "+N more"
aggregate), each with its recent spike history; edge width = current weight
(live numeric labels); spikes travel the edges as pulses during the
integrate phase, and outgoing spikes fly onto a spike-train track on the
right. Right: the gauge's HISTORY — the activation trace (plotted pre-reset
so spikes visibly cross the threshold), the moving target/threshold lines
with the same zone tints, spike ticks and reset drops, a hungry/sated
verdict ribbon, and the two slow knobs (target, mean incoming weight).

Within each step the event order encodes causality: pulses fly, then the
ball rises; the ball crosses threshold, then the outgoing spike fires and
the ball drops; the landing is judged, then the knobs move.

Everything is driven by a real neuron recorded from the tracking simulation
(paper configuration, seed 0 by default) — the script re-runs the simulation
and reads model state; no model logic is duplicated (the drive decomposition
is assert-checked against the recorded activations).

The video runs about a minute at a constant 1.2 s per model step, 60 fps,
with quintic easing and a continuously sliding timeline — every pulse,
rise, drop, and knob-slide is followable.

Usage: python scripts/render_neuron_rule_video.py [--seed 0] [--still N]
       [--neuron J --t0 T]  (see --help; prints the frame plan before
       rendering so you can pick stills)
"""

from __future__ import annotations

import argparse
import multiprocessing
import os
import pathlib
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.collections import LineCollection
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from homeostasis import (ReservoirConfig, VariableTrackingConfig,
                         VariableTrackingSimulation)
from viz.server import BASE_TRACKING_PARAMS, LOADOUT_BY_ID, RESERVOIR_PARAMS

INK = "#1c2733"
MUTED = "#7a8494"
EDGE = "#d9dee6"
AMBER = "#c9820a"        # sensor drive
SLATE = "#9aa3ad"        # spikes arriving from neighbors
SPIKE = "#d62839"        # firing
NODE = "#5b8fd6"         # activation
TARGET = "#7a4fbf"       # target level
GREEN = "#0c9c62"        # sated / weights
ROSE = "#c94f63"         # hungry (softer than SPIKE)
T_HUNGRY = "#f7e4e7"     # zone tints
T_SATED = "#e3f2ea"
T_SPIKE = "#fdf1dd"

W = 60                   # steps of history in the waveform band (half width)

plt.rcParams.update({
    "font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"],
    "text.color": INK,
    "axes.edgecolor": EDGE,
    "axes.labelcolor": MUTED,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
})


def smooth(u):
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


def silk(u):
    """Quintic ease — velocity AND acceleration are zero at both ends, so
    motion starts and stops without any visible kick."""
    u = np.clip(u, 0.0, 1.0)
    return u * u * u * (u * (6.0 * u - 15.0) + 10.0)


def fade(p, p0, w=0.06):
    """0 -> 1 over the window [p0, p0+w]; used to fade artists in."""
    return float(smooth((p - p0) / w))


# --------------------------------------------------------------------------
# pass 1: full-network run, used only to choose a photogenic neuron + window
# --------------------------------------------------------------------------

def make_sim(seed: int, loadout: str) -> VariableTrackingSimulation:
    params = LOADOUT_BY_ID[loadout]["params"]
    r_cfg = ReservoirConfig(**{k: v for k, v in params.items() if k in RESERVOIR_PARAMS})
    t_cfg = VariableTrackingConfig(
        **{k: v for k, v in params.items() if k in BASE_TRACKING_PARAMS})
    return VariableTrackingSimulation(r_cfg, t_cfg, seed=seed)


def simulate_pass1(seed: int, n_steps: int, loadout: str):
    sim = make_sim(seed, loadout)
    net = sim.network
    n = net.config.n_nodes
    rec = {
        "x": np.zeros((n_steps, n), dtype=np.float32),      # post-reset x'
        "thr": np.zeros((n_steps, n), dtype=np.float32),    # threshold in force
        "spiked": np.zeros((n_steps, n), dtype=bool),
    }
    for t in range(n_steps):
        thr = net.thresholds.copy()
        state, _ = sim.step()
        rec["x"][t] = state.x
        rec["thr"][t] = thr
        rec["spiked"][t] = state.spiked
    rec["adj"] = net.adjacency.copy()
    rec["in_adj"] = net.input_adjacency.copy()
    return rec


def branch_codes(x_post, thr, spiked):
    """0 hungry (E<0), 1 sated (E>=0), +2 if the step spiked."""
    e = x_post - thr / 2.0
    return np.where(e >= 0, 1, 0) + 2 * spiked.astype(int)


def pick_neuron_and_window(rec, slow_steps, seg_steps, t_min, t_max):
    """A moderately active neuron and a start step whose first slow_steps
    show the story: calm start, a visible charge-up into a spike, and both
    hungry and sated steps. The following segment should keep spiking and
    move the target."""
    x, thr, spiked = rec["x"], rec["thr"], rec["spiked"]
    rate = spiked.mean(0)
    in_deg = rec["adj"].sum(0)
    sens_links = rec["in_adj"].sum(0)
    cand = np.flatnonzero((rate > 0.15) & (rate < 0.5) & (in_deg >= 10)
                          & (sens_links >= 1))
    cand = cand[np.argsort(np.abs(rate[cand] - 0.30))][:40]
    best, best_score = None, -1e9
    for j in cand:
        br = branch_codes(x[:, j], thr[:, j], spiked[:, j])
        sp = spiked[:, j]
        for t0 in range(t_min, t_max):
            wbr = br[t0:t0 + slow_steps]
            wsp = sp[t0:t0 + slow_steps]
            n_sp = int(wsp.sum())
            # the slow-motion window must tell the story: a calm start, at
            # most a few spikes (not a burst), and no burst at the end
            if not 1 <= n_sp <= 3 or wsp[:2].any() or wsp[-2:].all():
                continue
            if np.flatnonzero(wsp)[0] > 6:   # first spike within ~7 s
                continue
            score = 3.0 * len(np.unique(wbr))
            # a spike preceded by >= 2 quiet steps: the charge-up story
            for i in np.flatnonzero(wsp):
                if i >= 2 and not wsp[i - 2:i].any():
                    score += 2.0
                    break
            seg = sp[t0:t0 + seg_steps]
            score -= 6.0 * abs(seg.mean() - 0.30)
            # a demo that goes quiet for hundreds of steps looks broken:
            # penalize the longest spikeless run and a dead final quarter
            runs = np.diff(np.flatnonzero(np.r_[True, seg, True]))
            score -= 0.15 * max(0, int(runs.max()) - 30)
            if seg[-seg_steps // 4:].sum() < 3:
                score -= 5.0
            seg_T = thr[t0:t0 + seg_steps, j] / 2.0
            score += min(float(seg_T.std()) * 40.0, 2.0)
            if (br[t0:t0 + seg_steps] == 3).any():   # spike-and-still-sated
                score += 0.5
            if score > best_score:
                best, best_score = (int(j), int(t0)), score
    if best is None:
        j = int(cand[0]) if len(cand) else int(np.argsort(np.abs(rate - 0.3))[0])
        best = (j, t_min)
    return best


# --------------------------------------------------------------------------
# pass 2: re-run and record the chosen neuron in full detail
# --------------------------------------------------------------------------

def simulate_detail(seed, loadout, j, t0, seg_steps):
    sim = make_sim(seed, loadout)
    net = sim.network
    c = net.config
    in_nbrs = np.flatnonzero(net.adjacency[:, j])
    in_w_col = net.input_weights[:, j].copy()
    rec_start = t0 - W
    n_rec = W + seg_steps
    d = {k: np.zeros(n_rec) for k in
         ("x_prev", "carry", "sens", "recur", "x_pre", "x_post", "T_pre",
          "T_post", "E", "dT", "dwlink", "wbar")}
    d["spiked"] = np.zeros(n_rec, dtype=bool)
    d["k"] = np.zeros(n_rec, dtype=int)
    d["nbr_spk"] = np.zeros((n_rec, len(in_nbrs)), dtype=bool)  # arriving now
    d["nbr_w"] = np.zeros((n_rec, len(in_nbrs)))                # weight in force
    for t in range(t0 + seg_steps):
        x_prev = float(net.x[j])
        spiked_prev = net.spiked.copy()
        w_col = net.weights[:, j].copy()
        T_pre = float(net.targets[j])
        state, _ = sim.step()
        if t < rec_start:
            continue
        i = t - rec_start
        sens = float(state.inputs @ in_w_col)
        recur = float(spiked_prev.astype(float) @ w_col)
        carry = x_prev * (1.0 - c.leak)
        x_pre = carry + sens + recur
        thr_pre = c.threshold_ratio * T_pre
        spk = bool(state.spiked[j])
        # cross-check the decomposition against the model's own arithmetic
        assert np.isclose(x_pre, state.x[j] + spk * thr_pre, atol=1e-9), \
            f"drive decomposition mismatch at t={t}"
        k = int(spiked_prev[in_nbrs].sum())
        E = float(state.error[j])
        d["x_prev"][i], d["carry"][i], d["sens"][i] = x_prev, carry, sens
        d["recur"][i], d["x_pre"][i] = recur, x_pre
        d["x_post"][i] = float(state.x[j])
        d["spiked"][i] = spk
        d["T_pre"][i], d["T_post"][i] = T_pre, float(state.targets[j])
        d["E"][i] = E
        d["dT"][i] = float(state.targets[j]) - T_pre
        d["k"][i] = k
        d["dwlink"][i] = (-E / k) if k > 0 else 0.0
        d["wbar"][i] = float(w_col[in_nbrs].mean())
        d["nbr_spk"][i] = spiked_prev[in_nbrs]
        d["nbr_w"][i] = w_col[in_nbrs]
    d["thr_pre"] = 2.0 * d["T_pre"]
    d["branch"] = branch_codes(d["x_post"], d["thr_pre"], d["spiked"])
    d["j"], d["in_deg"] = j, int(len(in_nbrs))
    d["n_sens"] = int((in_w_col > 0).sum())
    d["t0"], d["seg_steps"] = t0, seg_steps
    d["target_floor"] = c.target_floor
    return d


# --------------------------------------------------------------------------
# frame plan: slow motion -> ramp -> fast
# --------------------------------------------------------------------------

def build_schedule(slow_steps, ramp_steps, fast_steps, slow_fp, fast_fp):
    """Frames as (segment step index, fraction p within the step, frames/step)."""
    fpsteps = ([slow_fp] * slow_steps
               + [int(round(slow_fp + (fast_fp - slow_fp)
                            * smooth((i + 1) / ramp_steps)))
                  for i in range(ramp_steps)]
               + [fast_fp] * fast_steps)
    sched = []
    for s, fp in enumerate(fpsteps):
        for f in range(fp):
            sched.append((s, (f + 1) / fp, fp))
    return sched, fpsteps


# choreography fractions within one step. The order encodes causality: a
# pulse flies along its wire FIRST, the ball rises when it lands; the ball
# crosses the threshold FIRST, then the outgoing spike flies and the ball
# drops; the landing is judged, and only then do the knobs move.
P_LEAK = 0.10                    # leak sag done
F_SENS0, F_SENS1 = 0.06, 0.22    # sensor pulse in flight
F_RECUR0, F_RECUR1 = 0.22, 0.42  # neighbor pulses in flight
B_SENS1, B_RECUR1 = 0.34, 0.54   # ball rise ends after each landing
X_REVEAL = 0.54                  # integrated x committed to the trace
P_SPIKE0, P_SPIKE1 = 0.58, 0.72  # ball drop + outgoing spike flight
P_ERR, P_KNOB0, P_KNOB1 = 0.76, 0.82, 0.95


def ball_value(d, i, p):
    """Continuous within-step gauge value: sag by the leak, rise as each
    arrival lands, then (on spike steps) drop by the threshold."""
    carry = d["carry"][i]
    s1 = carry + d["sens"][i]
    segs = [(0.02, d["x_prev"][i]), (P_LEAK, carry), (F_SENS1, carry),
            (B_SENS1, s1), (F_RECUR1, s1), (B_RECUR1, d["x_pre"][i])]
    if d["spiked"][i]:
        segs += [(P_SPIKE0, d["x_pre"][i]), (P_SPIKE1, d["x_post"][i])]
    if p < segs[0][0]:
        return segs[0][1]
    for (p0, v0), (p1, v1) in zip(segs, segs[1:]):
        if p0 <= p < p1:
            return v0 + (v1 - v0) * silk((p - p0) / (p1 - p0))
    return segs[-1][1]


def knob_display(d, i, p):
    """Target/threshold lines slide to their new values late in the step."""
    u = silk((p - P_KNOB0) / (P_KNOB1 - P_KNOB0)) if p > P_KNOB0 else 0.0
    T = d["T_pre"][i] + (d["T_post"][i] - d["T_pre"][i]) * u
    return T, 2.0 * T


# --------------------------------------------------------------------------
# figure
# --------------------------------------------------------------------------

def card_box(ax, tint):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    box = FancyBboxPatch((0.01, 0.01), 0.98, 0.98,
                         boxstyle="round,pad=0.005,rounding_size=0.028",
                         facecolor=tint, edgecolor="#e4e8ee", lw=1.1,
                         transform=ax.transAxes, zorder=0)
    ax.add_patch(box)
    return box


def build_figure(d, args):
    fig = plt.figure(figsize=(12.8, 7.2), dpi=args.dpi)
    fig.patch.set_facecolor("white")
    art = {}
    y_max = max(float(np.percentile(d["x_pre"], 99.5)),
                1.35 * float(d["thr_pre"].max())) * 1.04
    y_min = min(0.0, float(d["x_post"].min()), float(d["x_pre"].min())) * 1.05
    art["ylim"] = (y_min, y_max)
    ys = y_max - y_min

    art["clock"] = fig.text(0.985, 0.945, "t = 0", fontsize=12, color=MUTED,
                            ha="right", family="monospace")
    art["speed"] = fig.text(0.985, 0.974, "", fontsize=10, color=SPIKE,
                            ha="right", fontweight="bold")

    # ---------------- stage headings, one shared baseline ----------------
    for x, txt in ((0.040, "1 · integrate"), (0.288, "2 · spike?"),
                   (0.490, "3 · adapt")):
        fig.text(x, 0.9455, txt, fontsize=13, fontweight="bold", color=INK)

    # ---------------- stage 1: integrate ----------------
    ax1 = fig.add_axes([0.020, 0.50, 0.215, 0.43])
    card_box(ax1, "#fcfdff")
    rows = [("leak", "leak −25%", MUTED),
            ("sens", "sensors", AMBER),
            ("recur", "neighbor spikes", SLATE)]
    art["rows"] = {}
    for r, (key, label, col) in enumerate(rows):
        y = 0.80 - 0.225 * r
        ar = FancyArrowPatch((0.06, y), (0.155, y), arrowstyle="-|>",
                             mutation_scale=13, color=col, lw=2.0,
                             transform=ax1.transAxes)
        ax1.add_patch(ar)
        lab = ax1.text(0.20, y, label, fontsize=10, color=INK, va="center")
        val = ax1.text(0.94, y - 0.095, "", fontsize=10, color=col,
                       va="center", ha="right", family="monospace",
                       fontweight="bold")
        art["rows"][key] = (ar, lab, val, col)
    ax1.text(0.06, 0.095, "x =", fontsize=11, color=MUTED)
    art["total"] = ax1.text(0.94, 0.095, "", fontsize=12, color=NODE,
                            ha="right", family="monospace", fontweight="bold")

    art["to_gauge"] = FancyArrowPatch((0.243, 0.69), (0.283, 0.69),
                                      arrowstyle="-|>", mutation_scale=17,
                                      color=EDGE, lw=2.4,
                                      transform=fig.transFigure, figure=fig)
    fig.patches.append(art["to_gauge"])

    # ---------------- stage 2: the gauge ----------------
    axg = fig.add_axes([0.288, 0.455, 0.135, 0.475])
    axg.set_xlim(0, 1)
    axg.set_ylim(*art["ylim"])
    axg.set_xticks([])
    axg.set_yticks([])
    art["zone_hungry"] = Rectangle((0, y_min), 1, 0, facecolor=T_HUNGRY,
                                   edgecolor="none", zorder=0)
    art["zone_sated"] = Rectangle((0, 0), 1, 0, facecolor=T_SATED,
                                  edgecolor="none", zorder=0)
    art["zone_spike"] = Rectangle((0, 0), 1, 0, facecolor=T_SPIKE,
                                  edgecolor="none", zorder=0)
    for z in ("zone_hungry", "zone_sated", "zone_spike"):
        axg.add_patch(art[z])
    (art["g_thr"],) = axg.plot([0, 1], [0, 0], color=INK, lw=1.5,
                               ls=(0, (4, 2)), zorder=3)
    (art["g_tgt"],) = axg.plot([0, 1], [0, 0], color=TARGET, lw=1.5,
                               ls=(0, (4, 2)), zorder=3)
    art["g_thr_lab"] = axg.text(0.05, 0, "threshold", fontsize=8.5,
                                color=INK, va="bottom", zorder=3)
    art["g_tgt_lab"] = axg.text(0.05, 0, "target", fontsize=8.5,
                                color=TARGET, va="bottom", zorder=3)
    if d["target_floor"] > y_min:
        axg.plot([0, 0.10], [d["target_floor"]] * 2, color=TARGET, lw=2.4,
                 zorder=3)
        axg.text(0.05, d["target_floor"] - 0.012 * ys, "floor", fontsize=7,
                 color=TARGET, va="top")
    art["ghost"] = axg.scatter([], [], zorder=4)
    art["ball"] = axg.scatter([0.45], [0], s=190, color=NODE, zorder=5,
                              edgecolors="white", linewidths=1.2)
    (art["drop"],) = axg.plot([], [], color=SPIKE, lw=2.2, alpha=0.65,
                              zorder=4, solid_capstyle="round")
    art["bolt"] = axg.annotate("spike!", xy=(0.97, 0), xytext=(0.50, 0),
                               fontsize=11, fontweight="bold", color=SPIKE,
                               ha="left", va="center", zorder=7,
                               arrowprops=dict(arrowstyle="-|>", color=SPIKE,
                                               lw=2.2))
    art["bolt"].set_visible(False)
    art["axg"] = axg

    (art["err_line"],) = axg.plot([], [], lw=2.6, zorder=5,
                                  solid_capstyle="round")
    art["err_lab"] = axg.text(0.76, 0, "", fontsize=9, fontweight="bold",
                              ha="left", va="center", zorder=6)

    art["to_adapt"] = FancyArrowPatch((0.428, 0.69), (0.468, 0.69),
                                      arrowstyle="-|>", mutation_scale=17,
                                      color=EDGE, lw=2.4,
                                      transform=fig.transFigure, figure=fig)
    fig.patches.append(art["to_adapt"])

    # ---------------- stage 3: adapt ----------------
    ax3 = fig.add_axes([0.473, 0.50, 0.512, 0.43])
    card_box(ax3, "#fcfdff")
    art["half"] = {}
    halves = [("sated", 0.555, T_SATED, GREEN, "above target — too much",
               "target ↑", "weights ↓"),
              ("hungry", 0.165, T_HUNGRY, ROSE, "below target — too little",
               "target ↓", "weights ↑")]
    for key, y0, tint, col, head, k1, k2 in halves:
        box = FancyBboxPatch((0.03, y0), 0.585, 0.335,
                             boxstyle="round,pad=0.004,rounding_size=0.02",
                             facecolor=tint, edgecolor="#e4e8ee", lw=1.0,
                             transform=ax3.transAxes, zorder=1, alpha=0.45)
        ax3.add_patch(box)
        h = ax3.text(0.06, y0 + 0.23, head, fontsize=10.5, fontweight="bold",
                     color=col, zorder=2)
        t1 = ax3.text(0.09, y0 + 0.085, k1, fontsize=12.5, fontweight="bold",
                      color=TARGET, zorder=2)
        t2 = ax3.text(0.33, y0 + 0.085, k2, fontsize=12.5, fontweight="bold",
                      color=GREEN, zorder=2)
        art["half"][key] = (box, h, t1, t2, col, tint)
    art["adapt_vals"] = []
    for r, lab in enumerate(("error", "Δ target", "Δ weight")):
        y = 0.775 - 0.185 * r
        ax3.text(0.665, y, lab, fontsize=10, color=MUTED, va="center")
        v = ax3.text(0.965, y, "", fontsize=11, color=INK, va="center",
                     ha="right", family="monospace", fontweight="bold")
        art["adapt_vals"].append(v)
    ax3.text(0.03, 0.045, "ΔT = 0.01·E   ·   Δw = −E/k on the k links that "
             "spiked   ·   threshold = 2×target", fontsize=9, color=MUTED)

    # ---------------- wiring panel (bottom-left) ----------------
    axn = fig.add_axes([0.025, 0.030, 0.425, 0.355])
    axn.set_xlim(0, 1)
    axn.set_ylim(0, 1)
    axn.axis("off")
    contrib = (np.abs(d["nbr_w"]) * d["nbr_spk"])[W:].sum(0)
    shown = np.argsort(contrib)[::-1][:6]
    n_other = d["k"] - d["nbr_spk"][:, shown].sum(1)
    NX, NEU = 0.33, (0.70, 0.52)
    SY, OY = 0.92, 0.12
    rows = np.linspace(0.81, 0.25, len(shown))
    art["net"] = {"shown": shown, "n_other": n_other, "NX": NX, "NEU": NEU,
                  "SY": SY, "OY": OY, "rows": rows,
                  "wref": max(float(np.abs(d["nbr_w"][:, shown]).max()), 1e-6),
                  "sens_max": max(float(d["sens"].max()), 0.01),
                  "oth_max": max(float(n_other.max()), 1.0)}
    axn.text(0.005, SY + 0.048, "sensors", fontsize=8.5, color=AMBER)
    axn.text(0.005, rows[0] + 0.048,
             f"{len(shown)} of {d['in_deg']} in-neighbors",
             fontsize=8.5, color=MUTED)
    axn.text(0.005, OY + 0.048, f"+{d['in_deg'] - len(shown)} more",
             fontsize=8.5, color=MUTED)
    art["n_edges"] = []
    for r in range(len(shown)):
        (ln,) = axn.plot([NX + 0.02, NEU[0]], [rows[r], NEU[1]], color=SLATE,
                         lw=1.5, zorder=2, solid_capstyle="round")
        art["n_edges"].append(ln)
    axn.plot([NX + 0.02, NEU[0]], [SY, NEU[1]], color=EDGE, lw=1.6, zorder=1)
    axn.plot([NX + 0.02, NEU[0]], [OY, NEU[1]], color=EDGE, lw=1.6,
             ls=(0, (2, 2)), zorder=1)
    art["wlabs"] = []
    for r in range(len(shown)):
        # stagger the labels along their edges so they don't stack into an
        # ambiguous column between the fanned-in lines
        frac = 0.30 + 0.09 * abs(r - (len(shown) - 1) / 2)
        x = NX + 0.02 + (NEU[0] - NX - 0.02) * frac
        y = rows[r] + (NEU[1] - rows[r]) * frac
        t = axn.text(x, y, "", fontsize=7.5, family="monospace", color=INK,
                     ha="center", va="center", zorder=6,
                     bbox=dict(fc="white", ec="none", alpha=0.65, pad=0.4))
        art["wlabs"].append(t)
    art["n_nodes"] = axn.scatter([NX] * len(shown), rows, s=230, c="#eef1f5",
                                 edgecolors=EDGE, linewidths=1.0, zorder=5)
    art["s_node"] = axn.scatter([NX], [SY], s=200, marker="s", c="#f7f8fa",
                                edgecolors=AMBER, linewidths=1.0, zorder=5)
    art["o_node"] = axn.scatter([NX], [OY], s=230, c="#f7f8fa",
                                edgecolors=EDGE, linewidths=1.0, zorder=5)
    art["neuron_node"] = axn.scatter([NEU[0]], [NEU[1]], s=1500, c="#f2f5f9",
                                     edgecolors=INK, linewidths=1.5, zorder=5)
    axn.text(NEU[0], NEU[1] - 0.150, f"neuron #{d['j']}", fontsize=8.5,
             color=MUTED, ha="center")
    axn.add_patch(FancyArrowPatch((0.762, NEU[1]), (0.985, NEU[1]),
                                  arrowstyle="-|>", mutation_scale=15,
                                  color=EDGE, lw=2.0, zorder=1))
    axn.text(0.985, NEU[1] + 0.058, "spikes out", fontsize=8.5, color=SPIKE,
             ha="right")
    art["in_ticks"] = LineCollection([], colors=SLATE, linewidths=1.4,
                                     zorder=3)
    art["s_bars"] = LineCollection([], colors=AMBER, linewidths=2.2, zorder=3)
    art["o_bars"] = LineCollection([], colors=SLATE, linewidths=2.2,
                                   alpha=0.55, zorder=3)
    art["out_ticks"] = LineCollection([], colors=SPIKE, linewidths=1.6,
                                      zorder=3)
    for c in ("in_ticks", "s_bars", "o_bars", "out_ticks"):
        axn.add_collection(art[c])
    art["pulses"] = axn.scatter([], [], zorder=4)

    # ---------------- waveform band (bottom-right) ----------------
    fig.text(0.98, 0.408, f"last {W} steps", fontsize=8.5, color=MUTED,
             ha="right")

    xlim = (-W + 0.5, 0.5)
    ax_in = fig.add_axes([0.535, 0.310, 0.445, 0.075])
    ax_in.set_xlim(*xlim)
    ax_in.set_xticks([])
    ax_in.set_yticks([])
    for side in ("top", "right"):
        ax_in.spines[side].set_visible(False)
    ax_in.axhline(0, color=EDGE, lw=0.6)
    top_in = max(float(d["sens"].max()), float(d["recur"].max()), 0.1)
    lo_in = min(0.0, float(d["recur"].min()))
    ax_in.set_ylim(lo_in * 1.1 - 0.02, top_in * 1.15)
    art["in_sens"] = LineCollection([], colors=AMBER, linewidths=2.4)
    art["in_rec"] = LineCollection([], colors=SLATE, linewidths=2.4)
    ax_in.add_collection(art["in_sens"])
    ax_in.add_collection(art["in_rec"])
    ax_in.text(0.005, 0.82, "arrivals", transform=ax_in.transAxes, fontsize=9,
               color=INK, fontweight="bold")
    ax_in.text(0.125, 0.82, "sensors", transform=ax_in.transAxes, fontsize=9,
               color=AMBER)
    ax_in.text(0.235, 0.82, "neighbor spikes", transform=ax_in.transAxes,
               fontsize=9, color=SLATE)

    ax_x = fig.add_axes([0.535, 0.132, 0.445, 0.163])
    ax_x.set_xlim(*xlim)
    ax_x.set_ylim(*art["ylim"])
    ax_x.set_xticks([])
    ax_x.set_yticks([])
    for side in ("top", "right"):
        ax_x.spines[side].set_visible(False)
    art["ax_x"] = ax_x
    art["fills"] = []
    (art["ln_thr"],) = ax_x.plot([], [], color=INK, lw=1.0, ls=(0, (4, 2)),
                                 zorder=4)
    (art["ln_tgt"],) = ax_x.plot([], [], color=TARGET, lw=1.0, ls=(0, (4, 2)),
                                 zorder=4)
    art["drops"] = LineCollection([], colors=SPIKE, linewidths=1.2, alpha=0.35,
                                  zorder=3)
    ax_x.add_collection(art["drops"])
    (art["ln_x"],) = ax_x.plot([], [], color=NODE, lw=1.6, zorder=5,
                               solid_capstyle="round",
                               solid_joinstyle="round")
    art["ticks"] = ax_x.scatter([], [], s=42, marker="|", color=SPIKE,
                                linewidths=1.1, zorder=6)
    (art["pen"],) = ax_x.plot([], [], "o", ms=6, color=NODE, zorder=7,
                              markeredgecolor="white", markeredgewidth=0.8)
    for dx, txt, col in ((0.005, "activation", NODE), (0.150, "target", TARGET),
                         (0.235, "threshold", INK), (0.350, "spikes", SPIKE)):
        ax_x.text(dx, 0.88, txt, transform=ax_x.transAxes, fontsize=9,
                  color=col, fontweight="bold" if dx == 0.005 else "normal",
                  bbox=dict(fc="white", ec="none", alpha=0.65, pad=0.6),
                  zorder=8)

    ax_rb = fig.add_axes([0.535, 0.111, 0.445, 0.016])
    ax_rb.set_xlim(*xlim)
    ax_rb.set_ylim(0, 1)
    ax_rb.axis("off")
    art["ribbon"] = ax_rb.imshow(np.zeros((1, W, 4)), aspect="auto",
                                 extent=(*xlim, 0, 1), interpolation="nearest",
                                 zorder=2)

    ax_k = fig.add_axes([0.535, 0.030, 0.445, 0.068])
    ax_k.set_xlim(*xlim)
    ax_k.set_xticks([])
    ax_k.spines["top"].set_visible(False)
    T_seg, wb_seg = d["T_pre"], d["wbar"]
    pad_T = max(float(T_seg.max() - T_seg.min()), 0.02) * 0.25
    ax_k.set_ylim(float(T_seg.min()) - pad_T, float(T_seg.max()) + pad_T)
    ax_k.set_yticks([])
    ax_k2 = ax_k.twinx()
    ax_k2.set_xlim(*xlim)
    pad_w = max(float(wb_seg.max() - wb_seg.min()), 0.02) * 0.25
    ax_k2.set_ylim(float(wb_seg.min()) - pad_w, float(wb_seg.max()) + pad_w)
    ax_k2.set_yticks([])
    ax_k2.spines["top"].set_visible(False)
    (art["ln_T"],) = ax_k.plot([], [], color=TARGET, lw=1.3)
    (art["ln_w"],) = ax_k2.plot([], [], color=GREEN, lw=1.3)
    ax_k.text(0.005, 0.80, "target", transform=ax_k.transAxes, fontsize=9,
              color=TARGET)
    ax_k.text(0.095, 0.80, "mean incoming weight", transform=ax_k.transAxes,
              fontsize=9, color=GREEN)
    return fig, art


# --------------------------------------------------------------------------
# per-frame update
# --------------------------------------------------------------------------

def make_update(d, art, sched):
    y_min, y_max = art["ylim"]
    ys = y_max - y_min
    rgba = {0: matplotlib.colors.to_rgba(ROSE, 0.75),
            1: matplotlib.colors.to_rgba(GREEN, 0.70),
            2: matplotlib.colors.to_rgba(ROSE, 0.75),
            3: matplotlib.colors.to_rgba(GREEN, 0.70)}
    base_face = np.array(matplotlib.colors.to_rgba("#f2f5f9"))
    node_rgb = np.array(matplotlib.colors.to_rgba(NODE))

    def fmt2(v):
        return f"{0.0 if abs(v) < 0.005 else v:.2f}"

    def stems(collection, xs, vals, mask, dx):
        segs = [((x + dx, 0.0), (x + dx, v)) for x, v, m in
                zip(xs, vals, mask) if m and abs(v) > 1e-6]
        collection.set_segments(segs)

    def update(f):
        s_idx, p, fp = sched[f]
        i = W + s_idx                    # index into the recorded arrays
        t_abs = d["t0"] + s_idx
        staged = fp >= 12                # slow phase: reveal events in order
        cp = p if staged else 1.0        # commit fraction for reveals
        lo = i - W
        idx = np.arange(lo, i + 1)
        xs = idx - (i - 1.0 + p)         # window slides left continuously

        # -------- stage 1 rows --------
        vals = {"leak": f"{fmt2(d['x_prev'][i])} → {fmt2(d['carry'][i])}",
                "sens": f"+{d['sens'][i]:.2f}",
                "recur": f"+{fmt2(d['recur'][i])} ({d['k'][i]} spiked)"
                         if d['recur'][i] >= 0 else
                         f"{fmt2(d['recur'][i])} ({d['k'][i]} spiked)"}
        gates = {"leak": 0.04, "sens": F_SENS1, "recur": F_RECUR1}
        row_zero = {"leak": False, "sens": d["sens"][i] < 0.005,
                    "recur": abs(d["recur"][i]) < 0.005}
        for key, (ar, lab, val, col) in art["rows"].items():
            on = cp >= gates[key] - 0.02
            ar.set_color(col if on else EDGE)
            ar.set_linewidth(2.8 if staged and on and p < gates[key] + 0.12
                             else 2.0)
            lab.set_color(INK if on else MUTED)
            val.set_text(vals[key] if on else "")
            val.set_color(MUTED if row_zero[key] else col)
            val.set_alpha(fade(cp, gates[key] - 0.02) if on else 1.0)
        art["total"].set_text(f"{d['x_pre'][i]:.2f}" if cp >= X_REVEAL else "")
        art["to_gauge"].set_color(
            NODE if staged and 0.04 <= p <= B_RECUR1 else EDGE)

        # -------- gauge --------
        v = ball_value(d, i, p)
        Td, THd = knob_display(d, i, p)
        art["ball"].set_offsets([[0.45, v]])
        ghosts = [(ball_value(d, i, p - dq), sz, al) for dq, sz, al in
                  ((0.05, 150, 0.22), (0.10, 110, 0.13), (0.15, 78, 0.06))
                  if p - dq > 0]
        if ghosts:
            art["ghost"].set_offsets([[0.45, g] for g, _, _ in ghosts])
            art["ghost"].set_sizes([s for _, s, _ in ghosts])
            art["ghost"].set_facecolors(
                [matplotlib.colors.to_rgba(NODE, a) for _, _, a in ghosts])
        else:
            art["ghost"].set_offsets(np.empty((0, 2)))
        art["zone_hungry"].set_y(y_min)
        art["zone_hungry"].set_height(Td - y_min)
        art["zone_sated"].set_y(Td)
        art["zone_sated"].set_height(THd - Td)
        art["zone_spike"].set_y(THd)
        art["zone_spike"].set_height(max(y_max - THd, 0.0))
        art["g_tgt"].set_ydata([Td, Td])
        art["g_thr"].set_ydata([THd, THd])
        art["g_tgt_lab"].set_position((0.05, Td + 0.008 * ys))
        art["g_thr_lab"].set_position((0.05, THd + 0.008 * ys))
        spiking = bool(d["spiked"][i])
        # the bolt fades in as the ball crosses and out again after the drop
        bolt_a = (fade(p, P_SPIKE0 - 0.04, 0.05)
                  * (1.0 - fade(p, P_SPIKE1 + 0.04, 0.08))) if spiking else 0.0
        art["bolt"].set_visible(bolt_a > 0.01)
        if bolt_a > 0.01:
            y_b = min(d["x_pre"][i], y_min + 0.90 * ys)
            art["bolt"].xy = (0.97, y_b)
            art["bolt"].set_position((0.50, min(y_b + 0.06 * ys,
                                                y_min + 0.96 * ys)))
            art["bolt"].set_alpha(bolt_a)
            art["bolt"].arrow_patch.set_alpha(bolt_a)
        if spiking and p >= P_SPIKE0:
            art["drop"].set_data([0.45, 0.45], [d["x_pre"][i], v])
            art["drop"].set_alpha(0.65 * fade(p, P_SPIKE0, 0.05))
        else:
            art["drop"].set_data([], [])
        art["axg"].set_facecolor(
            matplotlib.colors.to_rgba(SPIKE, 0.10 * bolt_a))

        # -------- error bracket + adapt card --------
        sated = d["E"][i] >= 0
        col = GREEN if sated else ROSE
        err_a = fade(cp, P_ERR)
        if err_a > 0:
            land = d["x_post"][i]
            art["err_line"].set_data([0.70, 0.70], [Td, land])
            art["err_line"].set_color(col)
            art["err_line"].set_alpha(err_a)
            art["err_lab"].set_position((0.76, (Td + land) / 2))
            art["err_lab"].set_text(f"{d['E'][i]:+.2f}")
            art["err_lab"].set_color(col)
            art["err_lab"].set_alpha(err_a)
        else:
            art["err_line"].set_data([], [])
            art["err_lab"].set_text("")
        art["to_adapt"].set_color(col if staged and p >= P_ERR else EDGE)
        active_key = "sated" if sated else "hungry"
        for key, (box, h, t1, t2, hcol, tint) in art["half"].items():
            on = key == active_key
            box.set_alpha(0.35 + 0.65 * err_a if on else 0.35)
            box.set_edgecolor(hcol if on and err_a > 0.5 else "#e4e8ee")
            box.set_linewidth(1.0 + 0.6 * err_a if on else 1.0)
            for txt in (h, t1, t2):
                txt.set_alpha(1.0 if on else 1.0 - 0.65 * err_a)
        if cp >= P_ERR:
            floored = (not sated) and d["dT"][i] == 0.0
            dT_txt = f"{d['dT'][i]:+.4f}" + (" (floor)" if floored else "")
            k = d["k"][i]
            dw_txt = f"{d['dwlink'][i]:+.3f}" if k else "—"
            for v_art, txt, c in zip(
                    art["adapt_vals"],
                    (f"{d['E'][i]:+.2f}", dT_txt, dw_txt),
                    (col, TARGET, GREEN)):
                v_art.set_text(txt)
                v_art.set_color(c)
                v_art.set_alpha(err_a)
        else:
            for v_art in art["adapt_vals"]:
                v_art.set_text("")

        # -------- waveforms --------
        committed = idx < i
        last = np.arange(len(xs)) == len(xs) - 1
        sens_v = d["sens"][idx].copy()
        rec_v = d["recur"][idx].copy()
        sens_v[-1] *= smooth((p - F_SENS0) / (F_SENS1 - F_SENS0))
        rec_v[-1] *= smooth((p - F_RECUR0) / (F_RECUR1 - F_RECUR0))
        stems(art["in_sens"], xs, sens_v, committed | last, -0.18)
        stems(art["in_rec"], xs, rec_v, committed | last, +0.18)

        x_plot = d["x_pre"][idx].copy()
        show_last_x = cp >= X_REVEAL
        if not show_last_x:
            x_plot[-1] = np.nan
        art["ln_x"].set_data(xs, x_plot)
        art["pen"].set_data([xs[-1]], [x_plot[-1]] if show_last_x
                            else [np.nan])
        tgt_plot = d["T_pre"][idx].astype(float).copy()
        thr_plot = d["thr_pre"][idx].astype(float).copy()
        tgt_plot[-1], thr_plot[-1] = Td, THd
        # extend the level lines and zone fills one step past "now" so the
        # sliding window never leaves a bare sliver at the right edge
        xs_ext = np.append(xs, xs[-1] + 1.0)
        tgt_ext = np.append(tgt_plot, Td)
        thr_ext = np.append(thr_plot, THd)
        art["ln_tgt"].set_data(xs_ext, tgt_ext)
        art["ln_thr"].set_data(xs_ext, thr_ext)

        for coll in art["fills"]:
            coll.remove()
        art["fills"] = [
            art["ax_x"].fill_between(xs_ext, y_min, tgt_ext, color=T_HUNGRY,
                                     alpha=0.45, zorder=1, lw=0),
            art["ax_x"].fill_between(xs_ext, tgt_ext, thr_ext, color=T_SATED,
                                     alpha=0.45, zorder=1, lw=0),
            art["ax_x"].fill_between(xs_ext, thr_ext, y_max, color=T_SPIKE,
                                     alpha=0.45, zorder=1, lw=0),
        ]

        sp_mask = d["spiked"][idx] & (committed | (cp >= P_SPIKE0))
        art["ticks"].set_offsets(
            np.c_[xs[sp_mask], np.full(sp_mask.sum(), y_min + 0.965 * ys)]
            if sp_mask.any() else np.empty((0, 2)))
        art["drops"].set_segments(
            [((x, min(d["x_pre"][ii], y_max)), (x, d["x_post"][ii]))
             for x, ii in zip(xs[sp_mask], idx[sp_mask])])

        rb = np.zeros((1, len(xs), 4))
        done = committed | (cp >= P_KNOB0)
        for c_i, ii in enumerate(idx):
            if done[c_i]:
                rb[0, c_i] = rgba[d["branch"][ii]]
        art["ribbon"].set_data(rb)
        art["ribbon"].set_extent((xs[0] - 0.5, xs[-1] + 0.5, 0, 1))

        knob_mask = committed | (cp >= P_KNOB1)
        art["ln_T"].set_data(xs, np.where(knob_mask, d["T_post"][idx], np.nan))
        art["ln_w"].set_data(xs, np.where(knob_mask, d["wbar"][idx], np.nan))

        # -------- wiring panel --------
        net = art["net"]
        shown, rows = net["shown"], net["rows"]
        NX, NEU, SY, OY = net["NX"], net["NEU"], net["SY"], net["OY"]
        wnow = d["nbr_w"][i, shown]
        arr = d["nbr_spk"][i, shown]
        wref = net["wref"]
        in_flight = F_RECUR0 <= p <= F_RECUR1
        for r, (ln, w_v) in enumerate(zip(art["n_edges"], wnow)):
            ln.set_linewidth(0.6 + 3.4 * abs(w_v) / wref
                             + (1.2 if arr[r] and in_flight else 0.0))
            ln.set_color(SLATE if w_v >= 0 else ROSE)
        for txt, w_v in zip(art["wlabs"], wnow):
            txt.set_text(f"{w_v:+.2f}")
            txt.set_color("#5d6673" if w_v >= 0 else ROSE)
        firing = [bool(a) and p <= F_RECUR1 for a in arr]
        art["n_nodes"].set_edgecolors(
            [SPIKE if f else EDGE for f in firing])
        art["n_nodes"].set_facecolors(
            ["#fbe9ec" if f else "#eef1f5" for f in firing])
        sens_now = d["sens"][i]
        art["s_node"].set_facecolor(
            matplotlib.colors.to_rgba(AMBER, 0.15 + 0.75 * min(
                sens_now / net["sens_max"], 1.0)) if sens_now > 0.02
            else "#f7f8fa")
        n_oth = net["n_other"][i]
        art["o_node"].set_facecolor(
            matplotlib.colors.to_rgba(SLATE, 0.15 + 0.60 * min(
                n_oth / net["oth_max"], 1.0)) if n_oth else "#f7f8fa")
        # history tracks march continuously (one slot per step)
        M_in, dxt = 16, 0.0165
        segs = []
        for r, s in enumerate(shown):
            for a in range(1, M_in + 1):
                if d["nbr_spk"][i - a, s]:
                    x0 = NX - 0.03 - (a - 1 + p) * dxt
                    if x0 > 0.035:
                        segs.append(((x0, rows[r] - 0.026),
                                     (x0, rows[r] + 0.026)))
        art["in_ticks"].set_segments(segs)
        segs = []
        for a in range(1, M_in + 1):
            v_s = d["sens"][i - a]
            if v_s > 0.02:
                x0 = NX - 0.03 - (a - 1 + p) * dxt
                if x0 > 0.035:
                    segs.append(((x0, SY - 0.028),
                                 (x0, SY - 0.028 + 0.07 * v_s / net["sens_max"])))
        art["s_bars"].set_segments(segs)
        segs = []
        for a in range(1, M_in + 1):
            v_o = net["n_other"][i - a]
            if v_o:
                x0 = NX - 0.03 - (a - 1 + p) * dxt
                if x0 > 0.035:
                    segs.append(((x0, OY - 0.028),
                                 (x0, OY - 0.028 + 0.07 * v_o / net["oth_max"])))
        art["o_bars"].set_segments(segs)
        M_out, dxo = 14, 0.0125
        segs = []
        for a in range(M_out + 1):
            if a == 0 and not (spiking and p >= P_SPIKE1):
                continue          # the newest tick appears when the pulse lands
            if d["spiked"][i - a]:
                x0 = 0.785 + max(a - 1 + p, 0.0) * dxo
                if x0 < 0.955:
                    segs.append(((x0, NEU[1] - 0.030), (x0, NEU[1] + 0.030)))
        art["out_ticks"].set_segments(segs)

        def fly(p0, p1, src, dst=(NEU[0] - 0.03, NEU[1])):
            u = silk((p - p0) / (p1 - p0))
            return (src[0] + (dst[0] - src[0]) * u,
                    src[1] + (dst[1] - src[1]) * u)

        offs, sizes, cols = [], [], []
        if sens_now > 0.02 and F_SENS0 <= p <= F_SENS1:
            offs.append(fly(F_SENS0, F_SENS1, (NX + 0.03, SY)))
            sizes.append(90)
            cols.append(AMBER)
        if in_flight:
            for r in np.flatnonzero(arr):
                offs.append(fly(F_RECUR0, F_RECUR1, (NX + 0.03, rows[r])))
                sizes.append(45 + 150 * abs(wnow[r]) / wref)
                cols.append(SLATE)
            if n_oth:
                offs.append(fly(F_RECUR0, F_RECUR1, (NX + 0.03, OY)))
                sizes.append(40 + 110 * n_oth / net["oth_max"])
                cols.append(matplotlib.colors.to_rgba(SLATE, 0.55))
        if spiking and P_SPIKE0 <= p <= P_SPIKE1:
            offs.append(fly(P_SPIKE0, P_SPIKE1, (NEU[0] + 0.05, NEU[1]),
                            (0.965, NEU[1])))
            sizes.append(110)
            cols.append(SPIKE)
        if offs:
            art["pulses"].set_offsets(np.array(offs))
            art["pulses"].set_sizes(np.array(sizes))
            art["pulses"].set_facecolors(cols)
            art["pulses"].set_edgecolors("white")
            art["pulses"].set_linewidths(0.9)
        else:
            art["pulses"].set_offsets(np.empty((0, 2)))
        level = float(np.clip(v / max(THd, 1e-9), 0, 1))
        art["neuron_node"].set_facecolors(
            [base_face + (node_rgb - base_face) * (0.12 + 0.88 * level)])
        art["neuron_node"].set_edgecolors(
            [SPIKE if spiking and P_SPIKE0 - 0.02 <= p <= P_SPIKE1 + 0.06
             else INK])

        art["clock"].set_text(f"t = {t_abs:,}")
        art["speed"].set_text(
            "" if fp >= sched[0][2] else f"×{sched[0][2] // fp}")
        return []

    return update


# --------------------------------------------------------------------------

def _render_chunk(payload):
    """Render frames [a, b) into an mp4 part — runs in a worker process.

    update(f) depends only on the frame index, so any frame range can be
    rendered independently; the parts are stream-copy concatenated after.
    """
    d, cfg, sched, a, b, part_path = payload
    args = argparse.Namespace(**cfg)
    fig, art = build_figure(d, args)
    update = make_update(d, art, sched)
    writer = animation.FFMpegWriter(fps=args.fps, bitrate=6000,
                                    extra_args=["-pix_fmt", "yuv420p"])
    with writer.saving(fig, part_path, dpi=args.dpi):
        for f in range(a, b):
            update(f)
            writer.grab_frame()
    plt.close(fig)
    return part_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--loadout", type=str, default="paper",
                    choices=sorted(LOADOUT_BY_ID))
    ap.add_argument("--neuron", type=int, default=None,
                    help="override the automatic neuron choice")
    ap.add_argument("--t0", type=int, default=None,
                    help="override the automatic start step")
    ap.add_argument("--slow-steps", type=int, default=50)
    ap.add_argument("--ramp-steps", type=int, default=0)
    ap.add_argument("--fast-steps", type=int, default=0)
    ap.add_argument("--slow-fpstep", type=int, default=72,
                    help="frames per model step (72 @ 60 fps = 1.2 s/step)")
    ap.add_argument("--fast-fpstep", type=int, default=12)
    ap.add_argument("--fps", type=int, default=60)
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--out", type=str, default="scripts/out/neuron_rule.mp4")
    ap.add_argument("--still", type=int, default=None,
                    help="render frame N to a PNG instead of the video")
    ap.add_argument("--jobs", type=int,
                    default=max(1, (os.cpu_count() or 4) - 2),
                    help="parallel render workers (1 = sequential)")
    args = ap.parse_args()

    label = LOADOUT_BY_ID[args.loadout]["label"]
    print(f"loadout {args.loadout} ('{label}'), seed {args.seed}", flush=True)
    seg_steps = args.slow_steps + args.ramp_steps + args.fast_steps
    if args.neuron is None or args.t0 is None:
        n_pass1 = W + 700 + seg_steps + 50
        print(f"pass 1: simulating {n_pass1} steps to pick a neuron...",
              flush=True)
        rec = simulate_pass1(args.seed, n_pass1, args.loadout)
        # the "story" constraints (calm start, a charge-up into a spike)
        # apply to the opening ~12 steps regardless of how long the video is
        j, t0 = pick_neuron_and_window(rec, min(args.slow_steps, 12),
                                       seg_steps, t_min=W + 400,
                                       t_max=n_pass1 - seg_steps - 1)
        if args.neuron is not None:
            j = args.neuron
        if args.t0 is not None:
            t0 = args.t0
    else:
        j, t0 = args.neuron, args.t0
    print(f"neuron #{j}, steps {t0}..{t0 + seg_steps}", flush=True)

    print("pass 2: recording that neuron in detail...", flush=True)
    d = simulate_detail(args.seed, args.loadout, j, t0, seg_steps)
    counts = {name: int((d["branch"][W:] == code).sum()) for code, name in
              enumerate(("hungry", "sated", "spike→hungry", "spike→sated"))}
    print(f"  branches in segment: {counts}", flush=True)

    sched, fpsteps = build_schedule(args.slow_steps, args.ramp_steps,
                                    args.fast_steps, args.slow_fpstep,
                                    args.fast_fpstep)
    n_frames = len(sched)
    slow_f = args.slow_steps * args.slow_fpstep
    ramp_f = sum(fpsteps[args.slow_steps:args.slow_steps + args.ramp_steps])
    print(f"frame plan: {n_frames} frames = {n_frames / args.fps:.1f} s  "
          f"(slow 0..{slow_f - 1}, ramp ..{slow_f + ramp_f - 1}, "
          f"fast ..{n_frames - 1})", flush=True)
    spike_steps = np.flatnonzero(d["spiked"][W:W + args.slow_steps])
    frame_of_step = np.cumsum([0] + fpsteps)
    for s in spike_steps:
        print(f"  slow-phase spike at step {s}: frames "
              f"{frame_of_step[s]}..{frame_of_step[s + 1] - 1}", flush=True)

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.still is not None:
        fig, art = build_figure(d, args)
        update = make_update(d, art, sched)
        update(args.still)
        png = out.with_suffix("").as_posix() + f"_frame{args.still}.png"
        fig.savefig(png, dpi=args.dpi, facecolor="white")
        print(f"saved {png}", flush=True)
        return

    t_start = time.time()
    if args.jobs <= 1:
        fig, art = build_figure(d, args)
        update = make_update(d, art, sched)
        print(f"rendering {n_frames} frames at {args.fps} fps...", flush=True)
        anim = animation.FuncAnimation(fig, update, frames=n_frames,
                                       blit=False)
        writer = animation.FFMpegWriter(fps=args.fps, bitrate=6000,
                                        extra_args=["-pix_fmt", "yuv420p"])
        anim.save(out.as_posix(), writer=writer, dpi=args.dpi,
                  progress_callback=lambda i, n: (i % 250 == 0) and print(
                      f"  frame {i}/{n}", flush=True))
    else:
        bounds = np.linspace(0, n_frames, args.jobs + 1).astype(int)
        parts = [out.with_name(f"{out.stem}_part{k:02d}.mp4")
                 for k in range(args.jobs)]
        payloads = [(d, vars(args), sched, int(a), int(b), p.as_posix())
                    for a, b, p in zip(bounds, bounds[1:], parts) if b > a]
        print(f"rendering {n_frames} frames at {args.fps} fps on "
              f"{len(payloads)} workers...", flush=True)
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=args.jobs,
                                 mp_context=ctx) as ex:
            for done, path in enumerate(ex.map(_render_chunk, payloads), 1):
                print(f"  part {done}/{len(payloads)} done "
                      f"({pathlib.Path(path).name})", flush=True)
        concat = out.with_name(f"{out.stem}_parts.txt")
        concat.write_text("".join(f"file '{p.resolve()}'\n"
                                  for p in parts if p.exists()))
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "concat",
                        "-safe", "0", "-i", concat.as_posix(),
                        "-c", "copy", out.as_posix()], check=True)
        for p in parts:
            p.unlink(missing_ok=True)
        concat.unlink()
    print(f"saved {out} in {time.time() - t_start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
