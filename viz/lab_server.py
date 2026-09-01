"""Single-node lab: interactive explorer for one homeostatic node.

First of the lab viewers — small pages that expose the design-space
campaign's protocols (scripts/lab/) interactively. A self-contained FastAPI
sub-application, mounted at /lab by viz.server. Same ground rule as the other
visualizers: no model logic lives here — every trace comes from the tested
`homeostasis` package, run server-side; the frontend only renders JSON.

The protocol is exactly scripts/lab/k2_single_node.py: a single node (no
recurrence possible at p_link=0) driven by one always-on input of weight mu,
so it receives constant drive mu per step. The end state over the last 500
steps is classified with k2's rules (dead-floor / silent-comf / spiking /
frozen-cycle), and the duty-law prediction f = (mu/T - leak)/rho is returned
alongside the observed late firing rate.

Run via the main app:  uvicorn viz.server:app --port 8471  ->  /lab
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from homeostasis.reservoir import HomeostaticReservoir, ReservoirConfig

STATIC_DIR = Path(__file__).parent / "static"

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


lab_app = FastAPI(title="homeostasis single-node lab")


@lab_app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "lab.html")


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
