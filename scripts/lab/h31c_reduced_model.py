"""H31c: the 2-ODE reduced entrainment model, parameters DERIVED not fitted.

State: heading error theta (deg, stimulus - heading), stored bias b (deg/step).
    vis(theta) = 1 if |theta| <= 92 else 0            (retinal aperture; the pawl)
    theta' = v*d - b_eff,   b_eff = b * vis(theta)    (darkness stalls turning)
    b'     = vis(theta) * (alpha*(v*d - b) - beta*b)  (writing needs input)
Per config, alpha = asym/tau_w and beta = (1-asym)/tau_w with
    tau_w  = measured OPEN-LOOP pool-asymmetry flip time (h31_openloop.json)
    asym   = measured entrainment asymptote (b6b_reentrainment.json)
Protocol mirrors b6b: entrain at d=+1 to fixed point, flip d, record b(t)
toward the new direction; report tau63 and the excursion depth. Zero free
parameters. Success: reproduces the measured tau ordering 30/90/135/225 and
the excursion ordering.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

LAB = Path(__file__).resolve().parents[1] / "out" / "lab"
V = 1.0
VIEW = 92.0


def simulate(alpha, beta, n=1500):
    asym = alpha / (alpha + beta)
    b = asym * V          # entrained to old direction +1
    theta = 0.0
    d = -1.0              # reversal at t=0
    bs = np.empty(n)
    for t in range(n):
        vis = 1.0 if abs(theta) <= VIEW else 0.0
        theta += V * d - b * vis
        theta = (theta + 180.0) % 360.0 - 180.0
        b += vis * (alpha * (V * d - b) - beta * b)
        bs[t] = b * d     # + = toward NEW direction
    return bs


def main():
    ol = json.loads((LAB / "h31_openloop.json").read_text())
    b6b = json.loads((LAB / "b6b_reentrainment.json").read_text())
    print("config     tau_w  asym | model tau63  model exc | measured tau63  exc")
    for name, wlr in (("wlr0.03", 0.03), ("wlr0.1", 0.1), ("wlr0.3", 0.3), ("wlr1.0", 1.0)):
        tau_w = float(np.median([r["flip_steps"] for r in ol if r["wlr"] == wlr and r["flip_steps"] >= 0]))
        curves = np.array([r["curve"] for r in b6b if r["name"] == name])
        asym = float(np.clip(curves.mean(axis=0)[-8:].mean(), 0.05, 0.98))
        alpha = asym / tau_w
        beta = (1 - asym) / tau_w
        bs = simulate(alpha, beta)
        a_model = bs[-200:].mean()
        tau63 = next((t for t, v in enumerate(bs) if v >= 0.63 * a_model), -1)
        exc = float(-(bs[bs < 0]).sum()) if (bs < 0).any() else 0.0
        c = curves.mean(axis=0)
        m_asym = c[-8:].mean()
        m_tau = next((k * 30 for k, v in enumerate(c) if v >= 0.63 * m_asym), -1)
        m_exc = float(-(c[:np.argmax(c >= 0)] * 30).sum()) if c[0] < 0 else 0.0
        print(f"{name:9s} {tau_w:5.0f}  {asym:.2f} | {tau63:11d}  {exc:9.1f} | {m_tau:14d}  {m_exc:5.1f}")


if __name__ == "__main__":
    main()
