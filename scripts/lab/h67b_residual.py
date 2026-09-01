"""H67b: lap-periodic vs aperiodic decomposition of chain link motion."""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
from h50_depth import CHAIN_FILE, PACE_CFG, PACE_SEED, START_Y, make_follower, LAB

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from homeostasis.simulation import WallSimulation  # noqa: E402

def main():
    chain = [(g, s) for g, s in json.loads(CHAIN_FILE.read_text())["chain"]]
    A = WallSimulation(wall_config=PACE_CFG, seed=PACE_SEED)
    links = [make_follower(g, s, START_Y[i]) for i, (g, s) in enumerate(chain)]
    n = 21600
    P = np.zeros((n, 1 + len(links), 2))
    for i in range(n):
        A.step()
        tx, ty = A.env.x, A.env.y
        P[i, 0] = (tx, ty)
        for j, (net, env) in enumerate(links):
            env.sx, env.sy = tx, ty
            st = net.step(env.sense())
            env.apply_action(*map(float, st.outputs)); env.steps += 1
            tx, ty = env.x, env.y
            P[i, 1 + j] = (env.x, env.y)
    # estimate lap period from the pacemaker's angle
    th = np.unwrap(np.arctan2(P[3600:, 0, 1] - 15, P[3600:, 0, 0] - 15))
    T = int(round(2 * np.pi / abs(np.diff(th).mean())))
    names = ["A", "B", "C", "D"]
    out = {"period": T}
    half = P[7200:, :, :]
    laps = len(half) // T
    k61 = np.ones(61) / 61.0
    for k, nm in enumerate(names):
        seg = half[:laps * T, k].reshape(laps, T, 2)
        mean_shape = seg.mean(axis=0)
        resid = seg - mean_shape
        rsd = float(np.sqrt((resid ** 2).sum(axis=2).mean()))
        # fast component: deviation from a 61-step moving average (OU tau=60 scale)
        fx = half[:, k, 0] - np.convolve(half[:, k, 0], k61, "same")
        fy = half[:, k, 1] - np.convolve(half[:, k, 1], k61, "same")
        fast = float(np.sqrt((fx[61:-61] ** 2 + fy[61:-61] ** 2).mean()))
        # phase-aligned residual: bin this link's position by PACEMAKER angle
        ang = np.arctan2(half[:, 0, 1] - 15, half[:, 0, 0] - 15)
        bins = ((ang + np.pi) / (2 * np.pi) * 72).astype(int) % 72
        pr = 0.0
        cnt = 0
        for b in range(72):
            m = bins == b
            if m.sum() > 10:
                q = half[m, k, :]
                pr += float(((q - q.mean(0)) ** 2).sum(1).mean()) * m.sum()
                cnt += m.sum()
        phase_resid = float(np.sqrt(pr / max(cnt, 1)))
        out[nm] = dict(lap_resid=rsd, fast=fast, phase_resid=phase_resid)
        print(f"{nm}: lap-resid {rsd:.3f} | fast sd {fast:.3f} | phase-aligned resid {phase_resid:.3f}")
    (LAB / "h67b_residual.json").write_text(json.dumps(out))

if __name__ == "__main__":
    main()
