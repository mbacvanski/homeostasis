"""H87: many followers, one pacemaker — the homeostatic flock."""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from homeostasis.reservoir import HomeostaticReservoir, ReservoirConfig  # noqa: E402
from homeostasis.simulation import WallSimulation  # noqa: E402
from homeostasis.pursuit import PursuitConfig, PursuitEnv  # noqa: E402
from h50_depth import PACE_CFG, PACE_SEED, LAB, RES_KEYS  # noqa: E402

CHAMP = json.loads((LAB / "h48e_warm.json").read_text())
STARTS = [(15.0, 10.0), (15.0, 6.0), (24.0, 15.0), (6.0, 15.0)]
RATIO, PATIENCE = 5.0, 300

class F:
    def __init__(self, start):
        g = CHAMP["champion"]
        pc = PursuitConfig(eye_offsets=(0.0,), sensors_per_eye=91, box_size=30.0,
                          initial_agent_x=start[0], initial_agent_y=start[1],
                          wheel_base=g["wheel_base"],
                          intensity_scale=g["intensity_scale"])
        res = ReservoirConfig(n_inputs=pc.n_sensors, **{k: g[k] for k in RES_KEYS})
        self.net = HomeostaticReservoir(res, seed=CHAMP["champ_seed"])
        self.env = PursuitEnv(pc, rng=self.net.rng)
        self.sel, self.streak = 0, 0

    def step(self, sources):
        bumps = []
        for sx, sy in sources:
            self.env.sx, self.env.sy = sx, sy
            bumps.append(self.env.sense())
        sums = [b.sum() for b in bumps]
        riv = int(np.argmax(sums))
        if riv != self.sel and sums[riv] >= RATIO * max(sums[self.sel], 1e-9):
            self.streak += 1
        else:
            self.streak = 0
        if self.streak >= PATIENCE:
            self.sel, self.streak = riv, 0
        st = self.net.step(bumps[self.sel])
        self.env.apply_action(*map(float, st.outputs))
        self.env.steps += 1
        return self.env.x, self.env.y

def main():
    A = WallSimulation(wall_config=PACE_CFG, seed=PACE_SEED)
    fs = [F(s) for s in STARTS]
    pos = list(STARTS)
    n = 10800
    D = np.zeros((n, 4))
    TH = np.zeros((n, 4))
    for i in range(n):
        A.step()
        pa = (A.env.x, A.env.y)
        for j, f in enumerate(fs):
            others = [pos[k] for k in range(4) if k != j]
            newp = f.step([pa] + others)
            D[i, j] = np.hypot(newp[0] - pa[0], newp[1] - pa[1])
            TH[i, j] = np.arctan2(newp[1] - 19.7, newp[0] - 19.7)
            pos[j] = newp
    late = slice(n // 2, None)
    locks = [(D[late, j] < 4.8).mean() for j in range(4)]
    print("locks on A:", [round(float(v), 3) for v in locks])
    seps = []
    for a in range(4):
        for b in range(a + 1, 4):
            d = np.rad2deg(np.abs((TH[late, a] - TH[late, b] + np.pi) % (2 * np.pi) - np.pi))
            seps.append(float(d.mean()))
    print("pairwise mean angular separations (deg):", [round(v, 1) for v in seps])
    (LAB / "h87_flock.json").write_text(json.dumps(dict(
        locks=[float(v) for v in locks], seps=seps)))

if __name__ == "__main__":
    main()
