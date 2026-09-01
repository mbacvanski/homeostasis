"""H86: the saved H50 chain, all-visible, capture-resistant sticky attention."""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from homeostasis.reservoir import HomeostaticReservoir, ReservoirConfig  # noqa: E402
from homeostasis.simulation import WallSimulation  # noqa: E402
from homeostasis.pursuit import PursuitConfig, PursuitEnv  # noqa: E402
from h50_depth import CHAIN_FILE, PACE_CFG, PACE_SEED, START_Y, LAB, RES_KEYS  # noqa: E402

RATIO, PATIENCE = 5.0, 300

class Follower:
    def __init__(self, genome, net_seed, start_y, n_sources):
        pc = PursuitConfig(eye_offsets=(0.0,), sensors_per_eye=91, box_size=30.0,
                          initial_agent_x=15.0, initial_agent_y=start_y,
                          wheel_base=genome["wheel_base"],
                          intensity_scale=genome["intensity_scale"])
        res = ReservoirConfig(n_inputs=pc.n_sensors, **{k: genome[k] for k in RES_KEYS})
        self.net = HomeostaticReservoir(res, seed=net_seed)
        self.env = PursuitEnv(pc, rng=self.net.rng)
        self.sel = 0
        self.streak = 0
        self.switches = []

    def step(self, sources, t):
        bumps = []
        for (sx, sy) in sources:
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
            self.switches.append(t)
        st = self.net.step(bumps[self.sel])
        self.env.apply_action(*map(float, st.outputs))
        self.env.steps += 1
        return self.env.x, self.env.y

def main():
    chain = [(g, s) for g, s in json.loads(CHAIN_FILE.read_text())["chain"]]
    A = WallSimulation(wall_config=PACE_CFG, seed=PACE_SEED)
    F = [Follower(g, s, START_Y[i], 3) for i, (g, s) in enumerate(chain)]
    # source order for follower i: [its h50-era target (i-1), then the others]
    n = 10800
    pos = [(15.0, START_Y[i]) for i in range(3)]
    D = np.zeros((n, 3))
    for i in range(n):
        A.step()
        agents = [(A.env.x, A.env.y)] + list(pos)
        for j, f in enumerate(F):
            own_target = agents[j]           # chain predecessor, fresh (sequential)
            others = [agents[k] for k in range(4) if k != j + 1 and agents[k] != own_target]
            srcs = [own_target] + others
            newpos = f.step(srcs[:3], i)
            D[i, j] = np.hypot(newpos[0] - own_target[0], newpos[1] - own_target[1])
            agents[j + 1] = newpos
        pos = agents[1:]
    late = D[n // 2:]
    out = {}
    for j, nm in enumerate(("B_on_A", "C_on_B", "D_on_C")):
        thr = [4.8, 3.9, 4.2][j]  # h50-era link dist +1
        out[nm] = float((late[:, j] < thr).mean())
        print(f"{nm}: lock {out[nm]:.3f} (dist {late[:, j].mean():.2f})"
              f" switches {len(F[j].switches)}")
    (LAB / "h86_chain_shared.json").write_text(json.dumps(out))

if __name__ == "__main__":
    main()
