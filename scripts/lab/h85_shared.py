"""H85: shared-visibility three-agent ecology with sticky attention."""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from homeostasis.reservoir import HomeostaticReservoir, ReservoirConfig  # noqa: E402
from homeostasis.simulation import WallSimulation  # noqa: E402
from homeostasis.pursuit import PursuitConfig, PursuitEnv  # noqa: E402
from h50_depth import PACE_CFG, LAB, RES_KEYS  # noqa: E402

CHAMP = json.loads((LAB / "h48e_warm.json").read_text())
A_SEED = 3

class StickyFollower:
    def __init__(self, net_seed, start_xy, n_sources, sticky=True,
                 ratio=2.0, patience=100):
        self.ratio = ratio
        self.patience = patience
        g = CHAMP["champion"]
        pc = PursuitConfig(eye_offsets=(0.0,), sensors_per_eye=91, box_size=30.0,
                          initial_agent_x=start_xy[0], initial_agent_y=start_xy[1],
                          wheel_base=g["wheel_base"],
                          intensity_scale=g["intensity_scale"])
        res = ReservoirConfig(n_inputs=pc.n_sensors, **{k: g[k] for k in RES_KEYS})
        self.net = HomeostaticReservoir(res, seed=net_seed)
        self.env = PursuitEnv(pc, rng=self.net.rng)
        self.sticky = sticky
        self.sel = 0
        self.streak = 0
        self.switches = []
        self.n_sources = n_sources

    def step(self, sources, t):
        """sources: list of (x, y) of every OTHER agent, fixed order."""
        bumps = []
        for (sx, sy) in sources:
            self.env.sx, self.env.sy = sx, sy
            bumps.append(self.env.sense())
        sums = [b.sum() for b in bumps]
        if self.sticky:
            riv = int(np.argmax(sums))
            if riv != self.sel and sums[riv] >= self.ratio * max(sums[self.sel], 1e-9):
                self.streak += 1
            else:
                self.streak = 0
            if self.streak >= self.patience:
                self.sel = riv
                self.streak = 0
                self.switches.append(t)
            acts = bumps[self.sel]
        else:
            acts = np.sum(bumps, axis=0)
        st = self.net.step(acts)
        self.env.apply_action(*map(float, st.outputs))
        self.env.steps += 1
        return self.env.x, self.env.y

def run(sticky, n=10800, c_seed=None, c_start=(15.0, 5.0), ratio=2.0, patience=100):
    A = WallSimulation(wall_config=PACE_CFG, seed=A_SEED)
    B = StickyFollower(CHAMP["champ_seed"], (15.0, 10.0), 2, sticky,
                       ratio=ratio, patience=patience)
    C = StickyFollower(CHAMP["champ_seed"] if c_seed is None else c_seed,
                       c_start, 2, sticky, ratio=ratio, patience=patience)
    D = np.zeros((n, 4))  # B-A, C-A, C-B, B-C
    posB, posC = (15.0, 10.0), (15.0, 5.0)
    for i in range(n):
        A.step()
        pa = (A.env.x, A.env.y)
        newB = B.step([pa, posC], i)
        newC = C.step([pa, posB], i)
        posB, posC = newB, newC
        D[i] = (np.hypot(posB[0] - pa[0], posB[1] - pa[1]),
                np.hypot(posC[0] - pa[0], posC[1] - pa[1]),
                np.hypot(posC[0] - posB[0], posC[1] - posB[1]), 0.0)
    late = D[n // 2:]
    return dict(B_A=float((late[:, 0] < 4).mean()),
                C_A=float((late[:, 1] < 4).mean()),
                C_B=float((late[:, 2] < 4).mean()),
                B_switches=B.switches, C_switches=C.switches)

def main():
    for label, kw in (("resist5x@15,8", dict(c_start=(15.0, 8.0), ratio=5.0, patience=300)),
                      ("twin@15,8", dict(c_start=(15.0, 8.0))),
                      ("twin@24,15", dict(c_start=(24.0, 15.0))),
                      ("fresh@15,5", dict(c_seed=CHAMP["champ_seed"] + 1))):
        r = run(True, **kw)
        print(f"C={label}: B_A {r['B_A']:.3f} | C_A {r['C_A']:.3f} | C_B {r['C_B']:.3f}"
              f" | switches C{len(r['C_switches'])}", flush=True)
    for sticky in (True, False):
        r = run(sticky)
        print(f"sticky={sticky}: B locks A {r['B_A']:.3f} | C on A {r['C_A']:.3f}"
              f" | C on B {r['C_B']:.3f} | switches B{len(r['B_switches'])}/C{len(r['C_switches'])}",
              flush=True)
        if sticky:
            out = r
    (LAB / "h85_shared.json").write_text(json.dumps(out))

if __name__ == "__main__":
    main()
