"""H60b: are the chain rings stationary? Radii by thirds over 21600 steps."""
from __future__ import annotations
import json
import numpy as np
from h50_depth import CHAIN_FILE, PACE_CFG, PACE_SEED, START_Y, make_follower, LAB
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from homeostasis.simulation import WallSimulation  # noqa: E402

def main():
    chain = [(g, s) for g, s in json.loads(CHAIN_FILE.read_text())["chain"]]
    A = WallSimulation(wall_config=PACE_CFG, seed=PACE_SEED)
    links = [make_follower(g, s, START_Y[i]) for i, (g, s) in enumerate(chain)]
    n = 21600
    R = np.zeros((n, 1 + len(links)))
    for i in range(n):
        A.step()
        tx, ty = A.env.x, A.env.y
        R[i, 0] = np.hypot(tx - 15.0, ty - 15.0)
        for j, (net, env) in enumerate(links):
            env.sx, env.sy = tx, ty
            st = net.step(env.sense())
            env.apply_action(*map(float, st.outputs)); env.steps += 1
            tx, ty = env.x, env.y
            R[i, 1 + j] = np.hypot(env.x - 15.0, env.y - 15.0)
    names = ["A", "B", "C", "D"]
    out = {}
    for k, nm in enumerate(names):
        thirds = [float(R[i * n // 3:(i + 1) * n // 3, k].mean()) for i in range(3)]
        out[nm] = thirds
        print(f"{nm}: thirds {['%.2f' % t for t in thirds]}")
    (LAB / "h60b_drift.json").write_text(json.dumps(out))

if __name__ == "__main__":
    main()
