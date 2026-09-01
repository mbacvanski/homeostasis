"""H60: is depth 4 the ceiling? Radii instrumentation + GA attempt at link E."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from h50_depth import (LAB, CHAIN_FILE, PACE_CFG, PACE_SEED, START_Y,
                       make_follower, cosim_chain, ga_link)
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from homeostasis.simulation import WallSimulation  # noqa: E402

def radii(chain, n=7200):
    """Mean orbit radius (about box center 15,15) of pacemaker + each link."""
    A = WallSimulation(wall_config=PACE_CFG, seed=PACE_SEED)
    links = [make_follower(g, s, START_Y[i]) for i, (g, s) in enumerate(chain)]
    acc = np.zeros(1 + len(links))
    half = n // 2
    for i in range(n):
        A.step()
        tx, ty = A.env.x, A.env.y
        if i >= half:
            acc[0] += np.hypot(tx - 15.0, ty - 15.0)
        for j, (net, env) in enumerate(links):
            env.sx, env.sy = tx, ty
            st = net.step(env.sense())
            env.apply_action(*map(float, st.outputs)); env.steps += 1
            tx, ty = env.x, env.y
            if i >= half:
                acc[1 + j] += np.hypot(env.x - 15.0, env.y - 15.0)
    return (acc / half).tolist()

def main():
    saved = json.loads(CHAIN_FILE.read_text())
    chain = [(g, s) for g, s in saved["chain"]]
    r = radii(chain)
    print("orbit radii A..D:", [round(v, 2) for v in r], flush=True)
    rng = np.random.default_rng(601)
    best = ga_link(chain, chain[-1], rng)
    print(f"link E: near4 {best['near4']:.2f} dist {best['dist']:.2f} sd {best['sd']:.2f}")
    out = dict(radii=r, E=dict(near4=best["near4"], dist=best["dist"], sd=best["sd"],
                               champion=best["champion"], champ_seed=best["champ_seed"]))
    if best["near4"] >= 0.6:
        chain2 = chain + [(best["champion"], best["champ_seed"])]
        r2 = radii(chain2)
        print("radii with E:", [round(v, 2) for v in r2])
        bestF = ga_link(chain2, chain2[-1], rng)
        print(f"link F: near4 {bestF['near4']:.2f} dist {bestF['dist']:.2f} sd {bestF['sd']:.2f}")
        out["radii_with_E"] = r2
        out["F"] = dict(near4=bestF["near4"], dist=bestF["dist"], sd=bestF["sd"])
    (LAB / "h60_depth5.json").write_text(json.dumps(out))

if __name__ == "__main__":
    main()
