"""H81: two pacemakers, one follower — homeostatic attention."""
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
A1_SEED, A2_SEED = 3, 33

def make_follower(start_xy):
    g = CHAMP["champion"]
    pc = PursuitConfig(eye_offsets=(0.0,), sensors_per_eye=91, box_size=30.0,
                       initial_agent_x=start_xy[0], initial_agent_y=start_xy[1],
                       wheel_base=g["wheel_base"],
                       intensity_scale=g["intensity_scale"])
    res = ReservoirConfig(n_inputs=pc.n_sensors, **{k: g[k] for k in RES_KEYS})
    net = HomeostaticReservoir(res, seed=CHAMP["champ_seed"])
    env = PursuitEnv(pc, rng=net.rng)
    return net, env

def cosim(start_xy, pacemakers, n=10800, teleport_at=None, teleport_to=None):
    """Follower senses the SUM of all pacemakers' retinal bumps."""
    sims = [WallSimulation(wall_config=PACE_CFG, seed=s) for s in pacemakers]
    net, env = make_follower(start_xy)
    D = np.zeros((n, len(sims)))
    for i in range(n):
        if teleport_at is not None and i == teleport_at:
            env.x, env.y = teleport_to
        acts = None
        for j, A in enumerate(sims):
            A.step()
            env.sx, env.sy = A.env.x, A.env.y
            D[i, j] = env.distance()
            a = env.sense()
            acts = a if acts is None else acts + a
        st = net.step(acts)
        env.apply_action(*map(float, st.outputs))
        env.steps += 1
    return D

def summarize(D, label):
    late = D[D.shape[0] // 2:]
    lock = [(late[:, j] < 4).mean() for j in range(D.shape[1])]
    out = dict(label=label, lock=[round(v, 3) for v in lock])
    if D.shape[1] == 2:
        locked = int(np.argmax(lock))
        rival_closer = late[:, 1 - locked] < late[:, locked]
        out["rival_closer_frac"] = round(float(rival_closer.mean()), 3)
    print(label, out)
    return out

def main():
    res = {}
    res["a2_only"] = summarize(cosim((24.0, 10.0), [A2_SEED]), "A2-only followability")
    res["startA1"] = summarize(cosim((15.0, 10.0), [A1_SEED, A2_SEED]), "start near A1")
    res["startA2"] = summarize(cosim((24.0, 10.0), [A1_SEED, A2_SEED]), "start near A2")
    res["teleport"] = summarize(cosim((15.0, 10.0), [A1_SEED, A2_SEED],
                                      teleport_at=5400, teleport_to=(24.0, 10.0)),
                                "teleport to A2 at 5400")
    (LAB / "h81_choice.json").write_text(json.dumps(res))

if __name__ == "__main__" and "--tol" not in sys.argv:
    main()

def cosim_scaled(start_xy, lam, n=10800):
    sims = [WallSimulation(wall_config=PACE_CFG, seed=s) for s in (A1_SEED, A2_SEED)]
    net, env = make_follower(start_xy)
    D = np.zeros((n, 2))
    for i in range(n):
        acts = None
        for j, A in enumerate(sims):
            A.step()
            env.sx, env.sy = A.env.x, A.env.y
            D[i, j] = env.distance()
            a = env.sense() * (1.0 if j == 0 else lam)
            acts = a if acts is None else acts + a
        st = net.step(acts)
        env.apply_action(*map(float, st.outputs))
        env.steps += 1
    return D

def main_b():
    out = {}
    for lam in (0.0, 0.1, 0.3, 1.0):
        D = cosim_scaled((15.0, 10.0), lam)
        late = D[len(D) // 2:]
        lock = [(late[:, j] < 4).mean() for j in range(2)]
        out[str(lam)] = [round(float(v), 3) for v in lock]
        print(f"lambda={lam}: lock A1 {lock[0]:.3f} | A2 {lock[1]:.3f}", flush=True)
    (LAB / "h81b_tolerance.json").write_text(json.dumps(out))

if __name__ == "__main__" and "--tol" in sys.argv:
    main_b()
