"""H97: reproduction-at-comfort vs random starts — viability evolution in the arena."""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from homeostasis.reservoir import HomeostaticReservoir, ReservoirConfig  # noqa: E402
from homeostasis.simulation import WallSimulation  # noqa: E402
from homeostasis.pursuit import PursuitConfig, PursuitEnv  # noqa: E402
from h33_evolve_pursuit import mutate  # noqa: E402
from h50_depth import PACE_CFG, PACE_SEED, LAB, RES_KEYS  # noqa: E402

CHAMP = json.loads((LAB / "h48e_warm.json").read_text())
LOCK_D, SPAWN_AFTER, CAP = 4.8, 1200, 6
INHERIT_WIRING = False
CLONE_TEST = False
BUDDING = False

class Agent:
    def __init__(self, genome, net_seed, xy):
        self.genome = genome
        pc = PursuitConfig(eye_offsets=(0.0,), sensors_per_eye=91, box_size=30.0,
                          initial_agent_x=xy[0], initial_agent_y=xy[1],
                          wheel_base=genome["wheel_base"],
                          intensity_scale=genome["intensity_scale"])
        res = ReservoirConfig(n_inputs=pc.n_sensors, **{k: genome[k] for k in RES_KEYS})
        self.net = HomeostaticReservoir(res, seed=net_seed)
        self.env = PursuitEnv(pc, rng=self.net.rng)
        self.lock_streak = 0
        self.spawned = 0
        self.mutant = False

    def step(self, target):
        self.env.sx, self.env.sy = target
        d = self.env.distance()
        st = self.net.step(self.env.sense())
        self.env.apply_action(*map(float, st.outputs))
        self.env.steps += 1
        self.lock_streak = self.lock_streak + 1 if d < LOCK_D else 0
        return d

def run(reproduce, n=21600, lineage_seed=971):
    rng = np.random.default_rng(lineage_seed)
    A = WallSimulation(wall_config=PACE_CFG, seed=PACE_SEED)
    agents = [Agent(dict(CHAMP["champion"]), CHAMP["champ_seed"], (15.0, 10.0))]
    if not reproduce:
        for _ in range(CAP - 1):
            ang = rng.uniform(0, 2 * np.pi)
            r = 7.8 + rng.uniform(-2, 2)
            xy = (float(np.clip(19.7 + r * np.cos(ang), 1, 29)),
                  float(np.clip(19.7 + r * np.sin(ang), 1, 29)))
            agents.append(Agent(dict(CHAMP["champion"]), int(rng.integers(0, 100000)), xy))
    D = []
    for i in range(n):
        A.step()
        target = (A.env.x, A.env.y)
        ds = [ag.step(target) for ag in agents]
        D.append(ds + [np.nan] * (CAP - len(ds)))
        if reproduce and len(agents) < CAP:
            for ag in list(agents):
                if ag.lock_streak >= SPAWN_AFTER and ag.spawned == 0 and len(agents) < CAP:
                    child_g = (dict(ag.genome) if CLONE_TEST
                               else mutate(dict(ag.genome), rng))
                    child_seed = (CHAMP["champ_seed"] if INHERIT_WIRING
                                  else int(rng.integers(0, 100000)))
                    child = Agent(child_g, child_seed,
                                  (ag.env.x, ag.env.y))
                    if BUDDING:
                        child.net.x = ag.net.x.copy()
                        child.net.targets = ag.net.targets.copy()
                        child.net.weights = ag.net.weights.copy()
                        child.net.spiked = ag.net.spiked.copy()
                        child.net._spiked_f = ag.net._spiked_f.copy()
                        child.env.heading = ag.env.heading
                    child.mutant = child_g != ag.genome
                    agents.append(child)
                    ag.spawned = 1
                    ag.lock_streak = 0
    D = np.array(D)
    late = D[-3600:]
    locked = [(bool(np.nanmean(late[:, j] < LOCK_D) >= 0.8) if not np.all(np.isnan(late[:, j])) else False)
              for j in range(CAP)]
    mutants_locked = sum(1 for j, ag in enumerate(agents)
                         if j < CAP and locked[j] and getattr(ag, "mutant", False))
    return dict(n_agents=len(agents), locked=int(sum(locked)),
                locked_list=locked, mutants_locked=int(mutants_locked))

def main():
    global INHERIT_WIRING, CLONE_TEST, BUDDING
    if "--budding" in sys.argv:
        INHERIT_WIRING = True
        CLONE_TEST = True
        BUDDING = True
        r = run(True)
        print(f"budding (full state copy): agents {r['n_agents']},"
              f" locked late {r['locked']}/{r['n_agents']}")
        (LAB / "h97d_budding.json").write_text(json.dumps(r))
        return
    if "--clone" in sys.argv:
        INHERIT_WIRING = True
        CLONE_TEST = True
        r = run(True)
        print(f"pure-clone spawn: agents {r['n_agents']},"
              f" locked late {r['locked']}/{r['n_agents']}")
        (LAB / "h97c_clone.json").write_text(json.dumps(r))
        return
    if "--heredity" in sys.argv:
        INHERIT_WIRING = True
        r = run(True)
        print(f"reproduce+wiring-heredity: agents {r['n_agents']},"
              f" locked late {r['locked']}/{r['n_agents']},"
              f" locked mutants {r['mutants_locked']}")
        (LAB / "h97b_heredity.json").write_text(json.dumps(r))
        return
    for label, rep in (("reproduce", True), ("random-ctrl", False)):
        r = run(rep)
        print(f"{label}: agents {r['n_agents']}, locked late {r['locked']}/{r['n_agents']}"
              f", locked mutants {r['mutants_locked']}", flush=True)
        if rep:
            out = r
    (LAB / "h97_reproduce.json").write_text(json.dumps(out))

if __name__ == "__main__":
    main()
