import os, sys, json, pathlib
for v in ("VECLIB_MAXIMUM_THREADS","OPENBLAS_NUM_THREADS","OMP_NUM_THREADS"): os.environ.setdefault(v,"1")
import dataclasses
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from homeostasis import PONG_RESERVOIR_CONFIG, PongConfig, PongSimulation

champ = json.loads(pathlib.Path("/Users/marc/Code/homeostasis4/scripts/out/evolution_pong/champions.json").read_text())[0]["params"]
RES_KEYS = ("n_nodes","p_link","input_weight","weight_init_mean","weight_init_sd","inhibitory_fraction","leak","target_lr","threshold_ratio")

def run(task):
    seed, mode = task
    r_cfg = dataclasses.replace(PONG_RESERVOIR_CONFIG, **{k: champ[k] for k in RES_KEYS})
    sim = PongSimulation(r_cfg, PongConfig(gain=champ["gain"]), seed=seed)
    if mode == "off-init":
        sim.network.learning_enabled = False
    spike_tail = 0.0
    for t in range(30000):
        if mode == "freeze-mid" and t == 15000:
            sim.network.learning_enabled = False
        state, _, _ = sim.step()
        if t >= 25000:
            spike_tail += state.prop_spiked
    hits = np.asarray(sim.env.hits, dtype=float)
    n = hits.size
    half = next((i for i, _ in enumerate(hits)), 0)
    # split hits into opportunities before/after the freeze point isn't tracked
    # per-step; approximate by first/second half of opportunities
    return {"seed": seed, "mode": mode, "hit": float(hits.mean()),
            "hit_2nd_half": float(hits[n//2:].mean()) if n else 0.0,
            "spike_tail": spike_tail/5000, "opps": int(n)}

if __name__ == "__main__":
    tasks = [(s, "on") for s in range(24)] + [(s, m) for s in range(12) for m in ("off-init","freeze-mid")]
    with ProcessPoolExecutor(10) as pool:
        res = list(pool.map(run, tasks, chunksize=2))
    for mode in ("on","off-init","freeze-mid"):
        rs = [r for r in res if r["mode"]==mode]
        h = np.array([r["hit"] for r in rs]); h2 = np.array([r["hit_2nd_half"] for r in rs])
        st = np.array([r["spike_tail"] for r in rs])
        print(f"{mode:>10} (n={len(rs)}): hit {h.mean():.3f} ± {h.std():.3f} "
              f"(median {np.median(h):.2f}, min {h.min():.2f}, max {h.max():.2f}) "
              f"| 2nd-half {h2.mean():.3f} | spike@tail {st.mean():.2f}")
    on = {r["seed"]: r["hit"] for r in res if r["mode"]=="on"}
    print(f"\nseed 0 learning-on: {on[0]:.3f}; per-seed (0-11):",
          " ".join(f"{on[s]:.2f}" for s in range(12)))
    fm = {r["seed"]: r for r in res if r["mode"]=="freeze-mid"}
    print("freeze-mid per-seed hit/2nd-half/spike:",
          " ".join(f"{fm[s]['hit']:.2f}/{fm[s]['hit_2nd_half']:.2f}/{fm[s]['spike_tail']:.1f}" for s in range(6)))
