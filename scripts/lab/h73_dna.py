"""H73: evolve a wiring-statistics generator (the DNA question)."""
from __future__ import annotations
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from common import make_configs, run_closed_loop  # noqa: F401
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from homeostasis.reservoir import HomeostaticReservoir  # noqa: E402
from homeostasis.tracking import TrackingEnv  # noqa: E402

LAB = Path(__file__).resolve().parents[1] / "out" / "lab"
SEG = 720

GENOME = {  # (lo, hi, log)
    "p": (0.02, 0.3, True),
    "reg": (0.0, 1.0, False),
    "b": (-2.0, 2.0, False),
    "rec": (0.0, 0.5, False),
}

def rand_g(rng):
    g = {}
    for k, (lo, hi, log) in GENOME.items():
        g[k] = float(np.exp(rng.uniform(np.log(lo), np.log(hi))) if log
                     else rng.uniform(lo, hi))
    return g

def mut(g, rng, rate=0.5):
    out = dict(g)
    for k, (lo, hi, log) in GENOME.items():
        if rng.random() < rate:
            v = out[k]
            v = float(np.exp(np.log(v) + rng.normal(0, 0.3))) if log else v + rng.normal(0, (hi - lo) * 0.15)
            out[k] = float(min(max(v, lo), hi))
    return out

def build_wiring(g, N, n_inputs, rng):
    """Sample an adjacency + input adjacency from generator statistics."""
    k = max(1, int(round(g["p"] * N)))
    adj = np.zeros((N, N), dtype=bool)
    regular = rng.random(N) < g["reg"]
    for n in range(N):
        deg = k if regular[n] else rng.binomial(N, g["p"])
        deg = int(min(max(deg, 0), N - 1))
        if deg:
            pre = rng.choice(N - 1, size=deg, replace=False)
            pre = pre + (pre >= n)  # skip self
            adj[pre, n] = True
    if g["rec"] > 0:
        s, t = np.nonzero(adj)
        m = rng.random(len(s)) < g["rec"]
        adj[t[m], s[m]] = True
    np.fill_diagonal(adj, False)
    indeg = adj.sum(axis=0).astype(float)
    w = (indeg + 1.0) ** g["b"]
    w = w / w.sum()
    inp = np.zeros((n_inputs, N), dtype=bool)
    n_links = int(round(0.1 * n_inputs * N))
    for _ in range(2):  # sample without worrying about exact count collisions
        pass
    flat = rng.choice(n_inputs * N, size=n_links, replace=False, p=np.tile(w, n_inputs) / n_inputs)
    inp[np.unravel_index(flat, (n_inputs, N))] = True
    return adj, inp

def eval_genome(task):
    g, seed = task
    rcfg, tcfg = make_configs({"weight_lr": 0.1, "target_lr": 0.01}, {})
    net = HomeostaticReservoir(rcfg, seed=seed)
    wrng = np.random.default_rng(seed + 550005)
    adj, inp = build_wiring(g, rcfg.n_nodes, rcfg.n_inputs, wrng)
    net.adjacency = adj
    net.weights = np.where(adj, wrng.normal(rcfg.weight_init_mean, rcfg.weight_init_sd,
                                            adj.shape), 0.0)
    net.input_adjacency = inp
    net.input_weights = inp * rcfg.input_weight
    orng = np.random.default_rng(seed + 880008)
    net.output_adjacency = orng.random((rcfg.n_nodes, rcfg.n_outputs)) < 0.1
    net._rebuild_structure_caches()
    env = TrackingEnv(tcfg)
    n = 7200
    n_seg = n // SEG
    seg_in45 = np.zeros(n_seg); seg_cnt = np.zeros(n_seg)
    for i in range(n):
        herr = env.heading_error()
        state = net.step(env.sense())
        env.apply_action(*state.outputs)
        env.advance_stimulus()
        s = min(i // SEG, n_seg - 1)
        seg_cnt[s] += 1
        seg_in45[s] += abs(herr) <= 45.0
    segs = seg_in45 / np.maximum(seg_cnt, 1)
    return float(np.mean(segs[5:]))

def fitness(g, seeds, pool):
    scores = list(pool.map(eval_genome, [(g, int(s)) for s in seeds]))
    return np.mean([s >= 0.35 for s in scores]) + 0.5 * np.mean(scores), scores

def main():
    rng = np.random.default_rng(73)
    with ProcessPoolExecutor(10) as pool:
        # baselines
        for name, g in (("bernoulli-p.1", dict(p=0.1, reg=0.0, b=0.0, rec=0.0)),
                        ("bernoulli-p.02", dict(p=0.02, reg=0.0, b=0.0, rec=0.0))):
            f, sc = fitness(g, range(200, 212), pool)
            print(f"baseline {name}: rel {np.mean([x>=0.35 for x in sc]):.2f} "
                  f"mean {np.mean(sc):.3f}", flush=True)
        pop = [rand_g(rng) for _ in range(16)]
        log = []
        for gen in range(6):
            seeds = rng.integers(0, 100000, size=4)
            fits = []
            for g in pop:
                f, _ = fitness(g, seeds, pool)
                fits.append(f)
            bi = int(np.argmax(fits))
            log.append(dict(gen=gen, fit=float(fits[bi]), champion=pop[bi]))
            print(f"gen {gen}: best fit {fits[bi]:.3f} genome "
                  f"{ {k: round(v, 3) for k, v in pop[bi].items()} }", flush=True)
            elite = pop[bi]
            new = [dict(elite)]
            while len(new) < len(pop):
                idx = rng.choice(16, 3, replace=False)
                a = pop[idx[int(np.argmax([fits[i] for i in idx]))]]
                new.append(mut(dict(a), rng))
            pop = new
        champ = log[-1]["champion"]
        _, sc = fitness(champ, range(300, 312), pool)
        print(f"champion 12-fresh: rel {np.mean([x>=0.35 for x in sc]):.2f} "
              f"mean {np.mean(sc):.3f}  genome { {k: round(v,3) for k,v in champ.items()} }")
        (LAB / "h73_dna.json").write_text(json.dumps(
            dict(log=log, champion=champ, fresh=sc)))

if __name__ == "__main__":
    main()
