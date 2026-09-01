"""H63: morphological low-pass link as a chain repeater."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from h33_evolve_pursuit import mutate, crossover, tournament, random_genome
from h50_depth import CHAIN_FILE, LAB, cosim_chain, ga_link, evaluate
from concurrent.futures import ProcessPoolExecutor

def ga_link_heavy(chain, warm, rng, gens=8, pop_n=16, wb_min=8.0):
    """ga_link with wheel_base projected into [wb_min, 16]."""
    def proj(g):
        g = dict(g)
        g["wheel_base"] = float(min(max(g["wheel_base"], wb_min), 16.0))
        return g
    wg, ws = warm
    wg = proj({**wg, "wheel_base": 10.0})
    pop = [(wg, ws)] + \
          [(proj(mutate(dict(wg), rng)), ws) for _ in range(4)] + \
          [(proj(mutate(dict(wg), rng)), int(rng.integers(0, 100000))) for _ in range(5)] + \
          [(proj(random_genome(rng)), int(rng.integers(0, 100000))) for _ in range(pop_n - 10)]
    best = None
    with ProcessPoolExecutor(10) as pool:
        for gen in range(gens):
            rows = list(pool.map(evaluate, [(chain, g, s) for g, s in pop], chunksize=1))
            fits = [r["fit"] for r in rows]
            bi = int(np.argmax(fits))
            if best is None or rows[bi]["fit"] > best["fit"]:
                best = dict(rows[bi], champion=pop[bi][0], champ_seed=pop[bi][1])
            elite = pop[bi]
            new = [elite]
            while len(new) < len(pop):
                pa = tournament(pop, fits, rng); pb = tournament(pop, fits, rng)
                g = proj(mutate(crossover(pa[0], pb[0], rng), rng))
                s = pa[1] if rng.random() < 0.5 else pb[1]
                if rng.random() < 0.2:
                    s = int(rng.integers(0, 100000))
                new.append((g, int(s)))
            pop = new
    return best

def main():
    chain = [(g, s) for g, s in json.loads(CHAIN_FILE.read_text())["chain"]]
    rng = np.random.default_rng(631)
    E = ga_link_heavy(chain, chain[-1], rng)
    print(f"heavy E: near4 {E['near4']:.2f} dist {E['dist']:.2f} sd {E['sd']:.2f}"
          f" wheel_base {E['champion']['wheel_base']:.1f}", flush=True)
    out = dict(E=dict(near4=E["near4"], dist=E["dist"], sd=E["sd"],
                      champion=E["champion"], champ_seed=E["champ_seed"]))
    if E["near4"] >= 0.6:
        chain2 = chain + [(E["champion"], E["champ_seed"])]
        r = cosim_chain(chain2[:-1], chain2[-1][0], chain2[-1][1], n=7200, jitter=True)
        print("per-link sd with E:", [round(v, 3) for v in r["link_sd"]], flush=True)
        out["sd_profile"] = r["link_sd"]
        F = ga_link(chain2, chain2[-1], rng)
        print(f"free F: near4 {F['near4']:.2f} dist {F['dist']:.2f} sd {F['sd']:.2f}")
        out["F"] = dict(near4=F["near4"], dist=F["dist"], sd=F["sd"],
                        champion=F["champion"], champ_seed=F["champ_seed"])
    (LAB / "h63_repeater.json").write_text(json.dumps(out))

if __name__ == "__main__":
    main()
