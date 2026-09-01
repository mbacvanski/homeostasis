"""H27b: random-config screen on wall avoidance — does the flow-performance
sign invert? ~90 configs x 6 seeds, 3600 steps. Performance = 1 - hit rate
over the last 1800 steps. Metrics per run: mean flow, flow std, duty, |E|,
prop_spiked, plus the paper-default config as id 0."""
from __future__ import annotations
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from common import make_configs  # noqa: F401
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from homeostasis.reservoir import ReservoirConfig  # noqa: E402
from homeostasis.simulation import WALL_RESERVOIR_CONFIG, run_wall  # noqa: E402
import dataclasses  # noqa: E402

LAB = Path(__file__).resolve().parents[1] / "out" / "lab"
SEEDS = list(range(6))


def sample_config(rng):
    return dict(
        n_nodes=int(rng.choice([100, 200, 300])),
        p_link=float(np.exp(rng.uniform(np.log(0.03), np.log(0.3)))),
        input_weight=float(np.exp(rng.uniform(np.log(1.0), np.log(8.0)))),
        weight_init_mean=float(np.exp(rng.uniform(np.log(0.5), np.log(8.0)))),
        leak=float(rng.uniform(0.05, 0.6)),
        target_lr=float(np.exp(rng.uniform(np.log(0.001), np.log(0.1)))),
        threshold_ratio=float(rng.uniform(1.2, 4.0)),
        weight_lr=float(np.exp(rng.uniform(np.log(0.01), np.log(2.0)))),
    )


def evaluate(task):
    cid, over, seed = task
    res = dataclasses.replace(WALL_RESERVOIR_CONFIG, **over)
    h = run_wall(n_steps=3600, seed=seed, reservoir_config=res)
    late = slice(1800, None)
    flow = h.inputs.sum(axis=1)
    return dict(cid=cid, seed=seed,
                perf=1.0 - float(h.hit[late].mean()),
                hits_late=int(h.hit[late].sum()),
                flow=float(flow[late].mean()),
                flow_sd=float(flow[late].std()),
                duty=float((flow[late] > 0.05).mean()),
                absE=float(h.mean_abs_error[late].mean()),
                f=float(h.prop_spiked[late].mean()),
                r_mean=float(np.hypot(h.x - 7.5, h.y - 7.5)[late].mean()))


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ra = np.argsort(np.argsort(a)).astype(float); rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    d = np.sqrt((ra**2).sum() * (rb**2).sum())
    return float((ra*rb).sum()/d) if d else 0.0


def main():
    rng = np.random.default_rng(42)
    configs = [dict()] + [sample_config(rng) for _ in range(90)]
    tasks = [(i, c, s) for i, c in enumerate(configs) for s in SEEDS]
    with ProcessPoolExecutor(10) as pool:
        rows = list(pool.map(evaluate, tasks, chunksize=4))
    (LAB / "wall_screen.json").write_text(json.dumps(dict(configs=configs, rows=rows)))

    # aggregate per config
    n = len(configs)
    agg = {}
    for k in ("perf", "flow", "flow_sd", "duty", "absE", "f", "r_mean"):
        agg[k] = np.array([np.mean([r[k] for r in rows if r["cid"] == i]) for i in range(n)])
    alive = agg["f"] > 0.005  # exclude totally dead nets from correlations, report both
    print(f"wall screen: {n} configs x {len(SEEDS)} seeds; alive {alive.sum()}/{n}")
    print(f"paper default (id 0): perf {agg['perf'][0]:.3f} flow {agg['flow'][0]:.2f} "
          f"flow_sd {agg['flow_sd'][0]:.2f} f {agg['f'][0]:.2f}")
    print("\n── Spearman with avoidance performance (all / alive-only):")
    for k in ("flow", "flow_sd", "duty", "absE", "f", "r_mean"):
        print(f"   {k:8s} {spearman(agg[k], agg['perf']):+.3f}  /  "
              f"{spearman(agg[k][alive], agg['perf'][alive]):+.3f}")
    stab = 1.0 / (1.0 + agg["flow_sd"])
    print(f"   {'flow_stab':8s} {spearman(stab, agg['perf']):+.3f}  /  "
          f"{spearman(stab[alive], agg['perf'][alive]):+.3f}")
    # top/bottom by flow
    order = np.argsort(agg["flow"])
    print(f"\n   mean perf of 10 LOWEST-flow configs:  {agg['perf'][order[:10]].mean():.3f}")
    print(f"   mean perf of 10 HIGHEST-flow configs: {agg['perf'][order[-10:]].mean():.3f}")


if __name__ == "__main__":
    main()
