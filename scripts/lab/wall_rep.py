"""Wall-avoidance behavioral replication (H27a)."""
from __future__ import annotations
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from common import make_configs  # noqa: F401  (BLAS pins)
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from homeostasis.simulation import run_wall  # noqa: E402
from homeostasis.wall import WallConfig  # noqa: E402

LAB = Path(__file__).resolve().parents[1] / "out" / "lab"
BIN = 600


def evaluate(task):
    arm, seed = task
    kw = {}
    n = 3600
    if arm == "no-learn":
        kw["learning_enabled"] = False
    if arm == "perturb":
        kw["wall_config"] = WallConfig(perturb_at=1000)
        n = 3000
    if arm == "noise":
        kw["wall_config"] = WallConfig(sensor_noise=0.2)
    h = run_wall(n_steps=n, seed=seed, **kw)
    bins = [float(h.hit[i:i+BIN].mean()) for i in range(0, n, BIN)]
    late = h.hit[-1000:]
    r = np.hypot(h.x - 7.5, h.y - 7.5)
    return dict(arm=arm, seed=seed, bins=bins,
                late_hits=int(late.sum()), hit_total=int(h.hit.sum()),
                late_turn=float(np.abs(h.d_heading[-1000:]).mean()),
                late_turn_signed=float(h.d_heading[-1000:].mean()),
                late_r_mean=float(r[-1000:].mean()), late_r_sd=float(r[-1000:].std()),
                late_flow=float(h.inputs[-1000:].sum(axis=1).mean()),
                late_f=float(h.prop_spiked[-1000:].mean()),
                # for perturb: hits in windows around the swap
                seg_hits=[int(h.hit[a:a+500].sum()) for a in range(0, n, 500)])


def main():
    tasks = [(arm, s) for arm in ("base", "no-learn", "perturb", "noise")
             for s in range(24)]
    with ProcessPoolExecutor(10) as pool:
        rows = list(pool.map(evaluate, tasks))
    LAB.mkdir(exist_ok=True)
    (LAB / "wall_rep.json").write_text(json.dumps(rows))
    for arm in ("base", "no-learn", "perturb", "noise"):
        sel = [r for r in rows if r["arm"] == arm]
        bins = np.array([r["bins"] for r in sel]).mean(axis=0)
        late0 = np.mean([r["late_hits"] == 0 for r in sel])
        print(f"\n══ {arm} (24 seeds)")
        print("   hits/step by 600-bin: " + " ".join(f"{b:.3f}" for b in bins))
        print(f"   zero-late-hit seeds: {late0:.2f}   median late hits {np.median([r['late_hits'] for r in sel]):.0f}")
        print(f"   late |turn| {np.mean([r['late_turn'] for r in sel]):.3f} rad/step  "
              f"signed {np.mean([np.abs(r['late_turn_signed']) for r in sel]):.3f}  "
              f"radius {np.mean([r['late_r_mean'] for r in sel]):.2f}±{np.mean([r['late_r_sd'] for r in sel]):.2f}")
        print(f"   late flow {np.mean([r['late_flow'] for r in sel]):.3f}  late f {np.mean([r['late_f'] for r in sel]):.3f}")
        if arm == "perturb":
            seg = np.array([r["seg_hits"] for r in sel]).mean(axis=0)
            print("   hits per 500-window: " + " ".join(f"{v:.1f}" for v in seg) + "  (swap at 1000)")


if __name__ == "__main__":
    main()
