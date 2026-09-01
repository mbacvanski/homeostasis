"""H38b: what do the ellipse and shuttle champions actually do?"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import make_configs  # noqa: F401
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from homeostasis.reservoir import ReservoirConfig  # noqa: E402
from homeostasis.simulation import run_pursuit  # noqa: E402
from homeostasis.pursuit import PursuitConfig  # noqa: E402

LAB = Path(__file__).resolve().parents[1] / "out" / "lab"
RES_KEYS = ("n_nodes", "p_link", "input_weight", "weight_init_mean",
            "leak", "target_lr", "threshold_ratio", "weight_lr")

def main():
    data = json.loads((LAB / "h38_manifold.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 6.2))
    for ax, motion in zip(axes, ("ellipse", "shuttle")):
        best = max(data[motion], key=lambda l: l["best_near"])
        g, seed = best["champion"], best["champ_seed"]
        kw = dict(stimulus_motion=motion) if motion == "ellipse" else \
             dict(stimulus_motion="wander", wander_sigma=0.0)
        pc = PursuitConfig(eye_offsets=(0.0,), sensors_per_eye=91,
                           wheel_base=g["wheel_base"],
                           intensity_scale=g["intensity_scale"], **kw)
        res = ReservoirConfig(n_inputs=pc.n_sensors, **{k: g[k] for k in RES_KEYS})
        h = run_pursuit(n_steps=3600, seed=seed, reservoir_config=res, pursuit_config=pc)
        late = slice(1800, None)
        speed = float(np.hypot(np.diff(h.x), np.diff(h.y))[1800:].mean())
        span = float(np.hypot(h.x[late].std(), h.y[late].std()))
        print(f"{motion}: near3 {(h.dist[late] < 3).mean():.2f} dist {h.dist[late].mean():.2f} "
              f"agent speed {speed:.3f} positional spread {span:.2f} "
              f"({'FOLLOWER' if speed > 0.05 and span > 1 else 'TOLL-BOOTH/PARKED'})")
        ax.plot(h.sx[late], h.sy[late], "-", lw=1.6, color="tab:red", label="stimulus (late)")
        ax.plot(h.x[late], h.y[late], "-", lw=0.8, color="tab:blue", label="agent (late)")
        ax.set_xlim(0, 15); ax.set_ylim(0, 15); ax.set_aspect("equal")
        ax.set_title(f"{motion} champion: near3 {(h.dist[late]<3).mean():.2f}")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(LAB / "fig_h38_champions.png", dpi=130)
    print("wrote fig_h38_champions.png")

if __name__ == "__main__":
    main()
