"""E: autopsy the perfect pursuer — what does near3=1.0 look like?"""
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

def main():
    champ = json.loads((LAB / "h33_evolve_pursuit.json").read_text())[-1]["champion"]
    ver = json.loads((LAB / "h33b_verify.json").read_text())
    best_i = int(np.argmax([r["near3"] for r in ver]))
    seed = 1000 + best_i
    pc = PursuitConfig(eye_offsets=(0.0,), sensors_per_eye=91,
                       wheel_base=champ["wheel_base"], intensity_scale=champ["intensity_scale"])
    res_keys = ("n_nodes", "p_link", "input_weight", "weight_init_mean",
                "leak", "target_lr", "threshold_ratio", "weight_lr")
    res = ReservoirConfig(n_inputs=pc.n_sensors, **{k: champ[k] for k in res_keys})
    h = run_pursuit(n_steps=3600, seed=seed, reservoir_config=res, pursuit_config=pc)
    late = slice(1800, None)
    rel = np.hypot(h.x - h.sx, h.y - h.sy)
    bearing = np.abs(h.bearing)
    print(f"perfect pursuer seed {seed}: late dist {h.dist[late].mean():.2f} "
          f"(sd {h.dist[late].std():.2f}), |bearing| median {np.median(bearing[late]):.0f} deg, "
          f"f {h.prop_spiked[late].mean():.2f}, hits {int(h.hit.sum())}")
    # is it orbiting the stimulus? angular progression of agent AROUND stimulus
    ang = np.unwrap(np.arctan2(h.y - h.sy, h.x - h.sx))
    orbit_rate = np.rad2deg(np.diff(ang)[1800:]).mean()
    print(f"revolution rate about the stimulus: {orbit_rate:+.1f} deg/step "
          f"({'orbiting' if abs(orbit_rate) > 1 else 'not orbiting'})")
    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    t0 = 1800
    ax.plot(h.x[t0:], h.y[t0:], "-", lw=0.7, color="tab:blue", label="agent (late)")
    ax.plot(h.sx[t0:], h.sy[t0:], "-", lw=1.6, color="tab:red", label="stimulus orbit")
    ax.plot(h.x[:t0], h.y[:t0], "-", lw=0.4, color="lightgray", label="agent (early)")
    ax.set_xlim(0, 15); ax.set_ylim(0, 15); ax.set_aspect("equal")
    ax.legend(fontsize=8); ax.set_title(f"The perfect pursuer (seed {seed}): late dist "
                                        f"{h.dist[late].mean():.2f}")
    fig.tight_layout(); fig.savefig(LAB / "fig_perfect_pursuer.png", dpi=130)
    print("wrote fig_perfect_pursuer.png")

if __name__ == "__main__":
    main()
