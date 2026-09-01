"""H46: two mutually-tracking agents."""
from __future__ import annotations
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from common import make_configs  # noqa: F401
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from homeostasis.reservoir import HomeostaticReservoir, ReservoirConfig  # noqa: E402
from homeostasis.tracking import TrackingConfig  # noqa: E402

LAB = Path(__file__).resolve().parents[1] / "out" / "lab"


def sense(tcfg, offs, heading_deg, stim_deg):
    theta = np.abs((stim_deg - (heading_deg + offs) + 180.0) % 360.0 - 180.0)
    acts = np.exp(-(theta ** 2) / tcfg.tuning_width)
    acts[theta <= tcfg.plateau_width] = 1.0
    return acts


def run_pair(task):
    wlr, seed = task
    tcfg = TrackingConfig()
    offs = tcfg.sensor_offsets
    if wlr == "mixed":
        ra = ReservoirConfig(n_inputs=tcfg.n_sensors, weight_lr=1.0, leak=0.25)
        rb = ReservoirConfig(n_inputs=tcfg.n_sensors, weight_lr=0.1, leak=0.25)
    else:
        ra = rb = ReservoirConfig(n_inputs=tcfg.n_sensors, weight_lr=wlr, leak=0.25)
    a = HomeostaticReservoir(ra, seed=seed)
    b = HomeostaticReservoir(rb, seed=seed + 50000)
    hA, hB = 90.0, 150.0  # gap 60: mutually in view
    n = 7200
    gap = np.empty(n); fA = np.empty(n); fB = np.empty(n)
    rotA = np.empty(n)
    for i in range(n):
        gap[i] = abs((hA - hB + 180.0) % 360.0 - 180.0)
        sA = a.step(sense(tcfg, offs, hA, hB))
        sB = b.step(sense(tcfg, offs, hB, hA))
        dA = tcfg.gain * (float(sA.outputs[0]) - float(sA.outputs[1]))
        dB = tcfg.gain * (float(sB.outputs[0]) - float(sB.outputs[1]))
        hA = (hA + dA) % 360.0
        hB = (hB + dB) % 360.0
        fA[i] = sA.prop_spiked; fB[i] = sB.prop_spiked
        rotA[i] = dA
    late = slice(3600, None)
    fB_only = float(fB[late].mean())
    # does B follow A? correlate signed motion
    return dict(wlr=str(wlr), seed=seed, fB=fB_only,
                gap_late=float(gap[late].mean()), gap_final=float(gap[-500:].mean()),
                aligned=float((gap[late] <= 45).mean()),
                f_late=float((fA[late].mean() + fB[late].mean()) / 2),
                rot_late=float(np.abs(rotA[late]).mean()),
                gap_traj=[round(float(v), 1) for v in gap[::72]])


def main():
    tasks = [(wlr, 100 + s) for wlr in (0.1, 1.0, "mixed") for s in range(8)]
    with ProcessPoolExecutor(10) as pool:
        rows = list(pool.map(run_pair, tasks, chunksize=1))
    (LAB / "h46_mutual.json").write_text(json.dumps(rows))
    for wlr in ("0.1", "1.0", "mixed"):
        sel = [r for r in rows if r["wlr"] == wlr]
        print(f"wlr={wlr}: gap_late {np.mean([r['gap_late'] for r in sel]):5.1f}°  "
              f"aligned {np.mean([r['aligned'] for r in sel]):.2f}  "
              f"f_pair {np.mean([r['f_late'] for r in sel]):.3f} (B alone {np.mean([r['fB'] for r in sel]):.3f})  "
              f"|rot| {np.mean([r['rot_late'] for r in sel]):.3f}°/step")
    # solo reference: same nets with a STATIC stimulus at fixed angle
    print("(solo static-stimulus reference f at wlr 0.1/1.0 ~ 0.0-0.03 / 0.39 from K1)")


if __name__ == "__main__":
    main()
