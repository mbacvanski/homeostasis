"""Generate cluster batch 1: fine ridge grid + A1 replication + N-line.

Writes chunk files scripts/out/lab/cluster1/chunk_###.jsonl (~120 tasks each)
plus a manifest. Held-out checkerboard cells are FLAGGED (not withheld from
running — withheld from FITTING; the flag travels with each row).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

LAB = Path(__file__).resolve().parents[1] / "out" / "lab"
DEST = LAB / "cluster1"
SEEDS = list(range(48))

LEAKS = [0.03, 0.05, 0.08, 0.12, 0.18, 0.25, 0.35, 0.5, 0.7, 0.9]
WLRS = [0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.2, 0.3, 0.5, 1.0]


def main():
    tasks = []
    # R1: fine ridge grid, tlr=0.01
    for i, leak in enumerate(LEAKS):
        for j, wlr in enumerate(WLRS):
            held = (i + j) % 4 == 0  # 25% checkerboard
            for s in SEEDS:
                tasks.append(dict(mode="closed", res={"leak": leak, "weight_lr": wlr},
                                  trk={}, seed=s, arm="full", snap_every=2400,
                                  _tag="R1", _leak=leak, _wlr=wlr, _held=held))
    # R2: A1 plane replication at 48 seeds
    for wlr in (0.0, 0.01, 0.03, 0.1, 0.3, 1.0):
        for tlr in (0.001, 0.01, 0.1):
            for s in SEEDS:
                tasks.append(dict(mode="closed",
                                  res={"weight_lr": wlr, "target_lr": tlr},
                                  trk={}, seed=s, arm="full", snap_every=2400,
                                  _tag="R2", _wlr=wlr, _tlr=tlr))
    # R3: N-line at leak=0.25 x ridge-bracketing wlr, 24 seeds
    for n in (50, 100, 200, 400, 800):
        for wlr in (0.05, 0.1, 0.2):
            for s in SEEDS[:24]:
                tasks.append(dict(mode="closed",
                                  res={"n_nodes": n, "weight_lr": wlr},
                                  trk={}, seed=s, arm="full", snap_every=2400,
                                  _tag="R3", _n=n, _wlr=wlr))

    DEST.mkdir(parents=True, exist_ok=True)
    for old in DEST.glob("chunk_*.jsonl"):
        old.unlink()
    rng = np.random.default_rng(0)
    order = rng.permutation(len(tasks))  # mix cheap/expensive across chunks
    chunk_size = 120
    n_chunks = 0
    for c, start in enumerate(range(0, len(tasks), chunk_size)):
        idx = order[start:start + chunk_size]
        with (DEST / f"chunk_{c:03d}.jsonl").open("w") as f:
            for k in idx:
                f.write(json.dumps(tasks[int(k)]) + "\n")
        n_chunks = c + 1
    (DEST / "manifest.json").write_text(json.dumps(
        dict(n_tasks=len(tasks), n_chunks=n_chunks, chunk_size=chunk_size,
             leaks=LEAKS, wlrs=WLRS)))
    print(f"{len(tasks)} tasks in {n_chunks} chunks -> {DEST}")


if __name__ == "__main__":
    main()
