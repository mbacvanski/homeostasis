"""Module-level Pong evaluator for cluster/pool use (spawn-safe)."""
from __future__ import annotations
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
import dataclasses
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from homeostasis.simulation import PONG_RESERVOIR_CONFIG, run_pong  # noqa: E402
from homeostasis.pong import PongConfig  # noqa: E402


def eval_pong(task: dict) -> dict:
    res = dataclasses.replace(PONG_RESERVOIR_CONFIG, **task.get("res", {}))
    h = run_pong(n_steps=int(task.get("n_steps", 100_000)), seed=task["seed"],
                 reservoir_config=res, pong_config=PongConfig.published(),
                 record=False)
    hits = h.hits.tolist()
    k = max(1, len(hits) // 2)
    out = dict(seed=task["seed"], hit_rate=h.hit_rate,
               n_opp=int(h.n_opportunities),
               first_half=sum(hits[:k]) / k,
               second_half=sum(hits[k:]) / max(1, len(hits) - k))
    for kk, v in task.items():
        if kk.startswith("_"):
            out[kk] = v
    return out
