"""Determinism cross-check for the /lab live sessions (viz.lab_server).

Drives each live session class headlessly (the exact objects the WebSocket
endpoints own) for a few hundred steps and compares the exposed numbers
against the corresponding BATCH code path — the existing /lab/api/* runner
functions and scripts/lab/common.py run_closed_loop — built from the same
parameters. Every comparison must match exactly (both sides round with the
same conventions); any mismatch means the live world is not the batch world
and is a bug.

Run:  .venv/bin/python scripts/check_lab_live.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from viz import lab_server as L  # noqa: E402

FAILURES = []


def check(label: str, live, batch) -> None:
    ok = live == batch
    print(f"  {'ok ' if ok else 'FAIL'} {label}: live={live!r} batch={batch!r}")
    if not ok:
        FAILURES.append(label)


def main() -> None:
    # -- wall: WallLive vs run_wall_variant (base, seed 0, wlr 1.0, tlr .01) --
    print("wall (base, seed 0, 500 steps)")
    s = L.WallLive()
    series = s.advance(500)
    b = L.run_wall_variant("base", 0, 500, 1.0, 0.01)
    e498 = series[498]
    i = 498 // L.WALL_SUBSAMPLE
    check("x @ t=498", e498["x"], b["trace"]["x"][i])
    check("y @ t=498", e498["y"], b["trace"]["y"][i])
    check("prop @ t=498", e498["prop"], b["trace"]["prop"][i])
    check("hits total @ 500", s.sim.env.hits, b["summary"]["hits_total"])
    # now-heading corresponds to the last step (499), not in the subsample;
    # compare against a fresh batch run's full-resolution history instead
    from homeostasis.simulation import run_wall
    h = run_wall(n_steps=500, seed=0)
    check("heading @ t=499 (rad, full-res run_wall)",
          round(float(s.sim.env.heading), 4), round(float(h.heading[499]), 4))

    # -- pursuit: PursuitLive vs run_pursuit_variant (orbit, champ seed) -----
    print("pursuit (h34-champion, orbit, speed 0.15, seed 66777, 500 steps)")
    s = L.PursuitLive()
    series = s.advance(500)
    b = L.run_pursuit_variant("h34-champion", "orbit", 0.15, L.H34_CHAMP_SEED, 500)
    e498 = series[498]
    i = 498 // L.PURSUIT_SUBSAMPLE
    for key in ("x", "y", "sx", "sy", "dist", "prop"):
        check(f"{key} @ t=498", e498[key], b["trace"][key][i])

    # ballistic crossing stats: live incremental counter vs h55_intercept's
    # batch slicing (the live counter excludes the trailing partial crossing;
    # add it back from the session's internal accumulator before comparing)
    print("pursuit ballistic (h34-champion, speed 0.15, seed 66777, 3600 steps)")
    s = L.PursuitLive()
    s.reset({"motion": "ballistic"})
    s.advance(3600)
    b = L.run_pursuit_variant("h34-champion", "ballistic", 0.15,
                              L.H34_CHAMP_SEED, 3600)
    live_n = s._n_crossings + (1 if s._cross_len >= 20 else 0)
    live_c = s._catches + (1 if s._cross_len >= 20 and s._cross_min < L.CATCH_R
                           else 0)
    check("n_crossings", live_n, b["summary"]["n_crossings"])
    check("catch_rate", round(live_c / max(live_n, 1), 4),
          b["summary"]["catch_rate"])

    # -- traj: TrajLive vs run_traj (ridge25, seed 0) ------------------------
    print("traj (ridge25, seed 0, 496 steps)")
    s = L.TrajLive()
    series = s.advance(496)
    b = L.run_traj("ridge25", 0, 500, None, 0.0)
    now = s._now()
    i = 496 // L.TRAJ_SUBSAMPLE
    check("err @ t=496", now["err"], b["trace"]["err"][i])
    check("heading @ t=496 (mod 360)", now["heading"],
          round(b["trace"]["heading"][i] % 360.0, 2))
    j = 492 // L.TRAJ_SUBSAMPLE
    check("prop @ t=492", series[492]["prop"], b["trace"]["prop"][j])

    # traj live kill == the kill-mid arm (common.py run_closed_loop):
    # kill 30% via the surgery command at t=720 of a 1440-step run
    print("traj live-kill vs common.py kill-mid (ridge25, seed 0, 1440 steps)")
    s = L.TrajLive()
    s.advance(720)
    s.handle({"cmd": "surgery", "op": "kill", "frac": 0.3})
    s.advance(720)
    r = L.lab_common.run_closed_loop(dict(
        res={"leak": 0.25, "weight_lr": 0.1}, seed=0, n_steps=1440,
        arm="kill-mid", kill_frac=0.3))
    check("seg_scores", s.seg_scores, r["seg_scores"])

    # -- repair: RepairLive vs the two H53 arms ------------------------------
    print("repair (seed 0, kill 30% at t=720, 1440 steps)")
    s = L.RepairLive()
    series = s.advance(720)
    s.handle({"cmd": "kill", "frac": 0.3})
    series += s.advance(720)
    arms = {}
    for key, arm in (("a", "kill-mid"), ("b", "kill-mid-frozen")):
        arms[key] = L.lab_common.run_closed_loop(dict(
            res=dict(L.REPAIR_RES), seed=0, n_steps=1440, arm=arm,
            kill_frac=0.3))
    check("learning twin seg_scores", s.twins[0]["seg"], arms["a"]["seg_scores"])
    check("frozen twin seg_scores", s.twins[1]["seg"], arms["b"]["seg_scores"])
    check("learning twin w-bar @ 1440", series[-1]["wa"],
          arms["a"]["snaps"]["w_mean"][-1])
    check("frozen twin w-bar @ 1440", series[-1]["wb"],
          arms["b"]["snaps"]["w_mean"][-1])
    fa = round(float(np.mean([e["fa"] for e in series])), 4)
    fb = round(float(np.mean([e["fb"] for e in series])), 4)
    check("learning twin mean f", fa, round(arms["a"]["prop_spiked"], 4))
    check("frozen twin mean f", fb, round(arms["b"]["prop_spiked"], 4))

    # -- ecology: exclusive 1-link vs run_ecology (h48c cosim) ---------------
    print("ecology (exclusive, 1 link, saved seed, 298 steps)")
    s = L.EcologyLive()
    s.advance(298)
    b = L.run_ecology(L.ECOLOGY_CHAIN[0][1], 300)
    now = s._now()
    i = 297 // L.ECOLOGY_SUBSAMPLE
    check("follower x @ t=297", now["followers"][0]["x"], b["trace"]["bx"][i])
    check("follower y @ t=297", now["followers"][0]["y"], b["trace"]["by"][i])
    check("pacemaker x @ t=297", now["ax"], b["trace"]["ax"][i])
    check("pacemaker y @ t=297", now["ay"], b["trace"]["ay"][i])

    print()
    if FAILURES:
        print(f"FAILED: {len(FAILURES)} mismatches: {FAILURES}")
        sys.exit(1)
    print("all live-vs-batch determinism checks passed")


if __name__ == "__main__":
    main()
