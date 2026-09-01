# scripts/lab — the design-space campaign

The overnight campaign of 2026-08-31/09-01: 100 preregistered hypotheses,
~28,000 runs, nine cluster batches. This directory is self-contained lab
code; the model lives in `src/homeostasis/` and is never duplicated here.

## How to read the campaign

0. **[docs/field_guide.md](../../docs/field_guide.md)** — the
   plain-language, illustrated version of everything (start here if any
   of the terms below are unfamiliar).
1. **[LEDGER.md](LEDGER.md)** — the running preregistration log
   (hypothesis → prediction → verdict, refutations and instrument bugs
   kept on the record). This is the authority for what was claimed and
   what happened, in order.
2. **[../../docs/design_space.md](../../docs/design_space.md)** — the
   synthesis (laws, mechanisms, cheat-sheet, open questions).
3. **[monday_update.md](monday_update.md)** — the pasteable summary.
4. The "Three Laws and a Ridge" artifact — built from
   [artifact_template.html](artifact_template.html) by base64-embedding
   the figures listed at its `{{FIG_*}}` placeholders from
   `scripts/out/lab/`.

Every number in the doc traces to a JSON in `scripts/out/lab/`; every
JSON traces to the `h*_*.py` / `k*_*.py` / `b*_*.py` script of the same
LEDGER entry. Scripts are deterministic (three bit-identical
cross-machine reproductions on record); rerunning one regenerates its
JSON exactly.

## The harness (`common.py`)

`run_closed_loop(task)` — one tracking run. Key task fields:

- `res` / `trk`: overrides for `ReservoirConfig` / `TrackingConfig`
  (`input_p_link` pins input wiring when sweeping `p_link`).
- `seed`, `n_steps` (default 7200; segments are 720 steps, "late" =
  segments 5+), `snap_every`.
- `arm`: `full` | `no-learn` | `lesion` | `freeze-mid[-resetT|-resetW]` |
  `shuffle-mid` | `freeze-W-only` | `freeze-T-only` | `freeze-T-mid` |
  `swap-mid` | `kill-mid` | `kill-mid-frozen` (with `kill_frac`; true
  adjacency-level node death, caches rebuilt so plasticity cannot
  regrow).
- `sensor_noise`: uniform ±σ on sensor activations, clamped ≥ 0 (own
  rng, `seed+900001`).
- `pin_output_p`: rebuild output pools at this density from a side rng
  (`seed+880008`) — REQUIRED when sweeping `p_link`, which otherwise
  also rewires the readout.
- `freeze_T_at` (+ optional `reset_T_on_freeze`): hold targets fixed
  from that step (at their current values, or reset to `target_init`).

Returns per-segment scores, duty and flow, policy bincounts, snap
trajectories (w̄, T̄, g, f), and summary stats.

`run_open_loop(task)` — scripted-drive runs (`stationary` / `slip` /
`jump` / `dark` / `sine`) with reconstruction gain and per-node law
recording; also accepts `sensor_noise`.

## Cluster lane (engaging / Slurm)

`cluster_runner.py TASKS.jsonl OUT.jsonl` runs task dicts (a `mode` key
selects closed/open/pong). Recipe: write chunk files under
`scripts/out/lab/<batch>/`, tar-pipe `scripts/lab` + the chunks to
`~/homeostasis4` on the cluster, submit an 8-chunk array on
`mit_quicktest` (14-min lane; a 6k-run batch finishes in ~4 minutes; keep
chunks under the wall-time). See the `run_*.sbatch` files on the remote
and the ship scripts referenced in the LEDGER.

## Reproduction gotchas (all learned the hard way, all in the LEDGER)

- ProcessPool workers must live in real files (macOS spawn; stdin
  scripts die with BrokenProcessPool).
- Wall-avoidance zero-collision metrics are confounded by the death
  solution — always pair with an aliveness check (late speed > 0.02).
- Hazard-triggered within-run classification selects on duration; use
  randomized/deterministic assignment.
- Measure orbits about the fitted center, not the arena center
  (eccentric-frame artifacts).
- Multi-agent cosims must update sequentially (a follower must see its
  target's same-step position; one step of staleness kills a lock).
- Replay instruments must reproduce the live lock at zero perturbation
  before their sweeps count (chirality, waveform, and wrap-jump
  closure all bit).
