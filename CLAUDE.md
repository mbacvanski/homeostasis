# homeostasis4 — working notes

Re-implementation of Falandays et al. (2024), homeostatic reservoir networks
(paper PDF in repo root). See README.md for the full picture.

## Ground rules

- **Code beats paper text.** Where the paper's methods and the authors'
  released Julia code disagree, follow the code (archived in
  `reference/original_julia/`, from https://osf.io/6hqrt/). Known
  discrepancies are tabled in README.md — extend that table when
  implementing the Pong / wall-avoidance case studies (download their
  original code from the same OSF repo first and diff it against the paper).
- The visualizer (`viz/`) must never duplicate model logic — it imports the
  `homeostasis` package. The fingerprint (Σx/ΣT/ΣW) cross-check with
  `scripts/fingerprint.py` only works if that stays true.
- Seed discipline: all randomness flows from `numpy.random.default_rng(seed)`
  in `HomeostaticReservoir.__init__`; the draw order (input adjacency →
  reservoir adjacency → weights → output adjacency) is part of the
  reproducibility contract — don't reorder it.

## Commands

- env: `uv venv --python 3.12 && uv pip install -e ".[dev,viz]"`
- tests: `.venv/bin/python -m pytest` (behavioral tests are `-m slow`;
  they use fixed seeds and are deterministic)
- experiments: `.venv/bin/python scripts/run_tracking_experiment.py`,
  `scripts/run_pong_experiment.py`
- visualizer: `.venv/bin/python -m uvicorn viz.server:app --port 8471`
  (also in `.claude/launch.json`; Pong is at `/pong`, served by
  `viz/pong_server.py` as a mounted sub-app)

## Compute budget (learned the hard way)

A Pong run is ~90x a tracking run (100k steps x N=500, cost per step ~N²):
~13 s each when the machine is quiet. Consequences:

- Never queue hundreds of them casually — 40 runs pins the mean to ±1.6%
  against the paper's SD of 0.10, which is enough for any comparison here.
- Run **one** batch at a time; two overlapping pools on 12 cores drove load
  average past 50 and got a job OOM-killed.
- Never pipe a long background job through `tail`/`head` — it buffers, so the
  log looks identical whether the job is running or dead. Use `python -u`
  and redirect to a file.

## Validation anchors (don't regress)

- Tracking: authors' published run scores within45=0.38, dir-agree=0.83, prop
  spiked=0.34 on `homeostasis.analysis.tracking_metrics`; our seeds 0–9
  bracket it. Behavioral tests in `tests/test_behavior.py` encode margins.
- Tracking learning-off ablation: agent stops moving entirely (both effectors
  saturate equally).
- Pong (paper, 500 runs x 1e5 steps, chance 0.20): hit rate 0.582 (SD 0.100);
  learning frozen 0.43; allocentric sensors 0.216; first-50 = last-50 (no
  learning curve). Margins encoded in `tests/test_pong_behavior.py`.
- `PongConfig` defaults to the fixed sensor test (`<= 2`, no 0° blind spot);
  `PongConfig.published()` restores the released strict `< 2` for replication.
  Paper comparisons must use `published()`.
