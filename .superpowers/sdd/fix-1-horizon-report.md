# Fix 1 — production horizon report

## Implemented behavior

- Both drift CLIs expose `--production-horizon-seconds`, with a stable default
  of `400.0`; the matrix `main` passes that value through its base
  `DriftEpisodeConfig` to every generated run.
- Under a fixed horizon, the episode controller computes
  `config.retrain_rounds * server.timeout` immediately before retraining. If
  the remaining production budget is smaller, it raises `RuntimeError` before
  executing a retraining round, so no overshot episode is returned or
  persisted as successful.
- A fitting conservative budget retains exact completion at the configured
  production endpoint for both agent and baseline.
- The handoff commands now state and pass the 400-second fixed horizon, and
  explain the insufficient-budget failure policy.

## TDD evidence

1. Added `test_fixed_horizon_rejects_retraining_before_any_round_when_budget_cannot_fit`.
   Its first run failed as intended: `AssertionError: RuntimeError not raised`.
   The fixture has a first-production-tick trigger, five 10-second conservative
   round bounds, only 10 seconds remaining, and asserts no retraining round
   occurred.
2. Added `test_matrix_cli_propagates_its_default_production_horizon_to_every_config`.
   Its first run failed as intended: `[None, None] != [400.0, 400.0]`.

## Verification

- Focused: `.venv/bin/python -m unittest tests.test_drift_episode tests.test_drift_runner -v` — 23 passed.
- Full: `.venv/bin/python -m unittest discover -s tests -v` — 84 passed.
- Both CLI `--help` outputs list `--production-horizon-seconds`.

## Concern

No production concern remains in this isolated fix. The fresh full-suite run
uses `MPLCONFIGDIR=/private/tmp/fl-mpl` and emits only the expected CPU-only
`UDDDetector` warning when its test intentionally requests CUDA.
