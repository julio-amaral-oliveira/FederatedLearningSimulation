# Fix 3 — atomic persistence report

## Implemented behavior

- `experiments.result_io.atomic_write_json` writes a uniquely named temporary
  file beside its destination, serializes the complete payload, flushes and
  `fsync`s it, then publishes it with `os.replace`. Failures before publication
  remove the temporary file and preserve an existing destination.
- `save_validated_pair` validates before publication, stages and `fsync`s the
  canonical `agent.json` and `baseline.json`, invalidates an older manifest,
  publishes both payloads, and publishes `pair-manifest.json` last. The
  manifest records schema version 1, a generation ID, both canonical filenames,
  and SHA-256 digests of the exact staged bytes.
- `load_persisted_pair` requires a valid manifest, verifies both byte digests
  before JSON decoding, constructs validated drift results, and validates the
  pair. Missing, partial, or tampered generations are rejected.
- Standalone drift results, the matrix `summary.json`, calibration output, the
  matrix runner, and the single-run CLI now share the centralized persistence
  implementation. Individual schema-v2 loading remains unchanged.

## TDD evidence

1. The first focused run failed with `ModuleNotFoundError:
   experiments.result_io`, establishing that the persistence seam did not
   exist.
2. The standalone writer regression failed with `AssertionError: OSError not
   raised`; after integration, injected pre-publication failures preserve the
   prior destination and remove temporary files.
3. The pair failure regression first failed with `NotImplementedError`; after
   implementation, a failure while publishing `baseline.json` leaves no
   completion manifest and `load_persisted_pair` rejects the directory.
4. The successful-pair regression first failed with `NotImplementedError`;
   after implementation, it proves the manifest is the last `os.replace`,
   independently recomputes both SHA-256 digests, loads the valid pair, and
   detects byte tampering.
5. Caller-integration regressions failed before their production changes:
   matrix and single-run output lacked `pair-manifest.json`, while direct
   summary and calibration writes did not raise the injected atomic-write
   failures. Each passed after routing through the shared implementation.

## Verification

- Focused:
  `MPLCONFIGDIR=/tmp/mplconfig XDG_CACHE_HOME=/tmp/cache .venv/bin/python -m unittest tests.test_drift_results tests.test_drift_runner tests.test_severity_calibration -v`
  — 43 passed.
- Full:
  `MPLCONFIGDIR=/tmp/mplconfig XDG_CACHE_HOME=/tmp/cache .venv/bin/python -m unittest discover -s tests -v`
  — 105 passed.
- `git diff --check` and Python bytecode compilation completed successfully.

## Concern

No production concern remains in this isolated fix. A crash after either
payload publication can leave mixed canonical files by design, but the absent
completion manifest makes that directory invalid until a later complete
generation is published.
