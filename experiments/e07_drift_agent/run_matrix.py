"""E07 matrix-runner entrypoint."""

from experiments.run_smoke_drift import (
    DEFAULT_SCENARIOS,
    _parse_scenario,
    aggregate_runs,
    build_run_matrix,
    main as _legacy_main,
    parse_args,
    run_matrix,
)

__all__ = [
    "DEFAULT_SCENARIOS",
    "_parse_scenario",
    "aggregate_runs",
    "build_run_matrix",
    "main",
    "parse_args",
    "run_matrix",
]


def main(argv=None):
    return _legacy_main(argv)


if __name__ == "__main__":
    main()
