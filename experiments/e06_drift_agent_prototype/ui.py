"""Compatibility entrypoint for the legacy E06 drift-agent UI."""

from experiments._legacy import run_legacy_module


def main(argv=None):
    return run_legacy_module("experiments.smoke_drift_ui", argv)


if __name__ == "__main__":
    main()
