"""Compatibility entrypoint for E05 temporal drift runs."""

from experiments._legacy import run_legacy_module


def main(argv=None):
    return run_legacy_module("experiments.temporal_drift", argv)


if __name__ == "__main__":
    main()
