"""Compatibility entrypoint for the original single-episode prototype."""

from experiments._legacy import run_legacy_module


def main(argv=None):
    return run_legacy_module("experiments.smoke_drift", argv)


if __name__ == "__main__":
    main()
