"""Compatibility entrypoint for the E04 comparison UI."""

from experiments._legacy import run_legacy_module


def main(argv=None):
    return run_legacy_module("experiments.comparison_ui", argv)


if __name__ == "__main__":
    main()
