"""Compatibility entrypoint for the E05 temporal drift UI."""

from experiments._legacy import run_legacy_module


def main(argv=None):
    return run_legacy_module("experiments.drift_ui", argv)


if __name__ == "__main__":
    main()
