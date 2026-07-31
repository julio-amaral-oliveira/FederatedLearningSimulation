"""Compatibility entrypoint for E04 result comparisons."""

from experiments._legacy import run_legacy_module


def main(argv=None):
    return run_legacy_module("experiments.compare_results", argv)


if __name__ == "__main__":
    main()
