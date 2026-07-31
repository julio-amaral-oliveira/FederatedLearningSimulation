"""Compatibility entrypoint for the E02 ablation runner."""

from experiments._legacy import run_legacy_module


def main(argv=None):
    return run_legacy_module("experiments.ablation_study", argv)


if __name__ == "__main__":
    main()
