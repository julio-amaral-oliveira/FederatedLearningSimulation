"""Compatibility entrypoint for E02 ablation plots."""

from experiments._legacy import run_legacy_module


def main(argv=None):
    return run_legacy_module("experiments.plot_ablation", argv)


if __name__ == "__main__":
    main()
