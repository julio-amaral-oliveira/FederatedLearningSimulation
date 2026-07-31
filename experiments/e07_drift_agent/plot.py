"""E07 scenario-plotting entrypoint."""

from experiments.plot_smoke_drift import (
    build_figure,
    load_scenario,
    main as _legacy_main,
    plot_scenario,
)

__all__ = ["build_figure", "load_scenario", "main", "plot_scenario"]


def main(argv=None):
    return _legacy_main(argv)


if __name__ == "__main__":
    main()
