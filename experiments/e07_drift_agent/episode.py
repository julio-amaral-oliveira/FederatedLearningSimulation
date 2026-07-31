"""E07 single-episode compatibility entrypoint."""

from experiments.smoke_drift import *  # noqa: F401,F403
from experiments.smoke_drift import main as _legacy_main


def main(argv=None):
    return _legacy_main(argv)


if __name__ == "__main__":
    main()
