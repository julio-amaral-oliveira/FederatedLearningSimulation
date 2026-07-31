"""E07 severity-calibration entrypoint."""

from experiments.severity_calibration import (
    CORRUPTIONS,
    SEVERITIES,
    calibrate_clean_checkpoint,
    calibrate_severities,
    main as _legacy_main,
    save_calibration,
)

__all__ = [
    "CORRUPTIONS",
    "SEVERITIES",
    "calibrate_clean_checkpoint",
    "calibrate_severities",
    "main",
    "save_calibration",
]


def main(argv=None):
    return _legacy_main(argv)


if __name__ == "__main__":
    main()
