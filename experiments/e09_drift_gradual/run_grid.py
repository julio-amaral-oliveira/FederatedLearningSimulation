"""Run the E09 grid: ramp duration x scenario x seed.

Scenario: global corruption on the calibrated severities, with the mixture
fraction rising linearly from 0 to 1 over a configured number of ticks.
Retraining keeps the E07 policy: quorum at the server, global retraining,
per-client current-distribution datasets.
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

_BASE = Path(__file__).resolve().parent.parent.parent
for _path in (str(_BASE), str(_BASE / "src")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from experiments.e07_drift_agent.episode import (
    DEFAULT_PRODUCTION_HORIZON_SECONDS,
    DriftEpisodeConfig,
)
from experiments.e07_drift_agent.run_matrix import run_matrix
from experiments.shared.registry import temporary_output_path

DEFAULT_SCENARIOS: tuple[tuple[str, int], ...] = (
    ("gaussian_noise", 3),
    ("frosted_glass_blur", 4),
    ("motion_blur", 1),
    ("fog", 4),
)


def build_grid(
    *,
    seeds: tuple[int, ...] = (42, 43, 44, 45, 46),
    scenarios: tuple[tuple[str, int], ...] = DEFAULT_SCENARIOS,
    ramp_ticks: tuple[int, ...] = (5, 10, 20),
) -> list[DriftEpisodeConfig]:
    """Materialize the scenario x seed x ramp-duration grid."""
    return [
        DriftEpisodeConfig(
            corruption=corruption,
            severity=severity,
            drift_ramp_ticks=ramp,
            seed=seed,
            retrain_rounds=1,
            production_horizon_seconds=DEFAULT_PRODUCTION_HORIZON_SECONDS,
        )
        for (corruption, severity), seed, ramp in itertools.product(
            scenarios, seeds, ramp_ticks
        )
    ]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the E09 gradual-drift grid")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--ramp-ticks", nargs="+", type=int, default=[5, 10, 20])
    parser.add_argument(
        "--scenario",
        nargs="+",
        default=[f"{corruption}:{severity}" for corruption, severity in DEFAULT_SCENARIOS],
        help="one or more corruption:severity pairs",
    )
    parser.add_argument(
        "--output-dir",
        default=str(
            temporary_output_path("e09-gradual-drift", "cifar-10") / "grid-v1"
        ),
    )
    args = parser.parse_args(argv)

    from src.utils.corruptions import apply_corruption

    scenarios: list[tuple[str, int]] = []
    for value in args.scenario:
        corruption, separator, severity = value.rpartition(":")
        if not separator:
            raise SystemExit(
                f"scenario must use corruption:severity, got {value!r}"
            )
        scenarios.append((corruption, int(severity)))
    configs = build_grid(
        seeds=tuple(args.seeds),
        scenarios=tuple(scenarios),
        ramp_ticks=tuple(args.ramp_ticks),
    )
    run_matrix(configs, output_dir=args.output_dir, corruption_fn=apply_corruption)
    print(f"E09 grid done: {len(configs)} pairs -> {args.output_dir}")


if __name__ == "__main__":
    main()
