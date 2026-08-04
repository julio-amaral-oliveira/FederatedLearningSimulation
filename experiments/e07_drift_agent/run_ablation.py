"""Run one-at-a-time sensitivity experiments for the drift agent."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable

from experiments.e07_drift_agent.episode import (
    DEFAULT_CLIENT_SPEED_PROFILE,
    DriftEpisodeConfig,
)
from experiments.e07_drift_agent.run_matrix import (
    DEFAULT_SCENARIOS,
    _parse_scenario,
    run_matrix,
)
from experiments.shared.registry import temporary_output_path


DEFAULT_ABLATION_PRODUCTION_HORIZON_SECONDS = 800.0


@dataclass(frozen=True, slots=True)
class AblationPlan:
    """Values for the four independent policy sensitivity axes."""

    quorums: tuple[float, ...] = (0.20, 0.30, 0.50)
    retrain_rounds: tuple[int, ...] = (1, 3, 5, 10)
    windows: tuple[int, ...] = (1, 2, 5)
    dropout_T: tuple[int, ...] = (5, 10, 25, 50)


DEFAULT_SEEDS: tuple[int, ...] = (42, 43, 44, 45, 46)


def build_ablation_matrix(
    *,
    seeds: Iterable[int],
    scenarios: Iterable[tuple[str, int]],
    plan: AblationPlan = AblationPlan(),
    base_config: DriftEpisodeConfig | None = None,
) -> list[DriftEpisodeConfig]:
    """Build unique one-at-a-time arms with one official baseline."""
    base = base_config or DriftEpisodeConfig(
        retrain_rounds=1,
        production_horizon_seconds=DEFAULT_ABLATION_PRODUCTION_HORIZON_SECONDS,
    )
    arms = [
        replace(base, trigger_threshold=quorum)
        for quorum in plan.quorums
    ]
    arms.extend(replace(base, retrain_rounds=rounds) for rounds in plan.retrain_rounds)
    arms.extend(
        replace(base, trigger_window_ticks=window)
        for window in plan.windows
    )
    arms.extend(replace(base, detector_T=dropout_t) for dropout_t in plan.dropout_T)
    unique_arms = list(
        {
            (
                arm.trigger_threshold,
                arm.retrain_rounds,
                arm.trigger_window_ticks,
                arm.detector_T,
            ): arm
            for arm in arms
        }.values()
    )
    return [
        replace(arm, seed=seed, corruption=corruption, severity=severity)
        for corruption, severity in scenarios
        for seed in seeds
        for arm in unique_arms
    ]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the ablation command-line options."""
    parser = argparse.ArgumentParser(
        description="Run one-at-a-time sensitivity experiments for E07"
    )
    parser.add_argument(
        "--output-dir",
        default=str(
            temporary_output_path("e07-drift-agent", "cifar-10")
            / "uniform"
            / "ablation-v1"
        ),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument(
        "--scenario",
        nargs="+",
        default=[f"{corruption}:{severity}" for corruption, severity in DEFAULT_SCENARIOS],
    )
    parser.add_argument("--quorums", nargs="+", type=float, default=[0.20, 0.30, 0.50])
    parser.add_argument("--retrain-rounds", nargs="+", type=int, default=[1, 3, 5, 10])
    parser.add_argument("--windows", nargs="+", type=int, default=[1, 2, 5])
    parser.add_argument("--dropout-T", nargs="+", type=int, default=[5, 10, 25, 50])
    parser.add_argument(
        "--client-speed-profile",
        "--speed-profile",
        choices=("uniform", "heterogeneous"),
        default=DEFAULT_CLIENT_SPEED_PROFILE,
    )
    parser.add_argument(
        "--production-horizon-seconds",
        type=float,
        default=DEFAULT_ABLATION_PRODUCTION_HORIZON_SECONDS,
    )
    parser.add_argument("--include-controls", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Build and execute the configured one-at-a-time ablation."""
    args = parse_args(argv)
    configs = build_ablation_matrix(
        seeds=args.seeds,
        scenarios=[_parse_scenario(value) for value in args.scenario],
        plan=AblationPlan(
            quorums=tuple(args.quorums),
            retrain_rounds=tuple(args.retrain_rounds),
            windows=tuple(args.windows),
            dropout_T=tuple(args.dropout_T),
        ),
        base_config=DriftEpisodeConfig(
            retrain_rounds=1,
            output_dir=args.output_dir,
            client_speed_profile=args.client_speed_profile,
            production_horizon_seconds=args.production_horizon_seconds,
        ),
    )
    print(f"ablation arms: {len(configs)}")
    for agent_path, baseline_path in run_matrix(
        configs,
        output_dir=Path(args.output_dir),
        include_controls=args.include_controls,
    ):
        print(f"agent: {agent_path}")
        print(f"baseline: {baseline_path}")


if __name__ == "__main__":
    main()
