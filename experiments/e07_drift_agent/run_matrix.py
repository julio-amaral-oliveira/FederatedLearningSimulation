"""Run reproducible seed/scenario matrices for synchronous drift episodes."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Callable, Iterable, Sequence

from experiments.e07_drift_agent.episode import (
    CLIENT_SPEED_PROFILES,
    DEFAULT_CLIENT_SPEED_PROFILE,
    DEFAULT_PRODUCTION_HORIZON_SECONDS,
    DriftEpisodeConfig,
    run_drift_comparison,
)
from experiments.shared.drift_controls import OracleMonitor, identity_corruption
from experiments.shared.registry import temporary_output_path
from experiments.shared.result_io import atomic_write_json, save_validated_pair


DEFAULT_SCENARIOS: tuple[tuple[str, int], ...] = (
    ("gaussian_noise", 5),
    ("frosted_glass_blur", 1),
    ("motion_blur", 1),
    ("fog", 1),
)


def build_run_matrix(
    *,
    seeds: Iterable[int],
    scenarios: Iterable[tuple[str, int]],
    quorums: Iterable[float] | None = None,
    retrain_rounds: Iterable[int] | None = None,
    window_ticks: Iterable[int] | None = None,
    base_config: DriftEpisodeConfig | None = None,
) -> list[DriftEpisodeConfig]:
    """Materialize the requested Cartesian experiment product.

    Sensitivity dimensions stay at the base configuration unless explicitly
    provided.  This prevents an ordinary multi-seed run from silently
    multiplying expensive training work.
    """
    base = base_config or DriftEpisodeConfig()
    requested_seeds = tuple(seeds)
    requested_scenarios = tuple(scenarios)
    requested_quorums = tuple(quorums) if quorums is not None else (base.trigger_threshold,)
    requested_rounds = tuple(retrain_rounds) if retrain_rounds is not None else (base.retrain_rounds,)
    requested_windows = (
        tuple(window_ticks) if window_ticks is not None else (base.trigger_window_ticks,)
    )

    return [
        replace(
            base,
            seed=seed,
            corruption=corruption,
            severity=severity,
            trigger_threshold=quorum,
            retrain_rounds=rounds,
            trigger_window_ticks=window,
        )
        for corruption, severity in requested_scenarios
        for seed in requested_seeds
        for quorum in requested_quorums
        for rounds in requested_rounds
        for window in requested_windows
    ]


def aggregate_runs(results: list[dict]) -> dict:
    """Aggregate each finite numeric metric across schema-v3 result payloads."""
    values_by_metric: dict[str, list[float]] = defaultdict(list)
    for result in results:
        for metric, value in result.get("metrics", {}).items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            numeric_value = float(value)
            if math.isfinite(numeric_value):
                values_by_metric[metric].append(numeric_value)

    return {
        "metrics": {
            metric: {
                "count": len(values),
                "mean": statistics.fmean(values),
                "std": statistics.pstdev(values),
                "min": min(values),
                "max": max(values),
            }
            for metric, values in sorted(values_by_metric.items())
        }
    }


def _summary_dimensions(
    config: DriftEpisodeConfig,
    *,
    population: str,
    trigger_policy: str,
    retrain_skipped: bool = False,
) -> dict:
    """Return the explicit run descriptor used to partition matrix summaries."""
    return {
        "population": population,
        "corruption": config.corruption,
        "severity": config.severity,
        "trigger_policy": trigger_policy,
        "quorum": config.trigger_threshold,
        "retrain_rounds": config.retrain_rounds,
        "window_ticks": config.trigger_window_ticks,
        "dropout_T": config.detector_T,
        "drifted_client_ids": (
            list(config.drifted_client_ids)
            if config.drifted_client_ids is not None
            else None
        ),
        "drift_onset_ticks": config.drift_onset_ticks,
        "drift_ramp_ticks": config.drift_ramp_ticks,
        "client_speed_profile": config.client_speed_profile,
        "production_horizon_seconds": config.production_horizon_seconds,
        "retrain_skipped": retrain_skipped,
    }


def _summary_group_key(dimensions: dict) -> str:
    """Encode the seed-independent descriptor as a stable JSON object key."""
    return json.dumps(dimensions, sort_keys=True, separators=(",", ":"))


def _summary_population(config: DriftEpisodeConfig, *, oracle_control: bool) -> str:
    if oracle_control:
        return "oracle_control"
    if config.corruption == "identity":
        return "identity_control"
    return "visual"


def _summarize_matrix_runs(
    runs: list[tuple[DriftEpisodeConfig, str, str, dict[str, dict]]],
) -> dict:
    """Partition arm aggregates by their explicit matrix-run descriptors."""
    grouped: dict[str, dict] = {}
    for config, population, trigger_policy, results in runs:
        dimensions = _summary_dimensions(
            config,
            population=population,
            trigger_policy=trigger_policy,
            retrain_skipped=bool(
                results["agent"]["metrics"].get("retrain_skipped_budget")
            ),
        )
        key = _summary_group_key(dimensions)
        group = grouped.setdefault(
            key,
            {
                "dimensions": dimensions,
                "seeds": set(),
                "agent_results": [],
                "baseline_results": [],
            },
        )
        group["seeds"].add(config.seed)
        group["agent_results"].append(results["agent"])
        group["baseline_results"].append(results["baseline"])

    return {
        "schema_version": 1,
        "groups": {
            key: {
                "dimensions": group["dimensions"],
                "seeds": sorted(group["seeds"]),
                "run_count": len(group["agent_results"]),
                "agent": aggregate_runs(group["agent_results"])["metrics"],
                "baseline": aggregate_runs(group["baseline_results"])["metrics"],
            }
            for key, group in sorted(grouped.items())
        },
    }


def _result_directory(
    output_dir: Path,
    config: DriftEpisodeConfig,
    duplicates: Counter[tuple[str, int, int]],
) -> Path:
    """Return a unique location for every scenario, seed, and sensitivity arm.

    Duplicate scenario/seed arms are disambiguated by their sensitivity
    dimensions plus the drift schedule, so staggered schedules never
    overwrite one another on disk.
    """
    path = output_dir / f"{config.corruption}_sev{config.severity}" / f"seed_{config.seed}"
    if duplicates[(config.corruption, config.severity, config.seed)] > 1:
        disambiguators = (
            f"quorum_{config.trigger_threshold:g}"
            f"_retrain_rounds_{config.retrain_rounds}"
            f"_window_{config.trigger_window_ticks}"
            f"_dropout_T_{config.detector_T}"
        )
        if config.drift_onset_ticks is not None:
            schedule = "_".join(
                f"{client}-{onset}"
                for client, onset in sorted(config.drift_onset_ticks.items())
            )
            disambiguators += f"_schedule_{schedule}"
        if config.drift_ramp_ticks is not None:
            disambiguators += f"_ramp_{config.drift_ramp_ticks}"
        if config.drifted_client_ids is not None:
            drifted = "-".join(str(client) for client in sorted(config.drifted_client_ids))
            disambiguators += f"_drifted_{drifted}"
        return path / disambiguators
    return path


def _write_pair(results: dict[str, dict], output_path: Path) -> tuple[Path, Path]:
    return save_validated_pair(
        output_path,
        results["agent"],
        results["baseline"],
    )


def run_matrix(
    configs: Sequence[DriftEpisodeConfig],
    *,
    output_dir: str | Path,
    runner: Callable[..., dict[str, dict]] = run_drift_comparison,
    corruption_fn: Callable | None = None,
    include_controls: bool = False,
) -> list[tuple[Path, Path]]:
    """Run and persist every paired episode without subprocess or cwd coupling."""
    if corruption_fn is None:
        from src.utils.corruptions import apply_corruption

        corruption_fn = apply_corruption

    root = Path(output_dir).expanduser()
    all_configs = list(configs)
    controls: set[tuple[int, float, int, int, int]] = set()
    if include_controls:
        for config in configs:
            control_key = (
                config.seed,
                config.trigger_threshold,
                config.retrain_rounds,
                config.trigger_window_ticks,
                config.detector_T,
            )
            if control_key not in controls:
                controls.add(control_key)
                all_configs.append(
                    replace(config, corruption="identity", severity=0)
                )

    duplicates = Counter((c.corruption, c.severity, c.seed) for c in all_configs)
    output_paths: list[tuple[Path, Path]] = []
    summarized_runs: list[tuple[DriftEpisodeConfig, str, str, dict[str, dict]]] = []
    original_count = len(configs)
    for index, config in enumerate(all_configs):
        is_control = index >= original_count
        results = runner(
            config,
            corruption_fn=identity_corruption if is_control else corruption_fn,
            **({"monitor_factory": lambda _server, _baseline: OracleMonitor()} if is_control else {}),
        )
        output_paths.append(_write_pair(results, _result_directory(root, config, duplicates)))
        oracle_control = is_control
        summarized_runs.append(
            (
                config,
                _summary_population(config, oracle_control=oracle_control),
                "oracle" if oracle_control else "detector",
                results,
            )
        )

    summary = _summarize_matrix_runs(summarized_runs)
    atomic_write_json(root / "summary.json", summary)
    return output_paths


def _parse_scenario(value: str) -> tuple[str, int]:
    corruption, separator, severity = value.rpartition(":")
    if not separator or not corruption:
        raise argparse.ArgumentTypeError("scenarios must use corruption:severity")
    try:
        return corruption, int(severity)
    except ValueError as error:
        raise argparse.ArgumentTypeError("scenario severity must be an integer") from error


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a reproducible drift experiment matrix")
    parser.add_argument("--dataset", default="cifar10")
    parser.add_argument(
        "--output-dir",
        default=str(
            temporary_output_path("e07-drift-agent", "cifar-10")
            / "uniform"
            / "matrix-v1"
        ),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument(
        "--scenario",
        nargs="+",
        default=[f"{corruption}:{severity}" for corruption, severity in DEFAULT_SCENARIOS],
        help="one or more corruption:severity pairs",
    )
    parser.add_argument("--quorums", nargs="+", type=float)
    parser.add_argument("--retrain-rounds", nargs="+", type=int)
    parser.add_argument("--window-ticks", nargs="+", type=int)
    parser.add_argument(
        "--client-speed-profile",
        "--speed-profile",
        choices=CLIENT_SPEED_PROFILES,
        default=DEFAULT_CLIENT_SPEED_PROFILE,
    )
    parser.add_argument(
        "--production-horizon-seconds",
        type=float,
        default=DEFAULT_PRODUCTION_HORIZON_SECONDS,
    )
    parser.add_argument("--include-controls", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    configs = build_run_matrix(
        seeds=args.seeds,
        scenarios=[_parse_scenario(value) for value in args.scenario],
        quorums=args.quorums,
        retrain_rounds=args.retrain_rounds,
        window_ticks=args.window_ticks,
        base_config=DriftEpisodeConfig(
            dataset=args.dataset,
            output_dir=args.output_dir,
            client_speed_profile=args.client_speed_profile,
            production_horizon_seconds=args.production_horizon_seconds,
        ),
    )
    paths = run_matrix(configs, output_dir=args.output_dir, include_controls=args.include_controls)
    for agent_path, baseline_path in paths:
        print(f"agent: {agent_path}")
        print(f"baseline: {baseline_path}")


if __name__ == "__main__":
    main()
