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

from experiments.drift_controls import OracleMonitor, identity_corruption
from experiments.drift_results import validate_pair
from experiments.smoke_drift import DriftEpisodeConfig, run_drift_comparison, save_drift_result


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

    return [
        replace(
            base,
            seed=seed,
            corruption=corruption,
            severity=severity,
            trigger_threshold=quorum,
            retrain_rounds=rounds,
        )
        for corruption, severity in requested_scenarios
        for seed in requested_seeds
        for quorum in requested_quorums
        for rounds in requested_rounds
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


def _result_directory(
    output_dir: Path,
    config: DriftEpisodeConfig,
    duplicates: Counter[tuple[str, int, int]],
) -> Path:
    """Return the canonical location, with a suffix only for sensitivities."""
    path = output_dir / f"{config.corruption}_sev{config.severity}" / f"seed_{config.seed}"
    if duplicates[(config.corruption, config.severity, config.seed)] > 1:
        return path / f"quorum_{config.trigger_threshold:g}" / f"retrain_rounds_{config.retrain_rounds}"
    return path


def _write_pair(results: dict[str, dict], output_path: Path) -> tuple[Path, Path]:
    validate_pair(results["agent"], results["baseline"])
    output_path.mkdir(parents=True, exist_ok=True)
    agent_path = Path(save_drift_result(results["agent"], str(output_path), "agent.json"))
    baseline_path = Path(save_drift_result(results["baseline"], str(output_path), "baseline.json"))
    return agent_path, baseline_path


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
    controls: set[tuple[int, float, int]] = set()
    if include_controls:
        for config in configs:
            control_key = (config.seed, config.trigger_threshold, config.retrain_rounds)
            if control_key not in controls:
                controls.add(control_key)
                all_configs.append(
                    replace(config, corruption="identity", severity=0)
                )

    duplicates = Counter((c.corruption, c.severity, c.seed) for c in all_configs)
    output_paths: list[tuple[Path, Path]] = []
    agent_results: list[dict] = []
    baseline_results: list[dict] = []
    original_count = len(configs)
    for index, config in enumerate(all_configs):
        is_control = index >= original_count
        results = runner(
            config,
            corruption_fn=identity_corruption if is_control else corruption_fn,
            **({"monitor_factory": lambda _server, _baseline: OracleMonitor()} if is_control else {}),
        )
        output_paths.append(_write_pair(results, _result_directory(root, config, duplicates)))
        agent_results.append(results["agent"])
        baseline_results.append(results["baseline"])

    summary = {
        "schema_version": 1,
        "agent": aggregate_runs(agent_results),
        "baseline": aggregate_runs(baseline_results),
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
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
    parser.add_argument("--output-dir", default="output/cifar-10/drift-agent")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument(
        "--scenario",
        nargs="+",
        default=[f"{corruption}:{severity}" for corruption, severity in DEFAULT_SCENARIOS],
        help="one or more corruption:severity pairs",
    )
    parser.add_argument("--quorums", nargs="+", type=float)
    parser.add_argument("--retrain-rounds", nargs="+", type=int)
    parser.add_argument("--include-controls", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    configs = build_run_matrix(
        seeds=args.seeds,
        scenarios=[_parse_scenario(value) for value in args.scenario],
        quorums=args.quorums,
        retrain_rounds=args.retrain_rounds,
        base_config=DriftEpisodeConfig(dataset=args.dataset, output_dir=args.output_dir),
    )
    paths = run_matrix(configs, output_dir=args.output_dir, include_controls=args.include_controls)
    for agent_path, baseline_path in paths:
        print(f"agent: {agent_path}")
        print(f"baseline: {baseline_path}")


if __name__ == "__main__":
    main()
