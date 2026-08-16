"""Run the E09 review-plan arms B1-B4.

B1 compares the drift detector against fixed-schedule retraining baselines
(``ScheduledMonitor``) and a tight first-alarm configuration; B3 measures the
no-drift false-positive rate; B4 sweeps the quorum and window sensitivities
plus a staggered per-client onset schedule.  B2 is a pure post-processing
step that recomputes downtime across persisted pairs for a tau sweep without
ever retraining.

Run everything::

    python -m experiments.e09_drift_gradual.run_b1_b4

Inspect the full matrix without executing anything::

    python -m experiments.e09_drift_gradual.run_b1_b4 --dry-run
"""

from __future__ import annotations

import argparse
import itertools
import json
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

_BASE = Path(__file__).resolve().parent.parent.parent
for _path in (str(_BASE), str(_BASE / "src")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from experiments.e07_drift_agent.episode import (
    DriftEpisodeConfig,
    run_drift_comparison,
)
from experiments.e07_drift_agent.run_matrix import build_run_matrix, run_matrix
from experiments.e09_drift_gradual.run_grid import (
    DEFAULT_SCENARIOS,
    E09_PRODUCTION_HORIZON_SECONDS,
)
from experiments.shared.comparison_core import compute_downtime, write_csv
from experiments.shared.drift_controls import ScheduledMonitor, identity_corruption
from experiments.shared.registry import output_path, temporary_output_path

DEFAULT_SEEDS: tuple[int, ...] = (42, 43, 44, 45, 46)
DEFAULT_RAMP_TICKS: tuple[int, ...] = (5, 10, 20)
DEFAULT_SCHEDULED_TIMES: tuple[float, ...] = (100.0, 150.0)
DEFAULT_QUORUMS: tuple[float, ...] = (0.20, 0.50)
DEFAULT_WINDOWS: tuple[int, ...] = (1,)
DEFAULT_TAU_VALUES: tuple[float, ...] = (0.40, 0.45, 0.50, 0.55)

# The published grid is the read-only tau source; freshly generated arm
# roots under --output-dir are appended when they exist.
GRID_TAU_SOURCE: Path = output_path("e09-gradual-drift", "cifar-10") / "grid-v1"

ARM_CHOICES: tuple[str, ...] = (
    "scheduled",
    "first-alarm",
    "no-drift",
    "quorum",
    "window",
    "staggered",
    "tau",
)
DEFAULT_ARMS: tuple[str, ...] = ARM_CHOICES[:-1]


@dataclass(frozen=True)
class ConfigGroup:
    """One ``run_matrix`` invocation: a stable subdirectory plus its configs."""

    subdir: str
    configs: tuple[DriftEpisodeConfig, ...]
    runner: Callable | None = None


@dataclass(frozen=True)
class TauSpec:
    """Post-processing spec for the B2 tau sweep (no training).

    ``sources`` is None only when the default must be resolved after the
    training arms finish: the published grid plus any freshly generated arm
    roots.
    """

    sources: tuple[Path, ...] | None
    tau_values: tuple[float, ...]


@dataclass(frozen=True)
class ArmPlan:
    """One requested arm: training groups, a tau sweep, or both inputs."""

    name: str
    groups: tuple[ConfigGroup, ...] = ()
    tau: TauSpec | None = None
    corruption_fn: Callable | None = None

    @property
    def episode_count(self) -> int:
        return sum(len(group.configs) for group in self.groups)

    @property
    def pair_count(self) -> int:
        return self.episode_count


def _scheduled_runner(trigger_seconds: float) -> Callable:
    """Return a runner that injects a fixed-timer ``ScheduledMonitor``."""

    def runner(config, *, corruption_fn, **kwargs):
        return run_drift_comparison(
            config,
            corruption_fn=corruption_fn,
            monitor_factory=lambda server, baseline: ScheduledMonitor(
                trigger_after_seconds=trigger_seconds
            ),
        )

    return runner


def _ramp_sensitivity_configs(
    *,
    seeds: tuple[int, ...],
    scenarios: tuple[tuple[str, int], ...],
    ramp_ticks: tuple[int, ...],
    quorums: tuple[float, ...] = (0.30,),
    window_ticks: tuple[int, ...] = (2,),
) -> tuple[DriftEpisodeConfig, ...]:
    """Grid configs spanning every ramp duration.

    ``build_run_matrix`` varies seed/scenario/quorum/window but keeps the
    ramp duration from ``base_config``, so each ramp value needs its own
    base config.
    """
    configs: list[DriftEpisodeConfig] = []
    for ramp in ramp_ticks:
        base = DriftEpisodeConfig(
            corruption=scenarios[0][0],
            severity=scenarios[0][1],
            drift_ramp_ticks=ramp,
            retrain_rounds=1,
            production_horizon_seconds=E09_PRODUCTION_HORIZON_SECONDS,
        )
        configs.extend(
            build_run_matrix(
                seeds=seeds,
                scenarios=scenarios,
                quorums=quorums,
                window_ticks=window_ticks,
                base_config=base,
            )
        )
    return tuple(configs)


def _scheduled_plan(
    seeds: tuple[int, ...],
    scenarios: tuple[tuple[str, int], ...],
    ramp_ticks: tuple[int, ...],
    scheduled_times: tuple[float, ...],
) -> ArmPlan:
    groups = tuple(
        ConfigGroup(
            subdir=f"scheduled_{time:g}s",
            configs=_ramp_sensitivity_configs(
                seeds=seeds, scenarios=scenarios, ramp_ticks=ramp_ticks
            ),
            runner=_scheduled_runner(time),
        )
        for time in scheduled_times
    )
    return ArmPlan(name="scheduled", groups=groups)


def _first_alarm_plan(
    seeds: tuple[int, ...],
    scenarios: tuple[tuple[str, int], ...],
    ramp_ticks: tuple[int, ...],
    windows: tuple[int, ...],
) -> ArmPlan:
    return ArmPlan(
        name="first-alarm",
        groups=(
            ConfigGroup(
                subdir="first_alarm",
                configs=_ramp_sensitivity_configs(
                    seeds=seeds,
                    scenarios=scenarios,
                    ramp_ticks=ramp_ticks,
                    quorums=(0.10,),
                    window_ticks=windows,
                ),
            ),
        ),
    )


def _no_drift_plan(
    seeds: tuple[int, ...],
    ramp_ticks: tuple[int, ...],
) -> ArmPlan:
    configs = tuple(
        DriftEpisodeConfig(
            corruption="identity",
            severity=0,
            drift_ramp_ticks=ramp,
            seed=seed,
            retrain_rounds=1,
            production_horizon_seconds=E09_PRODUCTION_HORIZON_SECONDS,
        )
        for seed, ramp in itertools.product(seeds, ramp_ticks)
    )
    return ArmPlan(
        name="no-drift",
        groups=(ConfigGroup(subdir="no_drift", configs=configs),),
        corruption_fn=identity_corruption,
    )


def _quorum_plan(
    seeds: tuple[int, ...],
    scenarios: tuple[tuple[str, int], ...],
    ramp_ticks: tuple[int, ...],
    quorums: tuple[float, ...],
) -> ArmPlan:
    groups = tuple(
        ConfigGroup(
            subdir=f"quorum_{quorum:g}",
            configs=_ramp_sensitivity_configs(
                seeds=seeds,
                scenarios=scenarios,
                ramp_ticks=ramp_ticks,
                quorums=(quorum,),
                window_ticks=(2,),
            ),
        )
        for quorum in quorums
    )
    return ArmPlan(name="quorum", groups=groups)


def _window_plan(
    seeds: tuple[int, ...],
    scenarios: tuple[tuple[str, int], ...],
    ramp_ticks: tuple[int, ...],
    windows: tuple[int, ...],
) -> ArmPlan:
    groups = tuple(
        ConfigGroup(
            subdir=f"window_{window}",
            configs=_ramp_sensitivity_configs(
                seeds=seeds,
                scenarios=scenarios,
                ramp_ticks=ramp_ticks,
                quorums=(0.30,),
                window_ticks=(window,),
            ),
        )
        for window in windows
    )
    return ArmPlan(name="window", groups=groups)


def _staggered_plan(
    seeds: tuple[int, ...],
    scenarios: tuple[tuple[str, int], ...],
) -> ArmPlan:
    num_clients = DriftEpisodeConfig().num_clients
    configs = tuple(
        DriftEpisodeConfig(
            corruption=corruption,
            severity=severity,
            seed=seed,
            drift_onset_ticks={index: index for index in range(num_clients)},
            drift_ramp_ticks=None,
            retrain_rounds=1,
            production_horizon_seconds=E09_PRODUCTION_HORIZON_SECONDS,
        )
        for (corruption, severity), seed in itertools.product(scenarios, seeds)
    )
    return ArmPlan(
        name="staggered",
        groups=(ConfigGroup(subdir="staggered", configs=configs),),
    )


def _tau_plan(
    tau_values: tuple[float, ...],
    tau_source: Iterable[str | Path] | None,
    *,
    dry_run: bool,
) -> ArmPlan:
    if tau_source is not None:
        sources: tuple[Path, ...] | None = tuple(Path(path) for path in tau_source)
    elif dry_run:
        sources = (GRID_TAU_SOURCE,)
    else:
        sources = None  # deferred until the training arms finish
    return ArmPlan(name="tau", tau=TauSpec(sources=sources, tau_values=tau_values))


def build_matrix(
    *,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    scenarios: tuple[tuple[str, int], ...] = DEFAULT_SCENARIOS,
    ramp_ticks: tuple[int, ...] = DEFAULT_RAMP_TICKS,
    arms: tuple[str, ...] = DEFAULT_ARMS,
    scheduled_times: tuple[float, ...] = DEFAULT_SCHEDULED_TIMES,
    quorums: tuple[float, ...] = DEFAULT_QUORUMS,
    windows: tuple[int, ...] = DEFAULT_WINDOWS,
    tau_values: tuple[float, ...] = DEFAULT_TAU_VALUES,
    tau_source: Iterable[str | Path] | None = None,
    dry_run: bool = False,
) -> list[ArmPlan]:
    """Materialize the requested B1-B4 arms without running anything."""
    plans: list[ArmPlan] = []
    requested = tuple(dict.fromkeys(arms))
    for arm in requested:
        if arm == "scheduled":
            plans.append(_scheduled_plan(seeds, scenarios, ramp_ticks, scheduled_times))
        elif arm == "first-alarm":
            plans.append(_first_alarm_plan(seeds, scenarios, ramp_ticks, windows))
        elif arm == "no-drift":
            plans.append(_no_drift_plan(seeds, ramp_ticks))
        elif arm == "quorum":
            plans.append(_quorum_plan(seeds, scenarios, ramp_ticks, quorums))
        elif arm == "window":
            plans.append(_window_plan(seeds, scenarios, ramp_ticks, windows))
        elif arm == "staggered":
            plans.append(_staggered_plan(seeds, scenarios))
        elif arm == "tau":
            plans.append(_tau_plan(tau_values, tau_source, dry_run=dry_run))
    return plans


def run_arms(
    plans: Iterable[ArmPlan],
    *,
    output_dir: str | Path,
) -> int:
    """Run every training arm; returns the number of executed episodes."""
    from src.utils.corruptions import apply_corruption

    root = Path(output_dir)
    total = 0
    for plan in plans:
        corruption_fn = plan.corruption_fn or apply_corruption
        for group in plan.groups:
            group_root = root / plan.name / group.subdir
            pairs = run_matrix(
                group.configs,
                output_dir=group_root,
                runner=group.runner or run_drift_comparison,
                corruption_fn=corruption_fn,
            )
            total += len(pairs)
            print(
                f"[{plan.name}/{group.subdir}] {len(pairs)} agent/baseline "
                f"pairs -> {group_root}"
            )
    return total


def _iter_pair_directories(source: Path) -> list[Path]:
    """Return every directory under ``source`` holding a validated pair."""
    return [
        agent_file.parent
        for agent_file in sorted(source.rglob("agent.json"))
        if (agent_file.parent / "baseline.json").is_file()
    ]


def _pair_context(payload: dict) -> dict:
    """Extract the reproducibility context shared by v2 and v3 payloads."""
    metadata = payload.get("metadata") or {}
    experiment = payload.get("experiment_config") or {}
    end_time = metadata.get("end_time_seconds")
    if end_time is None:
        raise ValueError("drift payload metadata must include end_time_seconds")
    return {
        "corruption": metadata.get("corruption") or experiment.get("corruption"),
        "severity": metadata.get("severity") or experiment.get("severity"),
        "seed": metadata.get("seed") or experiment.get("seed"),
        "ramp_ticks": experiment.get("drift_ramp_ticks"),
        "end_time_seconds": float(end_time),
    }


def compute_tau_sweep_rows(
    sources: Iterable[str | Path],
    tau_values: Iterable[float],
) -> list[dict]:
    """Recompute downtime across persisted pairs without touching sources."""
    rows: list[dict] = []
    for source in sources:
        root = Path(source)
        arm = root.name
        for pair_dir in _iter_pair_directories(root):
            agent = json.loads((pair_dir / "agent.json").read_text(encoding="utf-8"))
            baseline = json.loads(
                (pair_dir / "baseline.json").read_text(encoding="utf-8")
            )
            agent_context = _pair_context(agent)
            baseline_context = _pair_context(baseline)
            for tau in tau_values:
                rows.append(
                    {
                        "source": str(root),
                        "corruption": agent_context["corruption"],
                        "severity": agent_context["severity"],
                        "ramp_ticks": agent_context["ramp_ticks"],
                        "seed": agent_context["seed"],
                        "arm": arm,
                        "tau": tau,
                        "downtime_agent": compute_downtime(
                            agent["corrupted_accuracy_history"],
                            tau,
                            end_time=agent_context["end_time_seconds"],
                        ),
                        "downtime_baseline": compute_downtime(
                            baseline["corrupted_accuracy_history"],
                            tau,
                            end_time=baseline_context["end_time_seconds"],
                        ),
                    }
                )
    return rows


def _write_tau_summary_markdown(rows: list[dict], path: Path) -> None:
    """Persist a markdown table of mean downtime per group and tau."""
    grouped: dict[
        tuple, dict[str, dict[float, list[float]]]
    ] = {}
    for row in rows:
        key = (
            row["source"],
            row["arm"],
            row["corruption"],
            row["severity"],
            row["ramp_ticks"],
        )
        group = grouped.setdefault(
            key,
            {"agent": defaultdict(list), "baseline": defaultdict(list)},
        )
        group["agent"][row["tau"]].append(row["downtime_agent"])
        group["baseline"][row["tau"]].append(row["downtime_baseline"])

    lines = [
        "# B1-B4 tau sweep summary",
        "",
        "Downtime recomputed with compute_downtime(...) from persisted",
        "agent/baseline pairs; source JSONs are never modified.  Values are",
        "means across seeds within each group and tau.",
        "",
        "| source | arm | corruption | severity | ramp_ticks | tau |",
        "| mean_downtime_agent | mean_downtime_baseline |",
        "|---|---|---|---|---|---|---|---|",
    ]
    # ramp_ticks (key[4]) may be None or int across sources, so sort on the
    # homogeneous string repr instead of the raw tuples.
    for key in sorted(grouped, key=lambda k: str(k)):
        group = grouped[key]
        for tau in sorted(group["agent"]):
            agent_mean = statistics.fmean(group["agent"][tau])
            baseline_mean = statistics.fmean(group["baseline"][tau])
            lines.append(
                "| {} | {} | {} | {} | {} | {:g} | {:.6g} | {:.6g} |".format(
                    key[0],
                    key[1],
                    key[2],
                    key[3],
                    key[4],
                    tau,
                    agent_mean,
                    baseline_mean,
                )
            )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def run_tau_sweep(
    sources: Iterable[str | Path],
    tau_values: Iterable[float],
    output_dir: str | Path,
) -> tuple[Path, Path]:
    """Post-process persisted pairs into a tau downtime CSV and markdown."""
    rows = compute_tau_sweep_rows(sources, tau_values)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    csv_path = destination / "tau_sweep.csv"
    md_path = destination / "tau_sweep_summary.md"
    write_csv(rows, csv_path)
    _write_tau_summary_markdown(rows, md_path)
    return csv_path, md_path


def _existing_arm_roots(output_dir: Path) -> tuple[Path, ...]:
    """Subvariant roots under ``output_dir`` that already hold pairs."""
    if not output_dir.is_dir():
        return ()
    roots: list[Path] = []
    for arm_dir in sorted(path for path in output_dir.iterdir() if path.is_dir()):
        for sub_dir in sorted(path for path in arm_dir.iterdir() if path.is_dir()):
            if _iter_pair_directories(sub_dir):
                roots.append(sub_dir)
    return tuple(roots)


def default_tau_sources(output_dir: str | Path) -> tuple[Path, ...]:
    """The published grid plus any freshly generated arm roots."""
    return (GRID_TAU_SOURCE,) + _existing_arm_roots(Path(output_dir))


def _config_brief(config: DriftEpisodeConfig) -> str:
    ramp = (
        "ramp=None"
        if config.drift_ramp_ticks is None
        else f"ramp={config.drift_ramp_ticks}"
    )
    onset = ""
    if config.drift_onset_ticks is not None:
        onset = f" onset={{0..{len(config.drift_onset_ticks) - 1}}}"
    return (
        f"{config.corruption}/sev{config.severity} seed={config.seed} "
        f"{ramp} quorum={config.trigger_threshold:g} "
        f"window={config.trigger_window_ticks}{onset}"
    )


def print_dry_run(plans: Iterable[ArmPlan]) -> None:
    """Print the full matrix per arm and exit without executing anything."""
    total = 0
    for plan in plans:
        if plan.tau is not None:
            sources = plan.tau.sources or (GRID_TAU_SOURCE,)
            print(f"{plan.name}: post-processing sweep, no training")
            print(f"  sources: {', '.join(str(source) for source in sources)}")
            print(
                "  tau-values: "
                + ", ".join(f"{value:g}" for value in plan.tau.tau_values)
            )
            continue
        count = plan.episode_count
        total += count
        print(f"{plan.name}: {count} episodes, {count} agent/baseline pairs")
        for group in plan.groups:
            example = _config_brief(group.configs[0])
            print(f"  {group.subdir}: {len(group.configs)} configs, e.g. {example}")
    print(f"total episodes: {total}")


def _parse_scenarios(values: list[str]) -> list[tuple[str, int]]:
    scenarios: list[tuple[str, int]] = []
    for value in values:
        corruption, separator, severity = value.rpartition(":")
        if not separator:
            raise SystemExit(f"scenario must use corruption:severity, got {value!r}")
        scenarios.append((corruption, int(severity)))
    return scenarios


def _warn_scheduled_times(scheduled_times: Iterable[float]) -> None:
    """Warn on stderr about scheduled times that can never fire.

    ``ScheduledMonitor`` triggers at ``trigger_after_seconds`` of production
    time.  Under the default E09 horizon (600 s) a time at or beyond it
    never triggers; under the default monitor budget (25 ticks x 10 s =
    250 s) a time at or beyond it fires only if the horizon is extended.
    Warnings only: the caller may be sweeping longer-horizon runs
    deliberately, so no error is raised.
    """
    horizon = E09_PRODUCTION_HORIZON_SECONDS
    monitor_budget = (
        DriftEpisodeConfig.monitor_ticks * DriftEpisodeConfig.monitor_tick_seconds
    )
    for time in scheduled_times:
        if time >= horizon:
            print(
                f"WARNING: scheduled time {time:g} s >= production horizon "
                f"{horizon:g} s; the monitor will never trigger under the "
                f"default horizon — reduce the time or raise the horizon",
                file=sys.stderr,
            )
        elif time >= monitor_budget:
            print(
                f"WARNING: scheduled time {time:g} s >= default monitor "
                f"budget {monitor_budget:g} s ({DriftEpisodeConfig.monitor_ticks} "
                f"ticks x {DriftEpisodeConfig.monitor_tick_seconds:g} s); it "
                f"may never trigger unless the horizon is extended",
                file=sys.stderr,
            )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the E09 B1-B4 review-plan arms"
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument(
        "--scenario",
        nargs="+",
        default=[
            f"{corruption}:{severity}" for corruption, severity in DEFAULT_SCENARIOS
        ],
        help="one or more corruption:severity pairs",
    )
    parser.add_argument(
        "--ramp-ticks", nargs="+", type=int, default=list(DEFAULT_RAMP_TICKS)
    )
    parser.add_argument(
        "--arms",
        "--arm",
        nargs="+",
        choices=ARM_CHOICES,
        default=list(DEFAULT_ARMS),
        help=(
            "arms to run; tau runs as post-processing after any training "
            "arms"
        ),
    )
    parser.add_argument(
        "--scheduled-times",
        nargs="+",
        type=float,
        default=list(DEFAULT_SCHEDULED_TIMES),
        help="ScheduledMonitor trigger times for the scheduled arm",
    )
    parser.add_argument(
        "--quorums",
        nargs="+",
        type=float,
        default=list(DEFAULT_QUORUMS),
        help="trigger thresholds for the quorum arm",
    )
    parser.add_argument(
        "--windows",
        nargs="+",
        type=int,
        default=list(DEFAULT_WINDOWS),
        help="trigger window ticks for the window and first-alarm arms",
    )
    parser.add_argument(
        "--tau-values",
        nargs="+",
        type=float,
        default=list(DEFAULT_TAU_VALUES),
        help="accuracy thresholds to sweep in the tau post-processing",
    )
    parser.add_argument(
        "--tau-source",
        nargs="+",
        help=(
            "roots to scan for agent/baseline.json pairs; default is "
            f"{GRID_TAU_SOURCE} plus freshly generated arm roots under "
            "--output-dir"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=str(
            temporary_output_path("e09-gradual-drift", "cifar-10") / "b1b4"
        ),
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="print the matrix and exit"
    )
    args = parser.parse_args(argv)
    _warn_scheduled_times(args.scheduled_times)
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    plans = build_matrix(
        seeds=tuple(args.seeds),
        scenarios=tuple(_parse_scenarios(args.scenario)),
        ramp_ticks=tuple(args.ramp_ticks),
        arms=tuple(args.arms),
        scheduled_times=tuple(args.scheduled_times),
        quorums=tuple(args.quorums),
        windows=tuple(args.windows),
        tau_values=tuple(args.tau_values),
        tau_source=args.tau_source,
        dry_run=args.dry_run,
    )
    if args.dry_run:
        print_dry_run(plans)
        return

    train_plans = [plan for plan in plans if plan.groups]
    total = run_arms(train_plans, output_dir=output_dir)

    tau_output = output_dir / "tau_sweep"
    for plan in plans:
        if plan.tau is None:
            continue
        sources = plan.tau.sources
        if sources is None:
            sources = default_tau_sources(output_dir)
        csv_path, md_path = run_tau_sweep(sources, plan.tau.tau_values, tau_output)
        print(f"[tau] {len(sources)} source(s) -> {csv_path}")
        print(f"[tau] summary -> {md_path}")

    print(f"\nB1-B4 complete: {total} episodes")
    for plan in train_plans:
        for group in plan.groups:
            print(f"  {plan.name}/{group.subdir}: {output_dir / plan.name / group.subdir}")
    print(
        "summary.json files live next to each arm root above, e.g. "
        f"{output_dir / '<arm>' / '<subvariant>' / 'summary.json'}"
    )


if __name__ == "__main__":
    main()