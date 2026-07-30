"""Plot paired multi-seed drift experiments directly from their JSON results."""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter


_SEED_DIRECTORY = re.compile(r"seed_(\d+)$")
_ARM_FILES = {"agent": False, "baseline": True}
_PAIR_FIELDS = ("corruption", "severity", "tau", "horizon")


class ScenarioError(ValueError):
    """A scenario directory does not contain a compatible paired seed group."""


@dataclass(frozen=True)
class Episode:
    """The plotting fields from one result arm."""

    seed: int
    arm: str
    corruption: str
    severity: int
    tau: float
    onset: float
    horizon: float
    times: np.ndarray
    accuracies: np.ndarray
    final_accuracy: float
    downtime: float
    clean_retention_delta: float
    decisions: tuple[float, ...]
    retraining_periods: tuple[tuple[float, float], ...]
    oracle_trigger: bool


@dataclass(frozen=True)
class SeedPair:
    """Agent and baseline episodes generated from the same seed."""

    seed: int
    agent: Episode
    baseline: Episode


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ScenarioError(f"{field} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ScenarioError(f"{field} must be finite")
    return number


def _load_payload(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as source:
            payload = json.load(source)
    except (OSError, json.JSONDecodeError) as error:
        raise ScenarioError(f"cannot read {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ScenarioError(f"{path} must contain a JSON object")
    return payload


def _last_final_accuracy(history: list[dict[str, Any]], path: Path) -> float:
    finals = [entry for entry in history if entry.get("stage") == "episode_end"]
    candidate = finals[-1] if finals else (history[-1] if history else None)
    if candidate is None:
        raise ScenarioError(f"{path} has an empty corrupted_accuracy_history")
    return _finite_number(candidate.get("accuracy"), f"{path}: final accuracy")


def _episode_from_payload(path: Path, arm: str, directory_seed: int) -> Episode:
    payload = _load_payload(path)
    metadata = payload.get("metadata")
    config = payload.get("experiment_config")
    metrics = payload.get("metrics")
    history = payload.get("corrupted_accuracy_history")
    if (
        not isinstance(metadata, dict)
        or not isinstance(config, dict)
        or not isinstance(metrics, dict)
    ):
        raise ScenarioError(
            f"{path} requires experiment_config, metadata, and metrics objects"
        )
    if not isinstance(history, list):
        raise ScenarioError(f"{path} requires corrupted_accuracy_history")

    seed_value = metadata.get("seed")
    if isinstance(seed_value, bool) or not isinstance(seed_value, int):
        raise ScenarioError(f"{path}: metadata.seed must be an integer")
    if seed_value != directory_seed:
        raise ScenarioError(
            f"{path}: directory seed {directory_seed} differs from metadata seed {seed_value}"
        )
    expected_baseline = _ARM_FILES[arm]
    if metadata.get("baseline") is not expected_baseline:
        raise ScenarioError(f"{path}: baseline flag is incompatible with the {arm} arm")

    corruption = metadata.get("corruption")
    severity = metadata.get("severity")
    if not isinstance(corruption, str) or not corruption:
        raise ScenarioError(f"{path}: corruption must be a non-empty string")
    if isinstance(severity, bool) or not isinstance(severity, int):
        raise ScenarioError(f"{path}: severity must be an integer")
    tau = _finite_number(metadata.get("tau"), f"{path}: tau")
    onset = _finite_number(
        metadata.get("production_start_time"), f"{path}: production_start_time"
    )
    end = _finite_number(metadata.get("end_time_seconds"), f"{path}: end_time_seconds")
    horizon = _finite_number(
        metadata.get("production_horizon_seconds"),
        f"{path}: production_horizon_seconds",
    )
    for field in ("corruption", "severity", "seed", "tau", "baseline"):
        if config.get(field) != metadata.get(field):
            raise ScenarioError(
                f"{path}: {field} differs between experiment_config and metadata"
            )
    config_horizon = _finite_number(
        config.get("production_horizon_seconds"),
        f"{path}: experiment_config.production_horizon_seconds",
    )
    if not math.isclose(config_horizon, horizon, abs_tol=1e-9):
        raise ScenarioError(
            f"{path}: production horizon differs between experiment_config and metadata"
        )
    if horizon <= 0 or not math.isclose(end - onset, horizon, abs_tol=1e-6):
        raise ScenarioError(f"{path}: incompatible production horizon")

    points: list[tuple[float, float]] = []
    for index, entry in enumerate(history):
        if not isinstance(entry, dict):
            raise ScenarioError(f"{path}: history entry {index} must be an object")
        time = _finite_number(entry.get("time"), f"{path}: history time")
        accuracy = _finite_number(entry.get("accuracy"), f"{path}: history accuracy")
        points.append((time - onset, accuracy))
    if not points:
        raise ScenarioError(f"{path} has an empty corrupted_accuracy_history")
    if any(right[0] < left[0] for left, right in zip(points, points[1:])):
        raise ScenarioError(f"{path}: corrupted accuracy times must be ordered")
    if points[0][0] < -1e-6 or points[-1][0] > horizon + 1e-6:
        raise ScenarioError(f"{path}: trajectory lies outside the production horizon")

    decisions = tuple(
        _finite_number(item.get("time"), f"{path}: decision time") - onset
        for item in _object_list(payload, "retrain_decisions", path)
    )
    periods = tuple(
        (
            _finite_number(item.get("started_time"), f"{path}: retraining start") - onset,
            _finite_number(item.get("completed_time"), f"{path}: retraining end") - onset,
        )
        for item in _object_list(payload, "retrain_round_events", path)
    )
    if any(start > finish or start < 0 or finish > horizon + 1e-6 for start, finish in periods):
        raise ScenarioError(f"{path}: retraining period lies outside the horizon")
    drift_events = _object_list(payload, "drift_events", path)
    oracle_trigger = any(
        str(event.get("client_id", "")).casefold() == "oracle" for event in drift_events
    )

    return Episode(
        seed=seed_value,
        arm=arm,
        corruption=corruption,
        severity=severity,
        tau=tau,
        onset=onset,
        horizon=horizon,
        times=np.asarray([point[0] for point in points]),
        accuracies=np.asarray([point[1] for point in points]),
        final_accuracy=_last_final_accuracy(history, path),
        downtime=_finite_number(metrics.get("downtime_seconds"), f"{path}: downtime"),
        clean_retention_delta=_finite_number(
            metrics.get("clean_retention_delta"), f"{path}: clean_retention_delta"
        ),
        decisions=decisions,
        retraining_periods=periods,
        oracle_trigger=oracle_trigger,
    )


def _object_list(payload: dict[str, Any], field: str, path: Path) -> list[dict[str, Any]]:
    items = payload.get(field, [])
    if not isinstance(items, list) or any(not isinstance(item, dict) for item in items):
        raise ScenarioError(f"{path}: {field} must be a list of objects")
    return items


def _compatibility_key(episode: Episode) -> tuple[Any, ...]:
    return tuple(getattr(episode, field) for field in _PAIR_FIELDS)


def load_scenario(scenario_dir: str | Path) -> list[SeedPair]:
    """Discover and validate all ``seed_*`` agent/baseline pairs."""
    root = Path(scenario_dir)
    seed_directories: list[tuple[int, Path]] = []
    if root.is_dir():
        for candidate in sorted(root.iterdir()):
            match = _SEED_DIRECTORY.fullmatch(candidate.name)
            if candidate.is_dir() and match:
                seed_directories.append((int(match.group(1)), candidate))
    if not seed_directories:
        raise ScenarioError(f"{root} contains no seed_* result groups")

    pairs: list[SeedPair] = []
    seen_seeds: set[int] = set()
    for directory_seed, directory in seed_directories:
        if directory_seed in seen_seeds:
            raise ScenarioError(f"duplicate seed {directory_seed} in {root}")
        seen_seeds.add(directory_seed)
        missing = [arm for arm in _ARM_FILES if not (directory / f"{arm}.json").is_file()]
        if missing:
            raise ScenarioError(
                f"{directory} is missing arm(s): {', '.join(sorted(missing))}"
            )
        agent = _episode_from_payload(directory / "agent.json", "agent", directory_seed)
        baseline = _episode_from_payload(
            directory / "baseline.json", "baseline", directory_seed
        )
        if _compatibility_key(agent) != _compatibility_key(baseline):
            raise ScenarioError(f"seed {directory_seed} has incompatible paired arms")
        pairs.append(SeedPair(directory_seed, agent, baseline))

    expected = _compatibility_key(pairs[0].agent)
    for pair in pairs[1:]:
        if _compatibility_key(pair.agent) != expected:
            raise ScenarioError(
                "seed group has incompatible corruption, severity, tau, or horizon"
            )
    return pairs


def forward_fill_steps(
    times: Sequence[np.ndarray], values: Sequence[np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    """Align step trajectories to their time union using previous values."""
    if not times or len(times) != len(values):
        raise ValueError("times and values must contain the same non-empty trajectory set")
    union = np.unique(np.concatenate([np.asarray(item, dtype=float) for item in times]))
    aligned: list[np.ndarray] = []
    for trajectory_times, trajectory_values in zip(times, values):
        x = np.asarray(trajectory_times, dtype=float)
        y = np.asarray(trajectory_values, dtype=float)
        if x.ndim != 1 or y.ndim != 1 or len(x) != len(y) or not len(x):
            raise ValueError("each trajectory must contain matching one-dimensional values")
        indices = np.searchsorted(x, union, side="right") - 1
        indices = np.maximum(indices, 0)
        aligned.append(y[indices])
    return union, np.vstack(aligned)


def _plot_trajectories(axis, pairs: Sequence[SeedPair], colors: Sequence[Any]) -> None:
    all_episodes = [episode for pair in pairs for episode in (pair.agent, pair.baseline)]
    union, _ = forward_fill_steps(
        [episode.times for episode in all_episodes],
        [episode.accuracies for episode in all_episodes],
    )
    arm_style = {"agent": "-", "baseline": "--"}
    for pair, color in zip(pairs, colors):
        for episode in (pair.agent, pair.baseline):
            axis.step(
                episode.times,
                episode.accuracies,
                where="post",
                color=color,
                linestyle=arm_style[episode.arm],
                linewidth=0.9,
                alpha=0.62,
            )

    for arm, mean_color in (("agent", "#1557a0"), ("baseline", "#a84d10")):
        episodes = [getattr(pair, arm) for pair in pairs]
        arm_union, aligned = forward_fill_steps(
            [episode.times for episode in episodes],
            [episode.accuracies for episode in episodes],
        )
        # Both arms are evaluated over the same validated horizon. Extend their
        # summaries to the complete cross-arm time union with step semantics.
        indices = np.maximum(np.searchsorted(arm_union, union, side="right") - 1, 0)
        arm_values = aligned[:, indices]
        axis.fill_between(
            union,
            np.min(arm_values, axis=0),
            np.max(arm_values, axis=0),
            step="post",
            color=mean_color,
            alpha=0.12,
            linewidth=0,
        )
        axis.step(
            union,
            np.mean(arm_values, axis=0),
            where="post",
            color=mean_color,
            linestyle=arm_style[arm],
            linewidth=3.0,
            label=f"{arm.title()} mean",
        )

    sample = pairs[0].agent
    axis.axhline(sample.tau, color="#555555", linestyle=":", linewidth=1.4)
    axis.axvline(0, color="#222222", linewidth=1.1)
    axis.axvline(sample.horizon, color="#222222", linestyle=":", linewidth=1.1)
    for pair, color in zip(pairs, colors):
        for decision in pair.agent.decisions:
            axis.axvline(decision, color=color, alpha=0.28, linewidth=0.9)
        for start, finish in pair.agent.retraining_periods:
            axis.axvspan(start, finish, color=color, alpha=0.08, linewidth=0)

    axis.set_xlim(0, sample.horizon)
    axis.set_ylim(0, 1)
    axis.set_ylabel("Corrupted accuracy")
    axis.set_xlabel("Time since drift onset (s)")
    axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    axis.grid(True, alpha=0.2)
    axis.text(0, 1.015, "drift onset", transform=axis.get_xaxis_transform(), ha="left")
    axis.text(
        sample.horizon,
        1.015,
        "horizon",
        transform=axis.get_xaxis_transform(),
        ha="right",
    )

    seed_handles = [
        Line2D([0], [0], color=color, linewidth=2, label=f"Seed {pair.seed}")
        for pair, color in zip(pairs, colors)
    ]
    semantic_handles = [
        Line2D([0], [0], color="#1557a0", linewidth=3, label="Agent mean"),
        Line2D(
            [0], [0], color="#a84d10", linewidth=3, linestyle="--", label="Baseline mean"
        ),
        Line2D([0], [0], color="#555555", linestyle=":", label=f"τ = {sample.tau:g}"),
        Line2D([0], [0], color="#777777", alpha=0.4, label="Decision / retraining"),
    ]
    first_legend = axis.legend(handles=seed_handles, loc="lower right", ncol=2, fontsize=8)
    axis.add_artist(first_legend)
    axis.legend(handles=semantic_handles, loc="upper right", fontsize=8)


def _plot_dumbbell(
    axis,
    pairs: Sequence[SeedPair],
    colors: Sequence[Any],
    field: str,
    title: str,
    percent: bool,
    x_limits: tuple[float, float] | None = None,
) -> None:
    y = np.arange(len(pairs), dtype=float)
    agent_values = np.asarray([getattr(pair.agent, field) for pair in pairs])
    baseline_values = np.asarray([getattr(pair.baseline, field) for pair in pairs])
    for row, agent, baseline, color in zip(y, agent_values, baseline_values, colors):
        axis.plot([baseline, agent], [row, row], color=color, linewidth=1.5, alpha=0.75)
        axis.scatter(agent, row, s=34, color=color, marker="o", zorder=3)
        axis.scatter(
            baseline,
            row,
            s=72,
            facecolor="none",
            edgecolor=color,
            linewidth=1.5,
            zorder=4,
        )

    mean_row = len(pairs) + 0.55
    agent_mean = float(np.mean(agent_values))
    baseline_mean = float(np.mean(baseline_values))
    axis.plot([baseline_mean, agent_mean], [mean_row, mean_row], color="black", linewidth=2.4)
    axis.scatter(agent_mean, mean_row, s=50, color="black", zorder=3)
    axis.scatter(
        baseline_mean,
        mean_row,
        s=92,
        facecolor="none",
        edgecolor="black",
        linewidth=1.7,
        zorder=4,
    )
    axis.axvline(0, color="#777777", linewidth=0.8, alpha=0.55)
    axis.set_yticks([*y, mean_row], [*[f"Seed {pair.seed}" for pair in pairs], "Mean"])
    axis.invert_yaxis()
    axis.set_title(title, fontsize=10)
    axis.grid(axis="x", alpha=0.2)
    if percent:
        axis.xaxis.set_major_formatter(PercentFormatter(1.0))
    else:
        axis.set_xlabel("Seconds")
    if x_limits is None:
        axis.margins(x=0.12)
    else:
        axis.set_xlim(*x_limits)


def build_figure(pairs: Sequence[SeedPair]) -> Figure:
    """Build the grouped trajectory and paired-outcome figure."""
    if not pairs:
        raise ScenarioError("cannot plot an empty seed group")
    color_map = plt.get_cmap("tab10")
    colors = [color_map(index % 10) for index in range(len(pairs))]
    figure = plt.figure(figsize=(15, 10), constrained_layout=True)
    grid = figure.add_gridspec(2, 3, height_ratios=(2.15, 1.0))
    trajectory_axis = figure.add_subplot(grid[0, :])
    bottom_axes = [figure.add_subplot(grid[1, index]) for index in range(3)]

    _plot_trajectories(trajectory_axis, pairs, colors)
    _plot_dumbbell(
        bottom_axes[0],
        pairs,
        colors,
        "final_accuracy",
        "Final corrupted accuracy",
        True,
    )
    _plot_dumbbell(
        bottom_axes[1],
        pairs,
        colors,
        "downtime",
        "Downtime",
        False,
        x_limits=(0.0, pairs[0].agent.horizon),
    )
    _plot_dumbbell(
        bottom_axes[2],
        pairs,
        colors,
        "clean_retention_delta",
        "Clean retention Δ",
        True,
    )
    bottom_axes[1].tick_params(axis="y", labelleft=False)
    bottom_axes[2].tick_params(axis="y", labelleft=False)

    sample = pairs[0].agent
    control = sample.corruption == "identity" and sample.severity == 0 and any(
        pair.agent.oracle_trigger or pair.baseline.oracle_trigger for pair in pairs
    )
    label = f"{sample.corruption}:{sample.severity}"
    if control:
        label += " — oracle-trigger control"
    figure.suptitle(
        f"Drift comparison — {label} — {len(pairs)} paired seeds",
        fontsize=15,
        fontweight="bold",
    )
    figure.legend(
        handles=[
            Line2D(
                [0], [0], marker="o", color="black", linestyle="", label="Agent (filled)"
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                markerfacecolor="white",
                markeredgecolor="black",
                color="white",
                linestyle="",
                label="Baseline (open)",
            ),
        ],
        loc="outside lower center",
        ncol=2,
        frameon=False,
    )
    return figure


def plot_scenario(scenario_dir: str | Path, output: str | Path) -> Path:
    """Load a scenario, render it, and save a PNG."""
    pairs = load_scenario(scenario_dir)
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure = build_figure(pairs)
    try:
        figure.savefig(destination, dpi=180, bbox_inches="tight")
    finally:
        plt.close(figure)
    return destination


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot paired seed trajectories for one drift corruption scenario"
    )
    parser.add_argument("--scenario-dir", required=True, help="Directory containing seed_*")
    parser.add_argument("--output", required=True, help="Destination PNG file")
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        destination = plot_scenario(args.scenario_dir, args.output)
    except ScenarioError as error:
        parser.error(str(error))
    print(destination)


if __name__ == "__main__":
    main()
