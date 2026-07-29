"""Plot a validated agent/baseline drift result pair using raw evaluations."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Mapping, Sequence

_ROOT = os.path.join(os.path.dirname(__file__), "..")
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.drift_results import (
    DriftResult,
    load_drift_result,
    summarize_pair,
    validate_pair,
)


def _as_result(value: DriftResult | Mapping) -> DriftResult:
    return value if isinstance(value, DriftResult) else DriftResult.from_payload(value)


def _series(result: DriftResult) -> tuple[list[float], list[float]]:
    entries = sorted(
        result.corrupted_accuracy_history,
        key=lambda entry: float(entry["time"]),
    )
    return (
        [float(entry["time"]) for entry in entries],
        [float(entry["accuracy"]) for entry in entries],
    )


def _first_stage_time(result: DriftResult, stage: str) -> float | None:
    for entry in result.corrupted_accuracy_history:
        if entry.get("stage") == stage:
            return float(entry["time"])
    return None


def _first_recovery(
    result: DriftResult,
    *,
    decision_time: float | None,
    tau: float,
) -> tuple[float, float] | None:
    if decision_time is None:
        return None
    for entry in sorted(
        result.corrupted_accuracy_history,
        key=lambda item: float(item["time"]),
    ):
        time = float(entry["time"])
        accuracy = float(entry["accuracy"])
        if time > decision_time and accuracy >= tau:
            return time, accuracy
    return None


def _format_metric(value: float | None, suffix: str = "") -> str:
    return "N/A" if value is None else f"{value:.2f}{suffix}"


def plot_drift_pair(
    baseline: DriftResult | Mapping,
    agent: DriftResult | Mapping,
    output: str | Path,
) -> Path:
    """Render the accepted pair and return the generated PNG path."""

    baseline_result = _as_result(baseline)
    agent_result = _as_result(agent)
    validate_pair(agent_result, baseline_result)
    summary = summarize_pair(agent_result, baseline_result)

    agent_times, agent_accuracy = _series(agent_result)
    baseline_times, baseline_accuracy = _series(baseline_result)
    tau = float(agent_result.metadata["tau"])
    onset = _first_stage_time(agent_result, "drift_onset")
    if onset is None:
        onset = float(agent_result.metadata["production_start_time"])
    decisions = agent_result.get("retrain_decisions", [])
    decision_time = float(decisions[0]["time"]) if decisions else None
    recovery = _first_recovery(
        agent_result,
        decision_time=decision_time,
        tau=tau,
    )
    horizon = float(agent_result.metadata["end_time_seconds"])

    figure, axis = plt.subplots(figsize=(12, 7))
    axis.plot(
        baseline_times,
        baseline_accuracy,
        color="#6b7280",
        marker="o",
        markersize=3,
        linewidth=1.4,
        alpha=0.85,
        label="Baseline (raw)",
    )
    axis.plot(
        agent_times,
        agent_accuracy,
        color="#2563eb",
        marker="o",
        markersize=3,
        linewidth=1.8,
        label="Drift agent (raw)",
    )
    axis.axhline(tau, color="#dc2626", linestyle="--", linewidth=1.2, label=f"tau={tau:g}")
    axis.axvline(onset, color="#111827", linestyle=":", linewidth=1.2, label="Drift onset")
    axis.axvline(
        horizon,
        color="#7c3aed",
        linestyle="--",
        linewidth=1.2,
        label="Fixed horizon",
    )

    if decision_time is not None:
        axis.axvline(
            decision_time,
            color="#ea580c",
            linestyle="-.",
            linewidth=1.4,
            label="Retraining decision",
        )
    for index, event in enumerate(agent_result.get("retrain_round_events", []), start=1):
        started = float(event["started_time"])
        completed = float(event["completed_time"])
        axis.axvspan(
            started,
            completed,
            color="#f59e0b",
            alpha=0.10 if index % 2 else 0.18,
        )
        axis.text(
            (started + completed) / 2.0,
            0.02,
            f"R{index}",
            transform=axis.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=8,
            color="#92400e",
        )

    if recovery is not None:
        axis.scatter(
            [recovery[0]],
            [recovery[1]],
            color="#16a34a",
            marker="*",
            s=130,
            zorder=5,
            label="First recovery",
        )

    annotation = "\n".join(
        (
            f"Agent downtime: {_format_metric(summary['agent_downtime_seconds'], ' s')}",
            f"Baseline downtime: {_format_metric(summary['baseline_downtime_seconds'], ' s')}",
            f"Avoided: {_format_metric(summary['downtime_avoided_seconds'], ' s')}",
            f"Avoided: {_format_metric(summary['downtime_avoided_percent'], '%')}",
            f"Clean retention delta: {_format_metric(summary['clean_retention_delta'])}",
            "Pair: audited v3" if summary["audited_pair"] else "Pair: legacy v2 (unverified)",
        )
    )
    axis.text(
        0.99,
        0.02,
        annotation,
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9},
    )
    axis.set(
        title="Synchronous drift-agent comparison",
        xlabel="Virtual time (seconds)",
        ylabel="Accuracy",
        ylim=(-0.02, 1.02),
    )
    axis.grid(True, alpha=0.2)
    axis.legend(loc="best")
    figure.tight_layout()

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot a validated synchronous drift result pair"
    )
    parser.add_argument("--baseline", required=True, help="Path to baseline.json")
    parser.add_argument("--agent", required=True, help="Path to agent.json")
    parser.add_argument("--output", required=True, help="Destination PNG")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    output = plot_drift_pair(
        load_drift_result(args.baseline),
        load_drift_result(args.agent),
        args.output,
    )
    print(f"Plot saved: {output}")


if __name__ == "__main__":
    main()
