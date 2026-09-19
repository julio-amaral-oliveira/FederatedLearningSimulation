"""Compact single-panel paper figures: accuracy trajectories only.

The composite ``plot.py`` figure packs one trajectory panel plus three bar
panels into a 15x10 canvas, which becomes unreadable when scaled down to a
single IEEE column. This module renders only the trajectory panel, sized for
the final column width, so the fonts are laid out at the size they will be
printed instead of being scaled down by ~4x.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter

from experiments.e07_drift_agent.plot import (
    ScenarioError,
    SeedPair,
    forward_fill_steps,
    load_scenario,
)

AGENT_MEAN_COLOR = "#1557a0"
BASELINE_MEAN_COLOR = "#a84d10"
PAPER_WIDTH_IN = 3.5
PAPER_HEIGHT_IN = 2.1


def build_trajectory_figure(pairs: Sequence[SeedPair]) -> Figure:
    """Render the paired accuracy trajectories for one scenario."""
    if not pairs:
        raise ScenarioError("cannot plot an empty seed group")
    sample = pairs[0].agent
    start, end, onset = 0.0, sample.end, sample.onset

    figure = plt.figure(
        figsize=(PAPER_WIDTH_IN, PAPER_HEIGHT_IN), constrained_layout=True
    )
    axis = figure.add_subplot(111)

    if onset > start:
        axis.axvspan(start, onset, color="#ececec", linewidth=0, zorder=0)

    colors = [plt.get_cmap("tab10")(index % 10) for index in range(len(pairs))]

    for pair, color in zip(pairs, colors):
        for episode in (pair.agent, pair.baseline):
            axis.step(
                episode.times,
                episode.accuracies,
                where="post",
                color=color,
                linestyle="-" if episode.arm == "agent" else "--",
                linewidth=0.6,
                alpha=0.7,
                zorder=2,
            )

    all_episodes = [episode for pair in pairs for episode in (pair.agent, pair.baseline)]
    union, _ = forward_fill_steps(
        [episode.times for episode in all_episodes],
        [episode.accuracies for episode in all_episodes],
    )
    for arm, mean_color, style in (
        ("agent", AGENT_MEAN_COLOR, "-"),
        ("baseline", BASELINE_MEAN_COLOR, "--"),
    ):
        episodes = [getattr(pair, arm) for pair in pairs]
        arm_union, aligned = forward_fill_steps(
            [episode.times for episode in episodes],
            [episode.accuracies for episode in episodes],
        )
        indices = np.maximum(np.searchsorted(arm_union, union, side="right") - 1, 0)
        values = aligned[:, indices]
        axis.fill_between(
            union,
            np.min(values, axis=0),
            np.max(values, axis=0),
            step="post",
            color=mean_color,
            alpha=0.12,
            linewidth=0,
            zorder=1,
        )
        axis.step(
            union,
            np.mean(values, axis=0),
            where="post",
            color=mean_color,
            linestyle=style,
            linewidth=1.5,
            zorder=3,
        )

    for pair, color in zip(pairs, colors):
        for decision in pair.agent.decisions:
            axis.axvline(decision, color=color, alpha=0.30, linewidth=0.6, zorder=2)
        for start_retrain, finish in pair.agent.retraining_periods:
            axis.axvspan(start_retrain, finish, color=color, alpha=0.10, linewidth=0)

    axis.hlines(sample.tau, onset, end, color="#555555", linestyle=":", linewidth=1.0)
    axis.axvline(onset, color="#222222", linewidth=0.9)
    axis.axvline(end, color="#222222", linestyle=":", linewidth=0.9)

    gap = 0.02 * (end - start)
    axis.text(
        onset - gap,
        1.01,
        "warm-up",
        transform=axis.get_xaxis_transform(),
        ha="right",
        va="bottom",
        fontsize=6,
    )
    axis.text(
        onset + gap,
        1.01,
        "drift onset",
        transform=axis.get_xaxis_transform(),
        ha="left",
        va="bottom",
        fontsize=6,
    )
    axis.text(
        end,
        1.01,
        "horizon",
        transform=axis.get_xaxis_transform(),
        ha="right",
        va="bottom",
        fontsize=6,
    )

    axis.set_xlim(start, end)
    axis.set_ylim(0, 1)
    axis.set_xlabel("Simulated time (s)", fontsize=7)
    axis.set_ylabel("Accuracy", fontsize=7)
    axis.tick_params(labelsize=6.5)
    axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    axis.grid(True, alpha=0.2)

    tau_label = f"{sample.tau:g}"
    handles = [
        Line2D([0], [0], color=AGENT_MEAN_COLOR, linewidth=1.5, label="ReaQI (mean)"),
        Line2D(
            [0],
            [0],
            color=BASELINE_MEAN_COLOR,
            linewidth=1.5,
            linestyle="--",
            label="Baseline (mean)",
        ),
        Line2D(
            [0],
            [0],
            color="#555555",
            linestyle=":",
            linewidth=1.0,
            label=f"τ = {tau_label}",
        ),
        Patch(facecolor="#777777", alpha=0.25, label="Retraining"),
    ]
    figure.legend(
        handles=handles,
        loc="outside lower center",
        ncol=4,
        frameon=False,
        fontsize=6,
        handlelength=1.4,
        columnspacing=0.9,
    )
    return figure


def plot_trajectory_scenario(scenario_dir: str | Path, output: str | Path) -> Path:
    """Load a scenario, render the compact trajectory figure, and save it."""
    pairs = load_scenario(scenario_dir)
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure = build_trajectory_figure(pairs)
    try:
        figure.savefig(destination, bbox_inches="tight")
    finally:
        plt.close(figure)
    return destination


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compact paper figure: paired accuracy trajectories only"
    )
    parser.add_argument("--scenario-dir", required=True, help="Directory containing seed_*")
    parser.add_argument("--output", required=True, help="Destination file (.pdf)")
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        destination = plot_trajectory_scenario(args.scenario_dir, args.output)
    except ScenarioError as error:
        parser.error(str(error))
    print(destination)


if __name__ == "__main__":
    main()
