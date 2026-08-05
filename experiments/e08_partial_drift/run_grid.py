"""Run the E08 grid: quorum x window x drift rhythm.

Scenario: motion_blur:1, 5 of 10 clients drift on a staggered schedule.
Retraining keeps the E07 policy: quorum at the server, global retraining,
per-client current-distribution datasets.
"""

from __future__ import annotations

import argparse
import itertools
import sys
from dataclasses import replace
from pathlib import Path

_BASE = Path(__file__).resolve().parent.parent.parent
for _path in (str(_BASE), str(_BASE / "src")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from experiments.e07_drift_agent.episode import DriftEpisodeConfig
from experiments.e07_drift_agent.run_matrix import run_matrix
from experiments.shared.registry import temporary_output_path

NUM_DRIFTED = 5


def onset_ticks_for(rhythm: int) -> dict[int, int]:
    """Drift onset ticks for clients 0..4 at the given rhythm."""
    return {client: 1 + client * rhythm for client in range(NUM_DRIFTED)}


def build_grid(
    *,
    seeds: tuple[int, ...] = (42,),
    quorums: tuple[float, ...] = (0.2, 0.3, 0.5),
    windows: tuple[int, ...] = (1, 2, 5),
    rhythms: tuple[int, ...] = (1, 3, 5),
) -> list[DriftEpisodeConfig]:
    """Materialize the 27-config grid (3 x 3 x 3) for one seed."""
    return [
        replace(
            DriftEpisodeConfig(
                corruption="motion_blur",
                severity=1,
                drifted_client_ids=tuple(range(NUM_DRIFTED)),
                drift_onset_ticks=onset_ticks_for(rhythm),
                seed=seed,
                trigger_threshold=quorum,
                trigger_window_ticks=window,
                retrain_rounds=1,
            )
        )
        for seed in seeds
        for quorum, window, rhythm in itertools.product(quorums, windows, rhythms)
    ]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the E08 partial-drift grid")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--quorums", nargs="+", type=float, default=[0.2, 0.3, 0.5])
    parser.add_argument("--windows", nargs="+", type=int, default=[1, 2, 5])
    parser.add_argument("--rhythms", nargs="+", type=int, default=[1, 3, 5])
    parser.add_argument(
        "--output-dir",
        default=str(temporary_output_path("e08-partial-drift", "cifar-10") / "grid-v1"),
    )
    args = parser.parse_args(argv)

    from src.utils.corruptions import apply_corruption

    configs = build_grid(
        seeds=tuple(args.seeds),
        quorums=tuple(args.quorums),
        windows=tuple(args.windows),
        rhythms=tuple(args.rhythms),
    )
    run_matrix(configs, output_dir=args.output_dir, corruption_fn=apply_corruption)
    print(f"E08 grid done: {len(configs)} pairs -> {args.output_dir}")


if __name__ == "__main__":
    main()
