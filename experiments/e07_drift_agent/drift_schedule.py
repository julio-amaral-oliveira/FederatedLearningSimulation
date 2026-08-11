"""Per-client drift schedule helpers.

A schedule says which clients drift and on which production tick their
corruption starts.  Ticks are production-relative: 0 is the first
corrupted production tick and the warm-up phase does not count.  ``None``
fields mean "every client drifts from the first production tick", which
reproduces the E07 behavior exactly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from experiments.e07_drift_agent.episode import DriftEpisodeConfig


def drifted_clients(config: DriftEpisodeConfig) -> tuple[int, ...]:
    """Return the client indices that ever drift under this schedule."""
    if config.drifted_client_ids is None:
        return tuple(range(config.num_clients))
    return tuple(config.drifted_client_ids)


def onset_tick(client_index: int, config: DriftEpisodeConfig) -> int:
    """Return the production tick at which ``client_index`` starts drifting.

    Onsets are production-relative: 0 is the first corrupted production
    tick; the warm-up phase does not count.  With no schedule configured,
    every drifted client starts at production tick 0.
    """
    if config.drift_onset_ticks is None:
        return 0
    return config.drift_onset_ticks.get(client_index, 0)


def is_client_drifted_at_tick(
    client_index: int,
    tick: int,
    config: DriftEpisodeConfig,
) -> bool:
    """Return True when the client's corruption is active at ``tick``.

    ``tick`` must be production-relative (0 = first corrupted production
    tick; warm-up excluded), matching the ``onset_tick`` frame.
    """
    if client_index not in drifted_clients(config):
        return False
    return tick >= onset_tick(client_index, config)


def retrain_dataset_for(
    position: int,
    clean: tuple[Any, Any],
    corrupted: tuple[Any, Any],
    current_tick: int,
    config: DriftEpisodeConfig,
) -> tuple[Any, Any]:
    """Pick the dataset matching the client's current distribution."""
    if is_client_drifted_at_tick(position, current_tick, config):
        return corrupted
    return clean


def build_retrain_datasets(
    clean_client_datasets: list[tuple[Any, Any]],
    corrupted_client_datasets: list[tuple[Any, Any]],
    current_tick: int,
    config: DriftEpisodeConfig,
) -> list[tuple[Any, Any]]:
    """Build the per-client retraining datasets at ``current_tick``."""
    return [
        retrain_dataset_for(position, clean, corrupted, current_tick, config)
        for position, (clean, corrupted) in enumerate(
            zip(clean_client_datasets, corrupted_client_datasets)
        )
    ]


def ramp_fraction(
    virtual_seconds_since_production: float,
    config: DriftEpisodeConfig,
) -> float:
    """Return the mixture fraction at a production-relative virtual time.

    Without a configured ramp, the fraction is always 1, which reproduces
    the E07 all-corrupted regime.  With a ramp, the fraction grows linearly
    from 0 to 1 over ``drift_ramp_ticks`` monitor ticks.
    """
    if config.drift_ramp_ticks is None:
        return 1.0
    duration = config.drift_ramp_ticks * config.monitor_tick_seconds
    return min(1.0, max(0.0, virtual_seconds_since_production / duration))


def corrupt_count(size: int, fraction: float) -> int:
    """Return how many of ``size`` items are corrupted at a fraction."""
    return min(size, round(fraction * size))


def build_mixed_dataset(
    clean: tuple[Any, Any],
    corrupted: tuple[Any, Any],
    fraction: float,
    permutation: np.ndarray,
) -> tuple[Any, Any]:
    """Mix a dataset at a fraction using a deterministic permutation."""
    x = np.asarray(clean[0]).copy()
    y = np.asarray(clean[1]).copy()
    count = corrupt_count(len(x), fraction)
    x[permutation[:count]] = np.asarray(corrupted[0])[permutation[:count]]
    return x, y
