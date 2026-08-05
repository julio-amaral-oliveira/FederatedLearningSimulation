"""Per-client drift schedule helpers.

A schedule says which clients drift and on which production tick their
corruption starts. ``None`` fields mean "every client drifts from the
first production tick", which reproduces the E07 behavior exactly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from experiments.e07_drift_agent.episode import DriftEpisodeConfig


def drifted_clients(config: DriftEpisodeConfig) -> tuple[int, ...]:
    """Return the client indices that ever drift under this schedule."""
    if config.drifted_client_ids is None:
        return tuple(range(config.num_clients))
    return tuple(config.drifted_client_ids)


def onset_tick(client_index: int, config: DriftEpisodeConfig) -> int:
    """Return the production tick at which ``client_index`` starts drifting."""
    if config.drift_onset_ticks is None:
        return 0
    return config.drift_onset_ticks.get(client_index, 0)


def is_client_drifted_at_tick(
    client_index: int,
    tick: int,
    config: DriftEpisodeConfig,
) -> bool:
    """Return True when the client's corruption is active at ``tick``."""
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
