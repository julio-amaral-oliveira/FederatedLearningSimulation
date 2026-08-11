"""Explicit synchronous drift-retraining experiment.

The production phase never trains by itself: clients only provide unlabeled,
corrupted batches to ``DriftMonitor``.  A single alarm can start a bounded
block of synchronous FL rounds.  Corruptions are deliberately injected so
this module can be developed independently of their implementation.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import os
import random
import sys
from dataclasses import dataclass, replace
from typing import Callable

import numpy as np
import torch

_BASE = os.path.dirname(__file__)
_ROOT = os.path.join(_BASE, "..", "..")
_SRC = os.path.join(_ROOT, "src")
for _path in (_ROOT, _SRC, os.path.join(_SRC, "synchronous")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from synchronous.constants import SPEED_PROFILES
from experiments.shared.registry import temporary_output_path
from experiments.e07_drift_agent.drift_schedule import (
    build_mixed_dataset,
    build_retrain_datasets,
    corrupt_count,
    is_client_drifted_at_tick,
    ramp_fraction,
)

CorruptionFn = Callable[[torch.Tensor, str, int], torch.Tensor]
DEFAULT_PRODUCTION_HORIZON_SECONDS = 400.0
CORRUPTED_TEST_SEED_OFFSET = 9_999
DEFAULT_CLIENT_SPEED_PROFILE = "uniform"
CLIENT_SPEED_PROFILES = tuple(SPEED_PROFILES)


def corrupted_test_seed(base_seed: int) -> int:
    """Derive the shared corruption seed for test-set evaluation."""
    return base_seed + CORRUPTED_TEST_SEED_OFFSET


@dataclass(frozen=True)
class DriftEpisodeConfig:
    dataset: str = "cifar10"
    num_clients: int = 10
    initial_rounds: int = 20
    retrain_rounds: int = 5
    warmup_ticks: int = 20
    monitor_tick_seconds: float = 10.0
    monitor_ticks: int = 25
    trigger_threshold: float = 0.30
    trigger_window_ticks: int = 2
    detector_alpha: float = 0.002
    detector_T: int = 5
    local_epochs: int = 1
    batch_size: int = 32
    timeout_percentile: int = 75
    client_speed_profile: str = DEFAULT_CLIENT_SPEED_PROFILE
    drifted_client_ids: tuple[int, ...] | None = None
    drift_onset_ticks: dict[int, int] | None = None
    drift_ramp_ticks: int | None = None
    max_train_samples_per_client: int | None = None
    tau: float = 0.50
    corruption: str = "gaussian_noise"
    severity: int = 3
    seed: int = 42
    output_dir: str = str(
        temporary_output_path("e07-drift-agent", "cifar-10") / "episode"
    )
    baseline: bool = False
    production_horizon_seconds: float | None = None
    end_time_seconds: float | None = None


def _call_corruption(
    corruption_fn: CorruptionFn,
    x: np.ndarray | torch.Tensor,
    config: DriftEpisodeConfig,
    *,
    seed: int,
) -> np.ndarray:
    """Apply the collaborator-owned tensor transform without mutating input."""
    source = torch.as_tensor(x).detach().clone()
    corrupted = corruption_fn(source, config.corruption, config.severity, seed=seed)
    if not isinstance(corrupted, torch.Tensor):
        raise TypeError("corruption_fn must return torch.Tensor")
    if corrupted.shape != source.shape:
        raise ValueError("corruption_fn must preserve the input shape")
    return corrupted.detach().cpu().numpy().astype(np.asarray(x).dtype, copy=False)


def _sample_batch(x: np.ndarray, batch_size: int, rng: np.random.Generator) -> np.ndarray:
    if len(x) == 0:
        raise ValueError("cannot monitor a client with an empty dataset")
    count = min(batch_size, len(x))
    indices = rng.choice(len(x), size=count, replace=False)
    return x[indices]


def _evaluation_entry(server, dataset, stage: str) -> dict:
    loss, accuracy = server.evaluate_dataset(dataset)
    return {
        "time": float(server.virtual_time),
        "loss": float(loss),
        "accuracy": float(accuracy),
        "stage": stage,
    }


def _record_evaluation(
    destination: list[dict], server, dataset, stage: str) -> None:
    destination.append(_evaluation_entry(server, dataset, stage))


MIXTURE_PERMUTATION_SEED_OFFSET = 12_345


def _mixed_evaluation_test(
    clean_test: tuple[np.ndarray, np.ndarray],
    corrupted_test: tuple[np.ndarray, np.ndarray],
    permutation: np.ndarray | None,
    fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Mix the clean and corrupted tests at a fraction.

    Without a permutation, the fully corrupted test is returned, which
    reproduces the E07 evaluation.
    """
    if permutation is None:
        return corrupted_test
    count = corrupt_count(len(clean_test[0]), fraction)
    x = clean_test[0].copy()
    x[permutation[:count]] = corrupted_test[0][permutation[:count]]
    return x, clean_test[1]


def _current_evaluation_test(
    server,
    config: DriftEpisodeConfig,
    production_start: float,
    clean_test: tuple[np.ndarray, np.ndarray],
    corrupted_test: tuple[np.ndarray, np.ndarray],
    permutation: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the evaluation test for the current production fraction."""
    if config.drift_ramp_ticks is None:
        return corrupted_test
    fraction = ramp_fraction(
        float(server.virtual_time) - production_start, config
    )
    return _mixed_evaluation_test(
        clean_test, corrupted_test, permutation, fraction
    )


def _run_training_rounds(server, rounds: int, clean_test) -> list[dict]:
    """Run initial training one round at a time, recording clean evaluations.

    Each entry stores the clean-test accuracy right after the round so the
    plot can show the training period, not only the production phase.  The
    times are absolute virtual times and always precede production start.
    """
    history: list[dict] = []
    for _ in range(rounds):
        server.run_one_round(record_default_metrics=False)
        history.append(_evaluation_entry(server, clean_test, "train_round"))
    return history


def _monitor_batches(
    client_datasets: list[tuple[np.ndarray, np.ndarray]],
    config: DriftEpisodeConfig,
    corruption_fn: CorruptionFn | None,
    rng: np.random.Generator,
    *,
    tick: int,
    virtual_time: float | None = None,
) -> dict[str, torch.Tensor]:
    """Sample per-client monitor batches, corrupting drifted clients.

    ``tick`` is the absolute monitor tick (warm-up included).  The drift
    schedule is evaluated in the production-relative frame so that onset 0
    lands on the first corrupted production tick; the corruption seed stays
    on the absolute tick to keep the E07 sequence byte-identical.  With a
    configured ramp, the batch corrupts only ``round(fraction x batch)``
    images chosen through ``rng``, where the fraction follows the virtual
    time elapsed since production start.
    """
    batches: dict[str, torch.Tensor] = {}
    schedule_tick = tick - config.warmup_ticks
    fraction = None
    if config.drift_ramp_ticks is not None:
        production_seconds = (
            float(virtual_time - config.warmup_ticks * config.monitor_tick_seconds)
            if virtual_time is not None
            else schedule_tick * config.monitor_tick_seconds
        )
        fraction = ramp_fraction(production_seconds, config)
    for position, (x, _y) in enumerate(client_datasets):
        batch = _sample_batch(x, config.batch_size, rng)
        if corruption_fn is not None and is_client_drifted_at_tick(
            position, schedule_tick, config
        ):
            corruption_seed = config.seed + 10_000 * tick + position
            if fraction is None:
                batch = _call_corruption(
                    corruption_fn, batch, config, seed=corruption_seed
                )
            else:
                count = corrupt_count(len(batch), fraction)
                if count:
                    indices = rng.choice(len(batch), size=count, replace=False)
                    corrupted = _call_corruption(
                        corruption_fn, batch, config, seed=corruption_seed
                    )
                    batch[indices] = corrupted[indices]
        # The monitor contract is deliberately x-only.
        batches[str(position)] = torch.from_numpy(batch)
    return batches


def _run_retraining(
    result: dict,
    server,
    corrupted_client_datasets: list[tuple[np.ndarray, np.ndarray]],
    rounds: int,
    config: DriftEpisodeConfig,
    production_start_time: float,
    *,
    current_test_fn: Callable[[], tuple[np.ndarray, np.ndarray]],
    client_permutations: list[np.ndarray] | None = None,
) -> None:
    clean_client_datasets = [
        (np.asarray(client.dataset[0]).copy(), np.asarray(client.dataset[1]).copy())
        for client in server.clients
    ]
    if config.drift_ramp_ticks is None:
        current_tick = int(
            (float(server.virtual_time) - production_start_time)
            // config.monitor_tick_seconds
        )
        retrain_datasets = build_retrain_datasets(
            clean_client_datasets,
            corrupted_client_datasets,
            current_tick,
            config=config,
        )
    else:
        fraction = ramp_fraction(
            float(server.virtual_time) - production_start_time, config
        )
        permutations = client_permutations or [
            np.arange(len(x)) for x, _y in clean_client_datasets
        ]
        retrain_datasets = [
            build_mixed_dataset(clean, corrupted, fraction, permutation)
            for (clean, corrupted), permutation in zip(
                zip(clean_client_datasets, corrupted_client_datasets), permutations
            )
        ]
    for client, dataset in zip(server.clients, retrain_datasets):
        client.dataset = dataset
        if hasattr(client, "reset_optimizer"):
            client.reset_optimizer()
    for _ in range(rounds):
        event = server.run_one_round(record_default_metrics=False)
        result["retrain_round_events"].append(event)
        _record_evaluation(
            result["corrupted_accuracy_history"], server, current_test_fn(),
            "retrain_round",
        )


def _experiment_config(config: DriftEpisodeConfig) -> dict:
    """Return the effective experiment inputs that affect an episode."""
    return {
        "dataset": config.dataset,
        "num_clients": config.num_clients,
        "initial_rounds": config.initial_rounds,
        "retrain_rounds": config.retrain_rounds,
        "warmup_ticks": config.warmup_ticks,
        "monitor_tick_seconds": config.monitor_tick_seconds,
        "monitor_ticks": config.monitor_ticks,
        "local_epochs": config.local_epochs,
        "batch_size": config.batch_size,
        "timeout_percentile": config.timeout_percentile,
        "client_speed_profile": config.client_speed_profile,
        "drifted_client_ids": config.drifted_client_ids,
        "drift_onset_ticks": config.drift_onset_ticks,
        "drift_ramp_ticks": config.drift_ramp_ticks,
        "client_speed_tiers": [
            list(tier) for tier in _speed_tiers_for_profile(config.client_speed_profile)
        ],
        "max_train_samples_per_client": config.max_train_samples_per_client,
        "tau": config.tau,
        "corruption": config.corruption,
        "severity": config.severity,
        "seed": config.seed,
        "baseline": config.baseline,
        "production_horizon_seconds": config.production_horizon_seconds,
    }


def _speed_tiers_for_profile(
    profile: str,
) -> tuple[tuple[str, int, int, float], ...]:
    try:
        return tuple(SPEED_PROFILES[profile])
    except KeyError as error:
        choices = ", ".join(CLIENT_SPEED_PROFILES)
        raise ValueError(
            f"unknown client speed profile {profile!r}; expected one of: {choices}"
        ) from error


def _detector_config(config: DriftEpisodeConfig) -> dict:
    """Return the uncertainty detector settings used for this episode."""
    return {
        "detector_kind": "udd",
        "detector_alpha": config.detector_alpha,
        "detector_T": config.detector_T,
        "trigger_threshold": config.trigger_threshold,
        "trigger_window_ticks": config.trigger_window_ticks,
    }


def _effective_device(server) -> str:
    """Report the actual device of the model used by an injected or real server."""
    model = getattr(server, "global_model", None)
    if model is not None:
        parameter = next(model.parameters(), None)
        if parameter is not None:
            return str(parameter.device)
    return "cpu"


def _runtime_metadata(server) -> dict:
    """Capture environment facts required to reproduce a result."""
    cudnn = getattr(torch.backends, "cudnn", None)
    return {
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        "effective_device": _effective_device(server),
        "deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": bool(getattr(cudnn, "deterministic", False)),
        "cudnn_benchmark": bool(getattr(cudnn, "benchmark", False)),
        "clean_checkpoint_digest": None,
    }


def _new_result(config: DriftEpisodeConfig, server) -> dict:
    return {
        "schema_version": 3,
        "experiment_config": _experiment_config(config),
        "detector_config": _detector_config(config),
        "runtime": _runtime_metadata(server),
        "metadata": {
            "dataset": config.dataset,
            "corruption": config.corruption,
            "severity": config.severity,
            "seed": config.seed,
            "tau": config.tau,
            "baseline": config.baseline,
            "production_start_time": None,
            "warmup_completed_time": None,
            "preproduction_server_state": None,
            "end_time_seconds": None,
            "retrain_rounds_configured": config.retrain_rounds,
            "monitor_tick_seconds": config.monitor_tick_seconds,
            "production_horizon_seconds": config.production_horizon_seconds,
        },
        "corrupted_accuracy_history": [],
        "clean_evaluations": [],
        "clean_training_history": [],
        "drift_events": [],
        "retrain_decisions": [],
        "counterfactual_triggers": [],
        "retrain_round_events": [],
        "metrics": {
            "downtime_seconds": 0.0,
            "detection_delay_seconds": None,
            "retraining_duration_seconds": None,
            "time_to_recovery_seconds": None,
            "clean_retention_delta": None,
            "corrupted_accuracy_gain": None,
        },
    }


def _state_digest(state: object) -> str:
    """Return a compact, JSON-safe identifier for a reproducibility state."""
    return hashlib.sha256(repr(state).encode("utf-8")).hexdigest()


def _weights_digest(weights: list[torch.Tensor]) -> str:
    """Hash model weights with shape and dtype boundaries for stable audit IDs."""
    digest = hashlib.sha256()
    for index, weight in enumerate(weights):
        tensor = weight.detach().cpu().contiguous()
        digest.update(f"{index}:{tensor.dtype}:{tuple(tensor.shape)}\n".encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _global_weights_digest(server) -> str | None:
    if not hasattr(server, "global_model"):
        return None
    from utils.models import get_model_weights

    return _weights_digest(get_model_weights(server.global_model))


def _preproduction_server_state(server) -> dict[str, float | int | str | None]:
    """Record the restored state immediately before corrupted production."""
    server_rng_state = server.rng.getstate() if hasattr(server, "rng") else None
    return {
        "virtual_time": float(server.virtual_time),
        "next_round_index": getattr(server, "next_round_index", None),
        "server_rng_state_digest": (
            _state_digest(server_rng_state) if server_rng_state is not None else None
        ),
    }


def _build_server(config: DriftEpisodeConfig):
    """Build the real IID CIFAR-style server lazily, keeping unit tests light."""
    from synchronous.client import Client
    from synchronous.constants import (
        MAX_CONNECTION_TIME,
        MIN_CONNECTION_TIME,
        SPEED_TIER_SEED,
    )
    from synchronous.monte_carlo import get_percentiles_timeout
    from synchronous.server import Server
    from utils.data_loader import get_dataset_info, load_dataset
    from utils.data_split import split_iid_data
    from utils.experiment_runner import assign_speed_tiers

    dataset_info = get_dataset_info(config.dataset)
    training_data, testing_data = load_dataset(config.dataset)
    client_data = split_iid_data(training_data, config.num_clients, dataset_info["num_classes"])
    if config.max_train_samples_per_client:
        client_data = [
            (x[: config.max_train_samples_per_client], y[: config.max_train_samples_per_client])
            for x, y in client_data
        ]
    speed_tiers = _speed_tiers_for_profile(config.client_speed_profile)
    timeout = get_percentiles_timeout(
        [config.timeout_percentile],
        MIN_CONNECTION_TIME,
        MAX_CONNECTION_TIME,
        speed_tiers,
    )[0]
    speeds = assign_speed_tiers(config.num_clients, speed_tiers, SPEED_TIER_SEED)
    clients = [
        Client(dataset, index + 1, (speed[1], speed[2]), speed[0])
        for index, (dataset, speed) in enumerate(zip(client_data, speeds))
    ]
    server = Server(
        clients,
        config.num_clients,
        config.initial_rounds,
        timeout,
        config.local_epochs,
        config.batch_size,
        testing_data,
        dataset_info["model"],
    )
    server.setup_clients()
    return server


def _make_monitor(server, config: DriftEpisodeConfig):
    from src.orchestrator.orchestrator import DriftMonitor

    return DriftMonitor(
        server.global_model,
        alpha=config.detector_alpha,
        T=config.detector_T,
        trigger_threshold=config.trigger_threshold,
        window_ticks=config.trigger_window_ticks,
    )


def _mps_rng_api():
    backend = getattr(torch.backends, "mps", None)
    mps = getattr(torch, "mps", None)
    if (
        backend is not None
        and backend.is_available()
        and mps is not None
        and hasattr(mps, "get_rng_state")
        and hasattr(mps, "set_rng_state")
    ):
        return mps
    return None


def capture_random_state() -> dict:
    """Capture every active random generator used by a paired episode."""
    mps = _mps_rng_api()
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "torch_mps": mps.get_rng_state() if mps is not None else None,
    }


def restore_random_state(state: dict) -> None:
    """Restore a state captured by :func:`capture_random_state`."""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if state["torch_cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    mps = _mps_rng_api()
    if state["torch_mps"] is not None and mps is not None:
        mps.set_rng_state(state["torch_mps"])


def _snapshot_clean_checkpoint(server) -> dict:
    """Capture the post-warm-start state used by both comparison arms."""
    checkpoint = {
        "datasets": [(x.copy(), y.copy()) for x, y in (client.dataset for client in server.clients)],
        "virtual_time": float(server.virtual_time),
        "next_round_index": getattr(server, "next_round_index", None),
        "server_rng": copy.deepcopy(server.rng.getstate()) if hasattr(server, "rng") else None,
        "random_state": capture_random_state(),
    }
    if hasattr(server, "global_model"):
        from utils.models import get_model_weights

        checkpoint["global_weights"] = get_model_weights(server.global_model)
        checkpoint["clean_checkpoint_digest"] = _weights_digest(checkpoint["global_weights"])
    return checkpoint


def _restore_clean_checkpoint(server, checkpoint: dict) -> None:
    """Restore an independent server to a clean checkpoint without shared state."""
    for client, (x, y) in zip(server.clients, checkpoint["datasets"]):
        client.dataset = (x.copy(), y.copy())
    server.virtual_time = checkpoint["virtual_time"]
    if checkpoint["next_round_index"] is not None:
        server.next_round_index = checkpoint["next_round_index"]
    if checkpoint["server_rng"] is not None:
        server.rng.setstate(copy.deepcopy(checkpoint["server_rng"]))
    restore_random_state(checkpoint["random_state"])
    if "global_weights" in checkpoint:
        from utils.models import set_model_weights

        set_model_weights(server.global_model, checkpoint["global_weights"])
        for client in server.clients:
            if getattr(client, "local_model", None) is not None:
                client.set_model_weights(checkpoint["global_weights"])


def _finish_result(result: dict, server, clean_test, config: DriftEpisodeConfig) -> dict:
    from experiments.shared.comparison_core import compute_downtime

    _record_evaluation(result["corrupted_accuracy_history"], server, result["_corrupted_test"], "episode_end")
    _record_evaluation(result["clean_evaluations"], server, clean_test, "final")
    result["drift_events"] = list(result["_monitor"].drift_events)
    result["retrain_decisions"] = list(result["_monitor"].retrain_decisions)
    result["counterfactual_triggers"] = list(result["_monitor"].counterfactual_triggers)
    result["tick_history"] = copy.deepcopy(getattr(result["_monitor"], "tick_history", []))
    end_time = float(server.virtual_time)
    result["metadata"]["end_time_seconds"] = end_time
    result["metrics"]["downtime_seconds"] = compute_downtime(
        result["corrupted_accuracy_history"], config.tau, end_time=end_time
    )
    clean_pre_drift = next(
        entry for entry in result["clean_evaluations"] if entry["stage"] == "pre_drift"
    )
    clean_final = next(
        entry for entry in reversed(result["clean_evaluations"]) if entry["stage"] == "final"
    )
    corrupted_onset = next(
        entry for entry in result["corrupted_accuracy_history"] if entry["stage"] == "drift_onset"
    )
    corrupted_final = next(
        entry
        for entry in reversed(result["corrupted_accuracy_history"])
        if entry["stage"] == "episode_end"
    )
    result["metrics"]["clean_retention_delta"] = (
        float(clean_final["accuracy"]) - float(clean_pre_drift["accuracy"])
    )
    result["metrics"]["corrupted_accuracy_gain"] = (
        float(corrupted_final["accuracy"]) - float(corrupted_onset["accuracy"])
    )
    if result["retrain_decisions"]:
        decision_time = float(result["retrain_decisions"][0]["time"])
        result["metrics"]["detection_delay_seconds"] = (
            decision_time - float(result["metadata"]["production_start_time"])
        )
        result["metrics"]["retraining_duration_seconds"] = (
            float(result["retrain_round_events"][-1]["completed_time"]) - decision_time
        )
        recovery_time = next(
            (
                float(entry["time"])
                for entry in result["corrupted_accuracy_history"]
                if float(entry["time"]) > decision_time
                and float(entry["accuracy"]) >= config.tau
            ),
            None,
        )
        if recovery_time is not None:
            result["metrics"]["time_to_recovery_seconds"] = recovery_time - decision_time
    if config.drift_ramp_ticks is not None:
        if result["retrain_decisions"]:
            decision_time = float(result["retrain_decisions"][0]["time"])
            result["metrics"]["fraction_at_trigger"] = ramp_fraction(
                decision_time - float(result["metadata"]["production_start_time"]),
                config,
            )
        crossing = next(
            (
                float(entry["time"])
                for entry in result["corrupted_accuracy_history"]
                if float(entry["accuracy"]) < config.tau
            ),
            None,
        )
        if crossing is not None:
            result["metrics"]["fraction_at_tau_crossing"] = ramp_fraction(
                crossing - float(result["metadata"]["production_start_time"]),
                config,
            )
    del result["_monitor"]
    del result["_corrupted_test"]
    del result["_mixture_permutation"]
    del result["_client_permutations"]
    return result


def run_drift_episode(
    config: DriftEpisodeConfig,
    *,
    corruption_fn: CorruptionFn,
    server=None,
    monitor=None,
    _initial_training_complete: bool = False,
    _clean_checkpoint_digest: str | None = None,
    _clean_training_history: list[dict] | None = None,
) -> dict:
    """Run one clean-warmup/corrupted-production episode.

    ``server`` and ``monitor`` are injection points for fast controller tests.
    The detector receives only tensors of inputs; labels never leave datasets.
    """
    if config.retrain_rounds < 1:
        raise ValueError("retrain_rounds must be at least 1")
    if config.warmup_ticks < 20:
        raise ValueError("warmup_ticks must be at least 20 for ADWIN")
    if config.monitor_tick_seconds <= 0:
        raise ValueError("monitor_tick_seconds must be positive")
    if (
        config.production_horizon_seconds is not None
        and config.production_horizon_seconds <= 0
    ):
        raise ValueError("production_horizon_seconds must be positive")
    if config.drift_ramp_ticks is not None and (
        isinstance(config.drift_ramp_ticks, bool) or config.drift_ramp_ticks < 1
    ):
        raise ValueError("drift_ramp_ticks must be None or an integer >= 1")
    if config.drift_ramp_ticks is not None and (
        config.drift_onset_ticks is not None
        or config.drifted_client_ids is not None
    ):
        raise ValueError(
            "drift_ramp_ticks cannot be combined with drift_onset_ticks "
            "or drifted_client_ids"
        )

    server = server or _build_server(config)
    monitor = monitor or _make_monitor(server, config)
    rng = np.random.default_rng(config.seed)
    clean_test = server.testing_data
    clean_client_datasets = [(x.copy(), y.copy()) for x, y in (c.dataset for c in server.clients)]
    corrupted_client_datasets = [
        (_call_corruption(corruption_fn, x, config, seed=config.seed + index), y.copy())
        for index, (x, y) in enumerate(clean_client_datasets)
    ]
    corrupted_test = (
        _call_corruption(
            corruption_fn,
            clean_test[0],
            config,
            seed=corrupted_test_seed(config.seed),
        ),
        clean_test[1].copy(),
    )
    result = _new_result(config, server)
    result["_monitor"] = monitor
    result["_corrupted_test"] = corrupted_test
    mixture_permutation = None
    client_permutations = None
    if config.drift_ramp_ticks is not None:
        permutation_rng = np.random.default_rng(
            config.seed + MIXTURE_PERMUTATION_SEED_OFFSET
        )
        mixture_permutation = permutation_rng.permutation(len(clean_test[0]))
        client_permutations = [
            permutation_rng.permutation(len(x)) for x, _y in clean_client_datasets
        ]
    result["_mixture_permutation"] = mixture_permutation
    result["_client_permutations"] = client_permutations

    if not _initial_training_complete:
        result["clean_training_history"] = _run_training_rounds(
            server, config.initial_rounds, clean_test
        )
    elif _clean_training_history is not None:
        result["clean_training_history"] = copy.deepcopy(_clean_training_history)
    result["runtime"]["clean_checkpoint_digest"] = (
        _clean_checkpoint_digest
        if _clean_checkpoint_digest is not None
        else _global_weights_digest(server)
    )
    _record_evaluation(result["clean_evaluations"], server, clean_test, "pre_drift")

    # ADWIN sees stable, unlabeled clean inputs before production starts.
    for tick in range(config.warmup_ticks):
        monitor.observe_tick(
            _monitor_batches(clean_client_datasets, config, None, rng, tick=tick),
            virtual_time=float(server.virtual_time),
            warmup=True,
        )
    production_start = float(server.virtual_time)
    result["metadata"]["warmup_completed_time"] = production_start
    result["metadata"]["production_start_time"] = production_start
    result["metadata"]["preproduction_server_state"] = _preproduction_server_state(server)
    _record_evaluation(
        result["corrupted_accuracy_history"],
        server,
        _current_evaluation_test(
            server, config, production_start, clean_test, corrupted_test,
            mixture_permutation,
        ),
        "drift_onset",
    )

    tick = 0
    retrained = False
    target_time = (
        production_start + config.production_horizon_seconds
        if config.production_horizon_seconds is not None
        else config.end_time_seconds
    )
    while True:
        if target_time is None:
            if tick >= config.monitor_ticks or retrained:
                break
            advance = config.monitor_tick_seconds
        else:
            remaining = float(target_time) - float(server.virtual_time)
            if remaining <= 0:
                break
            advance = min(config.monitor_tick_seconds, remaining)

        server.virtual_time += advance
        outcome = monitor.observe_tick(
            _monitor_batches(
                clean_client_datasets, config, corruption_fn, rng,
                tick=config.warmup_ticks + tick,
                virtual_time=float(server.virtual_time),
            ),
            virtual_time=float(server.virtual_time),
            record_action=not config.baseline,
        )
        _record_evaluation(
            result["corrupted_accuracy_history"],
            server,
            _current_evaluation_test(
                server, config, production_start, clean_test, corrupted_test,
                mixture_permutation,
            ),
            "monitor_tick",
        )
        tick += 1
        if outcome.should_retrain and not config.baseline and not retrained:
            if config.production_horizon_seconds is not None:
                required_seconds = config.retrain_rounds * server.timeout
                remaining_seconds = float(target_time) - float(server.virtual_time)
                if remaining_seconds < required_seconds:
                    raise RuntimeError(
                        "retraining budget cannot fit within the fixed production horizon: "
                        f"{remaining_seconds:g}s remaining, {required_seconds:g}s required"
                    )
            _run_retraining(
                result,
                server,
                corrupted_client_datasets,
                config.retrain_rounds,
                config,
                production_start,
                current_test_fn=lambda: _current_evaluation_test(
                    server, config, production_start, clean_test,
                    corrupted_test, mixture_permutation,
                ),
                client_permutations=client_permutations,
            )
            retrained = True
            if target_time is None:
                break

    if target_time is not None and float(server.virtual_time) < float(target_time):
        raise RuntimeError("episode did not reach its configured end_time_seconds")
    return _finish_result(result, server, clean_test, config)


def run_drift_comparison(
    config: DriftEpisodeConfig,
    *,
    corruption_fn: CorruptionFn,
    server_factory=None,
    monitor_factory=None,
) -> dict[str, dict]:
    """Run paired agent and baseline episodes from reproducible clean starts."""
    def make_server():
        return server_factory() if server_factory is not None else _build_server(config)

    def seed_everything():
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)

    seed_everything()
    checkpoint_source = make_server()
    training_history = _run_training_rounds(
        checkpoint_source,
        config.initial_rounds,
        checkpoint_source.testing_data,
    )
    checkpoint = _snapshot_clean_checkpoint(checkpoint_source)

    agent_server = make_server()
    _restore_clean_checkpoint(agent_server, checkpoint)
    agent_monitor = monitor_factory(agent_server, False) if monitor_factory else None
    agent = run_drift_episode(
        replace(config, baseline=False, end_time_seconds=None),
        corruption_fn=corruption_fn,
        server=agent_server,
        monitor=agent_monitor,
        _initial_training_complete=True,
        _clean_checkpoint_digest=checkpoint.get("clean_checkpoint_digest"),
        _clean_training_history=training_history,
    )
    baseline_server = make_server()
    _restore_clean_checkpoint(baseline_server, checkpoint)
    baseline_monitor = monitor_factory(baseline_server, True) if monitor_factory else None
    baseline = run_drift_episode(
        replace(config, baseline=True, end_time_seconds=agent["metadata"]["end_time_seconds"]),
        corruption_fn=corruption_fn,
        server=baseline_server,
        monitor=baseline_monitor,
        _initial_training_complete=True,
        _clean_checkpoint_digest=checkpoint.get("clean_checkpoint_digest"),
        _clean_training_history=training_history,
    )
    from experiments.shared.drift_results import validate_pair

    validate_pair(agent, baseline)
    return {"agent": agent, "baseline": baseline}


def save_drift_result(result: dict, output_dir: str, filename: str = "smoke_drift.json") -> str:
    """Persist a complete schema-v3 result only after metrics are computed."""
    from experiments.shared.result_io import atomic_write_json

    path = os.path.join(output_dir, filename)
    atomic_write_json(path, result)
    return path


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Synchronous corrupted-data drift episode")
    parser.add_argument("--dataset", default="cifar10")
    parser.add_argument("--corruption", default="gaussian_noise")
    parser.add_argument("--severity", type=int, default=3)
    parser.add_argument(
        "--output-dir",
        default=str(temporary_output_path("e07-drift-agent", "cifar-10") / "episode"),
    )
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
    args = parser.parse_args(argv)
    try:
        # Deferred on purpose: the corruption module belongs to a parallel task.
        from src.utils.corruptions import apply_corruption
    except ImportError as error:
        raise SystemExit("src.utils.corruptions.apply_corruption is required to run this experiment") from error
    config = DriftEpisodeConfig(
        dataset=args.dataset,
        corruption=args.corruption,
        severity=args.severity,
        output_dir=args.output_dir,
        client_speed_profile=args.client_speed_profile,
        production_horizon_seconds=args.production_horizon_seconds,
    )
    results = run_drift_comparison(config, corruption_fn=apply_corruption)
    from experiments.shared.result_io import save_validated_pair

    agent_path, baseline_path = save_validated_pair(
        config.output_dir,
        results["agent"],
        results["baseline"],
    )
    print(f"agent: {agent_path}")
    print(f"baseline: {baseline_path}")


if __name__ == "__main__":
    main()
