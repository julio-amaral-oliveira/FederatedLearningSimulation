import unittest
import random
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import torch
import torch.nn as nn

from experiments.e07_drift_agent import episode as smoke_drift
from experiments.e07_drift_agent.episode import (
    DriftEpisodeConfig,
    capture_random_state,
    restore_random_state,
    run_drift_comparison,
    run_drift_episode,
)
from experiments.shared.comparison_core import compute_downtime
from src.orchestrator.orchestrator import DriftMonitor


class _FakeClient:
    def __init__(self, client_id):
        self.client_id = client_id
        self.dataset = (
            np.zeros((4, 1, 2, 2), dtype=np.float32),
            np.full(4, client_id, dtype=np.int64),
        )
        self.reset_calls = 0

    def reset_optimizer(self):
        self.reset_calls += 1


class _FakeServer:
    def __init__(
        self,
        round_durations=(2.0, 3.0, 5.0, 7.0, 11.0, 13.0),
        rng_seed=123,
        global_model=None,
        evaluation_accuracies=None,
        timeout=None,
    ):
        self.clients = [_FakeClient(0), _FakeClient(1), _FakeClient(2)]
        self.testing_data = (
            np.zeros((6, 1, 2, 2), dtype=np.float32),
            np.arange(6, dtype=np.int64) % 2,
        )
        self.virtual_time = 0.0
        self.round_durations = tuple(round_durations)
        self.timeout = max(self.round_durations) if timeout is None else timeout
        self.rng = random.Random(rng_seed)
        self.next_round_index = 0
        self.run_round_calls = []
        self.run_one_round_calls = 0
        self.evaluation_states = []
        self.evaluation_accuracies = iter(evaluation_accuracies or [0.25])
        self.last_evaluation_accuracy = 0.25
        if global_model is not None:
            self.global_model = global_model

    def run_rounds(self, count, *, record_default_metrics=False):
        self.run_round_calls.append(count)
        return [self.run_one_round(record_default_metrics=record_default_metrics) for _ in range(count)]

    def run_one_round(self, *, record_default_metrics=False):
        self.run_one_round_calls += 1
        started = self.virtual_time
        duration = self.round_durations[self.next_round_index]
        self.rng.random()  # Model a server that consumes deterministic scheduling RNG.
        self.virtual_time += duration
        self.next_round_index += 1
        return {
            "round": self.next_round_index,
            "started_time": started,
            "completed_time": self.virtual_time,
            "aggregated": True,
        }

    def evaluate_dataset(self, _dataset):
        self.evaluation_states.append(
            {
                "virtual_time": self.virtual_time,
                "next_round_index": self.next_round_index,
                "rng_state": self.rng.getstate(),
            }
        )
        self.last_evaluation_accuracy = next(
            self.evaluation_accuracies, self.last_evaluation_accuracy
        )
        return 1.0 - self.last_evaluation_accuracy, self.last_evaluation_accuracy


class _ScriptedMonitor:
    def __init__(self, alarms, warmup_ticks=20):
        self.alarms = iter(alarms)
        self.warmup_ticks = warmup_ticks
        self.tick_count = 0
        self.observed_batches = []
        self.drift_events = []
        self.retrain_decisions = []
        self.counterfactual_triggers = []

    def observe_tick(self, batches_by_client, *, virtual_time, record_action=True, warmup=False):
        self.observed_batches.extend(batches_by_client.values())
        self.tick_count += 1
        if warmup:
            return SimpleNamespace(should_retrain=False)
        should_retrain = next(self.alarms, False)
        if should_retrain:
            event = {"time": virtual_time, "flagged_fraction": 1.0}
            self.drift_events.append(event)
            if record_action:
                self.retrain_decisions.append(event)
            else:
                self.counterfactual_triggers.append(event)
        return SimpleNamespace(should_retrain=should_retrain)


class _WarmupFalseAlarmMonitor(_ScriptedMonitor):
    """A double whose detector would alarm during clean warm-up if allowed."""

    def __init__(self):
        super().__init__([True])
        self.warmup_false_alarm_attempts = 0

    def observe_tick(self, batches_by_client, *, virtual_time, record_action=True, warmup=False):
        if warmup:
            self.warmup_false_alarm_attempts += 1
            self.observed_batches.extend(batches_by_client.values())
            # A real monitor must suppress this alarm rather than latch it.
            return SimpleNamespace(should_retrain=False)
        return super().observe_tick(
            batches_by_client,
            virtual_time=virtual_time,
            record_action=record_action,
            warmup=warmup,
        )


def _config(**overrides):
    values = {
        "initial_rounds": 1,
        "warmup_ticks": 20,
        "monitor_ticks": 3,
        "monitor_tick_seconds": 10.0,
        "retrain_rounds": 5,
        "num_clients": 3,
    }
    values.update(overrides)
    return DriftEpisodeConfig(**values)


def _identity(batch, _kind, _severity, seed=None):
    return batch.clone()


class _DropoutTraceModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = nn.Dropout(p=0.5)
        self.anchor = nn.Parameter(torch.zeros(()))

    def forward(self, x):
        dropout_mean = self.dropout(x).mean(dim=(1, 2, 3))
        uncertain = (dropout_mean > 0.5).float()
        certainty = 10.0 * (1.0 - uncertain) + self.anchor * 0.0
        return torch.stack((certainty, -certainty), dim=1)


class TestDriftEpisode(unittest.TestCase):
    def test_capture_and_restore_random_state_uses_all_cuda_device_states(self):
        cuda_states = [
            torch.tensor([1, 2], dtype=torch.uint8),
            torch.tensor([3, 4], dtype=torch.uint8),
        ]

        with (
            patch.object(smoke_drift.torch.cuda, "is_available", return_value=True),
            patch.object(
                smoke_drift.torch.cuda,
                "get_rng_state_all",
                return_value=cuda_states,
            ) as get_rng_state_all,
            patch.object(smoke_drift.torch.cuda, "set_rng_state_all") as set_rng_state_all,
        ):
            state = capture_random_state()
            restore_random_state(state)

        self.assertIs(state["torch_cuda"], cuda_states)
        get_rng_state_all.assert_called_once_with()
        set_rng_state_all.assert_called_once_with(cuda_states)

    def test_capture_and_restore_random_state_uses_mps_rng_state(self):
        mps_state = torch.tensor([5, 6], dtype=torch.uint8)
        fake_mps = SimpleNamespace(
            get_rng_state=MagicMock(return_value=mps_state),
            set_rng_state=MagicMock(),
        )
        fake_mps_backend = SimpleNamespace(is_available=MagicMock(return_value=True))

        with (
            patch.object(smoke_drift.torch.cuda, "is_available", return_value=False),
            patch.object(smoke_drift.torch.backends, "mps", fake_mps_backend),
            patch.object(smoke_drift.torch, "mps", fake_mps),
        ):
            state = capture_random_state()
            restore_random_state(state)

        self.assertIs(state["torch_mps"], mps_state)
        fake_mps.get_rng_state.assert_called_once_with()
        fake_mps.set_rng_state.assert_called_once_with(mps_state)

    @unittest.skipUnless(
        torch.cuda.is_available() or torch.backends.mps.is_available(),
        "CUDA/MPS unavailable; a skipped test is not accelerator execution evidence.",
    )
    def test_restore_random_state_replays_dropout_on_an_available_accelerator(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "mps")
        dropout = torch.nn.Dropout(p=0.5).train().to(device)
        source = torch.ones(32, device=device)
        state = capture_random_state()

        first = dropout(source)
        restore_random_state(state)
        replayed = dropout(source)

        self.assertTrue(torch.equal(first, replayed))

    def test_dropout_trace_model_logits_depend_on_dropout_output(self):
        model = _DropoutTraceModel().train()
        x = torch.ones(4, 1, 2, 2)

        torch.manual_seed(1)
        first = model(x)
        torch.manual_seed(2)
        second = model(x)

        self.assertFalse(torch.equal(first, second))

    def test_restore_random_state_replays_active_dropout_sequence(self):
        dropout = torch.nn.Dropout(p=0.5).train()
        state = capture_random_state()

        first = dropout(torch.ones(32))
        restore_random_state(state)
        replayed = dropout(torch.ones(32))

        self.assertTrue(torch.equal(first, replayed))

    def test_schema_v3_persists_auditable_experiment_detector_and_runtime_metadata(self):
        config = _config(
            dataset="auditable-dataset",
            retrain_rounds=2,
            monitor_ticks=1,
            monitor_tick_seconds=7.5,
            trigger_threshold=0.4,
            trigger_window_ticks=3,
            detector_alpha=0.01,
            detector_T=7,
            local_epochs=2,
            batch_size=4,
            timeout_percentile=80,
            max_train_samples_per_client=9,
            tau=0.6,
            corruption="gaussian_blur",
            severity=4,
            seed=99,
            output_dir="must-not-be-persisted",
            production_horizon_seconds=10.0,
        )
        server = _FakeServer(
            global_model=nn.Linear(4, 2),
            evaluation_accuracies=[0.9, 0.8, 0.2, 0.3, 0.5, 0.5, 0.7],
        )

        result = run_drift_episode(
            config,
            server=server,
            monitor=_ScriptedMonitor([False]),
            corruption_fn=_identity,
        )

        self.assertEqual(result["schema_version"], 3)
        self.assertEqual(
            result["experiment_config"],
            {
                "dataset": "auditable-dataset",
                "num_clients": 3,
                "initial_rounds": 1,
                "retrain_rounds": 2,
                "warmup_ticks": 20,
                "monitor_tick_seconds": 7.5,
                "monitor_ticks": 1,
                "local_epochs": 2,
                "batch_size": 4,
                "timeout_percentile": 80,
                "client_speed_profile": "uniform",
                "drifted_client_ids": None,
                "drift_onset_ticks": None,
                "client_speed_tiers": [["uniform", 0, 10, 1.0]],
                "max_train_samples_per_client": 9,
                "tau": 0.6,
                "corruption": "gaussian_blur",
                "severity": 4,
                "seed": 99,
                "baseline": False,
                "production_horizon_seconds": 10.0,
            },
        )
        self.assertEqual(
            result["detector_config"],
            {
                "detector_kind": "udd",
                "detector_alpha": 0.01,
                "detector_T": 7,
                "trigger_threshold": 0.4,
                "trigger_window_ticks": 3,
            },
        )
        self.assertEqual(result["runtime"]["python_version"], sys.version)
        self.assertEqual(result["runtime"]["numpy_version"], np.__version__)
        self.assertEqual(result["runtime"]["torch_version"], torch.__version__)
        self.assertEqual(result["runtime"]["effective_device"], "cpu")
        self.assertIsInstance(result["runtime"]["deterministic_algorithms_enabled"], bool)
        self.assertIsInstance(result["runtime"]["cudnn_deterministic"], bool)
        self.assertIsInstance(result["runtime"]["cudnn_benchmark"], bool)
        self.assertRegex(result["runtime"]["clean_checkpoint_digest"], r"^[0-9a-f]{64}$")
        self.assertNotIn("output_dir", result["experiment_config"])
        self.assertAlmostEqual(result["metrics"]["clean_retention_delta"], -0.1)
        self.assertAlmostEqual(result["metrics"]["corrupted_accuracy_gain"], 0.3)

    def test_agent_executes_exactly_one_five_round_retraining_block(self):
        server = _FakeServer()
        monitor = _ScriptedMonitor([False, False, True])

        result = run_drift_episode(
            _config(), server=server, monitor=monitor, corruption_fn=_identity
        )

        self.assertEqual(server.run_round_calls, [])
        self.assertEqual(server.run_one_round_calls, 6)
        self.assertEqual(len(result["retrain_decisions"]), 1)
        self.assertEqual(len(result["retrain_round_events"]), 5)

    def test_training_rounds_are_recorded_as_clean_training_history(self):
        config = _config(initial_rounds=3)
        server = _FakeServer(round_durations=(2.0, 3.0, 5.0, 7.0, 11.0, 13.0, 2.0, 2.0))
        monitor = _ScriptedMonitor([False, False, True])

        result = run_drift_episode(
            config, server=server, monitor=monitor, corruption_fn=_identity
        )

        self.assertEqual(len(result["clean_training_history"]), 3)
        self.assertEqual(
            [entry["stage"] for entry in result["clean_training_history"]],
            ["train_round", "train_round", "train_round"],
        )
        self.assertEqual(
            [entry["time"] for entry in result["clean_training_history"]],
            [2.0, 5.0, 10.0],
        )
        self.assertLessEqual(
            result["clean_training_history"][-1]["time"],
            result["metadata"]["production_start_time"],
        )
        self.assertEqual(
            [entry["stage"] for entry in result["clean_evaluations"]],
            ["pre_drift", "final"],
        )

    def test_comparison_shares_the_same_training_history_between_arms(self):
        results = run_drift_comparison(
            _config(initial_rounds=2, production_horizon_seconds=400.0),
            corruption_fn=_identity,
            server_factory=lambda: _FakeServer(
                round_durations=(2.0, 3.0, 5.0, 7.0, 11.0, 13.0, 2.0),
                global_model=nn.Linear(4, 2),
            ),
            monitor_factory=lambda _server, _baseline: _ScriptedMonitor(
                [False, False, True]
            ),
        )

        self.assertEqual(
            results["agent"]["clean_training_history"],
            results["baseline"]["clean_training_history"],
        )
        self.assertEqual(len(results["agent"]["clean_training_history"]), 2)
        self.assertEqual(
            [entry["time"] for entry in results["agent"]["clean_training_history"]],
            [2.0, 5.0],
        )

    def test_retraining_uses_clean_data_for_clients_not_yet_drifted(self):
        from experiments.e07_drift_agent.episode import _run_retraining

        class _MiniClient:
            def __init__(self, dataset):
                self.dataset = dataset
                self.reset_calls = 0

            def reset_optimizer(self):
                self.reset_calls += 1

        class _MiniServer:
            def __init__(self):
                self.virtual_time = 100.0
                clean = (np.zeros(4, dtype=np.float32), np.zeros(4, dtype=np.int64))
                self.clients = [
                    _MiniClient((clean[0].copy(), clean[1].copy())) for _ in range(2)
                ]

            def run_one_round(self, *, record_default_metrics=False):
                return {"completed_time": float(self.virtual_time)}

            def evaluate_dataset(self, _dataset):
                return (0.5, 0.5)

        server = _MiniServer()
        clean = (np.zeros(4, dtype=np.float32), np.zeros(4, dtype=np.int64))
        corrupted = (np.ones(4, dtype=np.float32), np.zeros(4, dtype=np.int64))
        corrupted_test = (np.ones(4, dtype=np.float32), np.zeros(4, dtype=np.int64))
        result = {"retrain_round_events": [], "corrupted_accuracy_history": []}
        config = _config(num_clients=2, drifted_client_ids=(0,), drift_onset_ticks={0: 1})
        _run_retraining(
            result, server, [corrupted, corrupted], 1,
            config, production_start_time=0.0,
            current_test_fn=lambda: corrupted_test,
        )
        self.assertTrue(np.array_equal(server.clients[0].dataset[0], np.ones(4)))   # driftado
        self.assertTrue(np.array_equal(server.clients[1].dataset[0], np.zeros(4)))  # limpo
        self.assertEqual(server.clients[0].reset_calls, 1)
        self.assertEqual(server.clients[1].reset_calls, 1)

    def test_fixed_horizon_rejects_retraining_before_any_round_when_budget_cannot_fit(self):
        server = _FakeServer(
            round_durations=(2.0, 10.0, 10.0, 10.0, 10.0, 10.0),
            timeout=10.0,
        )

        with self.assertRaisesRegex(RuntimeError, "retraining budget"):
            run_drift_episode(
                _config(production_horizon_seconds=22.0),
                server=server,
                monitor=_ScriptedMonitor([True]),
                corruption_fn=_identity,
            )

        self.assertEqual(server.run_one_round_calls, 1)

    def test_fixed_production_horizon_continues_after_retraining_and_reports_distinct_recovery_metrics(self):
        config = _config(production_horizon_seconds=100.0)
        server = _FakeServer(
            round_durations=(2.0, 3.0, 5.0, 7.0, 11.0, 13.0),
            evaluation_accuracies=[
                0.9,  # training round (clean)
                0.9,  # pre-drift clean evaluation
                0.2,  # drift onset
                0.2, 0.2, 0.2,  # monitor ticks before decision
                0.2, 0.2, 0.2, 0.2, 0.4,  # five retraining rounds
                0.6, 0.6, 0.6, 0.6,  # production ticks after retraining
                0.6, 0.9,  # episode end and final clean evaluation
            ],
        )

        agent = run_drift_episode(
            config,
            server=server,
            monitor=_ScriptedMonitor([False, False, True]),
            corruption_fn=_identity,
        )
        baseline = run_drift_episode(
            _config(baseline=True, production_horizon_seconds=100.0),
            server=_FakeServer(),
            monitor=_ScriptedMonitor([False, False, True]),
            corruption_fn=_identity,
        )

        decision_time = agent["retrain_decisions"][0]["time"]
        last_round_end = agent["retrain_round_events"][-1]["completed_time"]
        first_recovery_time = 81.0
        self.assertLessEqual(
            config.retrain_rounds * server.timeout,
            agent["metadata"]["production_start_time"]
            + config.production_horizon_seconds
            - decision_time,
        )
        self.assertEqual(
            agent["metadata"]["end_time_seconds"],
            agent["metadata"]["production_start_time"] + config.production_horizon_seconds,
        )
        self.assertEqual(
            agent["metadata"]["end_time_seconds"],
            baseline["metadata"]["end_time_seconds"],
        )
        self.assertEqual(
            agent["metrics"]["time_to_recovery_seconds"],
            first_recovery_time - decision_time,
        )
        self.assertEqual(
            agent["metrics"]["retraining_duration_seconds"],
            last_round_end - decision_time,
        )
        self.assertNotIn("recovery_duration_seconds", agent["metrics"])

    def test_time_to_recovery_is_none_without_a_post_decision_accuracy_at_tau(self):
        result = run_drift_episode(
            _config(production_horizon_seconds=100.0),
            server=_FakeServer(
                evaluation_accuracies=[0.2] * 20,
            ),
            monitor=_ScriptedMonitor([False, False, True]),
            corruption_fn=_identity,
        )

        self.assertIsNone(result["metrics"]["time_to_recovery_seconds"])

    def test_non_positive_production_horizon_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "production_horizon_seconds must be positive"):
            run_drift_episode(
                _config(production_horizon_seconds=0.0),
                server=_FakeServer(),
                monitor=_ScriptedMonitor([]),
                corruption_fn=_identity,
            )

    def test_baseline_uses_the_given_horizon_without_retraining(self):
        agent = run_drift_episode(
            _config(),
            server=_FakeServer(),
            monitor=_ScriptedMonitor([False, False, True]),
            corruption_fn=_identity,
        )
        baseline = run_drift_episode(
            _config(baseline=True, end_time_seconds=agent["metadata"]["end_time_seconds"]),
            server=_FakeServer(),
            monitor=_ScriptedMonitor([False, False, True]),
            corruption_fn=_identity,
        )

        self.assertEqual(baseline["retrain_decisions"], [])
        self.assertEqual(len(baseline["counterfactual_triggers"]), 1)
        self.assertEqual(baseline["retrain_round_events"], [])
        self.assertEqual(
            baseline["metadata"]["end_time_seconds"],
            agent["metadata"]["end_time_seconds"],
        )

    def test_monitor_sees_only_corrupted_tensors_and_downtime_uses_corrupted_history(self):
        monitor = _ScriptedMonitor([False, False, True])
        result = run_drift_episode(
            _config(),
            server=_FakeServer(),
            monitor=monitor,
            corruption_fn=lambda x, *_args, **_kwargs: x + 1,
        )

        self.assertTrue(all(isinstance(batch, torch.Tensor) for batch in monitor.observed_batches))
        production_batches = monitor.observed_batches[_config().warmup_ticks * 3 :]
        self.assertTrue(all(torch.all(batch == 1) for batch in production_batches))
        self.assertEqual(
            result["metrics"]["downtime_seconds"],
            compute_downtime(
                result["corrupted_accuracy_history"],
                0.5,
                end_time=result["metadata"]["end_time_seconds"],
            ),
        )

    def test_clean_warmup_precedes_corrupted_production_and_each_stage_is_evaluated(self):
        result = run_drift_episode(
            _config(monitor_ticks=3),
            server=_FakeServer(),
            monitor=_ScriptedMonitor([False, False, True]),
            corruption_fn=_identity,
        )

        self.assertEqual(
            result["metadata"]["production_start_time"],
            result["metadata"]["warmup_completed_time"],
        )
        self.assertEqual(
            [point["stage"] for point in result["corrupted_accuracy_history"]],
            ["drift_onset", "monitor_tick"] * 0
            + ["drift_onset", "monitor_tick", "monitor_tick", "monitor_tick"]
            + ["retrain_round"] * 5
            + ["episode_end"],
        )
        self.assertEqual(
            [point["stage"] for point in result["clean_evaluations"]],
            ["pre_drift", "final"],
        )

    def test_warmup_false_alarm_cannot_consume_first_production_retraining_action(self):
        server = _FakeServer()
        monitor = _WarmupFalseAlarmMonitor()

        result = run_drift_episode(
            _config(monitor_ticks=1),
            server=server,
            monitor=monitor,
            corruption_fn=_identity,
        )

        self.assertEqual(monitor.warmup_false_alarm_attempts, _config().warmup_ticks)
        self.assertEqual(len(result["drift_events"]), 1)
        self.assertEqual(len(result["retrain_decisions"]), 1)
        self.assertEqual(len(result["retrain_round_events"]), 5)

    def test_comparison_pairs_production_start_and_absolute_horizon(self):
        created_servers = []

        def server_factory():
            server = _FakeServer(global_model=nn.Linear(4, 2))
            created_servers.append(server)
            return server

        results = run_drift_comparison(
            _config(production_horizon_seconds=400.0),
            corruption_fn=_identity,
            server_factory=server_factory,
            monitor_factory=lambda _server, _baseline: _ScriptedMonitor(
                [False, False, True]
            ),
        )

        self.assertEqual(
            results["agent"]["metadata"]["production_start_time"],
            results["baseline"]["metadata"]["production_start_time"],
        )
        self.assertEqual(
            results["agent"]["metadata"]["end_time_seconds"],
            results["baseline"]["metadata"]["end_time_seconds"],
        )
        self.assertEqual(results["baseline"]["retrain_round_events"], [])
        self.assertEqual(len(created_servers), 3)
        agent_server, baseline_server = created_servers[1:]
        self.assertEqual(
            results["agent"]["metadata"]["preproduction_server_state"],
            results["baseline"]["metadata"]["preproduction_server_state"],
        )
        self.assertEqual(
            agent_server.evaluation_states[0],
            baseline_server.evaluation_states[0],
        )
        self.assertEqual(
            agent_server.evaluation_states[0]["virtual_time"],
            created_servers[0].round_durations[0],
        )
        self.assertEqual(agent_server.evaluation_states[0]["next_round_index"], 1)
        self.assertRegex(
            results["agent"]["runtime"]["clean_checkpoint_digest"], r"^[0-9a-f]{64}$"
        )
        self.assertEqual(
            results["agent"]["runtime"]["clean_checkpoint_digest"],
            results["baseline"]["runtime"]["clean_checkpoint_digest"],
        )

    def test_comparison_replays_real_dropout_detector_trace_until_first_trigger(self):
        config = _config(
            monitor_ticks=12,
            batch_size=4,
            production_horizon_seconds=400.0,
        )
        results = run_drift_comparison(
            config,
            corruption_fn=lambda x, *_args, **_kwargs: x + 1,
            server_factory=lambda: _FakeServer(global_model=_DropoutTraceModel()),
        )

        self.assertTrue(results["agent"]["retrain_decisions"])
        decision_time = results["agent"]["retrain_decisions"][0]["time"]
        self.assertEqual(
            [
                entry
                for entry in results["agent"]["tick_history"]
                if entry["time"] <= decision_time
            ],
            [
                entry
                for entry in results["baseline"]["tick_history"]
                if entry["time"] <= decision_time
            ],
        )

    def test_result_tick_history_does_not_alias_monitor_trace(self):
        server = _FakeServer(global_model=_DropoutTraceModel())
        monitor = DriftMonitor(server.global_model)
        result = run_drift_episode(
            _config(monitor_ticks=1),
            server=server,
            monitor=monitor,
            corruption_fn=lambda x, *_args, **_kwargs: x + 1,
        )

        result["tick_history"][0]["clients"]["0"]["score"] = -1.0

        self.assertNotEqual(
            result["tick_history"][0]["clients"]["0"]["score"],
            monitor.tick_history[0]["clients"]["0"]["score"],
        )

    def test_monitor_corrupts_only_drifted_clients_after_their_onset(self):
        from experiments.e07_drift_agent.episode import _monitor_batches

        def tagging_corruption(batch, _kind, _severity, seed=None):
            return batch + 1.0

        rng = np.random.default_rng(42)
        x_clean = np.zeros((64, 32, 32, 3), dtype=np.float32)
        y_clean = np.zeros(64, dtype=np.int64)
        datasets = [(x_clean, y_clean) for _ in range(4)]
        config = _config(
            num_clients=4,
            warmup_ticks=0,
            drifted_client_ids=(0, 2),
            drift_onset_ticks={0: 3, 2: 5},
            batch_size=8,
            seed=42,
        )
        batches_tick4 = _monitor_batches(
            datasets, config, tagging_corruption, rng, tick=4
        )
        self.assertGreater(float(batches_tick4["0"].sum()), 0.0)   # onset 3
        self.assertEqual(float(batches_tick4["1"].sum()), 0.0)     # nunca drift
        self.assertEqual(float(batches_tick4["2"].sum()), 0.0)     # onset 5
        self.assertEqual(float(batches_tick4["3"].sum()), 0.0)     # nunca drift
        batches_tick6 = _monitor_batches(
            datasets, config, tagging_corruption, rng, tick=6
        )
        self.assertGreater(float(batches_tick6["2"].sum()), 0.0)   # onset 5

    def test_monitor_evaluates_schedule_in_production_relative_frame_with_warmup(self):
        from experiments.e07_drift_agent.episode import _monitor_batches

        def tagging_corruption(batch, _kind, _severity, seed=None):
            return batch + 1.0

        rng = np.random.default_rng(42)
        x_clean = np.zeros((64, 32, 32, 3), dtype=np.float32)
        y_clean = np.zeros(64, dtype=np.int64)
        datasets = [(x_clean, y_clean) for _ in range(2)]
        config = _config(
            num_clients=2,
            warmup_ticks=20,
            drifted_client_ids=(0,),
            drift_onset_ticks={0: 1},
            batch_size=8,
            seed=42,
        )
        first_production = _monitor_batches(
            datasets, config, tagging_corruption, rng, tick=20
        )
        self.assertEqual(float(first_production["0"].sum()), 0.0)  # produção tick 0 < onset 1
        self.assertEqual(float(first_production["1"].sum()), 0.0)  # nunca drift
        second_production = _monitor_batches(
            datasets, config, tagging_corruption, rng, tick=21
        )
        self.assertGreater(float(second_production["0"].sum()), 0.0)  # produção tick 1 >= onset 1
        self.assertEqual(float(second_production["1"].sum()), 0.0)    # nunca drift

    def test_ramp_corrupts_a_growing_fraction_of_monitor_batches(self):
        from experiments.e07_drift_agent.episode import _monitor_batches

        def tagging_corruption(batch, _kind, _severity, seed=None):
            return batch + 1.0

        rng = np.random.default_rng(42)
        x_clean = np.zeros((64, 1, 1, 1), dtype=np.float32)
        y_clean = np.zeros(64, dtype=np.int64)
        datasets = [(x_clean, y_clean) for _ in range(2)]
        config = _config(
            num_clients=2,
            drift_ramp_ticks=20,
            batch_size=32,
            seed=42,
            warmup_ticks=2,
        )
        first = _monitor_batches(
            datasets, config, tagging_corruption, rng,
            tick=20, virtual_time=20.0,
        )
        self.assertEqual(float(first["0"].sum()), 0.0)  # fração 0
        self.assertEqual(float(first["1"].sum()), 0.0)
        mid = _monitor_batches(
            datasets, config, tagging_corruption, rng,
            tick=30, virtual_time=120.0,
        )
        self.assertEqual(float(mid["0"].sum()), 16.0)  # fração 0.5 -> 16 de 32
        self.assertEqual(float(mid["1"].sum()), 16.0)
        end = _monitor_batches(
            datasets, config, tagging_corruption, rng,
            tick=40, virtual_time=300.0,
        )
        self.assertEqual(float(end["0"].sum()), 32.0)  # fração 1
        self.assertEqual(float(end["1"].sum()), 32.0)

    def test_ramp_batch_is_deterministic_across_arms(self):
        from experiments.e07_drift_agent.episode import _monitor_batches

        def tagging_corruption(batch, _kind, _severity, seed=None):
            return batch + 1.0

        x_clean = np.zeros((64, 32, 32, 3), dtype=np.float32)
        y_clean = np.zeros(64, dtype=np.int64)
        datasets = [(x_clean, y_clean) for _ in range(2)]
        config = _config(
            num_clients=2, drift_ramp_ticks=20, batch_size=32, seed=7, warmup_ticks=2
        )

        first = _monitor_batches(
            datasets, config, tagging_corruption, np.random.default_rng(7),
            tick=30, virtual_time=120.0,
        )
        second = _monitor_batches(
            datasets, config, tagging_corruption, np.random.default_rng(7),
            tick=30, virtual_time=120.0,
        )
        self.assertTrue(torch.equal(first["0"], second["0"]))
        self.assertTrue(torch.equal(first["1"], second["1"]))

    def test_ramp_records_fraction_metrics_and_uses_mixed_test(self):
        config = _config(drift_ramp_ticks=20, production_horizon_seconds=400.0)
        server = _FakeServer(
            round_durations=(2.0, 3.0, 5.0, 7.0, 11.0, 13.0, 2.0),
            evaluation_accuracies=[
                0.9,  # training round (clean)
                0.9,  # pre-drift clean evaluation
                0.9,  # drift onset (teste misto na fração 0 -> limpo)
                0.6, 0.6, 0.6,  # monitor ticks antes da decisão
                0.2, 0.2, 0.2, 0.2, 0.4,  # cinco rounds de retreino
                0.4, 0.4, 0.4, 0.4,  # ticks pós-retreino
                0.4, 0.9,  # episode end e final clean
            ],
        )
        result = run_drift_episode(
            config,
            server=server,
            monitor=_ScriptedMonitor([False, False, True]),
            corruption_fn=_identity,
        )

        production_start = result["metadata"]["production_start_time"]
        decision_time = result["retrain_decisions"][0]["time"]
        self.assertAlmostEqual(
            result["metrics"]["fraction_at_trigger"],
            (decision_time - production_start) / 200.0,
        )
        crossing = next(
            entry for entry in result["corrupted_accuracy_history"]
            if entry["accuracy"] < 0.5
        )
        self.assertAlmostEqual(
            result["metrics"]["fraction_at_tau_crossing"],
            (crossing["time"] - production_start) / 200.0,
        )
        onset = next(
            entry for entry in result["corrupted_accuracy_history"]
            if entry["stage"] == "drift_onset"
        )
        self.assertEqual(onset["accuracy"], 0.9)

    def test_ramp_metrics_are_absent_without_ramp(self):
        result = run_drift_episode(
            _config(production_horizon_seconds=400.0),
            server=_FakeServer(
                round_durations=(2.0, 3.0, 5.0, 7.0, 11.0, 13.0, 2.0)
            ),
            monitor=_ScriptedMonitor([False, False, True]),
            corruption_fn=_identity,
        )
        self.assertNotIn("fraction_at_trigger", result["metrics"])
        self.assertNotIn("fraction_at_tau_crossing", result["metrics"])

    def test_drift_schedule_fields_default_to_none_and_serialize(self):
        config = _config()
        self.assertIsNone(config.drifted_client_ids)
        self.assertIsNone(config.drift_onset_ticks)
        payload = smoke_drift._experiment_config(config)
        self.assertIsNone(payload["drifted_client_ids"])
        self.assertIsNone(payload["drift_onset_ticks"])

    def test_drift_schedule_fields_survive_replace(self):
        from dataclasses import replace

        config = replace(
            _config(),
            drifted_client_ids=(0, 1, 2),
            drift_onset_ticks={0: 1, 1: 4},
        )
        self.assertEqual(config.drifted_client_ids, (0, 1, 2))
        self.assertEqual(config.drift_onset_ticks, {0: 1, 1: 4})

    def test_drift_ramp_ticks_rejects_invalid_values_and_combinations(self):
        cases = (
            dict(drift_ramp_ticks=0),
            dict(drift_ramp_ticks=True),
            dict(drift_ramp_ticks=-3),
        )
        for overrides in cases:
            with self.subTest(overrides=overrides):
                with self.assertRaisesRegex(ValueError, "drift_ramp_ticks"):
                    run_drift_episode(
                        _config(**overrides),
                        server=_FakeServer(),
                        monitor=_ScriptedMonitor([]),
                        corruption_fn=_identity,
                    )

    def test_drift_ramp_ticks_rejects_combination_with_onset_fields(self):
        with self.assertRaisesRegex(ValueError, "drift_ramp_ticks"):
            run_drift_episode(
                _config(
                    drift_ramp_ticks=20,
                    drifted_client_ids=(0, 1),
                ),
                server=_FakeServer(),
                monitor=_ScriptedMonitor([]),
                corruption_fn=_identity,
            )
        with self.assertRaisesRegex(ValueError, "drift_ramp_ticks"):
            run_drift_episode(
                _config(drift_ramp_ticks=20, drift_onset_ticks={0: 1}),
                server=_FakeServer(),
                monitor=_ScriptedMonitor([]),
                corruption_fn=_identity,
            )


if __name__ == "__main__":
    unittest.main()
