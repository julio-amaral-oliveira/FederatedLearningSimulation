import unittest
import random
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import torch
import torch.nn as nn

from experiments.comparison_core import compute_downtime
from experiments import smoke_drift
from experiments.smoke_drift import (
    DriftEpisodeConfig,
    capture_random_state,
    restore_random_state,
    run_drift_comparison,
    run_drift_episode,
)
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
            evaluation_accuracies=[0.8, 0.2, 0.3, 0.5, 0.5, 0.7],
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

        self.assertEqual(server.run_round_calls, [1])
        self.assertEqual(server.run_one_round_calls, 6)
        self.assertEqual(len(result["retrain_decisions"]), 1)
        self.assertEqual(len(result["retrain_round_events"]), 5)

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


if __name__ == "__main__":
    unittest.main()
