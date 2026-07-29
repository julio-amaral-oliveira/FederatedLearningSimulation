import unittest
import random
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

from experiments.comparison_core import compute_downtime
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
    ):
        self.clients = [_FakeClient(0), _FakeClient(1), _FakeClient(2)]
        self.testing_data = (
            np.zeros((6, 1, 2, 2), dtype=np.float32),
            np.arange(6, dtype=np.int64) % 2,
        )
        self.virtual_time = 0.0
        self.round_durations = tuple(round_durations)
        self.rng = random.Random(rng_seed)
        self.next_round_index = 0
        self.run_round_calls = []
        self.run_one_round_calls = 0
        self.evaluation_states = []
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
        return 0.25, 0.25


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
            server = _FakeServer()
            created_servers.append(server)
            return server

        results = run_drift_comparison(
            _config(),
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

    def test_comparison_replays_real_dropout_detector_trace_until_first_trigger(self):
        config = _config(monitor_ticks=12, batch_size=4)
        results = run_drift_comparison(
            config,
            corruption_fn=lambda x, *_args, **_kwargs: x + 1,
            server_factory=lambda: _FakeServer(global_model=_DropoutTraceModel()),
        )

        self.assertTrue(results["agent"]["retrain_decisions"])
        self.assertEqual(
            results["agent"]["tick_history"],
            results["baseline"]["tick_history"][: len(results["agent"]["tick_history"])],
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
