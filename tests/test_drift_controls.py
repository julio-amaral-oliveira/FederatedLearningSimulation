import unittest
from typing import cast

import torch

from experiments.e07_drift_agent.episode import DriftEpisodeConfig, run_drift_episode
from experiments.e07_drift_agent.run_matrix import build_run_matrix
from experiments.shared.drift_controls import OracleMonitor, ScheduledMonitor
from src.orchestrator.orchestrator import TickOutcome
from tests.test_drift_episode import _FakeServer, _config, _identity


def _batches(client_count: int = 2) -> dict[str, torch.Tensor]:
    return {str(index): torch.zeros(2, 1, 1, 1) for index in range(client_count)}


class TestScheduledMonitor(unittest.TestCase):
    def test_warmup_ticks_never_trigger_or_record_decisions(self):
        monitor = ScheduledMonitor(trigger_after_seconds=0.0)

        outcomes = [
            monitor.observe_tick(_batches(), virtual_time=10.0, warmup=True)
            for _ in range(5)
        ]

        self.assertFalse(outcomes[-1].should_retrain)
        self.assertIsInstance(outcomes[-1], TickOutcome)
        self.assertEqual(monitor.retrain_decisions, [])
        self.assertEqual(monitor.counterfactual_triggers, [])
        self.assertEqual(monitor.drift_events, [])
        self.assertEqual(len(monitor.tick_history), 5)
        self.assertTrue(
            all(
                entry["warmup"] and entry["flagged_fraction"] == 0.0
                for entry in monitor.tick_history
            )
        )

    def test_triggers_exactly_once_on_the_scheduled_production_tick(self):
        monitor = ScheduledMonitor(trigger_after_seconds=30.0)
        times = [10.0, 20.0, 30.0, 40.0, 50.0]

        outcomes = [
            monitor.observe_tick(_batches(), virtual_time=time) for time in times
        ]

        self.assertEqual(
            [outcome.should_retrain for outcome in outcomes],
            [False, False, False, True, False],
        )
        self.assertEqual(len(monitor.retrain_decisions), 1)
        self.assertEqual(monitor.retrain_decisions[0]["time"], 40.0)
        self.assertEqual(monitor.retrain_decisions[0]["flagged_fraction"], 1.0)

    def test_latch_prevents_retriggering_on_later_ticks(self):
        monitor = ScheduledMonitor(trigger_after_seconds=10.0)

        for time in (10.0, 20.0, 30.0, 40.0, 50.0, 60.0):
            monitor.observe_tick(_batches(), virtual_time=time)

        self.assertEqual(len(monitor.retrain_decisions), 1)
        self.assertEqual(len(monitor.drift_events), 1)
        self.assertEqual(monitor.retrain_decisions[0]["time"], 20.0)

    def test_record_action_false_records_counterfactual_triggers(self):
        monitor = ScheduledMonitor(trigger_after_seconds=0.0)

        outcome = monitor.observe_tick(
            _batches(), virtual_time=10.0, record_action=False
        )

        self.assertTrue(outcome.should_retrain)
        self.assertEqual(monitor.retrain_decisions, [])
        self.assertEqual(len(monitor.counterfactual_triggers), 1)
        self.assertEqual(monitor.counterfactual_triggers[0]["time"], 10.0)

    def test_zero_trigger_after_fires_on_the_first_production_tick(self):
        scheduled = ScheduledMonitor()
        oracle = OracleMonitor()

        for monitor in (scheduled, oracle):
            monitor.observe_tick(_batches(), virtual_time=10.0, warmup=True)
            monitor.observe_tick(_batches(), virtual_time=10.0)

        self.assertEqual(scheduled.retrain_decisions, oracle.retrain_decisions)
        self.assertEqual(len(scheduled.retrain_decisions), 1)
        self.assertEqual(scheduled.retrain_decisions[0]["time"], 10.0)

    def test_non_tensor_batches_raise_type_error(self):
        monitor = ScheduledMonitor()
        # Deliberately violate the documented tensor contract.
        bad_batches = cast(dict[str, torch.Tensor], {"0": "not-a-tensor"})

        with self.assertRaisesRegex(TypeError, "must be torch.Tensor"):
            monitor.observe_tick(bad_batches, virtual_time=10.0)

    def test_tick_history_records_flagged_fraction_around_the_trigger(self):
        monitor = ScheduledMonitor(trigger_after_seconds=20.0)
        monitor.observe_tick(_batches(), virtual_time=5.0, warmup=True)
        for time in (10.0, 20.0, 30.0):
            monitor.observe_tick(_batches(), virtual_time=time)

        self.assertEqual(
            [entry["flagged_fraction"] for entry in monitor.tick_history],
            [0.0, 0.0, 0.0, 1.0],
        )
        self.assertEqual(
            [entry["warmup"] for entry in monitor.tick_history],
            [True, False, False, False],
        )

    def test_negative_trigger_after_seconds_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "trigger_after_seconds"):
            ScheduledMonitor(trigger_after_seconds=-1.0)

    def test_episode_retrains_exactly_once_at_the_scheduled_tick(self):
        server = _FakeServer()
        monitor = ScheduledMonitor(trigger_after_seconds=30.0)
        config = _config(monitor_ticks=5)

        result = run_drift_episode(
            config, server=server, monitor=monitor, corruption_fn=_identity
        )

        self.assertEqual(len(result["retrain_decisions"]), 1)
        self.assertEqual(len(result["drift_events"]), 1)
        self.assertEqual(len(result["retrain_round_events"]), 5)
        self.assertEqual(server.run_one_round_calls, 6)
        production_start = result["metadata"]["production_start_time"]
        self.assertEqual(
            result["retrain_decisions"][0]["time"],
            production_start + 4 * config.monitor_tick_seconds,
        )
        self.assertEqual(
            result["metrics"]["detection_delay_seconds"],
            4 * config.monitor_tick_seconds,
        )


class TestRunMatrixWindowTicks(unittest.TestCase):
    def test_expands_window_sensitivity_only_when_requested(self):
        base = DriftEpisodeConfig(trigger_window_ticks=2)

        defaulted = build_run_matrix(
            seeds=[7],
            scenarios=[("noise", 2)],
            base_config=base,
        )
        requested = build_run_matrix(
            seeds=[7],
            scenarios=[("noise", 2)],
            window_ticks=[1, 5, 50],
            base_config=base,
        )

        self.assertEqual(
            [config.trigger_window_ticks for config in defaulted],
            [2],
        )
        self.assertEqual(
            [config.trigger_window_ticks for config in requested],
            [1, 5, 50],
        )
        self.assertEqual(
            [
                (config.seed, config.corruption, config.severity)
                for config in requested
            ],
            [(7, "noise", 2)] * 3,
        )


if __name__ == "__main__":
    unittest.main()
