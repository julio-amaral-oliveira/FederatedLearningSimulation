import ast
import inspect
import unittest

import torch
import torch.nn as nn

from src.orchestrator.orchestrator import (
    DriftMonitor,
)


class _MockDetector:
    """Detector that returns configurable drift flags and scores."""

    def __init__(self, drift_flag: bool = False, score: float = 0.5):
        self.drift_flag = drift_flag
        self.score = score
        self.update_count = 0
        self.last_x = None

    def update(self, x_batch: torch.Tensor) -> tuple[bool, float]:
        self.update_count += 1
        self.last_x = x_batch
        return self.drift_flag, self.score


class _TinyModel(nn.Module):
    """Minimal model to satisfy Orchestrator construction."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class TestDriftMonitor(unittest.TestCase):
    def setUp(self):
        self.model = _TinyModel()
        self.batches = {f"client_{i}": torch.randn(2, 4) for i in range(10)}

    def _monitor(self, flags):
        detectors = {client_id: _MockDetector(flag, 0.42) for client_id, flag in flags.items()}
        return DriftMonitor(self.model, window_ticks=2, trigger_threshold=0.30, detectors=detectors)

    def test_distinct_flags_across_ticks_trigger_and_expire(self):
        monitor = self._monitor({"client_0": True, "client_1": True, "client_2": False})
        first = monitor.observe_tick(self.batches, virtual_time=10)
        monitor._detectors["client_2"].drift_flag = True
        second = monitor.observe_tick(self.batches, virtual_time=20)

        self.assertFalse(first.should_retrain)
        self.assertTrue(second.should_retrain)
        self.assertAlmostEqual(second.flagged_fraction, 0.30)
        monitor._action_latched = False
        for detector in monitor._detectors.values():
            detector.drift_flag = False
        monitor.observe_tick(self.batches, virtual_time=30)
        expired = monitor.observe_tick(self.batches, virtual_time=40)
        self.assertEqual(expired.flagged_fraction, 0.0)

    def test_action_latch_emits_only_one_action(self):
        monitor = self._monitor({f"client_{i}": i < 3 for i in range(10)})
        first = monitor.observe_tick(self.batches, virtual_time=10)
        second = monitor.observe_tick(self.batches, virtual_time=20)

        self.assertTrue(first.should_retrain)
        self.assertFalse(second.should_retrain)
        self.assertEqual(len(monitor.retrain_decisions), 1)

    def test_counterfactual_trigger_records_no_action(self):
        monitor = self._monitor({f"client_{i}": i < 3 for i in range(10)})

        outcome = monitor.observe_tick(self.batches, virtual_time=123, record_action=False)

        self.assertTrue(outcome.should_retrain)
        self.assertEqual(monitor.retrain_decisions, [])
        self.assertEqual(monitor.counterfactual_triggers, [{"time": 123, "flagged_fraction": 0.30}])

    def test_warmup_suppresses_false_alarms_without_latching_first_production_action(self):
        monitor = self._monitor({f"client_{i}": i < 3 for i in range(10)})

        warmup = monitor.observe_tick(self.batches, virtual_time=1, warmup=True)
        production = monitor.observe_tick(self.batches, virtual_time=2)

        self.assertFalse(warmup.should_retrain)
        self.assertEqual(warmup.flagged_fraction, 0.0)
        self.assertEqual(monitor.drift_events[0]["time"], 2)
        self.assertTrue(production.should_retrain)
        self.assertEqual(monitor.retrain_decisions[0]["time"], 2)

    def test_event_and_action_include_exact_virtual_time(self):
        monitor = self._monitor({f"client_{i}": i < 3 for i in range(10)})

        monitor.observe_tick(self.batches, virtual_time=17.5)

        self.assertEqual({event["time"] for event in monitor.drift_events}, {17.5})
        self.assertEqual(monitor.retrain_decisions[0]["time"], 17.5)

    def test_monitor_isolated_from_servers_and_training(self):
        """The monitor only emits a decision; the experiment owns training."""
        module = inspect.getmodule(DriftMonitor)
        source = inspect.getsource(module)
        tree = ast.parse(source)

        imported_modules = [
            node.module or ""
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        ]
        imported_modules.extend(
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        )
        self.assertFalse(
            any("server" in module_name.split(".") for module_name in imported_modules)
        )

        monitor_source = ast.parse(inspect.getsource(DriftMonitor))
        call_names = {
            node.func.id
            for node in ast.walk(monitor_source)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        call_names.update(
            node.func.attr
            for node in ast.walk(monitor_source)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        )
        self.assertTrue({"train", "retrain", "start_training", "aggregate_round"}.isdisjoint(call_names))


if __name__ == "__main__":
    unittest.main()
