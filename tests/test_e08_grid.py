"""Grid builder and smoke runner for the E08 partial-drift experiment."""

import json
import unittest

from experiments.e08_partial_drift.run_grid import build_grid


class TestBuildGrid(unittest.TestCase):
    def test_build_grid_returns_27_configs_with_declared_axes(self):
        configs = build_grid()
        self.assertEqual(len(configs), 27)
        quorums = {c.trigger_threshold for c in configs}
        windows = {c.trigger_window_ticks for c in configs}
        rhythms = {tuple(sorted(c.drift_onset_ticks.values())) for c in configs}
        self.assertEqual(quorums, {0.2, 0.3, 0.5})
        self.assertEqual(windows, {1, 2, 5})
        self.assertEqual(rhythms, {(1, 2, 3, 4, 5), (1, 4, 7, 10, 13), (1, 6, 11, 16, 21)})

    def test_build_grid_fixes_scenario_constants(self):
        configs = build_grid()
        for config in configs:
            self.assertEqual(config.corruption, "motion_blur")
            self.assertEqual(config.severity, 1)
            self.assertEqual(config.drifted_client_ids, (0, 1, 2, 3, 4))
            self.assertEqual(config.retrain_rounds, 1)
            self.assertEqual(config.detector_T, 5)
            self.assertEqual(config.seed, 42)


class TestRunGridSmoke(unittest.TestCase):
    def test_run_grid_smoke_with_fake_runner(self):
        import tempfile
        from pathlib import Path

        from experiments.e07_drift_agent.run_matrix import run_matrix

        calls = []

        def _payload(baseline):
            return {
                "schema_version": 2,
                "experiment_config": {
                    "baseline": baseline,
                    "corruption": "motion_blur",
                    "severity": 1,
                    "seed": 42,
                    "retrain_rounds": 1,
                },
                "detector_config": {},
                "runtime": {"clean_checkpoint_digest": "a"},
                "metadata": {
                    "production_start_time": 0.0,
                    "production_horizon_seconds": 400.0,
                    "end_time_seconds": 100.0,
                },
                "corrupted_accuracy_history": [],
                "clean_evaluations": [],
                "drift_events": [],
                "retrain_decisions": [],
                "counterfactual_triggers": [],
                "retrain_round_events": [],
                "tick_history": [],
                "metrics": {"downtime_seconds": 1.0 if not baseline else 100.0},
            }

        def fake_runner(config, **kwargs):
            calls.append(config)
            return {"agent": _payload(False), "baseline": _payload(True)}

        with tempfile.TemporaryDirectory() as tmp:
            configs = build_grid()
            run_matrix(configs, output_dir=tmp, runner=fake_runner)
            self.assertEqual(len(calls), 27)
            summary = json.loads((Path(tmp) / "summary.json").read_text())
            self.assertEqual(len(summary["groups"]), 27)


if __name__ == "__main__":
    unittest.main()
