import json
import tempfile
import unittest
from pathlib import Path

import torch

from experiments.drift_controls import OracleMonitor, identity_corruption
from experiments.run_smoke_drift import (
    aggregate_runs,
    build_run_matrix,
    parse_args,
    run_matrix,
)
from experiments.smoke_drift import DriftEpisodeConfig


class TestDriftRunMatrix(unittest.TestCase):
    def test_builds_explicit_seed_and_scenario_product_without_sensitivities(self):
        configs = build_run_matrix(
            seeds=[11, 12],
            scenarios=[("noise", 1), ("fog", 4)],
            base_config=DriftEpisodeConfig(trigger_threshold=0.3, retrain_rounds=5),
        )

        self.assertEqual(
            [
                (config.seed, config.corruption, config.severity, config.trigger_threshold, config.retrain_rounds)
                for config in configs
            ],
            [
                (11, "noise", 1, 0.3, 5),
                (12, "noise", 1, 0.3, 5),
                (11, "fog", 4, 0.3, 5),
                (12, "fog", 4, 0.3, 5),
            ],
        )

    def test_expands_quorum_and_retraining_sensitivities_only_when_requested(self):
        configs = build_run_matrix(
            seeds=[7],
            scenarios=[("noise", 2)],
            quorums=[0.2, 0.5],
            retrain_rounds=[3, 7],
        )

        self.assertEqual(
            [(config.trigger_threshold, config.retrain_rounds) for config in configs],
            [(0.2, 3), (0.2, 7), (0.5, 3), (0.5, 7)],
        )

    def test_oracle_triggers_once_on_the_first_production_tick(self):
        monitor = OracleMonitor()
        batches = {"client-0": torch.ones(2, 1, 2, 2)}

        warmup = monitor.observe_tick(batches, virtual_time=0.0, warmup=True)
        first_production = monitor.observe_tick(batches, virtual_time=10.0)
        later_production = monitor.observe_tick(batches, virtual_time=20.0)

        self.assertFalse(warmup.should_retrain)
        self.assertTrue(first_production.should_retrain)
        self.assertFalse(later_production.should_retrain)
        self.assertEqual(monitor.drift_events, [{"time": 10.0, "client_id": "oracle", "score": 1.0}])
        self.assertEqual(monitor.retrain_decisions, [{"time": 10.0, "flagged_fraction": 1.0}])
        self.assertEqual(len(monitor.tick_history), 3)

    def test_identity_corruption_returns_an_independent_tensor(self):
        source = torch.arange(8, dtype=torch.float32).reshape(2, 1, 2, 2)

        result = identity_corruption(source, "identity", 0, seed=17)
        result[0, 0, 0, 0] = -1

        self.assertTrue(torch.equal(source, torch.arange(8, dtype=torch.float32).reshape(2, 1, 2, 2)))
        self.assertEqual(result.dtype, source.dtype)
        self.assertEqual(result.device, source.device)

    def test_persists_each_pair_in_the_canonical_scenario_seed_layout(self):
        calls = []

        def runner(config, *, corruption_fn, **kwargs):
            calls.append((config, corruption_fn, kwargs))
            return {
                "agent": {"schema_version": 3, "metrics": {"downtime_seconds": 2.0}},
                "baseline": {"schema_version": 3, "metrics": {"downtime_seconds": 5.0}},
            }

        with tempfile.TemporaryDirectory() as directory:
            paths = run_matrix(
                build_run_matrix(seeds=[31], scenarios=[("noise", 2)]),
                output_dir=directory,
                runner=runner,
                corruption_fn=identity_corruption,
            )
            root = Path(directory) / "noise_sev2" / "seed_31"

            self.assertEqual(paths, [(root / "agent.json", root / "baseline.json")])
            self.assertEqual(json.loads((root / "agent.json").read_text()), {"schema_version": 3, "metrics": {"downtime_seconds": 2.0}})
            self.assertEqual(json.loads((root / "baseline.json").read_text()), {"schema_version": 3, "metrics": {"downtime_seconds": 5.0}})
        self.assertEqual(len(calls), 1)

    def test_aggregates_only_present_numeric_metrics(self):
        aggregate = aggregate_runs(
            [
                {"metrics": {"downtime_seconds": 2.0, "time_to_recovery_seconds": None, "label": "skip"}},
                {"metrics": {"downtime_seconds": 6.0, "time_to_recovery_seconds": 10.0, "enabled": True}},
                {"metrics": {"downtime_seconds": 4.0, "time_to_recovery_seconds": 14.0}},
            ]
        )

        self.assertEqual(
            aggregate["metrics"]["downtime_seconds"],
            {"count": 3, "mean": 4.0, "std": 1.632993161855452, "min": 2.0, "max": 6.0},
        )
        self.assertEqual(
            aggregate["metrics"]["time_to_recovery_seconds"],
            {"count": 2, "mean": 12.0, "std": 2.0, "min": 10.0, "max": 14.0},
        )
        self.assertNotIn("enabled", aggregate["metrics"])

    def test_cli_exposes_explicit_matrix_and_control_options(self):
        args = parse_args(
            [
                "--seeds", "1", "2",
                "--scenario", "noise:3", "fog:1",
                "--quorums", "0.2", "0.4",
                "--retrain-rounds", "3", "5",
                "--include-controls",
            ]
        )

        self.assertEqual(args.seeds, [1, 2])
        self.assertEqual(args.scenario, ["noise:3", "fog:1"])
        self.assertEqual(args.quorums, [0.2, 0.4])
        self.assertEqual(args.retrain_rounds, [3, 5])
        self.assertTrue(args.include_controls)


if __name__ == "__main__":
    unittest.main()
