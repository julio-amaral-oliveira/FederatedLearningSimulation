import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from experiments.e07_drift_agent.episode import DriftEpisodeConfig
from experiments.e07_drift_agent.run_ablation import (
    AblationPlan,
    DEFAULT_ABLATION_PRODUCTION_HORIZON_SECONDS,
    build_ablation_matrix,
    parse_args as parse_ablation_args,
)
from experiments.e07_drift_agent.run_matrix import (
    aggregate_runs,
    build_run_matrix,
    main,
    parse_args,
    run_matrix,
)
from experiments.shared.drift_controls import OracleMonitor, identity_corruption
from src.asynchronous.constants import (
    DEFAULT_SPEED_PROFILE as ASYNC_DEFAULT_SPEED_PROFILE,
    SPEED_PROFILES as ASYNC_SPEED_PROFILES,
)
from src.synchronous.constants import (
    DEFAULT_SPEED_PROFILE as SYNC_DEFAULT_SPEED_PROFILE,
    SPEED_PROFILES as SYNC_SPEED_PROFILES,
)


class TestDriftRunMatrix(unittest.TestCase):
    def test_sync_and_async_share_the_named_speed_profiles(self):
        self.assertEqual(SYNC_SPEED_PROFILES, ASYNC_SPEED_PROFILES)
        self.assertEqual(SYNC_DEFAULT_SPEED_PROFILE, "heterogeneous")
        self.assertEqual(ASYNC_DEFAULT_SPEED_PROFILE, "heterogeneous")

    def test_drift_config_and_matrix_cli_default_to_uniform_speed_profile(self):
        self.assertEqual(DriftEpisodeConfig().client_speed_profile, "uniform")
        self.assertEqual(parse_args([]).client_speed_profile, "uniform")

    def test_matrix_cli_defaults_to_the_temporary_e07_namespace(self):
        self.assertEqual(
            parse_args([]).output_dir,
            "output/e07-drift-agent/cifar-10/uniform/matrix-v1",
        )

    def test_matrix_cli_accepts_the_heterogeneous_speed_profile_alias(self):
        args = parse_args(["--speed-profile", "heterogeneous"])

        self.assertEqual(args.client_speed_profile, "heterogeneous")

    def test_ablation_cli_defaults_to_a_horizon_that_fits_ten_retraining_rounds(self):
        args = parse_ablation_args([])

        self.assertEqual(
            args.production_horizon_seconds,
            DEFAULT_ABLATION_PRODUCTION_HORIZON_SECONDS,
        )

    def test_build_run_matrix_preserves_the_requested_speed_profile(self):
        configs = build_run_matrix(
            seeds=[42, 43],
            scenarios=[("motion_blur", 1)],
            base_config=DriftEpisodeConfig(client_speed_profile="heterogeneous"),
        )

        self.assertEqual(
            [config.client_speed_profile for config in configs],
            ["heterogeneous", "heterogeneous"],
        )

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

    def test_builds_one_at_a_time_sensitivity_arms_in_ablation_runner(self):
        configs = build_ablation_matrix(
            seeds=[42, 43],
            scenarios=[("noise", 2)],
            plan=AblationPlan(
                quorums=(0.2, 0.3, 0.5),
                retrain_rounds=(1, 3, 5, 10),
                windows=(1, 2, 5),
                dropout_T=(5, 10, 25, 50),
            ),
        )

        self.assertEqual(len(configs), 11 * 2)
        self.assertTrue(
            all(
                config.production_horizon_seconds
                == DEFAULT_ABLATION_PRODUCTION_HORIZON_SECONDS
                for config in configs
            )
        )
        arms = {
            (
                config.trigger_threshold,
                config.retrain_rounds,
                config.trigger_window_ticks,
                config.detector_T,
            )
            for config in configs
        }
        self.assertEqual(
            arms,
            {
                (0.2, 1, 2, 5),
                (0.3, 1, 2, 5),
                (0.5, 1, 2, 5),
                (0.3, 3, 2, 5),
                (0.3, 5, 2, 5),
                (0.3, 10, 2, 5),
                (0.3, 1, 1, 5),
                (0.3, 1, 5, 5),
                (0.3, 1, 2, 10),
                (0.3, 1, 2, 25),
                (0.3, 1, 2, 50),
            },
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

            def result(*, baseline, downtime):
                return {
                    "schema_version": 2,
                    "metadata": {
                        "corruption": config.corruption,
                        "severity": config.severity,
                        "seed": config.seed,
                        "baseline": baseline,
                        "production_start_time": 0.0,
                        "end_time_seconds": 10.0,
                    },
                    "corrupted_accuracy_history": [],
                    "clean_evaluations": [],
                    "metrics": {"downtime_seconds": downtime},
                }

            return {
                "agent": result(baseline=False, downtime=2.0),
                "baseline": result(baseline=True, downtime=5.0),
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
            self.assertEqual(
                json.loads((root / "agent.json").read_text())["metrics"],
                {"downtime_seconds": 2.0},
            )
            self.assertEqual(
                json.loads((root / "baseline.json").read_text())["metrics"],
                {"downtime_seconds": 5.0},
            )
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

    def test_summary_partitions_identity_control_from_visual_metrics(self):
        def runner(config, **_kwargs):
            downtime = 100.0 if config.corruption == "identity" else 2.0

            def result(*, baseline):
                return {
                    "schema_version": 2,
                    "metadata": {
                        "corruption": config.corruption,
                        "severity": config.severity,
                        "seed": config.seed,
                        "baseline": baseline,
                        "production_start_time": 0.0,
                        "end_time_seconds": 400.0,
                    },
                    "corrupted_accuracy_history": [],
                    "clean_evaluations": [],
                    "metrics": {"downtime_seconds": downtime},
                }

            return {"agent": result(baseline=False), "baseline": result(baseline=True)}

        configs = [
            DriftEpisodeConfig(corruption="noise", severity=2, seed=1, production_horizon_seconds=400.0),
            DriftEpisodeConfig(corruption="identity", severity=0, seed=1, production_horizon_seconds=400.0),
        ]
        with tempfile.TemporaryDirectory() as directory:
            run_matrix(configs, output_dir=directory, runner=runner, corruption_fn=identity_corruption)
            summary = json.loads((Path(directory) / "summary.json").read_text())

        self.assertIn("groups", summary)
        groups_by_population = {
            group["dimensions"]["population"]: group
            for group in summary["groups"].values()
        }
        self.assertEqual(
            groups_by_population["visual"]["agent"]["downtime_seconds"]["mean"],
            2.0,
        )
        self.assertEqual(
            groups_by_population["identity_control"]["agent"]["downtime_seconds"]["mean"],
            100.0,
        )

    def test_summary_partitions_oracle_from_detector_triggered_runs(self):
        def runner(config, **_kwargs):
            def result(*, baseline):
                return {
                    "schema_version": 2,
                    "metadata": {
                        "corruption": config.corruption,
                        "severity": config.severity,
                        "seed": config.seed,
                        "baseline": baseline,
                        "production_start_time": 0.0,
                        "end_time_seconds": 400.0,
                    },
                    "corrupted_accuracy_history": [],
                    "clean_evaluations": [],
                    "metrics": {"downtime_seconds": 2.0},
                }

            return {"agent": result(baseline=False), "baseline": result(baseline=True)}

        with tempfile.TemporaryDirectory() as directory:
            run_matrix(
                [DriftEpisodeConfig(corruption="identity", severity=0, production_horizon_seconds=400.0)],
                output_dir=directory,
                runner=runner,
                corruption_fn=identity_corruption,
                include_controls=True,
            )
            summary = json.loads((Path(directory) / "summary.json").read_text())

        self.assertIn("groups", summary)
        dimensions = {tuple(sorted(group["dimensions"].items())) for group in summary["groups"].values()}
        self.assertEqual(len(dimensions), 2)
        self.assertIn(
            "detector",
            {group["dimensions"]["trigger_policy"] for group in summary["groups"].values()},
        )
        self.assertIn(
            "oracle",
            {group["dimensions"]["trigger_policy"] for group in summary["groups"].values()},
        )

    def test_summary_aggregates_equal_dimensions_across_sorted_unique_seeds(self):
        def runner(config, **_kwargs):
            def result(*, baseline):
                return {
                    "schema_version": 2,
                    "metadata": {
                        "corruption": config.corruption,
                        "severity": config.severity,
                        "seed": config.seed,
                        "baseline": baseline,
                        "production_start_time": 0.0,
                        "end_time_seconds": 400.0,
                    },
                    "corrupted_accuracy_history": [],
                    "clean_evaluations": [],
                    "metrics": {"downtime_seconds": float(config.seed * 2)},
                }

            return {"agent": result(baseline=False), "baseline": result(baseline=True)}

        with tempfile.TemporaryDirectory() as directory:
            run_matrix(
                build_run_matrix(
                    seeds=[2, 1],
                    scenarios=[("noise", 2)],
                    base_config=DriftEpisodeConfig(production_horizon_seconds=400.0),
                ),
                output_dir=directory,
                runner=runner,
                corruption_fn=identity_corruption,
            )
            summary = json.loads((Path(directory) / "summary.json").read_text())

        self.assertIn("groups", summary)
        group = next(iter(summary["groups"].values()))
        self.assertEqual(group["dimensions"]["population"], "visual")
        self.assertEqual(group["seeds"], [1, 2])
        self.assertEqual(group["run_count"], 2)
        self.assertEqual(
            group["agent"]["downtime_seconds"],
            {"count": 2, "mean": 3.0, "std": 1.0, "min": 2.0, "max": 4.0},
        )

    def test_summary_partitions_client_speed_profiles(self):
        def runner(config, **_kwargs):
            def result(*, baseline):
                return {
                    "schema_version": 2,
                    "metadata": {
                        "corruption": config.corruption,
                        "severity": config.severity,
                        "seed": config.seed,
                        "baseline": baseline,
                        "production_start_time": 0.0,
                        "end_time_seconds": 400.0,
                    },
                    "corrupted_accuracy_history": [],
                    "clean_evaluations": [],
                    "metrics": {"downtime_seconds": 2.0},
                }

            return {
                "agent": result(baseline=False),
                "baseline": result(baseline=True),
            }

        configs = [
            DriftEpisodeConfig(
                corruption="noise",
                severity=2,
                seed=1,
                client_speed_profile="uniform",
                production_horizon_seconds=400.0,
            ),
            DriftEpisodeConfig(
                corruption="noise",
                severity=2,
                seed=2,
                client_speed_profile="heterogeneous",
                production_horizon_seconds=400.0,
            ),
        ]
        with tempfile.TemporaryDirectory() as directory:
            run_matrix(
                configs,
                output_dir=directory,
                runner=runner,
                corruption_fn=identity_corruption,
            )
            summary = json.loads((Path(directory) / "summary.json").read_text())

        self.assertEqual(
            {
                group["dimensions"]["client_speed_profile"]
                for group in summary["groups"].values()
            },
            {"uniform", "heterogeneous"},
        )
        self.assertEqual(len(summary["groups"]), 2)

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

    def test_matrix_cli_propagates_its_default_production_horizon_to_every_config(self):
        captured = []

        def fake_run_matrix(configs, **_kwargs):
            captured.extend(configs)
            return []

        with patch("experiments.e07_drift_agent.run_matrix.run_matrix", side_effect=fake_run_matrix):
            main([
                "--seeds", "11", "12",
                "--scenario", "noise:3",
            ])

        self.assertEqual(
            [config.production_horizon_seconds for config in captured],
            [400.0, 400.0],
        )


if __name__ == "__main__":
    unittest.main()
