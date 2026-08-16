"""Construction-only tests for the E09 B1-B4 review-plan runner.

No test in this module executes training: the scheduled runner is inspected
through ``unittest.mock``, and the staggered schedule is validated with the
fast ``_FakeServer``/``_ScriptedMonitor`` doubles from ``test_drift_episode``,
following the reuse pattern of ``test_drift_controls``.
"""

import contextlib
import io
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from experiments.e07_drift_agent.episode import (
    DriftEpisodeConfig,
    run_drift_episode,
)
from experiments.e09_drift_gradual.run_b1_b4 import (
    DEFAULT_ARMS,
    DEFAULT_QUORUMS,
    DEFAULT_RAMP_TICKS,
    DEFAULT_SCHEDULED_TIMES,
    DEFAULT_TAU_VALUES,
    DEFAULT_WINDOWS,
    GRID_TAU_SOURCE,
    _scheduled_runner,
    build_matrix,
    compute_tau_sweep_rows,
    main,
    parse_args,
    run_tau_sweep,
)
from experiments.shared.comparison_core import compute_downtime
from experiments.shared.drift_controls import ScheduledMonitor, identity_corruption
from experiments.shared.registry import temporary_output_path
from tests.test_drift_episode import _FakeServer, _ScriptedMonitor, _identity


class TestBuildMatrixCounts(unittest.TestCase):
    def test_default_matrix_counts_per_arm(self):
        plans = build_matrix(dry_run=True)
        by_name = {plan.name: plan for plan in plans}

        self.assertEqual(
            [plan.name for plan in plans],
            ["scheduled", "first-alarm", "no-drift", "quorum", "window", "staggered"],
        )
        self.assertEqual(by_name["scheduled"].episode_count, 120)
        self.assertEqual(by_name["first-alarm"].episode_count, 60)
        self.assertEqual(by_name["no-drift"].episode_count, 15)
        self.assertEqual(by_name["quorum"].episode_count, 120)
        self.assertEqual(by_name["window"].episode_count, 60)
        self.assertEqual(by_name["staggered"].episode_count, 20)
        self.assertEqual(
            sum(plan.episode_count for plan in plans),
            395,
        )

    def test_default_subvariant_directories_are_stable(self):
        plans = build_matrix(dry_run=True)
        by_name = {plan.name: plan for plan in plans}

        self.assertEqual(
            [group.subdir for group in by_name["scheduled"].groups],
            ["scheduled_100s", "scheduled_150s"],
        )
        self.assertEqual(
            [group.subdir for group in by_name["first-alarm"].groups],
            ["first_alarm"],
        )
        self.assertEqual(
            [group.subdir for group in by_name["no-drift"].groups],
            ["no_drift"],
        )
        self.assertEqual(
            [group.subdir for group in by_name["quorum"].groups],
            ["quorum_0.2", "quorum_0.5"],
        )
        self.assertEqual(
            [group.subdir for group in by_name["window"].groups],
            ["window_1"],
        )
        self.assertEqual(
            [group.subdir for group in by_name["staggered"].groups],
            ["staggered"],
        )

    def test_sensitivity_arms_vary_quorum_and_window_only_as_requested(self):
        plans = build_matrix(dry_run=True)
        by_name = {plan.name: plan for plan in plans}

        quorum_groups = {group.subdir: group for group in by_name["quorum"].groups}
        self.assertEqual(
            {config.trigger_threshold for config in quorum_groups["quorum_0.2"].configs},
            {0.2},
        )
        self.assertEqual(
            {config.trigger_threshold for config in quorum_groups["quorum_0.5"].configs},
            {0.5},
        )
        self.assertEqual(
            {config.trigger_window_ticks for group in by_name["quorum"].groups
             for config in group.configs},
            {2},
        )
        self.assertEqual(
            {config.trigger_threshold for config in by_name["window"].groups[0].configs},
            {0.30},
        )
        self.assertEqual(
            {config.trigger_window_ticks for config in by_name["window"].groups[0].configs},
            {1},
        )

    def test_cli_defaults_match_the_contract(self):
        args = parse_args([])

        self.assertEqual(args.seeds, [42, 43, 44, 45, 46])
        self.assertEqual(args.ramp_ticks, list(DEFAULT_RAMP_TICKS))
        self.assertEqual(args.arms, list(DEFAULT_ARMS))
        self.assertEqual(args.scheduled_times, list(DEFAULT_SCHEDULED_TIMES))
        self.assertEqual(args.quorums, list(DEFAULT_QUORUMS))
        self.assertEqual(args.windows, list(DEFAULT_WINDOWS))
        self.assertEqual(args.tau_values, list(DEFAULT_TAU_VALUES))
        self.assertIsNone(args.tau_source)
        self.assertEqual(
            Path(args.output_dir),
            temporary_output_path("e09-gradual-drift", "cifar-10") / "b1b4",
        )


class TestScheduledRunner(unittest.TestCase):
    def test_scheduled_runner_injects_scheduled_monitor_at_the_requested_time(self):
        runner = _scheduled_runner(100.0)

        with patch(
            "experiments.e09_drift_gradual.run_b1_b4.run_drift_comparison"
        ) as comparison:
            runner(DriftEpisodeConfig(), corruption_fn=_identity)

        comparison.assert_called_once()
        monitor = comparison.call_args.kwargs["monitor_factory"](None, None)
        self.assertIsInstance(monitor, ScheduledMonitor)
        self.assertEqual(monitor.trigger_after_seconds, 100.0)

    def test_scheduled_plan_groups_carry_the_timed_runner(self):
        plan = build_matrix(arms=("scheduled",), scheduled_times=(100.0, 150.0))[0]

        self.assertEqual(
            [group.subdir for group in plan.groups],
            ["scheduled_100s", "scheduled_150s"],
        )
        for time, group in zip((100.0, 150.0), plan.groups):
            runner = group.runner
            if runner is None:
                self.fail("scheduled groups must carry a runner")
            with patch(
                "experiments.e09_drift_gradual.run_b1_b4.run_drift_comparison"
            ) as comparison:
                runner(DriftEpisodeConfig(), corruption_fn=_identity)
            monitor = comparison.call_args.kwargs["monitor_factory"](None, None)
            self.assertEqual(monitor.trigger_after_seconds, time)


class TestNoDriftArm(unittest.TestCase):
    def test_no_drift_configs_are_identity_and_use_identity_corruption(self):
        plan = build_matrix(arms=("no-drift",))[0]

        self.assertIs(plan.corruption_fn, identity_corruption)
        self.assertEqual(plan.episode_count, 15)
        for config in plan.groups[0].configs:
            self.assertEqual(config.corruption, "identity")
            self.assertEqual(config.severity, 0)
        self.assertEqual(
            {config.drift_ramp_ticks for config in plan.groups[0].configs},
            {5, 10, 20},
        )
        self.assertEqual(
            {config.seed for config in plan.groups[0].configs},
            {42, 43, 44, 45, 46},
        )


class TestStaggeredArm(unittest.TestCase):
    def test_staggered_configs_pass_episode_validation(self):
        plan = build_matrix(arms=("staggered",))[0]
        config = plan.groups[0].configs[0]

        self.assertEqual(
            config.drift_onset_ticks,
            {index: index for index in range(config.num_clients)},
        )
        self.assertIsNone(config.drift_ramp_ticks)
        self.assertEqual(config.retrain_rounds, 1)
        self.assertEqual(config.production_horizon_seconds, 600.0)

        lightweight = replace(
            config,
            initial_rounds=1,
            monitor_ticks=3,
            monitor_tick_seconds=10.0,
            production_horizon_seconds=None,
        )
        result = run_drift_episode(
            lightweight,
            server=_FakeServer(),
            monitor=_ScriptedMonitor([False]),
            corruption_fn=_identity,
        )
        self.assertEqual(result["schema_version"], 3)
        self.assertEqual(result["metadata"]["corruption"], config.corruption)

    def test_staggered_plan_covers_scenarios_times_seeds(self):
        plan = build_matrix(arms=("staggered",))[0]

        self.assertEqual(plan.episode_count, 20)
        self.assertEqual(
            len({(c.corruption, c.severity) for c in plan.groups[0].configs}),
            4,
        )
        self.assertEqual(len({c.seed for c in plan.groups[0].configs}), 5)


class TestTauArm(unittest.TestCase):
    def test_tau_arm_dry_run_lists_sources_without_reading_data(self):
        plans = build_matrix(arms=("tau",), dry_run=True)

        self.assertEqual(len(plans), 1)
        tau = plans[0].tau
        if tau is None:
            self.fail("expected a tau plan")
        self.assertEqual(tau.sources, (GRID_TAU_SOURCE,))
        self.assertEqual(tau.tau_values, (0.40, 0.45, 0.50, 0.55))

    def test_tau_dry_run_cli_does_not_execute_any_run(self):
        with patch(
            "experiments.e09_drift_gradual.run_b1_b4.run_matrix"
        ) as run_matrix_mock:
            with contextlib.redirect_stdout(io.StringIO()):
                main(["--dry-run", "--arms", "tau"])

        run_matrix_mock.assert_not_called()

    def test_tau_sweep_recomputes_downtime_across_v2_and_v3_payloads(self):
        history = [
            {"time": 0.0, "loss": 0.5, "accuracy": 0.80},
            {"time": 100.0, "loss": 0.5, "accuracy": 0.30},
            {"time": 200.0, "loss": 0.5, "accuracy": 0.60},
        ]
        end_time = 250.0
        agent = {
            "schema_version": 3,
            "metadata": {
                "corruption": "fog",
                "severity": 4,
                "seed": 42,
                "end_time_seconds": end_time,
            },
            "experiment_config": {
                "corruption": "fog",
                "severity": 4,
                "seed": 42,
                "drift_ramp_ticks": 10,
            },
            "corrupted_accuracy_history": history,
        }
        baseline = {
            "schema_version": 2,
            "metadata": {
                "corruption": "fog",
                "severity": 4,
                "seed": 42,
                "end_time_seconds": end_time,
            },
            "corrupted_accuracy_history": history,
        }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "scheduled_100s"
            pair = root / "fog_sev4" / "seed_42" / "variant"
            pair.mkdir(parents=True)
            (pair / "agent.json").write_text(json.dumps(agent), encoding="utf-8")
            (pair / "baseline.json").write_text(json.dumps(baseline), encoding="utf-8")

            rows = compute_tau_sweep_rows((root,), (0.40, 0.50))

            self.assertEqual(len(rows), 2)
            row = next(row for row in rows if row["tau"] == 0.50)
            self.assertEqual(row["arm"], "scheduled_100s")
            self.assertEqual(row["source"], str(root))
            self.assertEqual(row["corruption"], "fog")
            self.assertEqual(row["severity"], 4)
            self.assertEqual(row["seed"], 42)
            self.assertEqual(row["ramp_ticks"], 10)
            expected = compute_downtime(history, 0.50, end_time=end_time)
            self.assertAlmostEqual(row["downtime_agent"], expected)
            self.assertAlmostEqual(row["downtime_baseline"], expected)
            # The sources are never modified.
            self.assertEqual(
                json.loads((pair / "agent.json").read_text(encoding="utf-8")),
                agent,
            )
            self.assertEqual(
                json.loads((pair / "baseline.json").read_text(encoding="utf-8")),
                baseline,
            )

    def test_tau_sweep_persists_csv_and_markdown(self):
        history = [
            {"time": 0.0, "loss": 0.5, "accuracy": 0.90},
            {"time": 100.0, "loss": 0.5, "accuracy": 0.85},
        ]
        payload = {
            "schema_version": 3,
            "metadata": {
                "corruption": "fog",
                "severity": 4,
                "seed": 42,
                "end_time_seconds": 150.0,
            },
            "experiment_config": {
                "corruption": "fog",
                "severity": 4,
                "seed": 42,
                "drift_ramp_ticks": None,
            },
            "corrupted_accuracy_history": history,
        }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "no_drift"
            pair = root / "fog_sev4" / "seed_42" / "variant"
            pair.mkdir(parents=True)
            (pair / "agent.json").write_text(json.dumps(payload), encoding="utf-8")
            (pair / "baseline.json").write_text(json.dumps(payload), encoding="utf-8")

            csv_path, md_path = run_tau_sweep(
                (root,), (0.5,), Path(directory) / "tau_sweep"
            )

            self.assertTrue(csv_path.is_file())
            self.assertTrue(md_path.is_file())
            csv_text = csv_path.read_text(encoding="utf-8")
            self.assertIn(
                "source,corruption,severity,ramp_ticks,seed,arm,tau,"
                "downtime_agent,downtime_baseline",
                csv_text,
            )
            self.assertIn("no_drift", csv_text)
            md_text = md_path.read_text(encoding="utf-8")
            self.assertIn("mean_downtime_agent", md_text)
            self.assertIn("mean_downtime_baseline", md_text)


class TestDryRunNeverRuns(unittest.TestCase):
    def test_dry_run_never_executes_run_matrix(self):
        with patch(
            "experiments.e09_drift_gradual.run_b1_b4.run_matrix"
        ) as run_matrix_mock:
            with contextlib.redirect_stdout(io.StringIO()):
                main(["--dry-run"])

        run_matrix_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()