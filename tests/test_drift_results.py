import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from experiments.drift_results import (
    DriftResult,
    load_drift_result,
    summarize_pair,
    validate_pair,
)
from experiments.result_io import (
    atomic_write_json,
    load_persisted_pair,
    save_validated_pair,
)
from experiments.plot_smoke_drift import plot_drift_pair
from experiments.run_smoke_drift import run_matrix
from experiments.smoke_drift import (
    DriftEpisodeConfig,
    main as smoke_drift_main,
    save_drift_result,
)


def _trace(time, flagged_fraction):
    return {
        "time": float(time),
        "warmup": time < 0,
        "clients": {
            "0": {
                "score": float(flagged_fraction),
                "flag": bool(flagged_fraction),
            }
        },
        "flagged_fraction": float(flagged_fraction),
    }


def _v3_payload(*, baseline=False):
    decision = {"time": 20.0, "flagged_fraction": 0.5}
    return {
        "schema_version": 3,
        "experiment_config": {
            "dataset": "cifar10",
            "num_clients": 1,
            "initial_rounds": 1,
            "retrain_rounds": 2,
            "warmup_ticks": 1,
            "monitor_tick_seconds": 10.0,
            "monitor_ticks": 4,
            "local_epochs": 1,
            "batch_size": 4,
            "timeout_percentile": 75,
            "max_train_samples_per_client": None,
            "tau": 0.5,
            "corruption": "gaussian_noise",
            "severity": 3,
            "seed": 42,
            "baseline": baseline,
            "production_horizon_seconds": 40.0,
        },
        "detector_config": {
            "detector_kind": "udd",
            "detector_alpha": 0.002,
            "detector_T": 5,
            "trigger_threshold": 0.3,
            "trigger_window_ticks": 2,
        },
        "runtime": {
            "python_version": "3.test",
            "numpy_version": "2.test",
            "torch_version": "2.test",
            "effective_device": "cpu",
            "deterministic_algorithms_enabled": False,
            "cudnn_deterministic": False,
            "cudnn_benchmark": False,
            "clean_checkpoint_digest": "checkpoint-42",
        },
        "metadata": {
            "dataset": "cifar10",
            "corruption": "gaussian_noise",
            "severity": 3,
            "seed": 42,
            "tau": 0.5,
            "baseline": baseline,
            "production_start_time": 0.0,
            "warmup_completed_time": 0.0,
            "preproduction_server_state": {"virtual_time": 0.0},
            "end_time_seconds": 40.0,
            "retrain_rounds_configured": 2,
            "monitor_tick_seconds": 10.0,
            "production_horizon_seconds": 40.0,
        },
        "corrupted_accuracy_history": [
            {"time": 0.0, "loss": 1.0, "accuracy": 0.4, "stage": "drift_onset"},
            {"time": 10.0, "loss": 0.9, "accuracy": 0.45, "stage": "monitor_tick"},
            {"time": 20.0, "loss": 0.8, "accuracy": 0.48, "stage": "monitor_tick"},
            {"time": 30.0, "loss": 0.5, "accuracy": 0.6, "stage": "retrain_round"},
            {"time": 40.0, "loss": 0.4, "accuracy": 0.65, "stage": "episode_end"},
        ],
        "clean_evaluations": [
            {"time": 0.0, "loss": 0.2, "accuracy": 0.8, "stage": "pre_drift"},
            {"time": 40.0, "loss": 0.3, "accuracy": 0.78, "stage": "final"},
        ],
        "drift_events": [{"time": 20.0, "client_id": "0", "score": 0.5}],
        "retrain_decisions": [] if baseline else [decision],
        "counterfactual_triggers": [decision] if baseline else [],
        "tick_history": [
            _trace(-10.0, 0.0),
            _trace(10.0, 0.0),
            _trace(20.0, 0.5),
            _trace(30.0, 0.0),
            _trace(40.0, 0.0),
        ],
        "retrain_round_events": (
            []
            if baseline
            else [
                {
                    "round": 2,
                    "started_time": 20.0,
                    "completed_time": 25.0,
                    "aggregated": True,
                },
                {
                    "round": 3,
                    "started_time": 25.0,
                    "completed_time": 30.0,
                    "aggregated": True,
                },
            ]
        ),
        "metrics": {
            "downtime_seconds": 30.0 if baseline else 10.0,
            "detection_delay_seconds": None if baseline else 20.0,
            "retraining_duration_seconds": None if baseline else 10.0,
            "time_to_recovery_seconds": None if baseline else 10.0,
            "clean_retention_delta": 0.0 if baseline else -0.02,
            "corrupted_accuracy_gain": 0.0 if baseline else 0.25,
        },
    }


def _v2_payload(*, baseline=False):
    return {
        "schema_version": 2,
        "metadata": {
            "dataset": "cifar10",
            "corruption": "gaussian_noise",
            "severity": 3,
            "seed": 42,
            "tau": 0.5,
            "baseline": baseline,
            "production_start_time": 0.0,
            "end_time_seconds": 40.0,
        },
        "corrupted_accuracy_history": [
            {"time": 0.0, "accuracy": 0.4, "stage": "drift_onset"},
            {"time": 40.0, "accuracy": 0.6, "stage": "episode_end"},
        ],
        "clean_evaluations": [
            {"time": 0.0, "accuracy": 0.8, "stage": "pre_drift"},
            {"time": 40.0, "accuracy": 0.75, "stage": "final"},
        ],
        "retrain_decisions": [] if baseline else [{"time": 20.0, "flagged_fraction": 0.5}],
        "counterfactual_triggers": [],
        "retrain_round_events": [],
        "metrics": {
            "downtime_seconds": 30.0 if baseline else 10.0,
            "detection_delay_seconds": None if baseline else 20.0,
            "recovery_duration_seconds": None if baseline else 12.0,
        },
    }


class TestDriftResultLoading(unittest.TestCase):
    def _write(self, directory, payload, filename="result.json"):
        path = Path(directory) / filename
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_loads_legacy_v2_histories_and_existing_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            result = load_drift_result(self._write(directory, _v2_payload()))

        self.assertIsInstance(result, DriftResult)
        self.assertEqual(result.schema_version, 2)
        self.assertEqual(result.corrupted_accuracy_history[-1]["accuracy"], 0.6)
        self.assertEqual(result.clean_evaluations[-1]["accuracy"], 0.75)
        self.assertEqual(result.metrics["recovery_duration_seconds"], 12.0)

    def test_loads_legacy_v2_identity_severity_zero_without_v3_validation(self):
        payload = _v2_payload()
        payload["metadata"].update(corruption="identity", severity=0)

        result = DriftResult.from_payload(payload)

        self.assertEqual(result.schema_version, 2)
        self.assertEqual(result.metadata["corruption"], "identity")
        self.assertEqual(result.metadata["severity"], 0)

    def test_loads_auditable_v3(self):
        with tempfile.TemporaryDirectory() as directory:
            result = load_drift_result(self._write(directory, _v3_payload()))

        self.assertEqual(result.schema_version, 3)
        self.assertEqual(result.tick_history[-1]["time"], 40.0)
        self.assertEqual(result.detector_config["trigger_threshold"], 0.3)
        self.assertEqual(result.runtime["clean_checkpoint_digest"], "checkpoint-42")

    def test_v3_requires_trace_configs_and_runtime(self):
        required = (
            "tick_history",
            "drift_events",
            "retrain_decisions",
            "counterfactual_triggers",
            "retrain_round_events",
            "experiment_config",
            "detector_config",
            "runtime",
        )
        for key in required:
            with self.subTest(key=key), tempfile.TemporaryDirectory() as directory:
                payload = _v3_payload()
                del payload[key]
                path = self._write(directory, payload)

                with self.assertRaisesRegex(ValueError, key):
                    load_drift_result(path)

    def test_v3_rejects_an_overshot_end_time(self):
        payload = _v3_payload()
        payload["metadata"]["end_time_seconds"] = 41.0

        with self.assertRaisesRegex(ValueError, "end_time_seconds"):
            DriftResult.from_payload(payload)

    def test_v3_requires_every_detector_and_runtime_identity_field(self):
        cases = (
            ("detector_config", "detector_kind"),
            ("detector_config", "detector_alpha"),
            ("detector_config", "detector_T"),
            ("detector_config", "trigger_threshold"),
            ("detector_config", "trigger_window_ticks"),
            ("runtime", "python_version"),
            ("runtime", "numpy_version"),
            ("runtime", "torch_version"),
            ("runtime", "effective_device"),
        )
        for section, field in cases:
            with self.subTest(section=section, field=field):
                payload = _v3_payload()
                del payload[section][field]

                with self.assertRaisesRegex(ValueError, field):
                    DriftResult.from_payload(payload)

    def test_v3_rejects_invalid_detector_values_and_types(self):
        cases = (
            ("detector_kind", "other"),
            ("detector_alpha", 0.0),
            ("detector_alpha", 1.0),
            ("detector_alpha", True),
            ("detector_T", 0),
            ("detector_T", True),
            ("trigger_threshold", -0.1),
            ("trigger_threshold", 1.1),
            ("trigger_threshold", True),
            ("trigger_window_ticks", 0),
            ("trigger_window_ticks", True),
        )
        for field, value in cases:
            with self.subTest(field=field, value=value):
                payload = _v3_payload()
                payload["detector_config"][field] = value

                with self.assertRaisesRegex(ValueError, field):
                    DriftResult.from_payload(payload)

    def test_v3_rejects_invalid_experiment_identity_and_horizon_values(self):
        cases = (
            ("corruption", ""),
            ("severity", True),
            ("severity", 0),
            ("severity", 6),
            ("seed", True),
            ("seed", 42.0),
            ("tau", True),
            ("tau", -0.1),
            ("tau", 1.1),
            ("baseline", 0),
        )
        for field, value in cases:
            with self.subTest(field=field, value=value):
                payload = _v3_payload()
                payload["metadata"][field] = value
                payload["experiment_config"][field] = value

                with self.assertRaisesRegex(ValueError, field):
                    DriftResult.from_payload(payload)

        horizon_cases = (
            ("production_start_time", float("inf")),
            ("end_time_seconds", float("nan")),
            ("production_horizon_seconds", 0.0),
            ("production_horizon_seconds", True),
        )
        for field, value in horizon_cases:
            with self.subTest(field=field, value=value):
                payload = _v3_payload()
                payload["metadata"][field] = value
                if field == "production_horizon_seconds":
                    payload["experiment_config"][field] = value

                with self.assertRaisesRegex(ValueError, field):
                    DriftResult.from_payload(payload)

    def test_v3_allows_only_identity_severity_zero(self):
        identity = _v3_payload()
        identity["metadata"].update(corruption="identity", severity=0)
        identity["experiment_config"].update(corruption="identity", severity=0)

        result = DriftResult.from_payload(identity)

        self.assertEqual(result.metadata["corruption"], "identity")
        self.assertEqual(result.metadata["severity"], 0)

        invalid_identity = copy.deepcopy(identity)
        invalid_identity["metadata"]["severity"] = 1
        invalid_identity["experiment_config"]["severity"] = 1
        with self.assertRaisesRegex(ValueError, "identity severity"):
            DriftResult.from_payload(invalid_identity)

    def test_v3_requires_non_empty_runtime_identity_strings(self):
        for field in (
            "python_version",
            "numpy_version",
            "torch_version",
            "effective_device",
            "clean_checkpoint_digest",
        ):
            with self.subTest(field=field):
                payload = _v3_payload()
                payload["runtime"][field] = " "

                with self.assertRaisesRegex(ValueError, field):
                    DriftResult.from_payload(payload)

    def test_v3_rejects_invalid_or_unordered_history_timestamps(self):
        cases = (
            (
                "corrupted_accuracy_history",
                lambda payload: payload["corrupted_accuracy_history"][1].update(
                    time=float("nan")
                ),
            ),
            (
                "clean_evaluations",
                lambda payload: payload["clean_evaluations"][1].update(time=-1.0),
            ),
            (
                "tick_history",
                lambda payload: payload["tick_history"][3].update(time=15.0),
            ),
            (
                "drift_events",
                lambda payload: payload["drift_events"][0].update(time=41.0),
            ),
        )
        for field, mutate in cases:
            with self.subTest(field=field):
                payload = _v3_payload()
                mutate(payload)

                with self.assertRaisesRegex(ValueError, field):
                    DriftResult.from_payload(payload)

    def test_v3_does_not_apply_horizon_tolerance_to_event_ordering(self):
        cases = (
            lambda payload: payload["tick_history"][3].update(
                time=20.0 - 5e-10
            ),
            lambda payload: payload["retrain_round_events"][0].update(
                started_time=20.0 - 5e-10
            ),
            lambda payload: payload["retrain_round_events"][0].update(
                started_time=25.0,
                completed_time=25.0 - 5e-10,
            ),
        )
        for mutate in cases:
            with self.subTest(mutate=mutate):
                payload = _v3_payload()
                mutate(payload)

                with self.assertRaisesRegex(
                    ValueError, "tick_history|retrain_round_events"
                ):
                    DriftResult.from_payload(payload)

    def test_v3_rejects_a_baseline_retraining_event_at_construction(self):
        payload = _v3_payload(baseline=True)
        payload["retrain_round_events"].append(
            {
                "round": 2,
                "started_time": 20.0,
                "completed_time": 25.0,
                "aggregated": True,
            }
        )

        with self.assertRaisesRegex(ValueError, "baseline.*retrain_round_events"):
            DriftResult.from_payload(payload)

    def test_v3_rejects_an_agent_decision_with_the_wrong_round_count(self):
        payload = _v3_payload()
        payload["retrain_round_events"].pop()

        with self.assertRaisesRegex(ValueError, "retrain_round_events.*expected=2"):
            DriftResult.from_payload(payload)

    def test_v3_rejects_a_round_event_after_the_horizon(self):
        payload = _v3_payload()
        payload["retrain_round_events"][-1]["completed_time"] = 41.0

        with self.assertRaisesRegex(ValueError, "retrain_round_events.*horizon"):
            DriftResult.from_payload(payload)

    def test_unknown_schema_fails_clearly(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._write(directory, {"schema_version": 99})

            with self.assertRaisesRegex(ValueError, "unsupported drift result schema_version: 99"):
                load_drift_result(path)

    def test_v3_schema_version_must_be_the_integer_three(self):
        payload = _v3_payload()
        payload["schema_version"] = 3.0

        with self.assertRaisesRegex(ValueError, "schema_version"):
            DriftResult.from_payload(payload)


class TestDriftPairValidation(unittest.TestCase):
    def test_accepts_audited_v3_pair(self):
        validate_pair(_v3_payload(), _v3_payload(baseline=True))

    def test_rejects_each_audited_pair_divergence(self):
        mutations = {
            "corruption": lambda payload: (
                payload["metadata"].update(corruption="fog"),
                payload["experiment_config"].update(corruption="fog"),
            ),
            "seed": lambda payload: (
                payload["metadata"].update(seed=7),
                payload["experiment_config"].update(seed=7),
            ),
            "checkpoint": lambda payload: payload["runtime"].update(
                clean_checkpoint_digest="other"
            ),
            "production start": lambda payload: (
                payload["metadata"].update(
                    production_start_time=-1.0,
                    production_horizon_seconds=41.0,
                ),
                payload["experiment_config"].update(
                    production_horizon_seconds=41.0
                ),
            ),
            "horizon": lambda payload: (
                payload["metadata"].update(
                    production_horizon_seconds=50.0,
                    end_time_seconds=50.0,
                ),
                payload["experiment_config"].update(production_horizon_seconds=50.0),
            ),
            "trace": lambda payload: payload["tick_history"][1].update(
                flagged_fraction=0.25
            ),
            "trigger": lambda payload: payload["counterfactual_triggers"][0].update(
                time=30.0
            ),
        }
        for message, mutate in mutations.items():
            with self.subTest(message=message):
                baseline = _v3_payload(baseline=True)
                mutate(baseline)

                with self.assertRaisesRegex(ValueError, message):
                    validate_pair(_v3_payload(), baseline)

    def test_rejects_an_extra_unmatched_counterfactual_trigger(self):
        baseline = _v3_payload(baseline=True)
        baseline["counterfactual_triggers"].append(
            {"time": 30.0, "flagged_fraction": 0.5}
        )

        with self.assertRaisesRegex(ValueError, "counterfactual"):
            validate_pair(_v3_payload(), baseline)

    def test_accepts_a_v3_pair_when_the_control_takes_no_action(self):
        agent = _v3_payload()
        agent["retrain_decisions"] = []
        agent["retrain_round_events"] = []
        baseline = _v3_payload(baseline=True)
        baseline["counterfactual_triggers"] = []

        validate_pair(agent, baseline)
        summary = summarize_pair(agent, baseline)

        self.assertTrue(summary["audited_pair"])

    def test_accepts_v2_only_as_legacy_unverified_pair(self):
        agent = _v2_payload()
        baseline = _v2_payload(baseline=True)

        validate_pair(agent, baseline)
        summary = summarize_pair(agent, baseline)

        self.assertTrue(summary["legacy_unverified_pair"])
        self.assertFalse(summary["audited_pair"])
        self.assertAlmostEqual(summary["clean_retention_delta"], -0.05)

    def test_summarizes_audited_pair_metrics(self):
        summary = summarize_pair(_v3_payload(), _v3_payload(baseline=True))

        self.assertEqual(summary["downtime_avoided_seconds"], 20.0)
        self.assertAlmostEqual(summary["downtime_avoided_percent"], 100.0 * 20.0 / 30.0)
        self.assertEqual(summary["detection_delay_seconds"], 20.0)
        self.assertEqual(summary["time_to_recovery_seconds"], 10.0)
        self.assertEqual(summary["clean_retention_delta"], -0.02)
        self.assertTrue(summary["audited_pair"])
        self.assertFalse(summary["legacy_unverified_pair"])

    def test_runner_does_not_persist_a_rejected_pair(self):
        def invalid_runner(_config, **_kwargs):
            baseline = _v3_payload(baseline=True)
            baseline["runtime"]["clean_checkpoint_digest"] = "other"
            return {"agent": _v3_payload(), "baseline": baseline}

        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "checkpoint"):
                run_matrix(
                    [DriftEpisodeConfig(corruption="gaussian_noise", severity=3)],
                    output_dir=directory,
                    runner=invalid_runner,
                    corruption_fn=lambda value, *_args, **_kwargs: value,
                )

            self.assertEqual(list(Path(directory).rglob("*.json")), [])


class TestDriftResultPersistence(unittest.TestCase):
    def test_atomic_write_failure_preserves_existing_destination(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "result.json"
            destination.write_text('{"previous": true}', encoding="utf-8")

            with mock.patch(
                "experiments.result_io.os.fsync",
                side_effect=OSError("injected write failure"),
            ):
                with self.assertRaisesRegex(OSError, "injected write failure"):
                    atomic_write_json(destination, {"replacement": True})

            self.assertEqual(
                destination.read_text(encoding="utf-8"),
                '{"previous": true}',
            )
            self.assertEqual(list(Path(directory).glob(".result.json.*.tmp")), [])

    def test_standalone_result_write_failure_preserves_existing_destination(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "smoke_drift.json"
            destination.write_text('{"previous": true}', encoding="utf-8")

            with mock.patch(
                "experiments.result_io.os.fsync",
                side_effect=OSError("injected standalone write failure"),
            ):
                with self.assertRaisesRegex(
                    OSError,
                    "injected standalone write failure",
                ):
                    save_drift_result(_v3_payload(), directory)

            self.assertEqual(
                destination.read_text(encoding="utf-8"),
                '{"previous": true}',
            )

    def test_pair_publication_failure_invalidates_existing_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            save_validated_pair(
                output_dir,
                _v3_payload(),
                _v3_payload(baseline=True),
            )
            real_replace = __import__("os").replace

            def fail_baseline_publication(source, destination):
                if Path(destination).name == "baseline.json":
                    raise OSError("injected baseline publication failure")
                return real_replace(source, destination)

            with mock.patch(
                "experiments.result_io.os.replace",
                side_effect=fail_baseline_publication,
            ):
                with self.assertRaisesRegex(
                    OSError,
                    "injected baseline publication failure",
                ):
                    save_validated_pair(
                        output_dir,
                        _v3_payload(),
                        _v3_payload(baseline=True),
                    )

            self.assertFalse((output_dir / "pair-manifest.json").exists())
            with self.assertRaisesRegex(ValueError, "manifest"):
                load_persisted_pair(output_dir)

    def test_pair_manifest_is_last_and_detects_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            published = []
            real_replace = __import__("os").replace

            def record_publication(source, destination):
                published.append(Path(destination).name)
                return real_replace(source, destination)

            with mock.patch(
                "experiments.result_io.os.replace",
                side_effect=record_publication,
            ):
                agent_path, baseline_path = save_validated_pair(
                    output_dir,
                    _v3_payload(),
                    _v3_payload(baseline=True),
                )

            self.assertEqual(agent_path.name, "agent.json")
            self.assertEqual(baseline_path.name, "baseline.json")
            self.assertEqual(published[-1], "pair-manifest.json")
            manifest = json.loads(
                (output_dir / "pair-manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["schema_version"], 1)
            self.assertTrue(manifest["generation_id"])
            self.assertEqual(
                manifest["artifacts"]["agent"]["sha256"],
                hashlib.sha256(agent_path.read_bytes()).hexdigest(),
            )
            self.assertEqual(
                manifest["artifacts"]["baseline"]["sha256"],
                hashlib.sha256(baseline_path.read_bytes()).hexdigest(),
            )
            agent, baseline = load_persisted_pair(output_dir)
            self.assertFalse(agent.metadata["baseline"])
            self.assertTrue(baseline.metadata["baseline"])

            agent_path.write_bytes(agent_path.read_bytes() + b"\n")

            with self.assertRaisesRegex(ValueError, "digest"):
                load_persisted_pair(output_dir)

    def test_matrix_runner_publishes_canonical_manifested_pairs(self):
        def runner(_config, **_kwargs):
            return {
                "agent": _v3_payload(),
                "baseline": _v3_payload(baseline=True),
            }

        with tempfile.TemporaryDirectory() as directory:
            paths = run_matrix(
                [DriftEpisodeConfig(corruption="gaussian_noise", severity=3)],
                output_dir=directory,
                runner=runner,
                corruption_fn=lambda value, *_args, **_kwargs: value,
            )
            pair_directory = paths[0][0].parent

            self.assertEqual(paths[0][0].name, "agent.json")
            self.assertEqual(paths[0][1].name, "baseline.json")
            self.assertTrue((pair_directory / "pair-manifest.json").is_file())
            load_persisted_pair(pair_directory)

    def test_single_run_cli_publishes_one_manifested_pair(self):
        results = {
            "agent": _v3_payload(),
            "baseline": _v3_payload(baseline=True),
        }
        with tempfile.TemporaryDirectory() as directory:
            with (
                mock.patch(
                    "experiments.smoke_drift.run_drift_comparison",
                    return_value=results,
                ),
                mock.patch(
                    "sys.argv",
                    ["smoke_drift", "--output-dir", directory],
                ),
            ):
                smoke_drift_main()

            self.assertTrue((Path(directory) / "pair-manifest.json").is_file())
            load_persisted_pair(directory)

    def test_summary_write_failure_preserves_existing_destination(self):
        def runner(_config, **_kwargs):
            return {
                "agent": _v3_payload(),
                "baseline": _v3_payload(baseline=True),
            }

        with tempfile.TemporaryDirectory() as directory:
            summary_path = Path(directory) / "summary.json"
            summary_path.write_text('{"previous": true}', encoding="utf-8")
            real_replace = __import__("os").replace

            def fail_summary_publication(source, destination):
                if Path(destination).name == "summary.json":
                    raise OSError("injected summary publication failure")
                return real_replace(source, destination)

            with mock.patch(
                "experiments.result_io.os.replace",
                side_effect=fail_summary_publication,
            ):
                with self.assertRaisesRegex(
                    OSError,
                    "injected summary publication failure",
                ):
                    run_matrix(
                        [DriftEpisodeConfig(corruption="gaussian_noise", severity=3)],
                        output_dir=directory,
                        runner=runner,
                        corruption_fn=lambda value, *_args, **_kwargs: value,
                    )

            self.assertEqual(
                summary_path.read_text(encoding="utf-8"),
                '{"previous": true}',
            )


class TestDriftPlot(unittest.TestCase):
    def test_generates_non_empty_png_with_agg_backend(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "comparison.png"

            result = plot_drift_pair(
                _v3_payload(baseline=True),
                _v3_payload(),
                output,
            )

            self.assertEqual(result, output)
            self.assertTrue(output.is_file())
            self.assertGreater(output.stat().st_size, 0)
            import matplotlib

            self.assertEqual(matplotlib.get_backend().lower(), "agg")


if __name__ == "__main__":
    unittest.main()
