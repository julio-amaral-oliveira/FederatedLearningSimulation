import copy
import json
import tempfile
import unittest
from pathlib import Path

from experiments.drift_results import (
    DriftResult,
    load_drift_result,
    summarize_pair,
    validate_pair,
)
from experiments.plot_smoke_drift import plot_drift_pair
from experiments.run_smoke_drift import run_matrix
from experiments.smoke_drift import DriftEpisodeConfig


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

    def test_loads_auditable_v3(self):
        with tempfile.TemporaryDirectory() as directory:
            result = load_drift_result(self._write(directory, _v3_payload()))

        self.assertEqual(result.schema_version, 3)
        self.assertEqual(result.tick_history[-1]["time"], 40.0)
        self.assertEqual(result.detector_config["trigger_threshold"], 0.3)
        self.assertEqual(result.runtime["clean_checkpoint_digest"], "checkpoint-42")

    def test_v3_requires_trace_configs_and_runtime(self):
        required = ("tick_history", "experiment_config", "detector_config", "runtime")
        for key in required:
            with self.subTest(key=key), tempfile.TemporaryDirectory() as directory:
                payload = _v3_payload()
                del payload[key]
                path = self._write(directory, payload)

                with self.assertRaisesRegex(ValueError, key):
                    load_drift_result(path)

    def test_unknown_schema_fails_clearly(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._write(directory, {"schema_version": 99})

            with self.assertRaisesRegex(ValueError, "unsupported drift result schema_version: 99"):
                load_drift_result(path)


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
            "production start": lambda payload: payload["metadata"].update(
                production_start_time=1.0
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
