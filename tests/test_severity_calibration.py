import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from experiments.e07_drift_agent.calibrate import calibrate_severities, save_calibration


class TestCalibrateSeverities(unittest.TestCase):
    def test_calibration_write_failure_preserves_existing_destination(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "severity_calibration.json"
            destination.write_text('{"previous": true}', encoding="utf-8")

            with patch(
                "experiments.shared.result_io.os.fsync",
                side_effect=OSError("injected calibration write failure"),
            ):
                with self.assertRaisesRegex(
                    OSError,
                    "injected calibration write failure",
                ):
                    save_calibration({"replacement": True}, directory)

            self.assertEqual(
                destination.read_text(encoding="utf-8"),
                '{"previous": true}',
            )

    def test_materializes_iterables_to_keep_the_cartesian_product_complete(self):
        result = calibrate_severities(
            iter(["noise", "fog"]),
            iter([1, 2]),
            lambda _corruption, severity: severity / 10,
            minimum_accuracy=0.05,
            maximum_accuracy=0.25,
        )

        self.assertEqual(len(result["rows"]), 4)
        self.assertEqual(result["selected"]["noise"]["severity"], 1)
        self.assertEqual(result["selected"]["fog"]["severity"], 1)

    def test_evaluates_the_full_cartesian_table_after_a_match(self):
        calls = []

        def evaluate_accuracy(corruption, severity):
            calls.append((corruption, severity))
            return {("noise", 1): 0.40, ("noise", 2): 0.20, ("fog", 1): 0.60, ("fog", 2): 0.30}[
                (corruption, severity)
            ]

        result = calibrate_severities(
            ["noise", "fog"],
            [1, 2],
            evaluate_accuracy,
            minimum_accuracy=0.25,
            maximum_accuracy=0.50,
        )

        self.assertEqual(calls, [("noise", 1), ("noise", 2), ("fog", 1), ("fog", 2)])
        self.assertEqual(
            result["rows"],
            [
                {"corruption": "noise", "severity": 1, "accuracy": 0.40},
                {"corruption": "noise", "severity": 2, "accuracy": 0.20},
                {"corruption": "fog", "severity": 1, "accuracy": 0.60},
                {"corruption": "fog", "severity": 2, "accuracy": 0.30},
            ],
        )

    def test_selects_lowest_severity_strictly_inside_bounds_or_none(self):
        accuracies = {
            ("noise", 1): 0.25,
            ("noise", 2): 0.40,
            ("noise", 3): 0.30,
            ("fog", 1): 0.50,
            ("fog", 2): 0.20,
            ("fog", 3): 0.60,
        }

        result = calibrate_severities(
            ["noise", "fog"],
            [3, 1, 2],
            lambda corruption, severity: accuracies[(corruption, severity)],
            minimum_accuracy=0.25,
            maximum_accuracy=0.50,
        )

        self.assertEqual(result["selected"]["noise"], {"severity": 2, "accuracy": 0.40})
        self.assertIsNone(result["selected"]["fog"])
        self.assertEqual(len(result["rows"]), 6)
        self.assertEqual(
            result["accuracy_interval"], {"minimum": 0.25, "maximum": 0.50, "strict": True}
        )

    def test_calibrates_every_copy_with_episode_test_seed_from_one_clean_server(self):
        from experiments.e07_drift_agent.calibrate import calibrate_clean_checkpoint

        class FakeServer:
            def __init__(self):
                self.global_model = torch.nn.Linear(1, 1)
                self.testing_data = (
                    np.zeros((2, 1, 1, 1), dtype=np.float32),
                    np.array([0, 1], dtype=np.int64),
                )
                self.run_rounds_calls = []
                self.evaluated_datasets = []

            def run_rounds(self, count, *, record_default_metrics=False):
                self.run_rounds_calls.append((count, record_default_metrics))

            def evaluate_dataset(self, dataset):
                self.evaluated_datasets.append(dataset)
                return 0.0, float(dataset[0][0, 0, 0, 0])

        server = FakeServer()
        config = SimpleNamespace(dataset="tiny", initial_rounds=3, seed=17)

        def corruption(inputs, _corruption, severity, *, seed):
            self.assertEqual(seed, 10_016)
            return inputs + severity / 10

        with patch("experiments.e07_drift_agent.episode._build_server", return_value=server) as build_server:
            result = calibrate_clean_checkpoint(
                config,
                corruption_names=["noise", "fog"],
                severities=[1, 2],
                minimum_accuracy=0.15,
                maximum_accuracy=0.25,
                corruption_fn=corruption,
            )

        build_server.assert_called_once_with(config)
        self.assertEqual(server.run_rounds_calls, [(3, False)])
        self.assertEqual(len(server.evaluated_datasets), 4)
        self.assertEqual(result["selected"]["noise"]["severity"], 2)
        self.assertEqual(result["selected"]["fog"]["severity"], 2)
        self.assertAlmostEqual(result["selected"]["noise"]["accuracy"], 0.2)
        self.assertAlmostEqual(result["selected"]["fog"]["accuracy"], 0.2)
        self.assertEqual(result["dataset"], "tiny")
        self.assertEqual(result["seed"], 17)
        self.assertEqual(result["corrupted_test_seed"], 10_016)
        self.assertRegex(result["clean_checkpoint_digest"], r"^[0-9a-f]{64}$")


if __name__ == "__main__":
    unittest.main()
