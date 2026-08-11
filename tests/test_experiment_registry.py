import unittest
from pathlib import Path

from experiments.shared.registry import (
    EXPERIMENTS,
    get_experiment,
    output_path,
    temporary_output_path,
)


class TestExperimentRegistry(unittest.TestCase):
    def test_registers_all_nine_experiments(self):
        self.assertEqual(
            [spec.experiment_id for spec in EXPERIMENTS],
            [
                "e01-static",
                "e02-ablation",
                "e03-timeout",
                "e04-static-comparison",
                "e05-temporal-drift",
                "e06-drift-agent-prototype",
                "e07-drift-agent",
                "e08-partial-drift",
                "e09-gradual-drift",
            ],
        )

    def test_e09_uses_the_new_results_namespace(self):
        self.assertEqual(
            output_path("e09-gradual-drift", "cifar-10"),
            Path("results/e09-gradual-drift/cifar-10"),
        )
        self.assertEqual(
            temporary_output_path("e09-gradual-drift", "cifar-10"),
            Path("output/e09-gradual-drift/cifar-10"),
        )

    def test_e07_uses_the_new_results_namespace(self):
        self.assertEqual(
            output_path("e07-drift-agent", "cifar-10"),
            Path("results/e07-drift-agent/cifar-10"),
        )

    def test_e08_uses_the_new_results_namespace(self):
        self.assertEqual(
            output_path("e08-partial-drift", "cifar-10"),
            Path("results/e08-partial-drift/cifar-10"),
        )

    def test_unknown_experiment_is_rejected(self):
        with self.assertRaises(ValueError):
            get_experiment("drift")

    def test_temporary_path_is_separate_from_published_path(self):
        self.assertEqual(
            temporary_output_path("e07-drift-agent", "cifar-10"),
            Path("output/e07-drift-agent/cifar-10"),
        )
        self.assertNotEqual(
            temporary_output_path("e07-drift-agent", "cifar-10"),
            output_path("e07-drift-agent", "cifar-10"),
        )

    def test_e08_temporary_path_is_separate_from_published_path(self):
        self.assertEqual(
            temporary_output_path("e08-partial-drift", "cifar-10"),
            Path("output/e08-partial-drift/cifar-10"),
        )
        self.assertNotEqual(
            temporary_output_path("e08-partial-drift", "cifar-10"),
            output_path("e08-partial-drift", "cifar-10"),
        )

    def test_dataset_must_be_one_directory_name(self):
        with self.assertRaises(ValueError):
            output_path("e07-drift-agent", "../cifar-10")
