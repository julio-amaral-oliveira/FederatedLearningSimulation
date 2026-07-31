import unittest
from pathlib import Path

from experiments.registry import temporary_output_path
from src.utils.data_loader import get_dataset_info


class TestOutputPaths(unittest.TestCase):
    def test_dataset_info_accepts_an_explicit_output_root(self):
        output_dir = temporary_output_path("e04-static-comparison", "cifar-10")

        info = get_dataset_info("cifar10", output_dir=output_dir)

        self.assertEqual(info["output_dir"], str(output_dir))

    def test_explicit_output_root_does_not_mutate_dataset_defaults(self):
        output_dir = Path("/tmp/test-output")

        get_dataset_info("cifar10", output_dir=output_dir)

        self.assertEqual(get_dataset_info("cifar10")["output_dir"], "output-cifar-10")


if __name__ == "__main__":
    unittest.main()
